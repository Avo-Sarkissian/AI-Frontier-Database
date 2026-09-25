"""The static site's request plumbing, URL state and export arguments.

Covers the review findings owned by docs/app.js + static_api/static_helpers:
coalesced refreshes (#2), in-tick GPU presets (#7), the per-tab memo (#8),
the Run Local / EFFORT export (#9, #10), Add to Compare (#15, #16), URL sync
(#18), first-visit sizing (#26), boot order (#28) and deferred scripts (#29).

The JS behaviour is exercised in node against the real functions pulled out of
docs/app.js, not by grepping for their names.
"""
import io
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import static_api as api
from static_helpers import (
    LOCAL_ARG_NAMES, export_frame_for_tab, gpu_hw_meta, gpu_preset_options, local_frame,
)

ROOT = Path(__file__).resolve().parent.parent
APP = (ROOT / "docs" / "app.js").read_text()
HTML = (ROOT / "docs" / "index.html").read_text()
NODE = shutil.which("node")


def _js_function(name: str) -> str:
    """Source of a top-level `function name(` in docs/app.js, braces balanced."""
    start = APP.index(f"function {name}(")
    depth, i = 0, APP.index("{", start)
    while True:
        ch = APP[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return APP[start:i + 1]
        i += 1


def _node(script: str):
    if not NODE:
        pytest.skip("node not available")
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


# ── #2 / #8 — one coalescing refresher per tab ───────────────────────────────

_HARNESS = """
const window = { AF: { pyReady: true } };
%s
const calls = [], draws = [];
let inputs = ["a"];
const pending = [];
const call = (...args) => new Promise(res => { calls.push(args); pending.push({ args, res }); });
const draw = (out) => { draws.push(out); };
const refresh = makeRefresher("t", () => inputs.slice(), call, draw);
const tick = () => new Promise(r => setTimeout(r, 0));
async function answer() {            // resolve the oldest in-flight call
  const p = pending.shift();
  p.res("out:" + p.args[0]);
  await tick(); await tick();
}
"""


def _refresher_script(body: str) -> str:
    return (_HARNESS % _js_function("makeRefresher")) + "(async () => {\n" + body + \
        "\nprocess.stdout.write(JSON.stringify({ calls, draws }));\n})();"


def test_rapid_changes_run_one_call_in_flight_then_only_the_latest():
    """Five changes while one update_local runs: the worker sees exactly two
    calls, the second with the LATEST inputs, and the stale answer is never
    drawn (the old code queued and drew all six)."""
    got = _node(_refresher_script("""
      refresh();                                    // call #1 with "a"
      for (const v of ["b", "c", "d", "e", "f"]) { inputs = [v]; refresh(); }
      await tick();
      if (calls.length !== 1) throw new Error("more than one call in flight");
      await answer();                               // "a" is stale: dropped
      await answer();                               // "f" is drawn
    """))
    assert got["calls"] == [["a"], ["f"]]
    assert got["draws"] == ["out:f"]


def test_inputs_moved_inside_the_debounce_window_drop_the_stale_answer():
    """Every control reaches refresh() through a 150 ms debounce, so the DOM
    can move before any new request is counted. An answer for inputs the user
    has already left is still stale and must not be drawn."""
    got = _node(_refresher_script("""
      refresh(); await tick();                      // call #1 with "a"
      inputs = ["b"];                               // debounce still pending
      await answer();                               // "a" dropped, reruns "b"
      await answer();
      refresh(); await tick();                      // debounce fires: memo hit
    """))
    assert got["calls"] == [["a"], ["b"]]
    assert got["draws"] == ["out:b"]


def test_unchanged_inputs_skip_python_and_the_redraw():
    """A tab switch with nothing changed must not re-run update_local."""
    got = _node(_refresher_script("""
      refresh(); await tick(); await answer();
      refresh(); await tick();                      // same inputs: memo hit
      refresh(); await tick();
    """))
    assert got["calls"] == [["a"]]
    assert got["draws"] == ["out:a"]


def test_a_repeat_request_with_the_same_inputs_still_draws():
    """A newer request whose inputs did not move is not "stale"."""
    got = _node(_refresher_script("""
      refresh(); await tick();
      refresh();                                    // same inputs, mid-flight
      await answer();
    """))
    assert got["calls"] == [["a"]]
    assert got["draws"] == ["out:a"]


def test_a_failed_call_does_not_poison_the_memo():
    got = _node(_refresher_script("""
      const realCall = call;
      let fail = true;
      const r2 = makeRefresher("t2", () => inputs.slice(),
        (...a) => fail ? Promise.reject(new Error("boom")) : realCall(...a), draw);
      console.error = () => {};
      await r2(); fail = false;
      r2(); await tick(); await answer();
    """))
    assert got["calls"] == [["a"]]
    assert got["draws"] == ["out:a"]


def test_every_tab_refresh_goes_through_the_refresher():
    body = APP.split("async function rerenderActiveFilterCharts")[1].split("\n}\n")[0]
    assert "refreshers[tab]" in body
    for tab in ("overview", "landscape", "rankings", "budget", "table",
                "local", "image", "video", "recommend"):
        assert re.search(rf"\n  {tab}: makeRefresher\(", APP), f"{tab} is not coalesced"


def test_every_run_local_control_is_debounced():
    body = APP.split("function wireTabControls")[1]
    assert "LOCAL_DEBOUNCE_MS = 150" in APP
    for control in ("local-num-gpus", "local-quant", "local-context",
                    "local-speed-mode", "local-tags"):
        assert f'"{control}"' in body
    assert "el.onchange = scheduleLocal" in body
    # none of them calls refreshLocal directly any more
    assert not re.search(r"\.onchange = \(\) => refreshLocal\(\)", APP)


# ── #7 — GPU presets resolve in the browser, in the same tick ────────────────

def test_gpu_options_ship_each_presets_hardware():
    opts = json.loads(api.gpu_options())
    assert opts, "no GPU presets"
    for o in opts:
        meta = json.loads(api.local_hw_for_gpu(o["value"]))
        for k in ("vram_gb", "bandwidth_gbps", "hw_type", "fp16_tflops"):
            assert o[k] == meta[k], f"{o['value']} ships a different {k}"
        assert o["label"] and o["value"]


def test_an_unknown_preset_has_no_hardware():
    assert gpu_hw_meta("No Such Card") is None
    assert json.loads(api.local_hw_for_gpu("No Such Card")) is None


def test_the_preset_handlers_never_wait_on_the_worker():
    code = re.sub(r"^\s*//.*$", "", APP, flags=re.M)
    assert 'callPy("local_hw_for_gpu"' not in code, (
        "a preset change still queues a lookup behind update_local"
    )
    body = APP.split("function wireTabControls")[1]
    assert "window.AF.gpuMeta[localGpu.value]" in body
    assert "window.AF.gpuMeta[recGpu.value]" in body


def test_boot_default_never_overwrites_a_typed_vram():
    """setVramFromPreset(…, false) is the boot path; true is a real preset change."""
    fn = _js_function("setVramFromPreset")
    got = _node(fn + """
      const el = { value: "100", dataset: { userTyped: "1" } };
      const document = { getElementById: () => el };
      setVramFromPreset("local-vram", { vram_gb: 32 }, false);
      const afterBoot = el.value;
      setVramFromPreset("local-vram", { vram_gb: 24 }, true);
      process.stdout.write(JSON.stringify([afterBoot, el.value, el.dataset.userTyped || null]));
    """.replace("const document", "var document"))
    assert got == ["100", 24, None]


# ── #9 / #10 — the export describes what is on screen ───────────────────────

def _csv(text):
    return pd.read_csv(io.StringIO(text))


def test_the_export_honours_effort():
    full = _csv(api.export_csv([], 0, ""))
    high = _csv(api.export_csv([], 0, "", None, "high"))
    table = json.loads(api.update_table([], 0, "", "quality", "desc", "high"))
    assert len(high) == len(table), "↓CSV and the Table disagree under EFFORT"
    if len(high) == len(full):
        pytest.skip("this catalogue has no effort variants to collapse")
    assert len(high) < len(full)


def test_the_local_export_uses_the_selected_hardware():
    small = [8, 1, "Q4", 272, "nvidia", None, 8192, None]
    big = [80, 8, "Q4", 3350, "nvidia", None, 8192, None]
    a = _csv(api.export_csv([], 0, "", "local", None, small))
    b = _csv(api.export_csv([], 0, "", "local", None, big))
    fits_col = next(c for c in a.columns if c.startswith("fit"))
    assert (a[fits_col] != b[fits_col]).any(), "the export ignored the hardware"
    # and it is the same frame update_local plots
    ldf = local_frame(*small)[0]
    assert len(a) == len(ldf)


# ── ↓CSV stays reachable on the tabs that hide the global filters ───────────

STYLE = (ROOT / "assets" / "style.css").read_text()
APP_PY = (ROOT / "app.py").read_text()


def test_the_export_button_is_in_the_global_bar_that_gets_hidden():
    bar = HTML[HTML.index('<div class="filters">'):]
    bar = bar[:bar.index('<div class="tabs"')]
    assert 'id="btn-export"' in bar and 'class="export-btn"' in bar


def test_filterless_tabs_hide_the_filters_but_not_the_export():
    """showTabControls set display:none on the whole bar, taking ↓CSV with it:
    the Run Local / Image / Video exports were fixed but unclickable."""
    fn = _js_function("showTabControls")
    assert 'style.display =\n      TABS_WITHOUT_GLOBAL_FILTERS' not in fn
    got = _node("const TABS_WITHOUT_GLOBAL_FILTERS = %s;\n" % json.dumps(
        ["recommend", "local", "image", "video"]) + fn + """
      const cls = new Set();
      const bar = { style: {}, classList: { toggle: (c, on) => on ? cls.add(c) : cls.delete(c) } };
      var document = {
        querySelector: () => bar, querySelectorAll: () => [],
        getElementById: () => null,
      };
      const seen = {};
      for (const t of ["overview", "local", "image", "video", "recommend", "table"]) {
        try { showTabControls(t); } catch (e) {}
        seen[t] = [cls.has("export-only"), bar.style.display || ""];
      }
      process.stdout.write(JSON.stringify(seen));
    """)
    for t in ("local", "image", "video", "recommend"):
        assert got[t] == [True, ""], t
    for t in ("overview", "table"):
        assert got[t] == [False, ""], t
    # The class hides every control but the export button, in both renderings.
    assert re.search(r"\.filters\.export-only > :not\(\.export-btn\)\s*\{\s*display:\s*none", STYLE)
    assert 'Output("global-filters", "className")' in APP_PY
    assert '"filters export-only"' in APP_PY


def test_the_agent_stack_export_ignores_the_filters_it_hides():
    frame, _ = export_frame_for_tab("recommend", api._DF, ["Nobody"], 99, "zzz")
    assert len(frame) == len(api._DF)


def test_local_args_accept_a_dict_for_dash():
    kwargs = dict(zip(LOCAL_ARG_NAMES, [8, 1, "Q4", 272, "nvidia", None, 8192, None]))
    as_list, _ = export_frame_for_tab("local", None, None, 0, "",
                                      local_args=list(kwargs.values()))
    as_dict, _ = export_frame_for_tab("local", None, None, 0, "", local_args=kwargs)
    pd.testing.assert_frame_equal(as_list, as_dict)


def test_the_browser_sends_effort_and_the_local_hardware():
    assert 'callPyRaw("export_csv", p, q, s, tab, e, local)' in APP
    assert 'tab === "local" ? localArgs() : null' in APP
    # update_local reads the same builder, so the two cannot drift
    assert "...localArgs()" in APP


def test_local_arg_order_matches_update_local():
    import inspect
    params = list(inspect.signature(api.update_local).parameters)
    # speed_mode is update_local-only; the rest is LOCAL_ARG_NAMES in order
    shared = [p for p in params if p != "speed_mode"]
    assert tuple(shared) == LOCAL_ARG_NAMES


# ── #15 / #16 — Add to Compare ───────────────────────────────────────────────

def test_add_to_compare_goes_through_switch_tab_with_an_explicit_selection():
    body = APP.split('getElementById("detail-add-compare").onclick')[1].split("\n  };")[0]
    assert 'switchTab("compare")' in body
    assert "window.AF._compareWanted = keep" in body
    assert 'style.display = t.id === "compare"' not in body, "panels toggled by hand again"
    assert "wanted" in APP.split("async function refreshCompare")[1][:200]


# ── #18 — the URL follows the state ──────────────────────────────────────────

def test_state_url_carries_tab_filters_and_effort_and_keeps_foreign_params():
    got = _node(_js_function("stateUrl") + """
      var location = { origin: "https://x.io", pathname: "/AI/", search: "?v=123&tab=table&e=low" };
      var window = { AF: { state: { tab: "local", providers: ["Anthropic", "Google"],
                                    minQuality: 42.1, effort: "high" } } };
      const a = stateUrl();
      window.AF.state = { tab: "overview", providers: [], minQuality: 0, effort: "" };
      const b = stateUrl();
      process.stdout.write(JSON.stringify([a, b]));
    """)
    a, b = got
    assert a == "https://x.io/AI/?v=123&tab=local&p=Anthropic%2CGoogle&q=42.1&e=high"
    assert b == "https://x.io/AI/?v=123&tab=overview"


def test_the_url_is_synced_on_every_rerender_and_effort_is_restored():
    body = APP.split("async function rerenderActiveFilterCharts")[1][:900]
    assert body.index("syncUrl()") < body.index("if (!window.AF.pyReady) return;"), (
        "the URL only follows the state once Python has booted"
    )
    restore = _js_function("applyUrlState")
    assert 'u.get("e")' in restore and "effort_options" in restore


# ── #26 / #28 / #29 — load order and first paint ─────────────────────────────

def test_tabs_are_sized_in_the_next_frame_and_drawn_on_first_visit():
    body = _js_function("switchTab")
    assert "requestAnimationFrame" in body
    assert "setTimeout" not in body, "the 60 ms timer is back"
    assert "loadStaticFigures(id)" in body


def test_pyodide_boots_before_any_figure_is_drawn():
    body = _js_function("init")
    assert body.index("bootPyodide()") < body.index("loadStaticFigures(")
    assert "Promise.all(chartLoads)" not in body
    assert "uiReady" in body and "uiReady" in _js_function("bootPyodide")


def test_both_scripts_are_deferred_in_order():
    tags = re.findall(r"<script\b[^>]*>", HTML)
    srcs = [t for t in tags if "src=" in t]
    assert len(srcs) == 2
    assert all(re.search(r"\bdefer\b", t) for t in srcs), srcs
    assert "plotly" in srcs[0] and "app.js" in srcs[1], "execution order changed"


def test_image_catalogue_reads_open_weights_without_a_futurewarning(tmp_path, monkeypatch):
    """fillna(False) on the object column warned on every update_image call."""
    import warnings
    import data.image_models as im
    src = pd.read_csv(im._CACHE).head(3).copy()
    src["open_weights"] = ["True", None, "False"]
    path = tmp_path / "img.csv"
    src.to_csv(path, index=False)
    monkeypatch.setattr(im, "_CACHE", path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        df = im.get_image_df()
    assert df["open_weights"].tolist() == [True, False, False]
    assert df["open_weights"].dtype == bool
