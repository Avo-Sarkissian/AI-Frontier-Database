"""Workload MIX: the effective-price formula, its presets and every surface it reaches.

data/workload_mix.py prices the hosted catalogue for an input:output ratio and a
cache-hit rate. These tests pin the arithmetic against worked examples, the
provenance of every preset, the fill-in for missing cache prices, and that the
default mix changes nothing an untouched visitor sees.
"""
import base64
import io
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import static_api as api
from data.ingest import get_models
from data.workload_mix import (
    CACHE_FILL_RATIO, CUSTOM_RATIO_MAX, DEFAULT_MIX_KEY, MIX_PRESETS, apply_mix,
    effective_price, fmt_floor, fmt_price, mix_options, resolve_mix,
)
from static_helpers import ranking_frame

ROOT = Path(__file__).resolve().parent.parent
DF = get_models()
APP_JS = (ROOT / "docs" / "app.js").read_text()
NODE = shutil.which("node")


def _row(p_in, p_out, p_cache=float("nan")):
    return pd.DataFrame({"model": ["X"], "provider": ["P"], "quality": [50.0],
                         "price": [(p_in + 3 * p_out) / 4], "price_in": [p_in],
                         "price_out": [p_out], "cache_hit_price": [p_cache]})


def _price(frame, spec):
    return float(apply_mix(frame, resolve_mix(spec))["price"].iloc[0])


def _decoded(json_str):
    spec = json.loads(json_str)
    for trace in spec.get("data", []):
        for key, val in list(trace.items()):
            if isinstance(val, dict) and "bdata" in val and "dtype" in val:
                trace[key] = np.frombuffer(base64.b64decode(val["bdata"]),
                                           dtype=np.dtype(val["dtype"])).tolist()
    return spec


# ── The formula ──────────────────────────────────────────────────────────────

def test_default_mix_reproduces_the_stored_price_exactly():
    """The default must not move a single number: apply_mix hands back the same
    object. The weight form of the formula also reproduces the stored blend —
    to within one float ulp, because `price` went through a CSV text round trip
    (the reason the default short-circuits instead of recomputing)."""
    assert apply_mix(DF, resolve_mix(None)) is DF
    assert apply_mix(DF, None) is DF
    priced = DF[DF["price_in"].notna() & DF["price_out"].notna()]
    assert len(priced) > 50
    mix = resolve_mix(None)
    eff, filled = effective_price(priced["price_in"], priced["price_out"],
                                  priced["cache_hit_price"], mix)
    assert abs(eff.values - priced["price"].values).max() < 1e-12
    assert not filled.any()


def test_semianalysis_worked_example():
    """SemiAnalysis's 'AI Value Capture' (2026-05-01): a $5 / $25 model with a
    $0.50 cache, at 300:1 and 90% cached, costs ~$0.99/M. The formula gives
    310/301 = $1.030, within 5% of their figure."""
    got = _price(_row(5.0, 25.0, 0.5), "agentic")
    assert got == pytest.approx(310 / 301, abs=1e-12)
    assert abs(got - 0.99) / 0.99 < 0.05


def test_opus_55_presets():
    """Claude Opus 5.5: $4 in, $20 out, AA cache $0.20."""
    frame = _row(4.0, 20.0, 0.2)
    assert _price(frame, "ours") == pytest.approx(16.0)
    assert _price(frame, "chat") == pytest.approx(12.0)
    assert _price(frame, "aa") == pytest.approx(8.0)
    assert _price(frame, "observed") == pytest.approx((15 * 4 + 20) / 16)   # 5.00
    assert _price(frame, "agentic") == pytest.approx((300 * (0.9 * 0.2 + 0.1 * 4) + 20) / 301)
    assert _price(frame, "agentic") == pytest.approx(0.64452, abs=1e-5)


def test_h_zero_never_reads_the_cache_price():
    frame = _row(4.0, 20.0)          # no cache price
    for key in ("chat", "aa", "observed", ["custom", 50, 0]):
        out = apply_mix(frame, resolve_mix(key))
        assert math.isfinite(out["price"].iloc[0])
        assert not out["cache_price_filled"].any(), key


def test_missing_cache_is_filled_at_ten_percent_and_flagged():
    out = apply_mix(_row(4.0, 20.0), resolve_mix("agentic"))
    want = (300 * (0.9 * CACHE_FILL_RATIO * 4 + 0.1 * 4) + 20) / 301
    assert out["price"].iloc[0] == pytest.approx(want)
    assert bool(out["cache_price_filled"].iloc[0]) is True
    assert out["price_list_blend"].iloc[0] == pytest.approx(16.0)


def test_a_published_cache_price_beats_the_fill():
    published = apply_mix(_row(4.0, 20.0, 0.2), resolve_mix("agentic"))
    filled = apply_mix(_row(4.0, 20.0), resolve_mix("agentic"))
    assert bool(published["cache_price_filled"].iloc[0]) is False
    assert published["price"].iloc[0] < filled["price"].iloc[0]


def test_an_output_only_mix_ignores_the_cache_entirely():
    out = apply_mix(_row(4.0, 20.0), resolve_mix(["custom", 0, 90]))
    assert out["price"].iloc[0] == pytest.approx(20.0)
    assert not out["cache_price_filled"].any()


# ── Presets carry their provenance ───────────────────────────────────────────

def test_every_preset_cites_a_source_and_date():
    keys = [p.key for p in MIX_PRESETS]
    assert keys[0] == DEFAULT_MIX_KEY == "ours"
    assert set(keys) == {"ours", "chat", "aa", "observed", "agentic"}
    for p in MIX_PRESETS:
        assert p.source_url.startswith("https://"), p.key
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.as_of), p.key
        assert p.status in {"published", "reported", "derived", "estimated"}, p.key
        assert p.note.strip(), p.key
    by = {p.key: p for p in MIX_PRESETS}
    # Quoted, never scraped: the OpenRouter rankings page is off-limits, and the
    # observed preset is the published State of AI study instead.
    assert "quoted" in by["observed"].note.lower()
    assert "openrouter.ai/state-of-ai" in by["observed"].source_url
    assert "rankings" not in by["observed"].source_url
    assert (by["observed"].w_in, by["observed"].w_out) == (15.0, 1.0)
    assert "estimate" in by["agentic"].note.lower()
    assert "estimate" in by["chat"].note.lower()
    assert (by["agentic"].w_in, by["agentic"].w_out, by["agentic"].cache_hit) == (300.0, 1.0, 0.9)


def test_mix_options_ship_sources_and_end_with_custom():
    opts = mix_options()
    assert opts[0]["value"] == DEFAULT_MIX_KEY and opts[-1]["value"] == "custom"
    for o in opts[:-1]:
        assert "https://" in o["title"], o
        assert resolve_mix(o["value"]).key == o["value"]
    assert resolve_mix(opts[-1]["value"]).key == "custom"


def test_no_preset_uses_the_banned_rankings_count():
    """The 48:1 hand count from OpenRouter's rankings page is banned (terms)."""
    src = (ROOT / "data" / "workload_mix.py").read_text()
    assert "48:1" not in src and "openrouter.ai/rankings" not in src


# ── Input sanitising ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec", [
    None, "", "turbo", 42, ["turbo", 1, 2], {"m": "nope"}, object(),
])
def test_resolve_mix_falls_back_to_the_default(spec):
    assert resolve_mix(spec).key == DEFAULT_MIX_KEY


def test_resolve_mix_sanitises_hostile_custom_input():
    m = resolve_mix(["custom", -5, 150])
    assert (m.ratio, m.cache_hit) == (0.0, 1.0)
    m = resolve_mix(["custom", float("nan"), float("inf")])
    default = resolve_mix(None)
    assert m.ratio == pytest.approx(default.ratio) and m.cache_hit == 0.0
    m = resolve_mix(["custom", 1e9, "50"])
    assert m.ratio == CUSTOM_RATIO_MAX and m.cache_hit == 0.5
    m = resolve_mix({"m": "custom", "mr": "20", "mh": "50"})
    assert (m.ratio, m.cache_hit) == (20.0, 0.5)
    assert m.short == "custom 20:1 in:out · 50% cached"


def test_mixing_never_adds_drops_or_reorders_rows():
    full = ranking_frame(DF)
    for spec in ("chat", "aa", "observed", "agentic", ["custom", 7, 33]):
        out = apply_mix(full, resolve_mix(spec))
        assert out.index.equals(full.index)
        assert list(out["model"]) == list(full["model"])
        unpriced = full["price_in"].isna() | full["price_out"].isna()
        assert out.loc[unpriced, "price"].isna().all(), spec


# ── The data still supports the fill-in ──────────────────────────────────────

def test_fill_ratio_is_still_representative():
    """CACHE_FILL_RATIO rests on AA's own median cache/input ratio (0.10 over
    120 models on 2026-10-27). If the market moves, this fails and points at
    the constant instead of the site quietly using a stale guess."""
    has = DF["cache_hit_price"].notna() & DF["price_in"].gt(0)
    assert has.mean() >= 0.5, (
        f"only {has.mean():.0%} of rows carry an AA cache price — CACHE_FILL_RATIO "
        "would be doing most of the work; check the scraper's cache_hit_price")
    med = float((DF.loc[has, "cache_hit_price"] / DF.loc[has, "price_in"]).median())
    assert 0.05 <= med <= 0.20, (
        f"median AA cache/input ratio is {med:.3f}; revisit "
        "data/workload_mix.CACHE_FILL_RATIO and its source comment")


def test_the_mix_changes_the_answer():
    """The point of the control: at an input-heavy mix the value ranking is a
    different list, not a rescaled copy of the default one."""
    def top(spec):
        f = apply_mix(DF, resolve_mix(spec))
        f = f[(f["quality"] >= 20) & (f["price"] > 0)]
        return list((f["quality"] / f["price"]).nlargest(15).index)
    assert top(None) != top("observed")
    assert top(None) != top("agentic")


# ── Display ──────────────────────────────────────────────────────────────────

def test_fmt_price_never_rounds_a_sub_cent_price_to_zero():
    assert fmt_price(16) == "$16.00"
    assert fmt_price(0.08) == "$0.080"
    assert fmt_price(0.0041196) == "$0.00412"
    assert fmt_price(float("nan")) == "—"
    assert fmt_floor(0.08) == "$0.080" and fmt_floor(0.0041196) == "$0.00412"


def test_sub_cent_ticks_are_not_rounded_up():
    from components.charts.pareto import _log_price_ticks
    vals, text = _log_price_ticks(0.004, 1.0)
    for v, t in zip(vals, text):
        if v < 0.01:
            assert t == f"${v:g}", (v, t)
    # Cent-and-up keep the old format, so the default figure is unchanged.
    assert "$0.10" in text or "$0.20" in text or "$0.50" in text


# ── static_api ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("call", [
    lambda m: api.update_overview(None, 0, "", "price", None, *m),
    lambda m: api.update_overview(None, 0, "", "speed", None, *m),
    lambda m: api.update_rankings(None, 0, "", "value", None, *m),
    lambda m: api.update_value_leaders(None, 0, "", None, *m),
    lambda m: api.update_cost_calc(1, None, 0, "", 0, None, *m),
    lambda m: api.update_table(None, 0, "", "quality", "desc", None, *m),
    lambda m: api.update_treemap(None, 0, "", None, *m),
    lambda m: api.update_provider_leaderboard(None, 0, "", None, *m),
])
def test_mix_none_is_byte_identical_to_omitting_it(call):
    assert call([]) == call([None]) == call([["ours", None, None]])


def test_mix_is_the_last_parameter_of_every_hosted_function():
    """pyworker.js spreads the browser's list with no arity check."""
    import inspect
    for fn in (api.update_overview, api.update_treemap, api.update_provider_leaderboard,
               api.update_rankings, api.update_value_leaders, api.update_compare,
               api.update_cost_calc, api.update_table, api.export_csv, api.model_detail):
        params = list(inspect.signature(fn).parameters)
        assert params[-1] == "mix", fn.__name__
        assert inspect.signature(fn).parameters["mix"].default is None


def test_mix_info_floor_matches_manifest_at_default():
    info = json.loads(api.mix_info(None))
    assert info["floor_price"] == f"${DF['price'].min():.3f}"
    assert info["floor_label"] == "Floor price / 1M · ours 1:3 in:out (not AA's)"
    assert "not AA's" in info["table_header"]
    agentic = json.loads(api.mix_info(["agentic", None, None]))
    assert float(agentic["floor_price"].lstrip("$")) < float(info["floor_price"].lstrip("$"))
    assert "agentic" in agentic["floor_label"] and "agentic" in agentic["table_header"]
    assert agentic["n_filled"] > 0 and agentic["n_priced"] == info["n_priced"]


def test_detail_panel_names_cache_source_and_mix():
    with_cache = DF[DF["cache_hit_price"].notna()].iloc[0]
    without = DF[DF["cache_hit_price"].isna() & DF["price_in"].notna()].iloc[0]
    a = api.model_detail(with_cache["model"], with_cache["provider"], ["agentic", None, None])
    b = api.model_detail(without["model"], without["provider"])
    assert "Cached in" in a and "AA" in a and "agentic" in a
    assert "est. 10% of input" in b and "not AA" in b


def test_export_at_a_mix_carries_its_basis():
    default = pd.read_csv(io.StringIO(api.export_csv(None, 0, "", "table")))
    assert "price_basis" not in default.columns
    mixed = pd.read_csv(io.StringIO(api.export_csv(None, 0, "", "table", None, None,
                                                   ["agentic", None, None])))
    assert {"price_basis", "price_list_blend", "cache_price_filled"} <= set(mixed.columns)
    assert mixed["price_basis"].str.contains("agentic").all()
    want = apply_mix(DF, resolve_mix("agentic")).set_index("model")["price"]
    got = mixed.set_index("model")["price"]
    assert (got - want.loc[got.index]).abs().max() < 1e-9
    assert (mixed["price_list_blend"] - default.set_index("model").loc[
        mixed["model"], "price"].values).abs().max() < 1e-9


def test_update_cost_calc_best_moves_with_mix():
    base = json.loads(api.update_cost_calc(100, None, 0, "", 40))
    agentic = json.loads(api.update_cost_calc(100, None, 0, "", 40, None, ["agentic", None, None]))
    assert agentic["best"]["monthly_cost"] < base["best"]["monthly_cost"]
    assert agentic["basis"].startswith("agentic") and "not AA's" in base["basis"]


def test_the_agentic_floor_is_flagged_when_it_rests_on_a_fill_in():
    info = json.loads(api.mix_info(["agentic", None, None]))
    row = DF[DF["model"] == info["floor_model"]].iloc[0]
    assert info["floor_filled"] == bool(pd.isna(row["cache_hit_price"]))


# ── Static site wiring ───────────────────────────────────────────────────────

_HOSTED = ("update_overview", "update_treemap", "update_provider_leaderboard",
           "update_rankings", "update_value_leaders", "update_cost_calc",
           "update_table", "update_compare")


@pytest.mark.parametrize("fn", _HOSTED + ("export_csv",))
def test_every_hosted_refresher_sends_the_mix(fn):
    calls = re.findall(r'callPy(?:Raw)?\("' + fn + r'",([^)]*)\)', APP_JS)
    assert calls, f"docs/app.js never calls {fn}"
    for args in calls:
        assert args.strip().split(",")[-1].strip() == "m", (
            f"{fn} is called without the mix as its LAST argument: {args!r}")


def test_the_detail_panel_and_floor_stat_follow_the_mix():
    assert 'callPyRaw("model_detail", model, provider, readMix())' in APP_JS
    assert 'callPy("mix_info", mix)' in APP_JS
    body = APP_JS.split("async function rerenderActiveFilterCharts")[1][:1200]
    assert "refreshMixInfo()" in body


def _js_function(name: str) -> str:
    start = APP_JS.index(f"function {name}(")
    depth, i = 0, APP_JS.index("{", start)
    while True:
        ch = APP_JS[i]
        depth += ch == "{"
        depth -= ch == "}"
        if ch == "}" and depth == 0:
            return APP_JS[start:i + 1]
        i += 1


def test_state_url_writes_the_mix_only_when_it_is_not_the_default():
    if not NODE:
        pytest.skip("node not available")
    script = _js_function("stateUrl") + """
      var location = { origin: "https://x.io", pathname: "/", search: "?m=chat&v=1" };
      var window = { AF: { manifest: { default_mix: "ours" }, state: {
        tab: "overview", providers: [], minQuality: 0, effort: "", mix: ["ours", null, null] } } };
      const a = stateUrl();
      window.AF.state.mix = ["agentic", null, null];
      const b = stateUrl();
      window.AF.state.mix = ["custom", 20, 50];
      const c = stateUrl();
      process.stdout.write(JSON.stringify([a, b, c]));
    """
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    a, b, c = json.loads(out.stdout)
    assert a == "https://x.io/?v=1&tab=overview"
    assert b == "https://x.io/?v=1&tab=overview&m=agentic"
    assert c == "https://x.io/?v=1&tab=overview&m=custom&mr=20&mh=50"


def test_the_url_mix_is_validated_against_the_manifest():
    body = _js_function("applyUrlState")
    assert 'u.get("m")' in body and "mix_options" in body
    assert 'getElementById("filter-mix")' in APP_JS


def test_fmt_price_agrees_between_python_and_js():
    if not NODE:
        pytest.skip("node not available")
    vals = [16, 1, 0.999, 0.08, 0.0123, 0.00412, 0.0041196, 0]
    out = subprocess.run([NODE, "-e", _js_function("fmtPrice") +
                          f"process.stdout.write(JSON.stringify({vals}.map(fmtPrice)))"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [fmt_price(v) for v in vals]


# ── Dash parity ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dash_app():
    import app as module
    return module


def test_the_mix_setting_survives_a_shared_link(dash_app):
    assert dash_app.init_mix_from_url("?m=agentic") == ("agentic", None, None)
    assert dash_app.init_mix_from_url("?m=turbo") == ("ours", None, None)
    assert dash_app.init_mix_from_url("?m=custom&mr=20&mh=50") == ("custom", 20.0, 50.0)
    assert dash_app.init_mix_from_url("?m=custom&mr=-3&mh=500") == ("custom", 0.0, 100.0)
    assert dash_app.init_mix_from_url("") == ("ours", None, None)
    # init_from_url keeps its 4-tuple: shared-link tests destructure it.
    assert len(dash_app.init_from_url("?m=agentic")) == 4


@pytest.mark.parametrize("output", [
    "pareto-chart.figure", "treemap-chart.figure", "rankings-chart.figure",
    "value-leaders-chart.figure", "provider-leaderboard-chart.figure",
    "radar-chart.figure", "cost-calc-chart.figure", "model-table.data",
])
def test_every_dash_hosted_chart_listens_to_the_mix(dash_app, output):
    from dash._callback import GLOBAL_CALLBACK_MAP
    specs = [s for k, s in GLOBAL_CALLBACK_MAP.items() if output in k]
    assert specs, output
    ids = {d["id"] for d in specs[0]["inputs"]}
    assert {"filter-mix", "mix-ratio", "mix-cache"} <= ids, output


def test_dash_and_static_price_the_same_mix_identically(dash_app):
    mix = ("custom", 12, 40)
    dash_fig = json.loads(dash_app.update_value_leaders(None, 0, None, "", 0, *mix).to_json())
    static_fig = json.loads(api.update_value_leaders(None, 0, "", None, list(mix)))
    assert dash_fig["data"][0]["customdata"] == static_fig["data"][0]["customdata"]
    assert dash_fig["layout"]["title"]["text"] == static_fig["layout"]["title"]["text"]


def test_dash_csv_export_honours_the_mix(dash_app):
    out = dash_app.export_csv(1, "table", None, 0, "", None, None, None, None, None,
                              None, None, "agentic", None, None)
    exported = pd.read_csv(io.StringIO(out["content"]))
    assert exported["price_basis"].str.contains("agentic").all()


def test_dash_floor_stat_and_table_header_follow_the_mix(dash_app):
    floor, label, cols = dash_app.update_mix_labels(
        "agentic", None, None, [{"id": "price", "name": "Price"}, {"id": "model", "name": "Model"}])
    assert floor == json.loads(api.mix_info(["agentic", None, None]))["floor_price"]
    assert "agentic" in label and "agentic" in cols[0]["name"] and cols[1]["name"] == "Model"


# ── Build: manifest and pre-rendered figures ─────────────────────────────────

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import build_static
    out = tmp_path_factory.mktemp("figures")
    build_static.export_default_figures(out)
    return out


def test_the_manifest_carries_the_mix(built):
    m = json.loads((built / "manifest.json").read_text())
    assert m["default_mix"] == DEFAULT_MIX_KEY == m["mix_options"][0]["value"]
    for o in m["mix_options"]:
        assert resolve_mix(o["value"]).key == o["value"]
    assert m["floor_price_label"] == json.loads(api.mix_info(None))["floor_label"]
    assert m["floor_price"] == json.loads(api.mix_info(None))["floor_price"]
    assert m["table_price_header"] == json.loads(api.mix_info(None))["table_header"]


@pytest.mark.parametrize("fid,live", [
    ("pareto", lambda: api.update_overview(None, 0, "", "price", None, None)),
    ("quadrant", lambda: api.update_overview(None, 0, "", "speed", None, None)),
    ("rankings", lambda: api.update_rankings(None, 0, "", "intelligence", None, None)),
    ("value_leaders", lambda: api.update_value_leaders(None, 0, "", None, None)),
    ("treemap", lambda: api.update_treemap(None, 0, "", None, None)),
    ("provider_leaderboard", lambda: api.update_provider_leaderboard(None, 0, "", None, None)),
    ("cost_calc", lambda: json.dumps(json.loads(api.update_cost_calc(1, None, 0, "", 0, None, None))["figure"])),
])
def test_prerendered_figures_equal_the_default_live_render(built, fid, live):
    """The worker re-renders with the live default as soon as it boots. If the
    pre-rendered figure differs, the chart visibly changes a few seconds after
    every page load — so at the default mix they must be the same figure."""
    pre = _decoded((built / f"{fid}.json").read_text())
    now = _decoded(live())
    assert pre["layout"]["title"] == now["layout"]["title"]
    assert pre["layout"].get("xaxis", {}).get("title") == now["layout"].get("xaxis", {}).get("title")
    assert [t.get("x") for t in pre["data"]] == [t.get("x") for t in now["data"]]
    assert [t.get("hovertemplate") for t in pre["data"]] == [t.get("hovertemplate") for t in now["data"]]
