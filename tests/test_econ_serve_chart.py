"""Economics › Serve chart, the Economics tab scaffolding, and the $/GPU-HR
readout on Run Local and Agent Stack."""
import inspect
import json
import math
import re
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import build_static
import static_api as api
from captions import CAPTIONS
from components.charts.econ_serve import build_econ_serve, MEASURED_SYMBOL, MODELLED_SYMBOL
from data import serving_cost as sc
from static_helpers import (
    LOCAL_ARG_NAMES, TABS_WITHOUT_GLOBAL_FILTERS, ECON_EXPORT_FILENAMES,
    export_frame_for_tab, local_frame, gpu_preset_options,
)

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "docs" / "app.js").read_text()
INDEX = (ROOT / "docs" / "index.html").read_text()
APP_PY = (ROOT / "app.py").read_text()


def _fig(*args):
    return json.loads(api.update_econ_serve(*args))


def _points(fig):
    return [t for t in fig["data"] if t.get("customdata")]


# ── The chart ────────────────────────────────────────────────────────────────

def test_log_log_with_identity_diagonal():
    fig = _fig()
    xa, ya = fig["layout"]["xaxis"], fig["layout"]["yaxis"]
    assert xa["type"] == "log" and ya["type"] == "log"
    assert xa["range"] == ya["range"]
    diag = [t for t in fig["data"] if t.get("mode") == "lines"]
    assert len(diag) == 1
    lo, hi = xa["range"]
    assert diag[0]["x"] == diag[0]["y"]
    assert diag[0]["x"][0] == pytest.approx(10 ** lo)
    assert diag[0]["x"][-1] == pytest.approx(10 ** hi)


def test_axes_do_not_move_with_controls():
    ranges = {json.dumps([_fig(*a)["layout"]["xaxis"]["range"],
                          _fig(*a)["layout"]["yaxis"]["range"]])
              for a in ((None, None, None), ("NVIDIA H100 SXM", "rent", 10),
                        ("NVIDIA B200 SXM", "own", 100))}
    assert len(ranges) == 1


def test_measured_vs_modelled_symbols():
    fig = _fig()
    symbols = set()
    for t in _points(fig):
        sym = t["marker"]["symbol"]
        symbols |= set(sym if isinstance(sym, list) else [sym])
    assert symbols <= {MEASURED_SYMBOL, MODELLED_SYMBOL}
    assert not any("open" in s for s in symbols), "hollow is reserved for open weights"
    for t in fig["data"]:
        s = (t.get("marker") or {}).get("symbol")
        for v in (s if isinstance(s, list) else [s]):
            assert v is None or "open" not in str(v)
    keys = [t["name"] for t in fig["data"] if t.get("x") == [None]]
    assert any("measured" in k for k in keys) and any("modelled" in k for k in keys)


def test_modelled_points_are_fainter_than_measured():
    for t in _points(_fig()):
        for sym, op in zip(t["marker"]["symbol"], t["marker"]["opacity"]):
            assert op == (0.95 if sym == MEASURED_SYMBOL else 0.55)


def _frame_row(**kw):
    base = dict(model="<b>x</b>&", base_model="<b>x</b>&", provider="Meta",
                price_in=0.5, price_out=2.0, gpu="NVIDIA H100 SXM", gpu_count=2,
                basis="own", usd_per_gpu_hr=1.17, rate_source="SemiAnalysis InferenceX",
                rate_observed_on="2026-10-27", utilisation_pct=60.0, tok_s_per_gpu=500.0,
                usd_per_m_out=1.083, markup_out=1.85, markup_mix=5.5,
                evidence="measured", identity="same", ix_model="X%{y}",
                precision="fp8", framework="vllm", spec_method="none",
                deployment="single-node", interactivity_tok_s_user=40.0,
                run_date="2026-10-01",
                run_url="https://github.com/SemiAnalysisAI/InferenceX/actions/runs/123",
                calib_k=0.5, calib_models=3, calib_gsd=1.4, calib_fallback=False,
                sessions=float("nan"), per_session_tps=float("nan"), binding="",
                tp=2, ep=None)
    base.update(kw)
    return base


def test_hover_escapes_model_names():
    frame = pd.DataFrame([_frame_row()])
    fig = json.loads(build_econ_serve(frame, {"gpu": "cheapest"}, (-1, 1)).to_json())
    t = _points(fig)[0]
    hover = t["customdata"][0][0]
    assert "<b>x</b>&" not in hover and "&lt;b&gt;x&lt;/b&gt;&amp;" in hover
    assert "%{y}" not in hover, "a name must not smuggle a hovertemplate directive"
    assert t["customdata"][0][1] == "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/123"
    assert t["text"] == ["&lt;b&gt;x&lt;/b&gt;&amp;"]


def test_hover_names_both_markups_and_the_evidence():
    frame = pd.DataFrame([_frame_row(), _frame_row(
        base_model="m2", model="m2", evidence="modelled", run_url="", identity="",
        ix_model="", sessions=64, per_session_tps=36.0)])
    fig = json.loads(build_econ_serve(frame, {}, (-1, 1)).to_json())
    hovers = [cd[0] for t in _points(fig) for cd in t["customdata"]]
    assert all("on output price alone" in h and "Same-mix markup (8k in : 1k out)" in h
               for h in hovers)
    assert any("Measured · InferenceX" in h and "click to open the run" in h for h in hovers)
    assert any("Modelled · roofline, k = 0.50 fitted on 3 InferenceX models, residual ×1.4"
               in h for h in hovers)
    sib = json.loads(build_econ_serve(pd.DataFrame([_frame_row(identity="sibling")]), {},
                                      (-1, 1)).to_json())
    assert "(sibling checkpoint)" in _points(sib)[0]["customdata"][0][0]


def test_default_prerender_matches_live(tmp_path):
    figs = build_static.export_default_figures(tmp_path)
    assert "econ_serve.json" in figs
    pre = json.loads((tmp_path / "econ_serve.json").read_text())
    live = _fig(None, None, None)
    assert pre["data"] == live["data"]
    assert pre["layout"] == live["layout"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    opts = manifest["econ_serve_options"]
    assert opts["default_util_pct"] == sc.DEFAULT_UTILISATION_PCT
    assert opts["default_gpu"] == sc.DEFAULT_SERVE_GPU
    assert opts["default_basis"] == sc.DEFAULT_SERVE_BASIS
    assert [g["value"] for g in opts["gpus"]][0] == "cheapest"


def test_empty_frame_renders_empty_state():
    fig = build_econ_serve(pd.DataFrame(columns=sc.EXPORT_COLUMNS), {}, (-1, 1))
    assert fig.layout.annotations and "No open model" in fig.layout.annotations[0].text


def test_a_bad_control_value_falls_back_to_the_default():
    assert _fig("No Such GPU", "lease", "abc")["data"] == _fig()["data"]


# ── Scaffolding: tab, views, controls ────────────────────────────────────────

def test_app_js_economics_wiring():
    assert '{ id: "economics", label: "Economics",   charts: [] }' in APP_JS
    block = re.search(r"const TAB_CAPTIONS = \{(.*?)\n\};", APP_JS, re.S).group(1)
    assert 'economics: ["economics_serve", "economics_compute", "economics_open_gap"]' in block
    assert re.search(r'const TABS_WITHOUT_GLOBAL_FILTERS = \[[^\]]*"economics"', APP_JS)
    assert "economics" in TABS_WITHOUT_GLOBAL_FILTERS
    views = re.search(r"const ECON_VIEWS = \[(.*?)\n\];", APP_JS, re.S).group(1)
    assert "view:" in views and "{ id:" not in views
    assert 'view: "serve"' in views and 'view: "labs"' in views
    assert '"econ_open_gap_history", "econ_open_gap_lag", "econ_open_gap_price"' in views
    # no literal utilisation default near the slider: it comes from the manifest
    near = APP_JS[APP_JS.index("function fillEconControls"):][:1600]
    assert "o.default_util_pct" in near and not re.search(r"\b60\b", near)
    assert "econ-util" in APP_JS and "econ-serve-gpu" in APP_JS
    assert 'name="econ-serve-basis"' in INDEX and 'name="econ-view"' in INDEX
    # click opens only an InferenceX run
    assert 'const IX_RUN_PREFIX = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/"' in APP_JS
    assert 'window.open(url, "_blank", "noopener")' in APP_JS


def test_economics_controls_are_view_scoped_ids():
    for el in ('id="tab-controls-economics"', 'id="econ-shared-controls"',
               'id="econ-serve-controls"', 'id="econ-util"', 'id="econ-serve-gpu"'):
        assert el in INDEX
    # F's names must stay free
    assert 'id="econ-gpu"' not in INDEX and 'name="econ-basis"' not in INDEX


def test_the_economics_refresher_dispatches_per_view():
    assert "economics: () => (ECON_REFRESHERS[currentEconView()]" in APP_JS
    assert re.search(r'serve: makeRefresher\("economics-serve"', APP_JS)
    assert 'callPy("update_econ_serve", g, b, u)' in APP_JS
    assert "labs:" not in APP_JS.split("const ECON_REFRESHERS = {")[1].split("};")[0]


def test_tab_count_and_dash_parity():
    assert len(re.findall(r'\{ id: "\w+",', APP_JS)) == APP_PY.count("dcc.Tab(") == 11
    import app as dash_app
    assert "economics" in dash_app._VALID_TABS
    assert dash_app.init_from_url("?tab=economics")[0] == "economics"


def test_dash_economics_callback_draws_the_same_figure():
    import app as dash_app
    fig, label = dash_app.update_econ_serve(None, None, None)
    assert json.loads(fig.to_json())["data"] == _fig()["data"]
    assert label == "60%"
    styles = dash_app.toggle_econ_view("labs")
    assert styles[0] == {"display": "none"} and styles[1] == {}
    assert all(f is not None for f in styles[4:])


def test_readme_counts_eleven_tabs():
    readme = (ROOT / "README.md").read_text()
    assert "11 interactive tabs" in readme and "| **Economics** |" in readme


# ── ↓CSV ─────────────────────────────────────────────────────────────────────

def test_economics_export_is_the_serving_frame():
    frame, name = export_frame_for_tab("economics", api._DF, ["Nobody"], 99, "zzz",
                                       econ_args=["serve", "cheapest", "own", 60])
    assert name == "ai_frontier_serving_cost.csv" == ECON_EXPORT_FILENAMES["serve"]
    assert "usd_per_m_out" in frame.columns and "run_url" in frame.columns
    assert len(frame) == len(sc.serving_frame(hosted=api._DF))
    csv = api.export_csv(None, 0, "", "economics", None, None, None, None,
                         ["serve", "NVIDIA H100 SXM", "rent", 30])
    head = csv.splitlines()[0].split(",")
    assert head == sc.EXPORT_COLUMNS
    body = pd.read_csv(__import__("io").StringIO(csv))
    assert set(body["gpu"]) == {"NVIDIA H100 SXM"} and set(body["basis"]) == {"rent"}
    assert api.export_csv_filename("economics", ["serve"]) == "ai_frontier_serving_cost.csv"
    assert api.export_csv_filename("economics", ["bogus"]) == "ai_frontier_serving_cost.csv"


def test_export_csv_takes_econ_args_ninth():
    params = list(inspect.signature(api.export_csv).parameters)
    assert params.index("econ_args") == 8
    assert 'callPyRaw("export_csv", p, q, s, tab, e, local, m, w, econ)' in APP_JS
    assert list(inspect.signature(api.update_econ_serve).parameters) == [
        "gpu", "basis", "util_pct"]


# ── Run Local $/GPU-HR ───────────────────────────────────────────────────────

def test_run_local_rate_is_appended_last():
    assert LOCAL_ARG_NAMES[-1] == "gpu_rate"
    body = APP_JS[APP_JS.index("function localArgs()"):].split("\n}\n")[0]
    assert re.search(r"tags\.length \? tags : null, ctx, hw\.fp16_tflops \?\? null, rate\]",
                     body)
    assert ('window.AF.callPy("update_local", vram, gpus, quant, bw, hwType, tags, ctx, '
            'speedMode, fp16, rate)') in APP_JS
    assert list(inspect.signature(api.update_local).parameters)[-1] == "gpu_rate"


def test_local_frame_prices_the_on_screen_throughput():
    ldf, *_ = local_frame(80, 1, "Q8", 3350, "nvidia", None, 8192, 989.5, gpu_rate=2.0)
    ok = ldf["total_tps"] > 0
    assert ok.any()
    want = 2.0 * 1 * 1e6 / (ldf.loc[ok, "total_tps"] * 3600)
    assert ldf.loc[ok, "usd_per_m_out"].values == pytest.approx(want.values)
    assert ldf.loc[~ok, "usd_per_m_out"].isna().all()
    plain, *_ = local_frame(80, 1, "Q8", 3350, "nvidia", None, 8192, 989.5)
    assert plain["usd_per_m_out"].isna().all()


def test_without_a_rate_the_run_local_figures_are_unchanged():
    from data.local_models import (DEFAULT_VRAM_GB, DEFAULT_BANDWIDTH_GBPS,
                                   DEFAULT_CONTEXT_TOKENS, DEFAULT_FP16_TFLOPS)
    out = json.loads(api.update_local(DEFAULT_VRAM_GB, 1, "Q4", DEFAULT_BANDWIDTH_GBPS,
                                      "nvidia", None, DEFAULT_CONTEXT_TOKENS, None,
                                      DEFAULT_FP16_TFLOPS))
    pre = json.loads((ROOT / "docs" / "figures" / "local_compat.json").read_text())
    assert out["compat"]["data"] == pre["data"]
    assert "Serving cost" not in json.dumps(out)


def test_a_rate_adds_the_serving_cost_line_to_both_hovers():
    out = json.loads(api.update_local(80, 1, "Q8", 3350, "nvidia", None, 8192, "throughput",
                                      989.5, 2.0))
    for key in ("compat", "scatter"):
        txt = json.dumps(out[key], ensure_ascii=False)
        assert "Serving cost: $" in txt and "$2.00/GPU-hr ×1, full use" in txt
        assert "datacenter figures are on Economics" in txt


def test_gpu_options_carry_rates():
    opts = {o["value"]: o for o in gpu_preset_options()}
    for preset in sc.IX_HARDWARE_TO_PRESET.values():
        o = opts[preset]
        assert o["rate_own"] > 0 and o["rate_rent"] > 0 and o["rate_source"]
    assert opts["NVIDIA RTX 5090"]["rate_own"] is None
    assert opts["NVIDIA RTX 5090"]["rate_rent"] is None


def test_the_run_local_rate_box_is_filled_from_the_preset():
    assert 'id="local-gpu-rate"' in INDEX and 'name="local-rate-basis"' in INDEX
    body = APP_JS[APP_JS.index("function setRateFromPreset"):].split("\n}\n")[0]
    assert "hw.rate_own" in body and "hw.rate_rent" in body and "r.disabled = !has" in body
    assert 'id="local-gpu-rate"' in APP_PY.replace("id=\"local-gpu-rate\"", 'id="local-gpu-rate"')


# ── Agent Stack ──────────────────────────────────────────────────────────────

def test_agent_stack_prices_datacenter_local_and_not_consumer():
    dc = json.loads(api.update_recommend([], "hybrid", "NVIDIA H100 SXM", 80, 1, "Q4"))
    assert "/M out · rent, " in dc["cards_html"]
    consumer = json.loads(api.update_recommend([], "hybrid", "NVIDIA RTX 5090", 32, 1, "Q4"))
    assert "· rent, " not in consumer["cards_html"]
    assert "free" not in CAPTIONS["recommend"].lower()
    from components import stack_recommender
    assert "(free)" not in inspect.getsource(stack_recommender)


# ── Bundles ──────────────────────────────────────────────────────────────────

def test_new_modules_ship_to_the_browser():
    src = inspect.getsource(build_static.build_pybundle)
    for mod in ("data/econ_common.py", "data/serving_cost.py",
                "components/charts/econ_serve.py"):
        assert f'"{mod}"' in src
    assert "data/carried/econ/inferencex_rates.csv" in build_static.DATA_CSVS
    from data import open_gap, gpu_tco
    assert set(build_static.DERIVED_DATA) == {sc.DERIVED_POINTS, sc.DERIVED_CALIBRATION,
                                              open_gap.DERIVED_EXPORT, gpu_tco.DERIVED_WEEKLY,
                                              gpu_tco.DERIVED_SOFR, gpu_tco.DERIVED_EPOCH}
    assert (ROOT / ".gitignore").read_text().count("data/derived/") >= 1


def test_derived_files_and_the_data_zip_stay_small(tmp_path):
    for rel, compute in build_static.DERIVED_DATA.items():
        text = compute()
        limit = build_static.DERIVED_MAX_BYTES.get(rel, build_static.MAX_DERIVED_BYTES)
        assert len(text.encode()) < limit, rel
    build_static.build_data_bundle(tmp_path)
    z = tmp_path / build_static.DATA_BUNDLE
    assert z.stat().st_size < build_static.MAX_DATA_BUNDLE_BYTES
    with zipfile.ZipFile(z) as zz:
        assert sorted(zz.namelist()) == sorted(set(build_static.DATA_CSVS)
                                               | set(build_static.DERIVED_DATA))
        for rel, compute in build_static.DERIVED_DATA.items():
            assert zz.read(rel).decode() == compute()


def test_the_derived_slice_reads_back_as_the_computed_frame(tmp_path, monkeypatch):
    """Browser path (file present) and local path (computed) give one frame."""
    from data import econ_common
    computed = sc.fit_calibration()
    monkeypatch.setattr(econ_common, "ROOT", tmp_path)
    for rel, compute in build_static.DERIVED_DATA.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(compute())
    sc.reset_cache()
    try:
        assert sc.fit_calibration() == computed or all(
            (a.k, a.n_models, a.fallback) == (b.k, b.n_models, b.fallback)
            for a, b in zip(sc.fit_calibration().values(), computed.values()))
    finally:
        monkeypatch.undo()
        sc.reset_cache()
