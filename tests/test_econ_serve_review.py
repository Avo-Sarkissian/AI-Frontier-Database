"""Regression tests for the Economics › Serve review findings (2026-10-27).

Each test names the defect it pins: the claim the chart made, the export it
handed out, or the number it computed, and what it does now.
"""
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
from components.charts import econ_serve
from components.charts.econ_serve import build_econ_serve, label_positions
from data import econ_common, local_models as lm, open_gap, serving_cost as sc
from static_helpers import ECON_EXPORT_FILENAMES, econ_export, econ_view_args

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "docs" / "app.js").read_text()
INDEX = (ROOT / "docs" / "index.html").read_text()
APP_PY = (ROOT / "app.py").read_text()
README = (ROOT / "README.md").read_text()


def _fig(*args):
    return json.loads(api.update_econ_serve(*args))


def _point_traces(fig):
    return [t for t in fig["data"] if t.get("customdata")]


def _hovers(fig):
    return [cd[0] for t in _point_traces(fig) for cd in t["customdata"]]


# ── 1. The 35 tok/s floor on a measured point is a MEDIAN ───────────────────

def test_measured_hover_says_median_user_not_every_user():
    """InferenceX's 8k/1k views publish MEDIAN interactivity; the filter is on
    it, so "every user ≥ 35 tok/s" overclaimed for every measured point."""
    for args in ((None, None, None), ("NVIDIA B200 SXM", "own", 60)):
        hovers = _hovers(_fig(*args))
        measured = [h for h in hovers if "Measured · InferenceX" in h]
        modelled = [h for h in hovers if "Modelled · roofline" in h]
        assert measured and modelled
        assert all("median user ≥ 35 tok/s (InferenceX median interactivity)" in h
                   for h in measured)
        assert all("every session ≥ 35 tok/s" in h for h in modelled)
        assert not any("every user" in h for h in hovers)


def test_no_surface_claims_every_user():
    cap = CAPTIONS["economics_serve"]
    assert "MEDIAN user" in cap and "every user" not in cap
    row = next(l for l in README.splitlines() if l.startswith("| **Economics**"))
    assert "median user ≥ 35 tok/s" in row and "every user" not in row
    src = Path(sc.__file__).read_text()
    assert "every user" not in src


def test_the_derived_slice_only_holds_median_rows():
    pts = pd.read_csv(ROOT / sc.IX_POINTS_FILE)
    assert set(pts.loc[pts["sequence"] == "8k/1k", "x_field"]) == {"median_intvty"}


# ── 2. ↓CSV on Labs exports Labs ─────────────────────────────────────────────

def test_every_econ_view_has_its_own_export():
    """A view missing from ECON_EXPORT_FILENAMES fell back to Serve, so ↓CSV
    on Labs handed out the serving-cost table."""
    views = set(re.findall(r'^\s*\{\s*view:\s*"(\w+)"', APP_JS, re.M))
    assert {"serve", "labs"} <= views
    assert views <= set(ECON_EXPORT_FILENAMES)
    assert len(set(ECON_EXPORT_FILENAMES.values())) == len(ECON_EXPORT_FILENAMES)


def test_labs_export_is_the_labs_data():
    """Labs exports what Labs draws: the lab ledger, commitment multiples and
    Epoch rows (data/lab_economics.py), then the open-gap figures' rows."""
    name = api.export_csv_filename("economics", ["labs"])
    assert name == "ai_frontier_labs.csv"
    assert name != api.export_csv_filename("economics", ["serve"])
    # Dash sends the Serve controls along; they must not turn it into Serve.
    for args in (["labs"], ["labs", "NVIDIA H100 SXM", "rent", 40]):
        frame, fname = econ_export(args)
        assert fname == "ai_frontier_labs.csv"
        assert list(frame.columns)[0] == "chart"
        assert set(open_gap.EXPORT_CHARTS) <= set(frame["chart"])
        assert {"ledger", "runrate", "tokens", "commitment_multiple"} <= set(frame["chart"])
        assert "price_out" not in frame.columns          # not the Serve table
    assert econ_view_args(["labs", 1, 2]) == ("labs", [1, 2])
    assert econ_view_args(["nonsense"])[0] == "serve"


def test_labs_export_rows_match_the_figures():
    from data.ingest import get_models, load_history
    hist, df = load_history(), get_models()
    frame = open_gap.export_frame(hist, df)
    g, _ = open_gap.gap_history(hist)
    i, _ = open_gap.iso_quality_ratios(df, None)
    assert (frame["chart"] == "score_gap").sum() == len(g)
    assert (frame["chart"] == "iso_price").sum() == len(i)
    assert open_gap.export_frame(pd.DataFrame(), df.head(0)).columns.tolist()[0] == "chart"


# ── 3. Pre-render reads exactly what ships ───────────────────────────────────

def test_derived_points_slice_is_only_what_serving_reads():
    df = pd.read_csv(pd.io.common.StringIO(sc.derived_points_csv()))
    assert set(df["hw_key"]) <= set(sc.IX_HARDWARE_TO_PRESET)
    assert set(df["model"]) <= {m["ix_name"] for m in sc.IX_MODEL_MAP.values()}
    assert len(df) > 0


def test_an_oversize_slice_pre_renders_empty_like_the_browser(monkeypatch):
    """A slice shipped empty used to pre-render from the full archive: the
    published figure had InferenceX diamonds the live render then lost."""
    live = int((sc.serving_data()[0]["evidence"] == "measured").sum())
    assert live > 0
    monkeypatch.setattr(build_static, "MAX_DERIVED_BYTES", 500)
    with build_static.shipped_derived() as texts:
        assert texts[sc.DERIVED_POINTS] == ""
        frame, _ = sc.serving_data()
        assert int((frame["evidence"] == "measured").sum()) == 0
    sc.reset_cache()
    assert int((sc.serving_data()[0]["evidence"] == "measured").sum()) == live
    assert econ_common._OVERRIDES == {}


def test_the_zip_ships_the_texts_the_figures_were_drawn_from(tmp_path):
    with build_static.shipped_derived() as texts:
        build_static.build_data_bundle(tmp_path, texts)
        with zipfile.ZipFile(tmp_path / build_static.DATA_BUNDLE) as z:
            for rel, text in texts.items():
                assert z.read(rel).decode() == text
    src = Path(build_static.__file__).read_text()
    body = src[src.index("def main("):]
    assert body.index("shipped_derived()") < body.index("export_default_figures(")


# ── 4. Dash opens the run on click ───────────────────────────────────────────

def test_dash_serve_chart_has_the_run_click():
    i = APP_PY.index('Input("econ-serve-chart", "clickData")')
    block = APP_PY[APP_PY.rindex("clientside_callback(", 0, i):i]
    assert "_serving_cost.IX_RUNS_PREFIX" in block
    assert "noopener" in block and "customdata" in block
    assert 'dcc.Store(id="econ-click-sink")' in APP_PY


# ── 6. The headline share names its basis ────────────────────────────────────

def test_title_gives_both_shares_with_their_basis():
    frame, meta = sc.serving_data()
    out = (pd.to_numeric(frame["markup_out"]) > 1).mean()
    mix = (pd.to_numeric(frame["markup_mix"]) > 1).mean()
    title = _fig()["layout"]["title"]["text"]
    assert f"{out:.0%} on output price alone" in title
    assert f"{mix:.0%} at the same 8k:1k mix" in title
    assert mix > out


# ── 7. The modelled session count is not Run Local's display bound ─────────

def test_serving_session_cap_is_inferencex_sweep():
    pts = pd.read_csv(ROOT / sc.IX_POINTS_FILE)
    p = pts[(pts["sequence"] == "8k/1k") & (pts["deployment"] == "single-node")]
    swept = p.groupby("gpu_count")["concurrency"].max()
    for n in sc.GPU_COUNTS:
        assert sc.SERVING_MAX_SESSIONS[n] == int(swept[n]), n


def test_no_modelled_point_is_set_by_the_256_display_bound():
    frame, _ = sc.serving_data()
    mod = frame[frame["evidence"] == "modelled"]
    capped = mod[(pd.to_numeric(mod["sessions"]) == lm.MAX_REPORTED_CONCURRENCY)
                 & (pd.to_numeric(mod["per_session_tps"]) > sc.MIN_INTERACTIVITY * 1.05)]
    assert capped.empty
    assert "kv_vram" not in set(mod["binding"]) or all(
        s < sc.SERVING_MAX_SESSIONS[int(n)]
        for s, n, b in zip(mod["sessions"], mod["gpu_count"], mod["binding"])
        if b == "kv_vram")


def test_a_binding_cap_is_labelled_session_cap_not_kv():
    kw = dict(params_b=8, active_b=8, quant="Q8", vram_gb=80, bandwidth_gbps=3350,
              hw_type="nvidia", ctx_tokens=2048, kv_bytes_tok=131072, weights_gib=8.5,
              fp16_tflops=989, slo="datacenter")
    c = lm.optimal_concurrency(**kw, max_sessions=4)
    assert c.sessions == 4 and c.binding == "session_cap"
    # Run Local's default is unchanged.
    assert lm.optimal_concurrency(**kw).sessions <= lm.MAX_REPORTED_CONCURRENCY


# ── 8. The DeepSeek anchor is on the chart's basis ───────────────────────────

def test_deepseek_anchor_gives_the_like_for_like_figure():
    a = sc.deepseek_anchor()
    assert a["day_usd_per_m"] == pytest.approx(87072 / 168000, rel=1e-9)
    assert a["decode_usd_per_m"] == pytest.approx(16 * 1e6 / (14800 * 3600), rel=1e-9)
    cap = CAPTIONS["economics_serve"]
    assert f"${a['day_usd_per_m']:.2f}" in cap and f"{a['day_ratio']:.1f}×" in cap
    assert f"${a['decode_usd_per_m']:.2f}" in cap and "before prefill" in cap
    assert f"{a['decode_ratio']:.0f}×" in cap


# ── 9 / 12. The Agent Stack rent chip names its basis ────────────────────────

def test_rent_chip_names_output_basis_and_session_count():
    from components.stack_recommender import _rent_chip_text, _RENT_TITLE
    assert _rent_chip_text({"usd_per_m_out": 0.07, "sessions": 256}) == \
        "$0.07/M out · rent, 256 sessions, full use"
    assert _rent_chip_text({"usd_per_m_out": 0.5, "sessions": 1}) == \
        "$0.50/M out · rent, 1 session, full use"
    assert _rent_chip_text({"usd_per_m_out": float("nan"), "sessions": 3}) is None
    for word in ("OUTPUT", "max-throughput", "session", "full use", "Decode only"):
        assert word in _RENT_TITLE


def test_agent_stack_chip_carries_the_session_count():
    html = json.loads(api.update_recommend([], "local", "NVIDIA H100 SXM", 80, 1, "Q4"))
    assert re.search(r"/M out · rent, [\d,]+ sessions?, full use", html["cards_html"])


# ── 10. The rate basis is hyperscaler volume ─────────────────────────────────

def test_rates_are_called_hyperscaler_volume_everywhere():
    assert "large-hyperscaler-volume" in CAPTIONS["economics_serve"]
    o = sc.serve_options()
    assert all("hyperscaler" in b["title"] for b in o["bases"])
    assert "hyperscaler" in o["basis_title"]
    assert all("hyperscaler" in g["title"] for g in o["gpus"][1:])
    assert "hyperscaler-volume" in _fig()["layout"]["title"]["text"]
    assert "econ-serve-basis-label" in INDEX and "basis_title" in APP_JS


# ── 13. Provider legend swatches are one shape ───────────────────────────────

def test_provider_legend_is_circles_and_points_toggle_with_it():
    fig = _fig()
    pts = _point_traces(fig)
    assert pts and all(t.get("showlegend") is False for t in pts)
    keys = {t["legendgroup"]: t for t in fig["data"]
            if t.get("legendgroup") and t.get("showlegend") is not False}
    for t in pts:
        k = keys[t["legendgroup"]]
        assert k["marker"]["symbol"] == "circle"
        assert k["marker"]["color"] == t["marker"]["color"]
        assert k["x"] == [None]


# ── 14. Measured labels do not cover each other's markers ────────────────────

def _boxes_clear(gpu):
    # The go.Figure, not its JSON: Plotly 6 serialises numeric arrays as
    # base64 typed arrays.
    fig = build_econ_serve(*sc.serving_data(gpu), axis_range=sc.axis_range())
    lo, hi = fig.layout.xaxis.range
    span = hi - lo
    labelled = []
    for t in fig.data:
        if t.customdata is None or t.text is None:
            continue
        for x, y, txt, pos in zip(t.x, t.y, t.text, t.textposition):
            if txt:
                px = (math.log10(x) - lo) / span * econ_serve._PLOT_W_PX
                py = (math.log10(y) - lo) / span * econ_serve._PLOT_H_PX
                labelled.append((px, py, len(txt), pos))
    r = econ_serve._MARK_R_PX
    for i, (px, py, n, pos) in enumerate(labelled):
        box = econ_serve._label_box(px, py, pos, n)
        for j, (qx, qy, m, qpos) in enumerate(labelled):
            if i == j:
                continue
            assert not econ_serve._overlap(box, (qx - r, qx + r, qy - r, qy + r))
            assert not econ_serve._overlap(box, econ_serve._label_box(qx, qy, qpos, m))
    return labelled


def test_measured_labels_clear_each_other_at_every_gpu():
    for gpu in [None] + list(sc.IX_HARDWARE_TO_PRESET.values()):
        assert _boxes_clear(gpu) or gpu


def test_a_measured_label_avoids_unlabelled_points_and_the_diagonal():
    """MiniMax-M3's name printed over a modelled point at the same $1.20
    price; label_positions now treats every plotted point as an obstacle,
    and the dashed 1x line as a lesser one."""
    x = pd.Series([1.2], index=["mm"])
    y = pd.Series([0.68], index=x.index)
    rng = (-1.3, 1.1)
    free = label_positions(x, y, ["MiniMax-M3"], rng)
    assert free["mm"] == "top center"
    blocked = label_positions(x, y, ["MiniMax-M3"], rng, obstacles=[(1.2, 0.86)])
    assert blocked["mm"] != "top center"
    span = rng[1] - rng[0]
    px = (math.log10(1.2) - rng[0]) / span * econ_serve._PLOT_W_PX
    py = (math.log10(0.68) - rng[0]) / span * econ_serve._PLOT_H_PX
    box = econ_serve._label_box(px, py, blocked["mm"], len("MiniMax-M3"))
    ox = (math.log10(1.2) - rng[0]) / span * econ_serve._PLOT_W_PX
    oy = (math.log10(0.86) - rng[0]) / span * econ_serve._PLOT_H_PX
    assert not econ_serve._overlap(box, (ox - 4, ox + 4, oy - 4, oy + 4))
    # Just under the 1x line, "top center" would print across it; with every
    # side otherwise free the label goes below instead.
    on = label_positions(pd.Series([1.0], index=["d"]), pd.Series([0.75], index=["d"]),
                         ["Some Model Name"], rng)
    pxd = (0 - rng[0]) / span * econ_serve._PLOT_W_PX
    pyd = (math.log10(0.75) - rng[0]) / span * econ_serve._PLOT_H_PX
    assert econ_serve._on_diagonal(econ_serve._label_box(pxd, pyd, "top center", 15))
    assert not econ_serve._on_diagonal(econ_serve._label_box(pxd, pyd, on["d"], 15))


def test_label_positions_separates_a_vertical_stack():
    """DeepSeek V4 Pro / Kimi K2.7 Code / Qwen3.5 397B at ~$4: Kimi's name
    used to print over Qwen's diamond."""
    x = pd.Series([3.96, 4.0, 3.6], index=["ds", "kimi", "qwen"])
    y = pd.Series([0.903, 0.644, 0.471], index=x.index)
    pos = label_positions(x, y, ["DeepSeek V4 Pro 0813", "Kimi K2.7 Code",
                                 "Qwen3.5 397B A17B"], (-1.6, 2.6))
    assert pos["ds"] == "top center"
    assert pos["kimi"] in ("middle right", "middle left")
    assert pos["qwen"] == "bottom center"


# ── 15. Only the InferenceX-priced cards fill $/GPU-HR ───────────────────────

def test_rate_tooltips_name_the_seven_priced_cards():
    phrase = sc.priced_cards_phrase()
    assert phrase.startswith("one of the 7 ")
    assert all(sc.preset_label(p) in phrase for p in sc.IX_HARDWARE_TO_PRESET.values())
    assert phrase in INDEX
    assert "_serving_cost.priced_cards_phrase()" in APP_PY
    for text in (INDEX, APP_PY, CAPTIONS["local"]):
        assert "Picking a datacenter card" not in text
        assert "consumer cards have no published rate" not in text
    assert "seven datacenter cards InferenceX prices" in CAPTIONS["local"]


# ── 16. A figure's own height survives a resize ──────────────────────────────

def test_every_plotly_react_pins_the_figure_height():
    reacts = [m.start() for m in re.finditer(r"Plotly\.react\(", APP_JS)]
    assert len(reacts) == 2
    for i in reacts:
        before = APP_JS[max(0, i - 200):i]
        assert "pinFigureHeight(divId," in before
    fn = APP_JS[APP_JS.index("function pinFigureHeight"):][:400]
    assert "style.height" in fn and 'h > 0' in fn
