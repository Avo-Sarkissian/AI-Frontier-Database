"""Dash app and its Overview / Compare / Rankings charts.

Pins the fixes from the 2026-09-24 review: quadrant shapes in the right
coordinate space, missing radar metrics drawn as missing, Value Leaders wired to
the filters, the detail panel keyed by the raw name, EFFORT in the CSV export,
stable tie orders, the renamed IFM palette key, and an inert `import app`.
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from data.ingest import get_models
from static_helpers import apply_filters

ROOT = Path(__file__).resolve().parent.parent
DF = get_models()

os.environ.setdefault("DEBUG", "true")
os.environ.pop("WERKZEUG_RUN_MAIN", None)


def _fig(fig):
    return json.loads(fig.to_json())


# ── #3 quadrant: shapes take raw values on a log axis ────────────────────────

def test_quadrant_zones_and_speed_median_sit_inside_the_plotted_range():
    """Shapes on a log axis take DATA values; only annotations and the axis
    range take exponents. Passing log10() put every zone and the vertical
    median at 1-3 tok/s, left of a plot that starts at ~12."""
    from components.charts.quadrant import build_quadrant, _plottable

    fig = _fig(build_quadrant(DF, full_df=DF))
    lo, hi = (10 ** v for v in fig["layout"]["xaxis"]["range"])
    shapes = fig["layout"]["shapes"]
    assert shapes, "the quadrant drew no shapes"
    for sh in shapes:
        if sh.get("xref", "x") != "x":
            continue           # the horizontal median spans the paper, not data
        for key in ("x0", "x1"):
            assert lo * 0.999 <= sh[key] <= hi * 1.001, (
                f"shape {key}={sh[key]} lies outside the plotted {lo:.1f}-{hi:.1f} tok/s"
            )

    med = _plottable(DF)["speed"].median()
    vlines = [sh for sh in shapes if sh.get("type") == "line" and sh["x0"] == sh["x1"]]
    assert vlines and abs(vlines[0]["x0"] - med) < 1e-9, (
        "the speed-median crosshair is not at the raw median speed"
    )


def test_quadrant_zone_captions_stay_in_exponent_space():
    from components.charts.quadrant import build_quadrant

    layout = _fig(build_quadrant(DF, full_df=DF))["layout"]
    lx0, lx1 = layout["xaxis"]["range"]
    captions = [a for a in layout["annotations"] if a.get("xref") == "x"]
    assert len(captions) == 4
    for a in captions:
        assert lx0 <= a["x"] <= lx1, f"caption {a['text']!r} at {a['x']} is off the axis"


# ── #13 radar: unmeasured is missing, not the worst score ─────────────────────

def test_radar_draws_an_unmeasured_metric_as_a_gap_not_zero():
    from components.charts.radar import build_radar

    frame = DF.copy()
    target = frame.sort_values("quality", ascending=False).iloc[0]["model"]
    idx = frame.index[frame["model"] == target][0]
    frame.loc[idx, "speed"] = 0
    frame.loc[idx, "latency"] = float("nan")

    fig = _fig(build_radar(frame, [target], full_df=DF))
    trace = fig["data"][0]
    r = trace["r"]
    # DIMS = Intelligence, Speed, Affordability, Context, Latency
    assert r[1] is None and r[4] is None, f"missing speed/latency drew as {r[1]}, {r[4]}"
    assert r[0] is not None and r[0] > 0
    assert trace.get("connectgaps") is False
    assert "Speed: not measured" in trace["hovertemplate"]
    assert "Latency: not measured" in trace["hovertemplate"]
    assert "Speed: 0%" not in trace["hovertemplate"]
    assert "not measured" in fig["layout"]["title"]["text"]


def test_radar_gapped_outline_is_unfilled_and_its_area_is_one_polygon():
    """fill="toself" closes each gap-separated segment on its own, so a gapped
    outline drew a chord across the missing spoke and, with two gaps, no area
    at all. The outline stays gapped and hoverable but unfilled; a separate,
    non-hoverable trace fills one polygon with the missing vertices at 0."""
    from components.charts.radar import build_radar

    frame = DF.copy()
    target = frame.sort_values("quality", ascending=False).iloc[0]["model"]
    idx = frame.index[frame["model"] == target][0]
    frame.loc[idx, "speed"] = 0
    frame.loc[idx, "latency"] = float("nan")

    fig = _fig(build_radar(frame, [target], full_df=DF))
    outline, area = fig["data"]
    assert outline.get("fill") == "none"
    assert "markers" in outline["mode"]          # the lone Intelligence vertex shows
    assert area["fill"] == "toself"
    assert None not in area["r"]
    assert area["r"][1] == 0 and area["r"][4] == 0
    assert [a for a, o in zip(area["r"], outline["r"]) if o is not None] == \
        [o for o in outline["r"] if o is not None]
    assert area["hoverinfo"] == "skip" and area["showlegend"] is False
    assert not area.get("name")
    assert area["legendgroup"] == outline["legendgroup"]


def test_radar_fully_measured_model_keeps_one_filled_trace():
    from components.charts.radar import build_radar

    ok = DF[(DF["speed"] > 0) & (DF["latency"] > 0) & (DF["price"] > 0)
            & DF["context"].notna()]
    fig = _fig(build_radar(DF, [ok.iloc[0]["model"]], full_df=DF))
    assert len(fig["data"]) == 1 and fig["data"][0]["fill"] == "toself"


def test_radar_fully_measured_model_has_no_gaps_and_no_gap_note():
    from components.charts.radar import build_radar

    ok = DF[(DF["speed"] > 0) & (DF["latency"] > 0) & (DF["price"] > 0)
            & DF["context"].notna()]
    name = ok.iloc[0]["model"]
    fig = _fig(build_radar(DF, [name], full_df=DF))
    assert None not in fig["data"][0]["r"]
    assert "not measured" not in fig["layout"]["title"]["text"]


# ── #21 palette: keyed by the lab's current AA name ───────────────────────────

def test_aa_mirror_is_keyed_by_the_name_aa_publishes_today():
    """AA dropped the "MBZUAI " prefix on 2026-09-17; a key under the retired
    spelling matches no row anywhere."""
    from components.charts.constants import AA_CREATOR_COLORS
    from data.local_models import get_local_df

    families = set(get_local_df()["family"].dropna().astype(str))
    for fam in families:
        stale = [k for k in AA_CREATOR_COLORS if k != fam and k.endswith(" " + fam)]
        assert not stale or fam in AA_CREATOR_COLORS, (
            f"{fam!r} has an AA colour only under the stale key(s) {stale}"
        )


# ── stable tie orders ─────────────────────────────────────────────────────────

def test_treemap_breaks_count_ties_by_provider_name():
    from components.charts.treemap import build_treemap

    frame = pd.DataFrame({
        "provider": ["Zeta", "Alpha", "Mid", "Mid"],
        "model":    ["z1", "a1", "m1", "m2"],
        "quality":  [30.0, 40.0, 20.0, 25.0],
        "price":    [1.0, 2.0, 3.0, 4.0],
        "speed":    [10.0, 20.0, 30.0, 40.0],
    })
    labels = _fig(build_treemap(frame))["data"][0]["labels"]
    assert labels == ["Mid", "Alpha", "Zeta"]


def test_pareto_and_quadrant_are_deterministic_under_row_order():
    """Equal-size bubbles and equal-quality labels must not reorder when the
    input rows arrive in a different order (CI build vs in-browser render)."""
    from components.charts.pareto import build_pareto_scatter
    from components.charts.quadrant import build_quadrant

    shuffled = DF.sample(frac=1.0, random_state=7)
    for build in (build_pareto_scatter, build_quadrant):
        a = _fig(build(DF, full_df=DF))["data"]
        b = _fig(build(shuffled, full_df=shuffled))["data"]
        assert [t.get("customdata") for t in a] == [t.get("customdata") for t in b], (
            f"{build.__name__} depends on input row order"
        )


# ── Dash callbacks ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dash_app():
    import app as module
    return module


def test_value_leaders_follows_the_global_filters_and_data_version(dash_app):
    from dash._callback import GLOBAL_CALLBACK_MAP

    specs = [spec for key, spec in GLOBAL_CALLBACK_MAP.items()
             if "value-leaders-chart.figure" in key]
    assert specs, "Value Leaders has no callback"
    inputs = {d["id"] for d in specs[0]["inputs"]}
    assert {"filter-provider", "filter-quality", "filter-effort",
            "model-search", "data-version"} <= inputs

    fig = _fig(dash_app.update_value_leaders(["Anthropic"], 0, None, "", 0))
    providers = {row[1] for row in fig["data"][0]["customdata"]}
    assert providers == {"Anthropic"}, f"Value Leaders listed {providers} under PROVIDER=Anthropic"


def test_detail_panel_opens_for_a_name_that_needs_escaping(dash_app, monkeypatch):
    from components.charts.pareto import build_pareto_scatter

    frame = dash_app.df.copy()
    top = frame["quality"].idxmax()
    frame.loc[top, "model"] = "R&D <Coder> 100%"
    frame.loc[top, "provider"] = "Q&A Labs"
    monkeypatch.setattr(dash_app, "df", frame)

    class _Ctx:
        triggered_id = "pareto-chart"
    monkeypatch.setattr(dash_app, "ctx", _Ctx)

    fig = _fig(build_pareto_scatter(frame, full_df=frame))
    cd = next(c for t in fig["data"] for c in (t.get("customdata") or [])
              if len(c) > 6 and c[5] == "R&D <Coder> 100%")
    assert cd[0] != cd[5], "the hover copy should be escaped"

    class_name, body, name = dash_app.toggle_detail_panel(
        {"points": [{"customdata": cd}]}, None)
    assert class_name == "detail-panel open"
    assert name == "R&D <Coder> 100%"
    assert "Q&A Labs" in str(body) and "&amp;" not in str(body)


def test_dash_csv_export_honours_effort(dash_app):
    out = dash_app.export_csv(1, "table", None, 0, "", "high")
    exported = pd.read_csv(io.StringIO(out["content"]))
    on_screen = apply_filters(DF, None, 0, "", "high")
    assert len(exported) == len(on_screen) < len(DF)


def test_dash_csv_export_uses_the_picked_run_local_hardware(dash_app):
    small = pd.read_csv(io.StringIO(dash_app.export_csv(
        1, "local", None, 0, "", None, 8, 1, "Q4", None,
        {"bandwidth_gbps": 272, "hw_type": "nvidia", "fp16_tflops": None}, None)["content"]))
    big = pd.read_csv(io.StringIO(dash_app.export_csv(
        1, "local", None, 0, "", None, 80, 8, "Q4", None,
        {"bandwidth_gbps": 3350, "hw_type": "nvidia", "fp16_tflops": None}, None)["content"]))
    fits = lambda f: int(f["fits"].isin(["yes", "tight"]).sum())
    assert fits(big) > fits(small), "the export ignored the hardware on screen"


# ── importing app.py must not scrape ──────────────────────────────────────────

_PROBE = r"""
import json, os, sys
calls = []
import data.scraper, data.image_scraper, data.local_scraper, data.video_scraper
for mod, fn in ((data.scraper, "start_background_scraper"),
                (data.image_scraper, "start_background_image_scraper"),
                (data.local_scraper, "start_background_local_scraper"),
                (data.video_scraper, "start_background_video_scraper")):
    setattr(mod, fn, lambda interval_s=3600, _n=fn: calls.append(_n))
import app
after_import = list(calls)
client = app.server.test_client()
client.get("/_dash-layout"); client.get("/_dash-layout")
print(json.dumps({"after_import": after_import, "after_requests": calls}))
"""


def _probe(extra_env):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTEST_CURRENT_TEST", "AI_FRONTIER_SCRAPE", "WERKZEUG_RUN_MAIN")}
    env.update(extra_env)
    proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_import_app_starts_no_scraper_and_the_server_starts_them_once():
    got = _probe({})
    assert got["after_import"] == [], f"import app started {got['after_import']}"
    assert sorted(got["after_requests"]) == sorted([
        "start_background_scraper", "start_background_image_scraper",
        "start_background_local_scraper", "start_background_video_scraper",
    ]), f"serving requests started {got['after_requests']}"


def test_the_scrape_env_flag_forces_and_disables():
    assert len(_probe({"AI_FRONTIER_SCRAPE": "1"})["after_import"]) == 4
    assert _probe({"AI_FRONTIER_SCRAPE": "0"})["after_requests"] == []
