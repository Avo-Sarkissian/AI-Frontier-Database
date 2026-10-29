"""Landscape (treemap, provider leaderboard) and Compare (radar, listbox)
from the 2026-10-29 live audit.

Treemap: every tile was the same navy (the old ramp stayed dark for its first
55% and provider averages live there), the colour bar was titled "AvgScore",
small tiles printed ~7 px labels and the implicit root drew a grey band.
Leaderboard: "1 models", and 10 px labels in the provider hue at ~3:1.
Radar: the radial tick labels printed across the Intelligence spoke, which sat
at 3 o'clock and ran off a 356 px phone plot; legend names were cut at 28
characters ("Claude Opus 5.5 (Max, Defaul…"). The Compare listbox showed 4.2
rows.
"""
import json
import re
from pathlib import Path

import pytest

from components.charts import treemap as T
from components.charts.constants import BG, QUALITY_INDEX_MAX
from components.charts.provider_leaderboard import build_provider_leaderboard, count_label
from components.charts.radar import DIMS, build_radar
from data.ingest import get_models

ROOT = Path(__file__).resolve().parents[1]
DF = get_models()


def _lum(rgb):
    return T._luminance(rgb)


def _contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _rgb(hex_):
    return T._hex_rgb(hex_)


def _fig(f):
    return json.loads(f.to_json())


# ── Treemap ──────────────────────────────────────────────────────────────────

def test_treemap_ramp_is_monotonic_and_clear_of_the_page():
    lums = [_lum(T.ramp_color(v / 10)) for v in range(0, int(QUALITY_INDEX_MAX * 10) + 1)]
    assert all(b >= a for a, b in zip(lums, lums[1:])), "a higher score painted darker"
    first = _rgb(T.TREEMAP_COLORSCALE[0][1])
    assert max(abs(x - y) for x, y in zip(first, _rgb(BG))) >= 12


def test_treemap_tiles_are_visibly_different_across_the_live_spread():
    """The lowest- and highest-scoring providers on screen must not share a
    colour: the old ramp gave the whole live range a contrast ratio of ~1.3."""
    fig = T.build_treemap(DF)
    vals = [float(v) for v in fig.data[0].marker.colors if v == v]
    lo, hi = min(vals), max(vals)
    assert hi - lo > 10, "fixture assumption: providers spread by more than 10 points"
    assert _contrast(T.ramp_color(lo), T.ramp_color(hi)) >= 2.5


def test_treemap_labels_are_legible():
    fig = T.build_treemap(DF)
    trace, layout = fig.data[0], fig.layout
    assert layout.uniformtext.minsize >= 10
    assert layout.uniformtext.mode == "hide"
    assert len(trace.textfont.color) == len(trace.marker.colors)
    for v, col in zip(trace.marker.colors, trace.textfont.color):
        assert _contrast(T.ramp_color(v), _rgb(col)) >= 4.5, (v, col)


def test_treemap_label_colour_keeps_contrast_over_the_whole_domain():
    for i in range(0, int(QUALITY_INDEX_MAX * 10) + 1):
        v = i / 10
        assert _contrast(T.ramp_color(v), _rgb(T.label_color(v))) >= 4.5, v


def test_treemap_has_no_grey_root_and_a_human_colour_key():
    fig = _fig(T.build_treemap(DF))
    trace = fig["data"][0]
    assert trace["root"]["color"] == BG
    title = trace["marker"]["colorbar"]["title"]["text"]
    assert title == "Avg intelligence"
    assert "AvgScore" not in json.dumps(fig)


def test_treemap_wraps_only_long_provider_names():
    assert T.tile_label("Thinking Machines") == "Thinking<br>Machines"
    assert T.tile_label("Z AI") == "Z AI"
    assert T.tile_label("<b>x y z w v u</b>").startswith("&lt;b&gt;")


# ── Provider leaderboard ─────────────────────────────────────────────────────

def test_leaderboard_count_grammar():
    assert count_label(1) == "1 model"
    assert count_label(2) == "2 models"
    texts = [a["text"] for a in _fig(build_provider_leaderboard(DF))["layout"]["annotations"]]
    assert not any(re.match(r"1 models\b", t) for t in texts)


def test_leaderboard_right_labels_are_high_contrast():
    for a in _fig(build_provider_leaderboard(DF))["layout"]["annotations"]:
        assert _contrast(_rgb(a["font"]["color"]), _rgb(BG)) >= 7, a["text"]


# ── Radar ────────────────────────────────────────────────────────────────────

def _radar(models=None):
    picks = models or list(DF.sort_values("quality", ascending=False)["model"].head(5))
    return _fig(build_radar(DF, picks, full_df=DF))


def test_radar_puts_intelligence_at_the_top_and_out_of_the_tick_labels():
    polar = _radar()["layout"]["polar"]
    assert DIMS[0] == "Intelligence"
    assert polar["angularaxis"]["rotation"] == 90
    assert polar["angularaxis"]["direction"] == "clockwise"
    # The rings stay; their numbers, which printed over the spoke, do not.
    assert polar["radialaxis"]["showticklabels"] is False
    assert polar["radialaxis"]["tickvals"] == [0.25, 0.5, 0.75, 1.0]


def test_radar_legend_abbreviates_instead_of_cutting():
    long = [m for m in DF["model"] if "Default Fallback" in m]
    if not long:
        pytest.skip("no Default Fallback rows in the live catalogue")
    names = [t["name"] for t in _radar(long[:5])["data"] if t.get("showlegend", True) and t.get("name")]
    assert names
    for n in names:
        assert "Default Fallback" not in n and not n.endswith("…"), n


# ── Compare listbox ──────────────────────────────────────────────────────────

def test_compare_listbox_is_a_whole_number_of_rows():
    css = (ROOT / "docs" / "assets" / "shell.css").read_text()
    rule = re.search(r"#radar-model-select\s*\{([^}]*)\}", css).group(1)
    row = int(re.search(r"--row:\s*(\d+)px", rule).group(1))
    rows = int(re.search(r"height:\s*calc\((\d+)\s*\*\s*var\(--row\)\s*\+\s*2px\)", rule).group(1))
    assert rows >= 3 and row >= 20
    opt = re.search(r"#radar-model-select option\s*\{([^}]*)\}", css).group(1)
    assert "height: var(--row)" in opt
