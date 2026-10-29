"""The 2026-10-29 chart audit: phone layout, point labels, bar labels, one
row per model family, empty states, Overview's key, the detail panel, the
Agent Stack and quality floors that survive an index rescale.

Each test asserts behaviour through the artefact a visitor meets (the figure
JSON static_api returns, the cards' HTML, or docs/app.js run in node) rather
than through a helper's name.
"""
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import static_api as api
from components.charts.constants import (
    POINT_LABEL, base_model_name, clean_model_name, contrast_ratio, composite,
    parse_rgba, place_point_labels, text_on, PROVIDER_COLORS,
)
from data import quality_scale as qs
from data.ingest import get_models

ROOT = Path(__file__).resolve().parent.parent
APP = (ROOT / "docs" / "app.js").read_text()
NODE = shutil.which("node")
DF = get_models()


def _decode(o):
    """Plotly's JSON ships arrays as {dtype, bdata}; turn them back into lists."""
    import base64
    import numpy as np
    if isinstance(o, dict):
        if set(o) >= {"dtype", "bdata"}:
            arr = np.frombuffer(base64.b64decode(o["bdata"]), dtype=o["dtype"])
            if "shape" in o:
                arr = arr.reshape([int(v) for v in str(o["shape"]).split(",")])
            return arr.tolist()
        return {k: _decode(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_decode(v) for v in o]
    return o


def _fig(s):
    return _decode(json.loads(s) if isinstance(s, str) else s)


def _families(names) -> list[str]:
    return [base_model_name(str(n)) for n in names]


def _no_repeats(names, what):
    fams = _families(names)
    dup = sorted({f for f in fams if fams.count(f) > 1})
    assert not dup, f"{what} repeats a model family: {dup}"


# ── names ────────────────────────────────────────────────────────────────────

def test_the_default_fallback_qualifier_is_dropped_from_short_names():
    assert clean_model_name("Claude Opus 5.5 (Max, Default Fallback)") == "Claude Opus 5.5 (Max)"
    assert clean_model_name("Claude Fable 5.1 (Xhigh, Default Fallback)") == "Claude Fable 5.1 (XH)"
    assert clean_model_name("GPT-6 Astra (High)") == "GPT-6 Astra (H)"


# ── #2 point labels ──────────────────────────────────────────────────────────

def _box(a, w, h):
    return (a[0] - w / 2, a[1] - h / 2, a[0] + w / 2, a[1] + h / 2)


@pytest.mark.parametrize("xaxis", ["price", "speed", "task"])
def test_overview_point_labels_are_offset_plates_that_clear_every_mark(xaxis):
    """Labels used to be text centred ON their marker. Now each is an
    annotation offset from its mark, on a dark plate, and in the nominal
    desktop plot box none overlaps another label or any marker."""
    fig = _fig(api.update_overview(None, 0, "", xaxis))
    lay = fig["layout"]
    labels = [a for a in lay["annotations"] if a.get("name") == POINT_LABEL]
    assert labels, "Overview lost its direct labels"
    assert not any(isinstance(t.get("mode"), str) and "text" in t["mode"]
                   for t in fig["data"]), "a text trace is back"
    (x0, x1), (y0, y1) = lay["xaxis"]["range"], lay["yaxis"]["range"]
    W, H = 1200.0, 360.0

    def px(x, y):
        return ((x - x0) / (x1 - x0) * W, (y1 - y) / (y1 - y0) * H)

    marks = []
    for t in fig["data"]:
        if t.get("mode") != "markers" or not t.get("x"):
            continue
        sizes = t["marker"]["size"]
        sizes = sizes if isinstance(sizes, list) else [sizes] * len(t["x"])
        for x, y, s in zip(t["x"], t["y"], sizes):
            marks.append((*px(math.log10(x), y), s / 2))
    boxes = []
    for a in labels:
        assert a["bgcolor"] and (a["ax"] or a["ay"]), "a label sits on its own mark"
        cx, cy = px(a["x"], a["y"])
        w = len(a["text"]) * 6.0 + 8
        b = _box((cx + a["ax"], cy + a["ay"]), w, 15)
        assert 0 <= b[0] and b[2] <= W and 0 <= b[1] and b[3] <= H, f"{a['text']} leaves the plot"
        for mx, my, r in marks:
            nx, ny = min(max(mx, b[0]), b[2]), min(max(my, b[1]), b[3])
            assert (mx - nx) ** 2 + (my - ny) ** 2 >= r * r - 1e-6, f"{a['text']} covers a mark"
        for o in boxes:
            assert b[2] <= o[0] or o[2] <= b[0] or b[3] <= o[1] or o[3] <= b[1], \
                f"{a['text']} overlaps another label"
        boxes.append(b)
        assert "(" not in a["text"], "a family label still carries its effort suffix"


def test_place_point_labels_skips_a_label_with_no_free_position():
    # A ring of marks around the candidate leaves nowhere to put its plate.
    # 1200 x 360 px plot: 0.5 x-units and 8.33 y-units are both 30 px.
    ring = [(10 + i * 0.5, 50 + j * 8.33, 60) for i in range(-3, 4) for j in range(-3, 4)]
    out = place_point_labels([(10, 50, "Boxed in", 60)], ring, (0, 20), (0, 100))
    assert out == []


# ── #11 Overview key and legend ──────────────────────────────────────────────

def _legend_order(fig):
    seen = []
    for t in fig["data"]:
        if t.get("showlegend") is False or not t.get("name"):
            continue
        if t["name"] not in seen:
            seen.append(t["name"])
    return seen


def test_overview_legend_order_is_the_same_in_every_x_axis_view():
    orders = {x: _legend_order(_fig(api.update_overview(None, 0, "", x)))
              for x in ("price", "speed", "task")}
    providers = {x: [n for n in o if n != "Pareto Frontier"] for x, o in orders.items()}
    common = set.intersection(*(set(v) for v in providers.values()))
    trimmed = {x: [n for n in v if n in common] for x, v in providers.items()}
    assert trimmed["price"] == trimmed["speed"] == trimmed["task"]
    assert providers["price"][-1] == "Other"


def test_the_frontier_leads_the_legend_instead_of_trailing_other():
    fig = _fig(api.update_overview(None, 0, "", "price"))
    line = next(t for t in fig["data"] if t.get("name") == "Pareto Frontier")
    assert line.get("legendrank", 1000) < 1000


def test_the_size_key_and_correlation_sit_in_one_place_on_every_view():
    notes = {}
    for x in ("price", "speed", "task"):
        lay = _fig(api.update_overview(None, 0, "", x))["layout"]
        n = [a for a in lay["annotations"] if a.get("name") == "note"]
        assert len(n) == 1, x
        notes[x] = n[0]
    pos = {(a["x"], a["y"], a["xanchor"], a["yanchor"], a["xref"]) for a in notes.values()}
    assert len(pos) == 1, "the note moves between X AXIS views"
    assert notes["price"]["text"].startswith("size = speed (tok/s)")
    assert notes["task"]["text"].startswith("size = speed (tok/s)")
    assert notes["speed"]["text"].startswith("size = affordability")
    for a in notes.values():
        assert "r = " in a["text"]


# ── #3 bar value labels ──────────────────────────────────────────────────────

def test_text_on_picks_dark_text_for_openais_light_grey_bar():
    assert text_on(PROVIDER_COLORS["OpenAI"], 0.85) == "#000000"
    # and the choice is the one with more contrast, for every palette colour
    for c in set(PROVIDER_COLORS.values()):
        shown = composite(c, 0.85)
        chosen = parse_rgba(text_on(c, 0.85))
        other = parse_rgba("#ffffff" if text_on(c, 0.85) == "#000000" else "#000000")
        assert contrast_ratio(shown, chosen) >= contrast_ratio(shown, other)


@pytest.mark.parametrize("fig_of", [
    lambda: _fig(api.update_rankings(None, 0, "", "intelligence")),
    lambda: _fig(json.loads(api.update_cost_calc(1, None, 0, ""))["figure"]),
])
def test_bar_value_labels_are_upright_and_readable_on_their_bar(fig_of):
    fig = fig_of()
    bar = next(t for t in fig["data"] if t.get("type") == "bar" and t.get("text"))
    assert bar["textangle"] == 0
    assert bar.get("insidetextanchor") == "end"
    colors = bar["insidetextfont"]["color"]
    assert isinstance(colors, list) and len(colors) == len(bar["x"])
    fills = bar["marker"]["color"]
    op = bar["marker"].get("opacity", 1)
    ops = op if isinstance(op, list) else [op] * len(fills)
    for fill, o, txt in zip(fills, ops, colors):
        ratio = contrast_ratio(composite(fill, o), parse_rgba(txt))
        assert ratio >= 4.5, f"{txt} on {fill}@{o}: {ratio:.1f}:1"


def test_budget_axis_ticks_use_the_bar_labels_price_format():
    from data.workload_mix import fmt_price
    fig = _fig(json.loads(api.update_cost_calc(1, None, 0, ""))["figure"])
    ax = fig["layout"]["xaxis"]
    assert ax["ticktext"] == [fmt_price(v) for v in ax["tickvals"]]


# ── #4 one row per model family under "All variants" ─────────────────────────

def test_rankings_show_each_model_family_once():
    fig = _fig(api.update_rankings(None, 0, "", "intelligence"))
    bar = [t for t in fig["data"] if t.get("type") == "bar" and t.get("customdata")][0]
    names = [cd[0] for cd in bar["customdata"]]
    _no_repeats(names, "Top 25")
    assert f"Top {len(names)} " in fig["layout"]["title"]["text"]


def test_value_leaders_show_each_model_family_once():
    fig = _fig(api.update_value_leaders(None, 0, ""))
    _no_repeats([cd[0] for cd in fig["data"][0]["customdata"]], "Value Leaders")


def test_the_budget_list_and_answer_count_families():
    out = json.loads(api.update_cost_calc(1, None, 0, "", 20))
    bar = [t for t in out["figure"]["data"] if t.get("customdata")][0]
    _no_repeats([cd[0] for cd in bar["customdata"]], "Budget")
    fams = {base_model_name(m) for m in DF[(DF["quality"] >= 20) & (DF["price"] > 0)]["model"]}
    assert out["best"]["n_qualifying"] == len(fams)


def test_compare_defaults_are_five_different_families():
    picks = json.loads(api.update_compare(None, 0, "", None, None))["value"]
    assert len(picks) == 5
    _no_repeats(picks, "Compare defaults")


def test_agent_stack_alternates_and_tiers_never_repeat_a_family():
    from components.stack_recommender import select_stack
    out = select_stack(DF)
    claimed = []
    for t in out["tiers"]:
        _no_repeats(t["picks"]["model"], f"{t['name']} alternates")
        if not t["duplicate_of"] and not t["picks"].empty:
            claimed.append(base_model_name(t["picks"].iloc[0]["model"]))
    assert len(claimed) == len(set(claimed))


# ── #5 empty states ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("fig_of", [
    lambda: api.update_overview(None, 0, "zzzz", "price"),
    lambda: api.update_overview(None, 0, "zzzz", "speed"),
    lambda: api.update_overview(None, 0, "zzzz", "task"),
    lambda: api.update_treemap(None, 0, "zzzz"),
    lambda: api.update_provider_leaderboard(None, 0, "zzzz"),
    lambda: api.update_rankings(None, 0, "zzzz", "intelligence"),
    lambda: api.update_value_leaders(None, 0, "zzzz"),
    lambda: json.loads(api.update_compare(None, 0, "zzzz", [], None))["figure"],
])
def test_a_filter_matching_nothing_draws_a_titled_message_not_bare_axes(fig_of):
    fig = _fig(fig_of())
    lay = fig["layout"]
    assert not fig["data"]
    assert lay["xaxis"]["visible"] is False and lay["yaxis"]["visible"] is False
    assert lay["title"]["text"], "the empty state lost the chart's title"
    assert "No models match these filters" in lay["annotations"][0]["text"]


# ── #12 detail panel ─────────────────────────────────────────────────────────

def test_the_standing_line_is_a_true_share_of_families():
    from static_helpers import standing_note
    weak = DF[DF["quality"] > 0].sort_values("quality").iloc[2]
    note = standing_note(weak["quality"], DF)
    pct = int(re.search(r"above (\d+)%", note).group(1))
    assert pct < 25 and "Top" not in note
    assert standing_note(DF["quality"].max(), DF).startswith("Highest score")


def test_the_detail_panel_says_tokens_and_drops_absent_indices():
    row = DF.iloc[0].copy()
    row["context"], row["coding"], row["agentic"] = "1m", float("nan"), float("nan")
    out = api._detail_html(row, row["provider"])
    assert ">1M tokens</span>" in out
    assert "Coding" not in out and "not scored" not in out


def test_overview_hovers_say_intelligence_like_the_axis():
    for x in ("price", "speed"):
        fig = _fig(api.update_overview(None, 0, "", x))
        hov = [t.get("hovertemplate", "") for t in fig["data"] if t.get("hovertemplate")]
        assert hov and all("Quality:" not in h and "Intelligence:" in h for h in hov)


def test_clicking_a_bubble_rings_it_and_closing_the_panel_clears_it():
    handler = APP[APP.index("function attachParetoClickHandler"):]
    handler = handler[:handler.index("\n}\n")]
    assert "highlightPoint(ev.points[0])" in handler
    close = APP[APP.index("function wireDetailPanel"):]
    close = close[:close.index("detail-add-compare")]
    assert "highlightPoint(null)" in close


# ── #15 Agent Stack ─────────────────────────────────────────────────────────

def test_a_slow_first_token_is_flagged_on_the_card():
    from components.stack_recommender import _api_row_html, SLOW_TTFT_WARN_S
    row = pd.Series({"model": "Slowpoke", "provider": "Google", "price": 1.0,
                     "speed": 100.0, "quality": 40.0, "latency": SLOW_TTFT_WARN_S + 300})
    out = _api_row_html(row)
    assert "⚠ 330s to first token" in out
    fast = _api_row_html(row.copy().replace({SLOW_TTFT_WARN_S + 300: 1.2}))
    assert "1.20s TTFT" in fast and "⚠" not in fast


def test_balanced_excludes_a_minutes_long_first_token():
    from components.stack_recommender import select_stack, BALANCED_MAX_LATENCY_S
    bal = next(t for t in select_stack(DF)["tiers"] if t["key"] == "balanced")["picks"]
    lat = pd.to_numeric(bal["latency"], errors="coerce")
    assert not (lat > BALANCED_MAX_LATENCY_S).any()


def test_tier_marks_are_one_style_not_three_glyphs():
    out = json.loads(api.update_recommend(["__all__"], "api", None, None, None, None))["cards_html"]
    for glyph in ("⚡", "⚙", "🧠"):
        assert glyph not in out
    assert out.count("width:8px;height:8px;border-radius:2px") == 3


# ── #16 floors that survive a rescale ────────────────────────────────────────

def _rescaled(df):
    """A NON-uniform monotone rescale, like v5's (x1.08 top, x0.36 bottom)."""
    out = df.copy()
    q = out["quality"]
    out["quality"] = q.where(q.isna(), (q / 60.0) ** 2.2 * 64.0)
    return out


def test_tier_pools_are_unchanged_by_a_non_uniform_rescale():
    from components.stack_recommender import _API_TIERS, _tier_pool
    re_df = _rescaled(DF)
    for tier in _API_TIERS:
        before = set(_tier_pool(DF, tier)["model"])
        after = set(_tier_pool(re_df, tier)["model"])
        assert before == after, tier["key"]


@pytest.mark.parametrize("key", ["value_leaders", "budget_strong", "budget_mid",
                                 "compare_value", "compare_strong"])
def test_hosted_floors_admit_the_same_families_after_a_rescale(key):
    def admitted(df):
        f = qs.floor(df, key)
        best = df[df["quality"] > 0].groupby(df["model"].map(base_model_name))["quality"].max()
        return set(best[best >= f].index)
    assert admitted(DF) == admitted(_rescaled(DF))


def test_local_tiers_differ_where_v4_literal_floors_collapsed_them():
    """On v5 a 32 GB card runs one model above the old 10 / 12 floors, so all
    three local tiers named it. A quantile floor still lets the smaller model
    that fits field the Fast tier."""
    from components.stack_recommender import select_stack
    rows = [("Big A (Max)", 32.0, 300.0, "no", 50.0), ("Big B", 30.0, 250.0, "no", 40.0),
            ("Big C", 28.0, 200.0, "no", 45.0), ("Mid D", 25.0, 120.0, "no", 60.0),
            ("Mid E", 22.0, 100.0, "no", 70.0), ("Fits Q", 20.0, 17.0, "yes", 68.0),
            ("Mid F", 18.0, 90.0, "no", 80.0), ("Mid G", 14.0, 80.0, "no", 90.0),
            ("Mid H", 12.0, 70.0, "no", 100.0), ("Fits G", 6.4, 18.0, "yes", 63.0),
            ("Fits N", 1.8, 5.0, "yes", 285.0)]
    local = pd.DataFrame([{"name": n, "family": "X", "quality": q, "vram_req_gb": v,
                           "fits": f, "speed_tps": s} for n, q, v, f, s in rows])
    tiers = {t["key"]: t for t in select_stack(DF, mode="local", local_df=local)["tiers"]}
    tops = {k: t["picks"].iloc[0]["name"] for k, t in tiers.items()}
    assert tops["reasoning"] == "Fits Q"
    assert len(set(tops.values())) >= 2, f"every tier names one model: {tops}"
    assert "Fits G" in tops.values()
    for k in ("fast", "balanced"):
        assert "Fits N" not in set(tiers[k]["picks"]["name"]), f"the weakest model won {k}"


def test_a_tier_left_repeating_takes_back_a_family_another_tier_can_spare():
    """Balanced claimed Claude Haiku 5.5 — Fast's only candidate on the default
    providers — while holding four other families; Fast then rendered as a
    repeat."""
    from components.stack_recommender import select_stack
    out = select_stack(DF, providers=["Anthropic", "Google", "OpenAI"])
    tiers = {t["key"]: t for t in out["tiers"]}
    tops = [base_model_name(t["picks"].iloc[0]["model"]) for t in out["tiers"]
            if not t["picks"].empty]
    assert len(tops) == len(set(tops))
    assert all(t["duplicate_of"] is None for t in out["tiers"] if not t["picks"].empty)


def test_the_min_score_ladder_and_budget_slider_track_the_peak():
    from static_helpers import quality_options
    peak = float(DF["quality"].max())
    steps = [o["value"] for o in quality_options(None, None, peak=peak)]
    assert steps[0] == 0 and max(steps) < peak and max(steps) > peak - 5
    assert qs.ladder(92.0)[-1] == 90.0 and qs.ladder(130.0)[-1] == 120.0
    assert qs.slider_max(peak) == math.ceil(peak)
    m = json.loads((ROOT / "docs" / "figures" / "manifest.json").read_text())
    assert m["budget_slider_max"] == math.ceil(float(m["peak_quality"]))
    opts = [o["value"] for o in m["quality_options"]]
    assert max(v for v in opts) < float(m["peak_quality"])


def test_the_budget_slider_reads_its_max_from_the_manifest():
    assert "budgetSlider.max = String(m.budget_slider_max)" in APP


# ── #1 the phone transform ───────────────────────────────────────────────────

def _js(name):
    start = APP.index(f"function {name}(")
    depth, i = 0, APP.index("{", start)
    while True:
        ch = APP[i]
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return APP[start:i + 1]
        i += 1


def _phone(fig: dict, width: int) -> dict:
    if not NODE:
        pytest.skip("node not available")
    consts = re.search(r"const PHONE_PX = .*?\nconst PHONE_LEGEND_ROW_PX = \d+;", APP, re.S).group(0)
    script = (consts + "\n" + "\n".join(_js(n) for n in (
        "narrowFigure", "phoneTitle", "legendEntries", "phoneFigure"))
        + f"\nconst fig = {json.dumps(fig)};\n"
        + f"process.stdout.write(JSON.stringify(narrowFigure(fig, {width})));")
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_on_a_phone_every_figure_drops_its_subtitle_labels_and_parks_the_legend():
    fig = _fig(api.update_overview(None, 0, "", "price"))
    phone = _phone(fig, 356)
    lay = phone["layout"]
    assert "<span" not in lay["title"]["text"]
    assert all(len(ln) <= (356 - 32) / 8.4 for ln in lay["title"]["text"].split("<br>"))
    assert not any(a.get("name") == POINT_LABEL for a in lay["annotations"])
    assert lay["legend"]["yref"] == "container" and lay["legend"]["y"] == 0
    assert lay["margin"]["b"] > 64
    assert len(lay["xaxis"]["tickvals"]) <= (356 - 84) // 52 + 1
    assert lay["xaxis"]["tickangle"] == 0
    # The desktop figure is untouched.
    assert _phone(fig, 1374) == fig


def test_on_a_phone_text_traces_are_hidden():
    fig = {"data": [{"type": "scatter", "mode": "text", "x": [1], "y": [1], "text": ["a"]},
                    {"type": "scatter", "mode": "markers+text", "x": [1], "y": [1]}],
           "layout": {"title": {"text": "A chart title far too long to fit on one phone line "
                                        "<span>sub</span>"}}}
    phone = _phone(fig, 356)
    assert phone["data"][0]["visible"] is False
    assert phone["data"][1]["mode"] == "markers"
    assert phone["layout"]["title"]["text"].count("<br>") >= 1
