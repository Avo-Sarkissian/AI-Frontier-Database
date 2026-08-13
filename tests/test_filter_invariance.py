"""Scales, frontiers and thresholds must not be derived from the filtered frame.

Every render path receives a *user-filtered* frame. Any statistic computed from
that frame — a median crosshair, a Pareto frontier, a normalising maximum, a
chosen ELO column — silently redefines what the chart means as the user filters,
so the same model changes rank, zone or position for reasons that have nothing
to do with the model.

constants.py already applies this rule to bubble *size* (BUBBLE_PRICE_REF,
QUALITY_INDEX_MAX) and tests/test_pareto_chart.py guards it. It was never
applied to thresholds, frontier membership, metric selection or score
normalisation. These tests close that gap.
"""
import pandas as pd
import pytest

from data.ingest import get_models
from data.image_models import get_image_df
from data.video_models import get_video_df
from components.charts.pareto import _pareto_frontier, build_pareto_scatter
from components.charts.quadrant import build_quadrant
from components.charts.image_scatter import build_image_faceted, _CATEGORIES, _pick_elo_column
from components.charts.video_chart import build_video_scatter
from components.stack_recommender import _pick_api_tier, _API_TIERS


DF = get_models()


# ── Pareto frontier ───────────────────────────────────────────────────────────

def test_pareto_frontier_drops_a_dominated_model_on_a_price_tie():
    """`sort_values("price")` is an unstable quicksort with no secondary key, so
    when two models share a price the row order decides which is kept — and the
    strictly worse one can be appended first, putting it on a line labelled
    "Pareto Frontier"."""
    df = pd.DataFrame({
        "model":   ["cheap", "tie-weak", "tie-strong"],
        "price":   [1.0, 5.0, 5.0],
        "quality": [10.0, 20.0, 30.0],
    })
    front = set(_pareto_frontier(df)["model"])
    assert "tie-weak" not in front, "a model dominated at an identical price is on the frontier"
    assert front == {"cheap", "tie-strong"}


def test_pareto_frontier_is_the_non_dominated_set_of_the_live_catalogue():
    rows = DF[(DF["price"] > 0) & (DF["quality"] > 0)][["model", "price", "quality"]]
    recs = rows.to_dict("records")
    brute = {
        r["model"] for r in recs
        if not any(
            o["price"] <= r["price"] and o["quality"] >= r["quality"]
            and (o["price"] < r["price"] or o["quality"] > r["quality"])
            for o in recs
        )
    }
    assert set(_pareto_frontier(rows)["model"]) == brute


def _frontier_points(fig):
    for tr in fig.data:
        if getattr(tr, "name", None) == "Pareto Frontier":
            return set(zip(tr.x, tr.y))
    return set()


def test_pareto_frontier_membership_does_not_grow_under_a_filter():
    """Filtering may hide frontier points; it must never promote a model that
    the whole market already dominates."""
    full = _frontier_points(build_pareto_scatter(DF, full_df=DF))
    sub = DF[DF["provider"].isin(["OpenAI"])]
    narrowed = _frontier_points(build_pareto_scatter(sub, full_df=DF))
    assert narrowed <= full, f"filter invented frontier points: {narrowed - full}"


# ── Quadrant thresholds ───────────────────────────────────────────────────────

def _crosshairs(fig):
    """(vertical x, horizontal y) of the median crosshair shapes."""
    vx = hy = None
    for s in fig.layout.shapes:
        if s.type == "line":
            if s.x0 == s.x1 and s.y0 != s.y1 or (s.xref or "").startswith("x") and s.x0 == s.x1:
                vx = s.x0
            elif s.y0 == s.y1:
                hy = s.y0
    return vx, hy


@pytest.mark.parametrize("providers", [["OpenAI", "Anthropic"], ["Meta"], ["Google"]])
def test_quadrant_thresholds_are_invariant_under_a_provider_filter(providers):
    """"Fast · Smart" is an absolute-sounding claim. If the crosshairs move with
    the filter it is really a claim about who else is on screen."""
    full_vx, full_hy = _crosshairs(build_quadrant(DF, full_df=DF))
    sub = DF[DF["provider"].isin(providers)]
    sub_vx, sub_hy = _crosshairs(build_quadrant(sub, full_df=DF))
    assert (sub_vx, sub_hy) == (full_vx, full_hy)


def test_quadrant_y_axis_is_invariant_under_a_provider_filter():
    full = build_quadrant(DF, full_df=DF).layout.yaxis.range
    sub = build_quadrant(DF[DF["provider"] == "Meta"], full_df=DF).layout.yaxis.range
    assert sub == full


# ── Image Gen: which ELO column each facet uses ───────────────────────────────

def _chosen_elo_columns(df, full_df):
    return [_pick_elo_column(full_df, c["elo_cols"]) for c in _CATEGORIES]


def test_image_elo_columns_are_chosen_from_the_full_frame():
    """Each facet takes the first candidate column with data *in the frame it was
    handed*, and every category lists a retired 2025 column as fallback — so
    narrowing to one provider can silently switch a facet to a different metric,
    and models scored only on the current column vanish."""
    img = get_image_df()
    full_cols = _chosen_elo_columns(img, img)
    for provider in ["MiniMax", "DeepSeek", "Runway", "OpenGVLab"]:
        sub = img[img["provider"] == provider]
        if sub.empty:
            continue
        fig = build_image_faceted(sub, full_df=img)
        assert _chosen_elo_columns(sub, img) == full_cols
        assert fig is not None


# ── Video Gen frontier ────────────────────────────────────────────────────────

def test_video_frontier_membership_does_not_grow_under_a_filter():
    vdf = get_video_df()
    paid = vdf[vdf["price_per_sec"] > 0]
    full = _frontier_points(build_video_scatter(paid, full_df=paid))
    sub = paid[paid["provider"].isin(["Runway", "OpenAI"])]
    narrowed = _frontier_points(build_video_scatter(sub, full_df=paid))
    assert narrowed <= full, f"filter invented frontier points: {narrowed - full}"


# ── Agent Stack scoring ───────────────────────────────────────────────────────

def test_fast_tier_score_is_invariant_under_provider_selection():
    """The composite Fast score divides quality, value and speed by three
    separate pool-dependent maxima, so ticking an unrelated provider box changes
    the weighting and flips the rendered order."""
    tier = {**next(t for t in _API_TIERS if t["key"] == "fast"), "n": 10_000}
    full = _pick_api_tier(DF, tier, full_df=DF).set_index("model")["_score"]

    for providers in (["SpaceXAI"], ["Alibaba"], ["Anthropic", "Google"]):
        sub = DF[DF["provider"].isin(providers)]
        scored = _pick_api_tier(sub, tier, full_df=DF).set_index("model")["_score"]
        shared = scored.index.intersection(full.index)
        assert len(shared) > 0
        for model in shared:
            assert scored[model] == pytest.approx(full[model]), (
                f"{model} scores {scored[model]} under {providers} "
                f"but {full[model]} against the full catalogue"
            )


# ── The render paths users actually hit ───────────────────────────────────────
# The builders accepting a full_df argument proves nothing on its own; what
# matters is that static_api (the Pyodide bridge serving the public site) and
# app.py (Dash) both pass it. These tests go through the real entry points.

def _fig_from(json_str):
    import json as _json
    import plotly.graph_objects as go
    return go.Figure(_json.loads(json_str))


@pytest.mark.parametrize("providers", [["OpenAI", "Anthropic"], ["Meta"]])
def test_static_api_overview_thresholds_are_filter_invariant(providers):
    import static_api

    full = _crosshairs(_fig_from(static_api.update_overview(None, 0, "", "speed")))
    narrowed = _crosshairs(_fig_from(static_api.update_overview(providers, 0, "", "speed")))
    assert narrowed == full


def test_static_api_overview_frontier_does_not_grow_under_a_filter():
    import static_api

    full = _frontier_points(_fig_from(static_api.update_overview(None, 0, "", "price")))
    narrowed = _frontier_points(_fig_from(static_api.update_overview(["OpenAI"], 0, "", "price")))
    assert narrowed <= full


def test_static_api_overview_frontier_is_stable_across_min_score_presets():
    """One click on MIN SCORE was enough to promote a dominated model."""
    import static_api

    full = _frontier_points(_fig_from(static_api.update_overview(None, 0, "", "price")))
    for min_q in (10, 20, 30, 45):
        narrowed = _frontier_points(
            _fig_from(static_api.update_overview(None, min_q, "", "price"))
        )
        assert narrowed <= full, f"MIN SCORE >= {min_q} invented {narrowed - full}"


def test_static_api_image_facets_keep_their_metric_under_a_provider_filter(monkeypatch):
    """The builder taking a full_df argument is worthless unless update_image
    actually passes it, so capture what the render path really hands over."""
    import static_api

    img = get_image_df()
    seen = {}

    def _capture(df, full_df=None):
        seen["plotted"] = df
        seen["full"] = full_df
        return build_image_faceted(df, full_df=full_df)

    monkeypatch.setattr(static_api, "build_image_faceted", _capture)
    static_api.update_image(["MiniMax"], None)

    assert seen["full"] is not None, "update_image did not pass the full arena"
    assert len(seen["full"]) == len(img)
    assert len(seen["plotted"]) < len(img), "provider filter did not narrow the frame"
    assert _chosen_elo_columns(seen["plotted"], seen["full"]) == _chosen_elo_columns(img, img)


def test_static_api_video_frontier_does_not_grow_under_a_filter():
    import static_api

    full = _frontier_points(_fig_from(json_scatter(static_api.update_video(None, None))))
    narrowed = _frontier_points(
        _fig_from(json_scatter(static_api.update_video(["Runway", "OpenAI"], None)))
    )
    assert narrowed <= full


def json_scatter(payload: str) -> str:
    import json as _json
    return _json.dumps(_json.loads(payload)["scatter"])
