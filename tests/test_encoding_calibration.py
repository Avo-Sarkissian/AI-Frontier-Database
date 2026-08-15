"""Encoding constants must be checked against the population that will be drawn.

Theme 3 of audit/2026-08-12, and the fourth defect pattern this codebase names:
hand-picked round numbers used as ceilings, floors and colour stops, never
compared with the data. Every one of these shipped in docs/figures/*.json, so
they reached visitors with no interaction at all.

The rule these tests encode: if a constant decides how long a bar is, how big a
bubble is, how far out a spoke reaches, or what colour a tile takes, then the
range it implies has to match the range the data actually occupies.
"""
import json
import math

import pandas as pd
import pytest

from data.ingest import get_models
from data.image_models import get_image_df
from data.video_models import get_video_df
from components.charts.constants import (
    bubble_size, BUBBLE_PRICE_REF, BUBBLE_PRICE_FLOOR, BUBBLE_MIN_PX, BUBBLE_MAX_PX,
)
from components.charts.radar import build_radar, radar_reference, _log_norm, _context_k
from components.charts.image_scatter import build_image_faceted
from components.charts.video_chart import build_video_rankings

DF = get_models()


# ── 3.1 — radar ──────────────────────────────────────────────────────────────

def _spoke_values(axis: str, ref: dict) -> list[float]:
    if axis == "speed":
        return [_log_norm(v, ref["speed_lo"], ref["speed_hi"])
                for v in DF["speed"] if pd.notna(v) and v > 0]
    if axis == "latency":
        return [_log_norm(v, ref["lat_lo"], ref["lat_hi"], invert=True)
                for v in DF["latency"] if pd.notna(v) and v > 0]
    if axis == "context":
        return [_log_norm(_context_k(c), ref["ctx_lo"], ref["ctx_hi"])
                for c in DF["context"]]
    if axis == "price":
        return [_log_norm(v, ref["price_lo"], ref["price_hi"], invert=True)
                for v in DF["price"] if pd.notna(v) and v > 0]
    raise AssertionError(axis)


@pytest.mark.parametrize("axis", ["speed", "latency", "context", "price"])
def test_no_radar_axis_crushes_the_population_into_the_centre(axis):
    """132 of 148 models used to draw a Speed spoke inside the innermost 10% of
    the radius, so the axis carried no signal — a 298 tok/s model and a 52 tok/s
    model both read as the origin."""
    ref = radar_reference(DF)
    values = _spoke_values(axis, ref)
    assert values, f"no {axis} values to test"
    crushed = sum(1 for v in values if v < 0.10)
    assert crushed / len(values) < 0.15, (
        f"{crushed} of {len(values)} models render inside the innermost 10% of "
        f"the {axis} spoke — the axis carries almost no signal"
    )


def test_the_best_model_can_reach_the_edge_of_every_radar_axis():
    """The Intelligence ceiling was 70.0 against a population max of 63.05, so
    the best model in the catalogue could only ever draw 90%."""
    ref = radar_reference(DF)
    assert abs(DF["quality"].max() / ref["quality_max"] - 1.0) < 1e-9
    for axis in ("speed", "latency", "context", "price"):
        assert max(_spoke_values(axis, ref)) > 0.99, (
            f"nothing reaches the outer edge of the {axis} axis"
        )


def test_distinct_context_sizes_never_collapse_onto_one_spoke_value():
    """Models that share a context window should share a spoke — 68 really do
    have 1m — but two DIFFERENT windows must never draw the same radius.

    The old ceiling was 2000k against a population max of exactly 1000k, so
    every 1M model pinned at precisely 50.0%: a round-number coincidence that
    read as a deliberate midpoint rather than a measurement."""
    ref = radar_reference(DF)
    pairs = {}
    for ctx in DF["context"]:
        k = _context_k(ctx)
        if k <= 0:
            continue
        pairs.setdefault(round(_log_norm(k, ref["ctx_lo"], ref["ctx_hi"]), 4), set()).add(k)
    collisions = {v: sorted(ks) for v, ks in pairs.items() if len(ks) > 1}
    assert not collisions, f"different context sizes drawing the same spoke: {collisions}"

    # And the ceiling must not be a round multiple of the population max, which
    # is what produced the 50.0% artifact.
    pop_max = max(_context_k(c) for c in DF["context"])
    assert abs(ref["ctx_hi"] / pop_max - 1.0) < 1e-9, (
        f"context ceiling {ref['ctx_hi']} is not the population max {pop_max}"
    )


def test_models_with_very_different_latency_do_not_draw_the_same_spoke():
    """The audit's stated verification: no two models more than 2x apart in
    real TTFT may render within 1% of each other. The 30s ceiling clamped
    everything slower to a flat 0%, so 32.52s and 101.87s were identical."""
    ref = radar_reference(DF)
    rows = DF[DF["latency"].notna() & (DF["latency"] > 0)]
    pairs = [(a.latency, b.latency,
              _log_norm(a.latency, ref["lat_lo"], ref["lat_hi"], invert=True),
              _log_norm(b.latency, ref["lat_lo"], ref["lat_hi"], invert=True))
             for a in rows.itertuples() for b in rows.itertuples()
             if a.latency >= 2 * b.latency]
    bad = [(la, lb) for la, lb, na, nb in pairs if abs(na - nb) < 0.01]
    assert not bad, (
        f"{len(bad)} model pairs differ by >2x in TTFT but draw within 1% of "
        f"each other, e.g. {bad[:3]}"
    )


def test_radar_axes_do_not_move_with_the_filter():
    """Fixed reference is the whole promise of the subtitle. A ceiling taken
    from the frame being drawn would change a model's shape as the user
    filters."""
    subset = DF[DF["provider"] == DF["provider"].value_counts().index[0]]
    full_ref = radar_reference(DF)
    picked = list(DF.sort_values("quality", ascending=False)["model"].head(3))
    a = json.loads(build_radar(DF, picked, full_df=DF).to_json())
    b = json.loads(build_radar(subset, picked, full_df=DF).to_json())
    ra = {t["name"]: t["r"] for t in a["data"] if t.get("name") in picked}
    rb = {t["name"]: t["r"] for t in b["data"] if t.get("name") in picked}
    for name in ra:
        if name in rb:
            assert ra[name] == rb[name], f"{name} changed shape under a filter"
    assert radar_reference(DF) == full_ref


def test_the_radar_caption_does_not_promise_a_linear_scale():
    """The old caption said 'normalized 0-100 across all models' while four of
    the five axes were neither."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    radar_src = (root / "components" / "charts" / "radar.py").read_text()
    assert "normalized 0–100 across all models</span>" not in radar_src
    assert "log scale" in radar_src, "the caption does not disclose the log axes"
    assert "log scale" in (root / "captions.py").read_text(), (
        "the Compare tab caption does not disclose the log axes"
    )


# ── 3.2 — image facet axes ───────────────────────────────────────────────────

def test_no_provider_filter_produces_a_backwards_image_axis():
    """`x_min = max(900, ...)` against a filtered frame made the headroom term
    negative whenever the best model scored under 900, so x_max < x_min. 17 of
    39 providers produced at least one inverted axis."""
    full = get_image_df()
    inverted = []
    for provider in sorted(full["provider"].dropna().unique()):
        fig = json.loads(build_image_faceted(full[full["provider"] == provider],
                                             full_df=full).to_json())
        for key in ("xaxis", "xaxis2", "xaxis3"):
            rng = fig["layout"].get(key, {}).get("range")
            if rng and not rng[0] < rng[1]:
                inverted.append((provider, key, rng))
    assert not inverted, f"axes running high-to-low: {inverted[:5]}"


def test_every_image_bar_falls_inside_its_axis():
    """Stability.ai drew (900, 844) against bars from 513 to 858 — every bar
    below the floor."""
    full = get_image_df()
    outside = []
    for provider in sorted(full["provider"].dropna().unique()):
        fig = json.loads(build_image_faceted(full[full["provider"] == provider],
                                             full_df=full).to_json())
        axes = [fig["layout"].get(k, {}).get("range")
                for k in ("xaxis", "xaxis2", "xaxis3")]
        for trace in fig["data"]:
            if trace.get("type") != "bar" or not trace.get("x"):
                continue
            idx = int(str(trace.get("xaxis", "x")).replace("x", "") or 1) - 1
            rng = axes[idx] if 0 <= idx < len(axes) else None
            if not rng:
                continue
            for v in trace["x"]:
                if isinstance(v, (int, float)) and not (rng[0] <= v <= rng[1] * 1.001):
                    outside.append((provider, v, rng))
    assert not outside, f"bars drawn outside their axis: {outside[:5]}"


def test_image_axis_range_is_filter_invariant():
    """Bar length must mean the same thing under every filter."""
    full = get_image_df()
    def ranges(frame):
        fig = json.loads(build_image_faceted(frame, full_df=full).to_json())
        return [fig["layout"].get(k, {}).get("range")
                for k in ("xaxis", "xaxis2", "xaxis3")]
    top = full["provider"].value_counts().index[0]
    assert ranges(full) == ranges(full[full["provider"] == top])


# ── 3.3 — treemap ramp ───────────────────────────────────────────────────────

def _relative_luminance(rgb) -> float:
    def _c(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (_c(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _treemap_scale():
    from components.charts.treemap import build_treemap
    fig = json.loads(build_treemap(DF).to_json())
    for trace in fig["data"]:
        scale = (trace.get("marker") or {}).get("colorscale")
        if scale:
            return [(float(t), c) for t, c in scale]
    pytest.fail("no colorscale found on the treemap")


def test_the_treemap_ramp_never_darkens_as_quality_rises():
    """A sequential ramp that reverses is worse than no colour at all. This one
    darkened over its first 30% — 60 consecutive negative luminance steps —
    and 112 of 496 provider pairs rendered the better provider darker."""
    stops = _treemap_scale()

    def sample(t):
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                a, b = _hex_to_rgb(c0), _hex_to_rgb(c1)
                return tuple(a[i] + (b[i] - a[i]) * f for i in range(3))
        return _hex_to_rgb(stops[-1][1])

    lums = [_relative_luminance(sample(i / 200)) for i in range(201)]
    negative = [i for i in range(1, len(lums)) if lums[i] < lums[i - 1] - 1e-12]
    assert not negative, (
        f"{len(negative)} steps of the ramp go backwards, first at "
        f"t={negative[0] / 200:.3f}"
    )
    assert lums[0] == min(lums), "the darkest colour is not at the bottom"


def test_the_treemap_floor_is_distinguishable_from_the_page():
    """The stop before last time was within 9 RGB units of the #111111 page, so
    the weakest provider was painted as the background. Fixing the monotonicity
    must not undo that."""
    stops = _treemap_scale()
    floor = _hex_to_rgb(stops[0][1])
    page = _hex_to_rgb("#111111")
    assert sum(abs(a - b) for a, b in zip(floor, page)) > 20, (
        "the ramp's darkest colour is indistinguishable from the page"
    )


# ── 3.4 — bubble size ────────────────────────────────────────────────────────

def test_price_bubbles_separate_the_models_that_are_actually_different():
    """53 of 95 models were sub-$1 and spanned 0.43 px in total — a 12.5x price
    difference rendered as no visible difference."""
    prices = DF[DF["price"] > 0]["price"]
    sizes = bubble_size(prices, BUBBLE_PRICE_REF, invert=True,
                        log=True, floor=BUBBLE_PRICE_FLOOR)
    frame = pd.DataFrame({"price": prices.values, "size": sizes.values})
    cheap = frame[frame["price"] < 1]
    assert len(cheap) > 20, "not enough cheap models to test"
    span = cheap["size"].max() - cheap["size"].min()
    assert span > 2.0, (
        f"the {len(cheap)} sub-$1 models span only {span:.2f} px of diameter"
    )


def test_a_tenfold_price_difference_is_always_visible():
    for cheap, dear in ((0.1, 1.0), (0.5, 5.0), (1.0, 10.0)):
        a, b = bubble_size([cheap, dear], BUBBLE_PRICE_REF, invert=True,
                           log=True, floor=BUBBLE_PRICE_FLOOR)
        assert a - b > 1.5, (
            f"${cheap} and ${dear} draw at {a:.2f} and {b:.2f} px — a decade of "
            f"price is not visible"
        )


def test_bubble_sizes_stay_within_their_declared_bounds():
    sizes = bubble_size(DF["price"], BUBBLE_PRICE_REF, invert=True,
                        log=True, floor=BUBBLE_PRICE_FLOOR)
    assert sizes.min() >= BUBBLE_MIN_PX - 1e-9
    assert sizes.max() <= BUBBLE_MAX_PX + 1e-9


# ── 3.5 — video bar baseline ─────────────────────────────────────────────────

def test_the_video_bar_axis_starts_where_its_title_says():
    """Bar length was the sole encoding on an axis starting at 40 under a
    '0–100' title, overstating the gap by 1.8x in the default view."""
    fig = json.loads(build_video_rankings(get_video_df()).to_json())
    rng = fig["layout"]["xaxis"]["range"]
    title = fig["layout"]["xaxis"]["title"]["text"]
    assert rng[0] == 0, (
        f"axis titled {title!r} starts at {rng[0]}, so bar length misstates the "
        f"proportion between models"
    )
