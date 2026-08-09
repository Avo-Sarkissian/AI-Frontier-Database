# tests/test_pareto_chart.py
"""Invariants for the Overview tab's Cost vs. Intelligence scatter.

These pin the defects found in the 2026-08-09 audit: unreadable log-axis
ticks, a legend taller than the plot area, bubbles that rescale when filters
change, marker occlusion, and a literal "r = nan" annotation.
"""
import math

import pandas as pd
import pytest

from components.charts.constants import (
    PROVIDER_COLORS, PROVIDER_SHAPES, DEFAULT_SHAPE, MAX_LEGEND_PROVIDERS,
    SPOTLIGHT_PROVIDERS,
)


# --- OKLab ΔE, mirroring the data-viz palette validator ----------------------

def _srgb_to_oklab(hex_color):
    h = hex_color.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def _delta_e(a, b):
    pa, pb = _srgb_to_oklab(a), _srgb_to_oklab(b)
    return 100 * math.dist(pa, pb)


NORMAL_VISION_FLOOR = 15.0   # OKLab ΔE×100, the validator's hard gate
from components.charts.pareto import build_pareto_scatter
from data.ingest import get_models


@pytest.fixture(scope="module")
def real_df():
    return get_models()


@pytest.fixture(scope="module")
def fig(real_df):
    return build_pareto_scatter(real_df)


def _marker_traces(figure):
    return [t for t in figure.data if getattr(t, "mode", "") == "markers"]


def _legend_traces(figure):
    return [t for t in figure.data if getattr(t, "showlegend", None) is not False]


# --- x axis ------------------------------------------------------------------

def test_log_xaxis_declares_explicit_ticks(fig):
    """Without tickvals Plotly falls back to D2 minor ticks and renders the
    nonsense sequence '2 5 0.1 2 5 1 2 5 10 2 5 100'."""
    xa = fig.layout.xaxis
    assert xa.type == "log"
    assert xa.tickvals, "log x axis must declare explicit tickvals"
    assert xa.ticktext, "log x axis must declare explicit ticktext"
    assert len(xa.tickvals) == len(xa.ticktext)


def test_xaxis_tick_labels_are_prices_not_bare_mantissas(fig):
    """The leftmost label used to read '2' for a $0.02 model."""
    assert all(t.startswith("$") for t in fig.layout.xaxis.ticktext)


def test_xaxis_ticks_span_the_data(fig, real_df):
    priced = real_df[(real_df["price"] > 0) & (real_df["quality"] > 0)]
    lo, hi = priced["price"].min(), priced["price"].max()
    vals = list(fig.layout.xaxis.tickvals)
    assert min(vals) <= lo * 1.5 and max(vals) >= hi / 1.5


# --- legend ------------------------------------------------------------------

def test_legend_is_small_enough_not_to_clip(fig):
    """A 25-entry vertical legend is ~487px tall against a ~300px plot area,
    so Plotly clips it and hides the major labs behind a scrollbar."""
    assert len(_legend_traces(fig)) <= MAX_LEGEND_PROVIDERS + 2  # + Other + Pareto


def test_major_providers_appear_in_legend(fig, real_df):
    """Alphabetical ordering used to push OpenAI/Meta/Mistral below the fold."""
    top2 = list(real_df["provider"].value_counts().head(2).index)
    names = {t.name for t in _legend_traces(fig)}
    for provider in top2:
        assert provider in names, f"{provider} missing from legend"


def test_legend_is_ordered_by_model_count_not_alphabetically(fig):
    """Alphabetical order buried the majors; densest provider must lead."""
    named = [t for t in _marker_traces(fig) if t.name != "Other"]
    sizes = [len(t.x) for t in named]
    assert sizes == sorted(sizes, reverse=True), (
        f"legend not count-ordered: {[(t.name, len(t.x)) for t in named]}"
    )


def test_other_bucket_sorts_last(fig):
    names = [t.name for t in _marker_traces(fig)]
    if "Other" in names:
        assert names[-1] == "Other"


# --- encoding integrity ------------------------------------------------------

def test_legend_providers_have_unique_colors(fig):
    """OpenAI and AI21 Labs both shipped #34d399."""
    colors = [t.marker.color for t in _marker_traces(fig)]
    assert len(colors) == len(set(colors)), f"duplicate provider colors: {colors}"


def test_legend_providers_have_unique_shapes(fig):
    """The subtitle claims 'shape = provider family'; it must be true for
    every provider the legend actually names."""
    shapes = [t.marker.symbol for t in _marker_traces(fig) if t.name != "Other"]
    assert len(shapes) == len(set(shapes)), f"duplicate provider shapes: {shapes}"


def test_no_duplicate_colors_in_the_palette():
    dupes = {}
    for name, hexv in PROVIDER_COLORS.items():
        dupes.setdefault(hexv, []).append(name)
    collisions = {k: v for k, v in dupes.items() if len(v) > 1}
    assert not collisions, f"palette color collisions: {collisions}"


def test_every_colored_provider_also_has_a_shape():
    missing = sorted(set(PROVIDER_COLORS) - set(PROVIDER_SHAPES))
    assert not missing, f"providers with a color but no shape: {missing}"


def test_spotlight_shapes_are_distinct_and_not_all_one_family():
    """Shape is the secondary encoding that makes the 6-8 CVD floor legal, so
    it has to actually separate marks. star-triangle-up read as identical to
    triangle-up at bubble sizes, which is why NVIDIA moved to hourglass."""
    shapes = [PROVIDER_SHAPES[p] for p in SPOTLIGHT_PROVIDERS]
    assert len(shapes) == len(set(shapes)), f"duplicate spotlight shapes: {shapes}"
    families = {}
    for s in shapes:
        families.setdefault(s.split("-")[0], []).append(s)
    crowded = {k: v for k, v in families.items() if len(v) > 2}
    assert not crowded, f"too many look-alike marks in one family: {crowded}"


def test_spotlight_providers_are_fully_specified():
    for p in SPOTLIGHT_PROVIDERS:
        assert p in PROVIDER_COLORS, f"spotlight provider {p} has no color"
        assert p in PROVIDER_SHAPES, f"spotlight provider {p} has no shape"
    assert MAX_LEGEND_PROVIDERS == len(SPOTLIGHT_PROVIDERS)


def test_spotlight_palette_clears_the_normal_vision_floor():
    """The ten spotlight colors can all share the Overview scatter, so every
    pair must be separable — the data-viz validator's hard gate is ΔE >= 15
    (OKLab x100) with all pairs in play. Re-run the validator before editing
    any of these hexes:
      node scripts/validate_palette.js "<hexes>" --mode dark \
           --surface "#111111" --pairs all
    """
    hexes = [PROVIDER_COLORS[p] for p in SPOTLIGHT_PROVIDERS]
    worst, pair = math.inf, None
    for i in range(len(hexes)):
        for j in range(i + 1, len(hexes)):
            d = _delta_e(hexes[i], hexes[j])
            if d < worst:
                worst, pair = d, (SPOTLIGHT_PROVIDERS[i], SPOTLIGHT_PROVIDERS[j])
    assert worst >= NORMAL_VISION_FLOOR, (
        f"{pair} only ΔE {worst:.1f} apart (floor {NORMAL_VISION_FLOOR})"
    )


def test_only_spotlight_providers_get_their_own_series(fig):
    for t in _marker_traces(fig):
        assert t.name == "Other" or t.name in SPOTLIGHT_PROVIDERS, (
            f"{t.name} is charted as its own series but is not in the "
            f"validated spotlight palette"
        )


# --- bubble sizing -----------------------------------------------------------

def _sizes_by_model(figure):
    out = {}
    for t in _marker_traces(figure):
        for cd, size in zip(t.customdata, t.marker.size):
            out[cd[0]] = size
    return out


def test_bubble_size_is_invariant_under_filtering(real_df):
    """Sizes were normalized against the filtered subset's max speed, so a
    model appeared to get faster when the user narrowed the provider filter."""
    full = _sizes_by_model(build_pareto_scatter(real_df))
    top_provider = real_df["provider"].value_counts().index[0]
    subset = real_df[real_df["provider"] == top_provider]
    filtered = _sizes_by_model(build_pareto_scatter(subset))
    shared = set(full) & set(filtered)
    assert shared, "no overlapping models to compare"
    for model in shared:
        assert full[model] == pytest.approx(filtered[model]), (
            f"{model} resized {full[model]} -> {filtered[model]} under filtering"
        )


def test_bubbles_are_bounded_to_limit_occlusion(fig):
    """36px bubbles buried 19 models entirely, making them unhoverable."""
    for t in _marker_traces(fig):
        for s in t.marker.size:
            assert 0 < s <= 26, f"bubble diameter {s}px exceeds occlusion budget"


def test_markers_have_a_visible_separating_outline(fig):
    """line.color was rgba(0,0,0,0) — dead config that let the dense sub-$1
    cluster merge into an undifferentiated blob."""
    for t in _marker_traces(fig):
        line = t.marker.line
        assert line.width and line.width > 0
        assert "rgba(0,0,0,0)" not in str(line.color)


# --- correlation annotation --------------------------------------------------

def test_no_nan_correlation_annotation_for_single_model(real_df):
    """Searching one model left 'r = nan  (log price vs quality)' on screen."""
    one = real_df[real_df["price"] > 0].head(1)
    out = build_pareto_scatter(one)
    for ann in (out.layout.annotations or []):
        assert "nan" not in str(ann.text).lower()


def test_correlation_annotation_present_for_full_data(fig):
    texts = " ".join(str(a.text) for a in (fig.layout.annotations or []))
    assert "r = " in texts and "nan" not in texts.lower()


def test_empty_frame_does_not_raise(real_df):
    empty = real_df.iloc[0:0]
    out = build_pareto_scatter(empty)
    assert out is not None
