# tests/test_chart_contract.py
"""Invariants every provider-coloured scatter must hold.

The Overview tab renders build_pareto_scatter for the Price axis and
build_quadrant for the Speed axis, so a fix applied to only one of them leaves
half the tab broken — which is exactly what happened after the first pass.
These run against both.
"""
import pytest

from components.charts.constants import (
    PROVIDER_COLORS, BUBBLE_MIN_PX, BUBBLE_MAX_PX, SPOTLIGHT_PROVIDERS,
    bubble_size, safe_corr,
)
from components.charts.pareto import build_pareto_scatter
from components.charts.quadrant import build_quadrant
from data.ingest import get_models

BUILDERS = [
    pytest.param(build_pareto_scatter, id="pareto"),
    pytest.param(build_quadrant, id="quadrant"),
]


@pytest.fixture(scope="module")
def real_df():
    return get_models()


def _marker_traces(fig):
    return [t for t in fig.data if "markers" in str(getattr(t, "mode", ""))]


def _sizes_by_model(fig):
    out = {}
    for t in _marker_traces(fig):
        if not t.customdata:
            continue
        for cd, size in zip(t.customdata, t.marker.size):
            out[cd[0]] = size
    return out


# --- encoding stability ------------------------------------------------------

@pytest.mark.parametrize("build", BUILDERS)
def test_marker_size_is_invariant_under_filtering(build, real_df):
    """Both charts normalised marker size against the plotted frame's own max,
    so a model changed size when the user narrowed a filter."""
    full = _sizes_by_model(build(real_df))
    provider = real_df["provider"].value_counts().index[0]
    subset = _sizes_by_model(build(real_df[real_df["provider"] == provider]))
    shared = set(full) & set(subset)
    assert shared, "no overlapping models to compare"
    for model in shared:
        assert full[model] == pytest.approx(subset[model]), (
            f"{model} resized {full[model]:.1f} -> {subset[model]:.1f} under filtering"
        )


@pytest.mark.parametrize("build", BUILDERS)
def test_markers_stay_within_the_occlusion_budget(build, real_df):
    for t in _marker_traces(build(real_df)):
        for s in t.marker.size:
            assert BUBBLE_MIN_PX - 0.01 <= s <= BUBBLE_MAX_PX + 0.01, f"{s}px out of range"


@pytest.mark.parametrize("build", BUILDERS)
def test_markers_have_a_visible_separating_outline(build, real_df):
    for t in _marker_traces(build(real_df)):
        line = t.marker.line
        assert line.width and line.width > 0
        assert "rgba(0,0,0,0)" not in str(line.color)


# --- palette integrity -------------------------------------------------------

@pytest.mark.parametrize("build", BUILDERS)
def test_only_spotlight_providers_get_their_own_colour(build, real_df):
    for t in _marker_traces(build(real_df)):
        assert t.name == "Other" or t.name in SPOTLIGHT_PROVIDERS, (
            f"{t.name} is charted as its own series but is outside the "
            f"validated spotlight palette"
        )


@pytest.mark.parametrize("build", BUILDERS)
def test_provider_colours_are_unique_on_screen(build, real_df):
    colours = [t.marker.color for t in _marker_traces(build(real_df))]
    assert len(colours) == len(set(colours)), f"duplicate colours: {colours}"


@pytest.mark.parametrize("build", BUILDERS)
def test_legend_sits_below_the_plot_not_in_a_side_gutter(build, real_df):
    """The vertical legend clipped once it outgrew the plot height and cost the
    data most of the width on narrow viewports."""
    fig = build(real_df)
    assert fig.layout.legend.orientation == "h"
    assert fig.layout.margin.r <= 40, f"right gutter still {fig.layout.margin.r}px"


# --- no NaN reaches the canvas -----------------------------------------------

@pytest.mark.parametrize("build", BUILDERS)
def test_no_nan_annotation_for_a_single_model(build, real_df):
    one = real_df[(real_df["price"] > 0) & (real_df["speed"] > 0)].head(1)
    for ann in (build(one).layout.annotations or []):
        assert "nan" not in str(ann.text).lower()


@pytest.mark.parametrize("build", BUILDERS)
def test_correlation_is_present_for_the_full_catalogue(build, real_df):
    texts = " ".join(str(a.text) for a in (build(real_df).layout.annotations or []))
    assert "r = " in texts and "nan" not in texts.lower()


@pytest.mark.parametrize("build", BUILDERS)
def test_empty_frame_does_not_raise(build, real_df):
    assert build(real_df.iloc[0:0]) is not None


# --- direct labels stay legible ----------------------------------------------

@pytest.mark.parametrize("build", BUILDERS)
def test_direct_labels_are_not_stacked_on_one_another(build, real_df):
    """Taking the top-N by rank put three labels on the same pixels — the
    leaders cluster, so 'Gemini 3.6 Flash' and 'Gemini 3.5 Flash' collided."""
    import math

    fig = build(real_df)
    for t in fig.data:
        if getattr(t, "mode", None) != "text" or t.x is None or len(t.x) < 2:
            continue
        xs = [math.log10(v) if fig.layout.xaxis.type == "log" else float(v) for v in t.x]
        ys = [float(v) for v in t.y]
        xr = (max(xs) - min(xs)) or 1.0
        yr = (max(ys) - min(ys)) or 1.0
        pts = [((x - min(xs)) / xr, (y - min(ys)) / yr) for x, y in zip(xs, ys)]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = math.dist(pts[i], pts[j])
                assert d > 0.02, f"labels {t.text[i]!r} and {t.text[j]!r} overlap (d={d:.3f})"


@pytest.mark.parametrize("build", BUILDERS)
def test_direct_labels_are_selective(build, real_df):
    """A number on every point is noise; only a handful earn a label."""
    fig = build(real_df)
    labelled = sum(len(t.x) for t in fig.data if getattr(t, "mode", None) == "text")
    plotted = sum(len(t.x) for t in _marker_traces(fig))
    assert labelled <= max(8, plotted * 0.1), f"{labelled} labels over {plotted} points"


# --- the shared helpers themselves -------------------------------------------

def test_bubble_size_is_bounded_and_monotonic():
    sizes = bubble_size([0, 1, 50, 200, 900, 5000], ref=900.0)
    assert list(sizes) == sorted(sizes)
    assert sizes.min() >= BUBBLE_MIN_PX and sizes.max() <= BUBBLE_MAX_PX


def test_bubble_size_inverts_for_affordability():
    cheap, dear = bubble_size([0.1, 19.0], ref=20.0, invert=True)
    assert cheap > dear


def test_bubble_size_clamps_beyond_the_reference():
    at_ref, way_past = bubble_size([900.0, 100000.0], ref=900.0)
    assert at_ref == pytest.approx(way_past) == pytest.approx(BUBBLE_MAX_PX)


def test_bubble_size_handles_missing_and_nonpositive():
    assert list(bubble_size([None, 0, -5], ref=900.0)) == [BUBBLE_MIN_PX] * 3


@pytest.mark.parametrize("x,y", [
    ([1.0], [2.0]),                      # single point
    ([1.0, 2.0], [3.0, 4.0]),            # below min_points
    ([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]),  # zero variance -> nan
])
def test_safe_corr_returns_none_instead_of_nan(x, y):
    assert safe_corr(x, y) is None


def test_safe_corr_computes_a_real_correlation():
    r = safe_corr([1, 2, 3, 4], [2, 4, 6, 8])
    assert r == pytest.approx(1.0)


def test_spotlight_palette_is_not_silently_widened():
    """MAX_LEGEND_PROVIDERS tracks the validated set; widening it puts
    unvalidated colour pairs on screen.

    THE CEILING IS A MEASUREMENT, NOT A CONSTANT. It read "nine chromatic" for
    most of this file's life, then ten-with-one-achromatic, and it is now
    eleven chromatic plus achromatic OpenAI. Each move was a re-solve, never a
    loosened assertion: the set is only legal because every pair was re-checked
    against the floors, which tests/test_pareto_chart.py does independently.

    Kimi and Z AI were promoted on 2026-09-02 because the data had made the
    omission indefensible — by best model they ranked 4th and 5th, ahead of
    four labs that already had their own colour, and Kimi K3 (max) was the 12th
    strongest model in the catalogue while rendering as grey "Other".

    Twelve saturates this surface. Seating them cost the palette its slack: the
    ten-way set had room to spare and the twelve-way set meets the normal-vision
    floor at exactly 15.0. Marker shape carries proportionally more of the load
    now, which is why PROVIDER_SHAPES pins all twelve explicitly. A thirteenth
    should be assumed infeasible until a re-solve says otherwise.
    """
    assert all(p in PROVIDER_COLORS for p in SPOTLIGHT_PROVIDERS)

    def _chroma(hex_colour):
        r, g, b = (int(hex_colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        return max(r, g, b) - min(r, g, b)

    chromatic = [p for p in SPOTLIGHT_PROVIDERS
                 if _chroma(PROVIDER_COLORS[p]) > 12]
    assert len(chromatic) <= 11, (
        f"{len(chromatic)} chromatic spotlight hues ({chromatic}) — eleven is "
        f"what this surface was last shown to separate. Re-run the palette "
        f"validator before adding another, or make the newcomer achromatic."
    )
    achromatic = [p for p in SPOTLIGHT_PROVIDERS
                  if _chroma(PROVIDER_COLORS[p]) <= 12]
    assert len(achromatic) == 1, (
        f"expected exactly one achromatic spotlight entry (OpenAI, which is "
        f"AA's neutral inverted for this surface), got {achromatic}"
    )


def test_user_specified_brand_colours_are_honoured():
    """Anthropic keeps its orange and Meta its blue — explicit requirements,
    and the two hues four and three other labs respectively also claim."""
    import colorsys

    def hue_deg(hex_colour):
        r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return colorsys.rgb_to_hsv(r, g, b)[0] * 360

    anthropic = hue_deg(PROVIDER_COLORS["Anthropic"])
    meta = hue_deg(PROVIDER_COLORS["Meta"])
    assert 10 <= anthropic <= 45, f"Anthropic should read orange, got hue {anthropic:.0f}"
    assert 195 <= meta <= 240, f"Meta should read blue, got hue {meta:.0f}"


def test_run_local_families_share_the_provider_palette():
    """A lab must read as one colour across tabs. Run Local used to keep its own
    hand-maintained family palette, so Meta was indigo there and blue on Overview,
    and every lab added upstream since fell through to grey."""
    from data.local_models import family_color, DEFAULT_FAMILY_COLOR
    from data.local_models import get_local_df

    catalog = get_local_df()
    uncoloured = sorted({
        f for f in catalog["family"].dropna().unique()
        if family_color(f) == DEFAULT_FAMILY_COLOR
    })
    assert not uncoloured, f"families with no colour: {uncoloured}"

    for family in catalog["family"].dropna().unique():
        if family in PROVIDER_COLORS:
            assert family_color(family) == PROVIDER_COLORS[family], (
                f"{family} reads differently on Run Local than on Overview"
            )


def test_cost_chart_gives_every_model_its_own_row():
    """Truncated names can collide (the Nemotron 3 Nano Reasoning/Non-reasoning
    pair), and two models on one category makes Plotly stack both bars and draw
    both right-hand labels on top of each other."""
    from components.charts.cost_calc import build_cost_calc
    from data.ingest import get_models

    ys = list(build_cost_calc(get_models(), monthly_tokens_m=1.0).data[1].y)
    assert len(ys) == len(set(ys)), "duplicate categories in the cost chart"


# ── Right gutters: annotations must not cost the bars their width ────────────

def _annotated_charts():
    from data.ingest import get_models
    from data.local_models import get_local_df
    from data.video_models import get_video_df
    from components.charts.provider_leaderboard import build_provider_leaderboard
    from components.charts.cost_calc import build_cost_calc
    from components.charts.local_compat import build_local_compat
    from components.charts.video_chart import build_video_rankings

    df, loc, vid = get_models(), get_local_df(), get_video_df()
    return {
        "provider_leaderboard": build_provider_leaderboard(df),
        "cost_calc": build_cost_calc(df),
        "local_compat": build_local_compat(loc, "Q4"),
        "video_rankings": build_video_rankings(vid),
    }


def test_right_gutters_stay_under_the_shared_cap():
    """These four hard-coded 130-280px gutters; the leaderboard's bars got ~43%
    of the width on desktop and far less on a phone."""
    from components.charts.constants import MAX_RIGHT_GUTTER_PX
    for name, fig in _annotated_charts().items():
        assert fig.layout.margin.r <= MAX_RIGHT_GUTTER_PX, (
            f"{name} reserves {fig.layout.margin.r}px on the right"
        )


def test_annotated_charts_do_not_also_reserve_axis_headroom():
    """The gutter was doubled up with 30-55% of empty axis: bars were squeezed
    twice. Annotations are anchored to the paper edge now, so the axis needs
    almost none."""
    for name, fig in _annotated_charts().items():
        rng = fig.layout.xaxis.range
        if rng is None:
            continue
        lo, hi = rng
        bars = [t for t in fig.data if t.type == "bar" and t.x is not None]
        widest = max((max(t.x) for t in bars if len(t.x)), default=0)
        if widest <= 0:
            continue
        assert hi <= widest * 1.10, (
            f"{name} leaves {hi / widest:.2f}x headroom past its longest bar"
        )


def test_annotations_are_anchored_to_the_paper_edge():
    for name, fig in _annotated_charts().items():
        refs = {a.xref for a in fig.layout.annotations if a.text}
        assert refs <= {"paper"}, f"{name} still positions annotations in data space: {refs}"


def test_video_gen_shares_the_provider_palette():
    """Video Gen kept a private 13-colour list, so Google was blue there and
    amber everywhere else."""
    from data.video_models import PROVIDER_COLORS as VIDEO, DEFAULT_COLOR as VIDEO_DEFAULT
    from data.video_models import get_video_df

    catalog = get_video_df()
    uncoloured = sorted({
        p for p in catalog["provider"].dropna().unique()
        if VIDEO.get(p, VIDEO_DEFAULT) == VIDEO_DEFAULT
    })
    assert not uncoloured, f"video providers with no colour: {uncoloured}"

    for provider in catalog["provider"].dropna().unique():
        if provider in PROVIDER_COLORS:
            assert VIDEO[provider] == PROVIDER_COLORS[provider], (
                f"{provider} reads differently on Video Gen"
            )


def test_annotations_fit_inside_their_gutter():
    """Capping the gutter is only safe if the labels are trimmed to match --
    otherwise the longest one runs off the canvas and reads as a broken render.
    cost_calc is excluded: its annotation is span markup, so character length
    is not visible width.

    Width is the longest LINE, not the whole string: local_compat's annotation
    is two lines joined by <br> (single-stream speed and VRAM, then the
    concurrent sessions the same hardware supports), and summing them budgeted
    twice the width the render actually needs."""
    from components.charts.constants import MAX_RIGHT_GUTTER_PX
    for name, fig in _annotated_charts().items():
        if name == "cost_calc":
            continue
        gutter = fig.layout.margin.r
        for a in fig.layout.annotations:
            if not a.text or "<span" in a.text:
                continue
            widest = max(len(line) for line in a.text.split("<br>"))
            approx_px = widest * 10 * 0.55
            assert approx_px <= gutter + 8, (
                f"{name}: {a.text!r} needs ~{approx_px:.0f}px of a {gutter}px gutter"
            )


def test_palette_mirrors_artificial_analysis():
    """Every provider wears Artificial Analysis's own creator colour, or its hue.

    AA publishes a `modelCreatorColor` per company and paints it on their charts.
    Read 2026-09-01 from the leaderboard, text-to-image, video and text-to-speech
    payloads and checked against the rendered SVG fills, not just the JSON.
    Mirroring it is the whole point of this palette, so it is asserted:

      * AA_MIRROR_EXACT - painted byte for byte as AA paints it
      * AA_MIRROR_HUE   - AA's colour is unusable on #111111 or collides on the
                          Overview scatter, so L and C move but the HUE is AA's
      * AA_NO_HUE       - deliberately not mirrored, each with its reason

    A provider with an AA colour that appears in none of the three has drifted
    out of the mirror unnoticed, which is the failure this catches.
    """
    import math

    from components.charts.constants import (
        AA_CREATOR_COLORS, AA_MIRROR_EXACT, AA_MIRROR_HUE, AA_NO_HUE,
    )

    def oklch_hue(hex_colour):
        h = hex_colour.lstrip("#")
        srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                   for c in srgb]
        l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
        m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
        s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
        a_ = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
        b_ = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
        return math.degrees(math.atan2(b_, a_)) % 360

    for provider in sorted(AA_MIRROR_EXACT):
        assert PROVIDER_COLORS[provider] == AA_CREATOR_COLORS[provider], (
            f"{provider} should be AA's {AA_CREATOR_COLORS[provider]} verbatim, "
            f"got {PROVIDER_COLORS[provider]}"
        )

    for provider in sorted(AA_MIRROR_HUE):
        ours, theirs = PROVIDER_COLORS[provider], AA_CREATOR_COLORS[provider]
        delta = abs(oklch_hue(ours) - oklch_hue(theirs)) % 360
        delta = min(delta, 360 - delta)
        assert delta <= 1.0, (
            f"{provider} drifted {delta:.1f} degrees off AA's hue "
            f"({theirs} -> {ours}); mirroring AA means keeping the hue"
        )

    accounted = set(AA_MIRROR_EXACT) | set(AA_MIRROR_HUE) | set(AA_NO_HUE)
    mirrorable = {p for p in PROVIDER_COLORS if p in AA_CREATOR_COLORS}
    assert mirrorable <= accounted, (
        f"providers with an AA colour that no rule covers: "
        f"{sorted(mirrorable - accounted)}"
    )
