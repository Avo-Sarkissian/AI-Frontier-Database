"""
Provider ecosystem treemap.
Tile area = number of models; tile color = average quality score.
"""
import pandas as pd
import plotly.graph_objects as go

from data.workload_mix import resolve_mix

from components.charts.constants import (
    BG as _BG, FONT as _FONT, QUALITY_INDEX_MAX, plot_text,
    dedupe_to_best_variant, empty_state_figure,
)

# (position, colour) stops; see build_treemap for why this ramp.
TREEMAP_COLORSCALE = [
    [0.0,  "#1a2330"],
    [0.2,  "#1b3654"],
    [0.4,  "#1f5b88"],
    [0.6,  "#2585b6"],
    [0.8,  "#52b5dd"],
    [1.0,  "#a8e3f4"],
]


def _hex_rgb(c: str) -> tuple[int, int, int]:
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def _luminance(rgb) -> float:
    def ch(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ramp_color(value: float) -> tuple[int, int, int]:
    """The RGB the treemap paints for `value` on the fixed 0..QUALITY_INDEX_MAX domain."""
    t = 0.0 if value != value else min(1.0, max(0.0, float(value) / QUALITY_INDEX_MAX))
    for (p0, c0), (p1, c1) in zip(TREEMAP_COLORSCALE, TREEMAP_COLORSCALE[1:]):
        if t <= p1:
            f = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            a, b = _hex_rgb(c0), _hex_rgb(c1)
            return tuple(round(x + (y - x) * f) for x, y in zip(a, b))
    return _hex_rgb(TREEMAP_COLORSCALE[-1][1])


def tile_label(provider: str) -> str:
    """The provider name as tile text: escaped, and broken at spaces when it
    runs past 10 characters ("Thinking<br>Machines", but "Z AI" stays whole)."""
    text = plot_text(provider)
    return text.replace(" ", "<br>") if len(str(provider)) > 10 else text


def label_color(value: float) -> str:
    """White on dark tiles, black on light ones: whichever has the higher
    WCAG contrast against the tile's fill. Pure white and black, because the
    better of the two is then >= 4.58:1 on ANY fill; the softer #f2f2f2 /
    #111111 pair falls to 4.2:1 where the two cross over (a mid-ramp tile)."""
    lum = _luminance(ramp_color(value))
    on_white = 1.05 / (lum + 0.05)
    on_black = (lum + 0.05) / 0.05
    return "#ffffff" if on_white >= on_black else "#000000"


def build_treemap(df: pd.DataFrame, basis=None) -> go.Figure:
    """``basis``: the workload MIX the prices are at; the avg-price hover names it."""
    basis = resolve_mix(basis)
    # One row per model family (its best tier): a tile's size counts MODELS,
    # and five effort tiers of one model are not five models — Anthropic read
    # "20 models" for four.
    df = dedupe_to_best_variant(df[df["quality"] > 0]) if not df.empty else df
    if df.empty:
        return empty_state_figure("Provider Landscape", "No models match these filters",
                                  "Clear the search, widen PROVIDER, or lower MIN SCORE")
    # Aggregate by provider
    agg = (
        df.groupby("provider")
        .agg(
            model_count=("model", "count"),
            avg_quality=("quality", "mean"),
            avg_price=("price", "mean"),
            avg_speed=("speed", "mean"),
            best_model=("quality", lambda s: df.loc[s.idxmax(), "model"]),
        )
        .reset_index()
    )
    # Provider name breaks count ties, so equal-sized tiles keep one order in
    # the CI-built figure and the in-browser re-render instead of reshuffling.
    agg = agg[agg["model_count"] >= 1].sort_values(
        ["model_count", "provider"], ascending=[False, True], kind="mergesort"
    )

    hover = (
        "<b>%{label}</b><br>"
        "Models: %{customdata[0]}<br>"
        "Avg Intelligence: %{customdata[1]:.1f}<br>"
        "Avg Price: $%{customdata[2]:.4g}/M  ·  " + basis.short + "<br>"
        "Best model: %{customdata[3]}<br>"
        "<extra></extra>"
    )

    fig = go.Figure(go.Treemap(
        labels=agg["provider"].map(plot_text),
        parents=[""] * len(agg),
        values=agg["model_count"],
        # [4] the tile label, wrapped at spaces: Plotly never wraps tile text,
        # so "Thinking Machines" needed a 90 px-wide tile or was hidden.
        customdata=agg.assign(best_model=agg["best_model"].map(plot_text),
                              tile_label=agg["provider"].map(tile_label))
                     [["model_count", "avg_quality", "avg_price", "best_model",
                       "tile_label"]].values,
        hovertemplate=hover,
        marker=dict(
            colors=agg["avg_quality"],
            # Fixed domain. Without cmin/cmax Plotly autoscales the ramp to
            # whatever subset is on screen, so a provider changed colour with
            # the filter — and could go DARKER as its average quality rose,
            # inverting the encoding. A single-provider filter was worse still:
            # the degenerate range collapsed to v±0.5 and every provider painted
            # the same mid-ramp navy whatever its actual score.
            cmin=0.0,
            cmax=QUALITY_INDEX_MAX,
            # Sequential and perceptually even: lightness climbs steadily
            # from the first stop to the last. The ramp it replaces stayed
            # navy for its first 55%, and provider averages live in that
            # stretch (v5: roughly 10-45 of 70), so every tile read as the
            # same dark blue. Every stop is >= 12 RGB units clear of the
            # #111111 page, and the ramp is monotonic in luminance (pinned
            # by a test), so a higher score can never paint darker.
            colorscale=TREEMAP_COLORSCALE,
            showscale=True,
            # A short horizontal key under the tiles: the vertical bar beside
            # them took a fifth of a phone's width from the tiles.
            colorbar=dict(
                orientation="h",
                x=0, xanchor="left", y=-0.03, yanchor="top",
                lenmode="pixels", len=220,
                thickness=8,
                tickfont=dict(color="#999999", size=10, family=_FONT),
                title=dict(
                    text="Avg intelligence",
                    font=dict(color="#999999", size=10, family=_FONT),
                    side="right",
                ),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                outlinewidth=0,
            ),
        ),
        # Label colour per tile from its fill, so the light end of the ramp
        # carries dark text instead of white at under 2:1.
        textfont=dict(family=_FONT, size=13,
                      color=[label_color(c) for c in agg["avg_quality"]]),
        texttemplate="%{customdata[4]}",
        tiling=dict(packing="squarify", pad=2),
        pathbar=dict(visible=False),
        # The implicit root behind the top-level tiles drew as a grey frame
        # and band; painted the page colour it disappears.
        root=dict(color=_BG),
        marker_pad=dict(t=0, l=0, r=0, b=0),
    ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Provider Landscape"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  area = # models  ·  color = avg intelligence</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        margin=dict(l=20, r=20, t=52, b=56),
        # Plotly shrinks a label to whatever fits its tile, down to ~7 px on
        # the small ones. Below 10 px it hides the label instead (the hover
        # still names the provider).
        uniformtext=dict(minsize=10, mode="hide"),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )

    return fig
