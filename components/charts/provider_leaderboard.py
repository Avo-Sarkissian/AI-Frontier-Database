"""
Provider leaderboard chart.

Aggregates all models by provider and shows:
  - Best model intelligence score (bar length)
  - Model count
  - Average intelligence score
  - Cheapest model price

One bar per provider, sorted by peak intelligence.
"""
import pandas as pd
import plotly.graph_objects as go

from data.workload_mix import resolve_mix

from components.charts.constants import (
    dedupe_to_best_variant, empty_state_figure,
    plot_text, clean_model_name,
    right_gutter, fit_text, ANNOTATED_AXIS_HEADROOM,
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)


LABEL_GREY = "#aaaaaa"


def count_label(n) -> str:
    """"1 model", "3 models" — the label read "1 models" for single-model labs."""
    n = int(n)
    return f"{n} model" if n == 1 else f"{n} models"


def build_provider_leaderboard(df: pd.DataFrame, basis=None) -> go.Figure:
    """Horizontal bar: one per provider, ordered by best model quality.

    ``basis``: the workload MIX the prices are at; the floor-price hover names it."""
    basis = resolve_mix(basis)
    # One row per model family, as the treemap beside it counts them.
    work = dedupe_to_best_variant(df[df["quality"] > 0]) if not df.empty else df
    if work.empty:
        return empty_state_figure("Provider Leaderboard", "No models match these filters",
                                  "Clear the search, widen PROVIDER, or lower MIN SCORE")

    # Aggregation
    agg = work.groupby("provider").agg(
        best_quality=("quality", "max"),
        avg_quality=("quality", "mean"),
        model_count=("model", "count"),
        min_price=("price", "min"),
    ).reset_index()

    # Best model name per provider
    idx_best = work.groupby("provider")["quality"].idxmax()
    best_names = work.loc[idx_best, ["provider", "model"]].set_index("provider")["model"]
    agg["best_model"] = agg["provider"].map(best_names)

    agg = agg.sort_values("best_quality", ascending=True).reset_index(drop=True)
    # Escaped label used for every RENDERED sink (tick labels, annotations);
    # the raw name stays for palette lookups, which are keyed by it.
    agg["_label"] = agg["provider"].map(plot_text)

    colors = [PROVIDER_COLORS.get(p, DEFAULT_COLOR) for p in agg["provider"]]
    # Abbreviated, not cut: "Claude Opus 5.5 (Max, Default Fallback)" ran
    # into the gutter's ellipsis as "Claude Opus 5.5 (…". Escaped by
    # clean_model_name.
    short_names = agg["best_model"].apply(clean_model_name)

    fig = go.Figure()

    # Ghost track — scale to actual data max, not a hardcoded 100
    max_q = agg["best_quality"].max()
    fig.add_trace(go.Bar(
        y=agg["_label"],
        x=[max_q] * len(agg),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Best-quality bars
    fig.add_trace(go.Bar(
        y=agg["_label"],
        x=agg["best_quality"],
        orientation="h",
        marker=dict(color=colors, opacity=0.80, line=dict(width=0)),
        customdata=agg.assign(
            best_model=agg["best_model"].map(plot_text),
            provider=agg["_label"],
        )[["best_model", "provider", "best_quality",
                         "avg_quality", "model_count", "min_price"]].values,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Best model: %{customdata[0]}<br>"
            "Peak intelligence: %{customdata[2]:.0f}<br>"
            "Avg intelligence: %{customdata[3]:.0f}<br>"
            "Models: %{customdata[4]}<br>"
            "Floor price: $%{customdata[5]:.4g}/M  ·  " + basis.short + "<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # Avg quality markers
    fig.add_trace(go.Scatter(
        y=agg["_label"],
        x=agg["avg_quality"],
        mode="markers",
        marker=dict(
            symbol="line-ns",
            size=12,
            color="rgba(255,255,255,0.30)",
            line=dict(width=2, color="rgba(255,255,255,0.30)"),
        ),
        hovertemplate="Avg intelligence: %{x:.0f}<extra></extra>",
        showlegend=False,
        name="avg quality",
    ))

    _gutter = right_gutter(
        f"{count_label(c)}  ·  {n}" for c, n in zip(agg["model_count"], short_names)
    )
    # Right-side annotations: model count + best model name.
    # Collected as plain dicts and set once in update_layout below: one
    # fig.add_annotation per row re-validates the whole annotations tuple on
    # every call, which makes the build quadratic in rows.
    annotations = []
    for i, row in agg.iterrows():
        annotations.append(dict(
            x=1.01,
            y=row["_label"],
            text=fit_text(f"{count_label(row['model_count'])}  ·  {short_names[i]}", _gutter),
            showarrow=False,
            xanchor="left",
            # Body grey, not the provider hue: the bar and the tick label
            # already carry the provider, and several palette entries (Z AI's
            # green, Mistral's red, Reflection's olive) sat at ~3:1 on the
            # #111111 card at 10 px. #aaaaaa is 7.9:1. Same rule as Rankings.
            font=dict(size=10, family=_FONT, color=LABEL_GREY),
            xref="paper", yref="y",
        ))

    height = max(400, len(agg) * 30 + 80)

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Provider Leaderboard"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  bar = peak intelligence  ·  tick mark = average</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=12), standoff=12),
            range=[0, max_q * ANNOTATED_AXIS_HEADROOM],
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            tickfont=dict(color="#999999", size=11, family=_FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        ),
        annotations=annotations,
        barmode="overlay",
        margin=dict(l=20, r=_gutter, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )
    return fig


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888"),
        annotations=[dict(x=0.5, y=0.5, xref="paper", yref="paper",
                          text=msg, showarrow=False,
                          font=dict(color="#777777", size=13, family=_FONT))],
        margin=dict(l=40, r=40, t=60, b=40), height=400,
    )
    return fig
