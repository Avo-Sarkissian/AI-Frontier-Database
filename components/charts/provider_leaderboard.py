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

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)


def build_provider_leaderboard(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: one per provider, ordered by best model quality."""
    if df.empty:
        return _empty("No data available.")

    work = df[df["quality"] > 0].copy()

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

    colors = [PROVIDER_COLORS.get(p, DEFAULT_COLOR) for p in agg["provider"]]
    short_names = agg["best_model"].apply(lambda m: m[:32] + "…" if len(m) > 32 else m)

    fig = go.Figure()

    # Ghost track — scale to actual data max, not a hardcoded 100
    max_q = agg["best_quality"].max()
    fig.add_trace(go.Bar(
        y=agg["provider"],
        x=[max_q] * len(agg),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Best-quality bars
    fig.add_trace(go.Bar(
        y=agg["provider"],
        x=agg["best_quality"],
        orientation="h",
        marker=dict(color=colors, opacity=0.80, line=dict(width=0)),
        customdata=agg[["best_model", "provider", "best_quality",
                         "avg_quality", "model_count", "min_price"]].values,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Best model: %{customdata[0]}<br>"
            "Peak intelligence: %{customdata[2]:.0f}<br>"
            "Avg intelligence: %{customdata[3]:.0f}<br>"
            "Models: %{customdata[4]}<br>"
            "Floor price: $%{customdata[5]:.4f}/M<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # Avg quality markers
    fig.add_trace(go.Scatter(
        y=agg["provider"],
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

    # Right-side annotations: model count + best model name
    for i, row in agg.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        fig.add_annotation(
            x=max_q + 1,
            y=row["provider"],
            text=f"{int(row['model_count'])} models  ·  {short_names[i]}",
            showarrow=False,
            xanchor="left",
            font=dict(size=10, family=_FONT, color=color),
            xref="x", yref="y",
        )

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
            range=[0, max_q * 1.55],
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            tickfont=dict(color="#999999", size=11, family=_FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        ),
        barmode="overlay",
        margin=dict(l=20, r=280, t=52, b=36),
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
