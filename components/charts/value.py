"""
Value analysis chart: Intelligence per Dollar.
Shows quality/price ratio (efficiency) for all models.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR, clean_model_name,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)


def build_value_chart(df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """Bar chart showing intelligence-per-dollar for top value models."""
    plot_df = df[
        (df["quality"] > 0) &
        (df["price"] > 0) &
        df["quality"].notna() &
        df["price"].notna()
    ].copy()

    plot_df["value_score"] = plot_df["quality"] / plot_df["price"]

    ranked = plot_df.sort_values("value_score", ascending=False).head(top_n)
    ranked = ranked.iloc[::-1].reset_index(drop=True)

    colors = [PROVIDER_COLORS.get(p, DEFAULT_COLOR) for p in ranked["provider"]]
    short_names = ranked["model"].apply(clean_model_name)

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "Provider: %{customdata[1]}<br>"
        "Value Score: %{x:.2f} score/$<br>"
        "AA Score: %{customdata[2]:.1f}<br>"
        "Price: $%{customdata[3]:.4f}/M tokens<br>"
        "<extra></extra>"
    )

    fig = go.Figure()

    # Background track
    fig.add_trace(go.Bar(
        y=short_names,
        x=[ranked["value_score"].max() * 1.05] * len(ranked),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Main bars
    fig.add_trace(go.Bar(
        y=short_names,
        x=ranked["value_score"],
        orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        customdata=ranked[["model", "provider", "quality", "price"]].values,
        hovertemplate=hover,
        showlegend=False,
        text=ranked["value_score"].apply(lambda v: f"{v:.0f}"),
        textposition="inside",
        textfont=dict(color="rgba(255,255,255,0.6)", size=11, family=_FONT),
    ))

    # Provider label on right — use category string for reliable positioning
    for i, row in ranked.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        fig.add_annotation(
            x=ranked["value_score"].max() * 1.06,
            y=short_names[i],
            text=row["provider"],
            showarrow=False,
            xanchor="left",
            font=dict(size=10, family=_FONT, color=color),
            xref="x", yref="y",
        )

    height = max(400, top_n * 26)

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Intelligence per Dollar"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  quality ÷ price  (higher = better value)</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="AA Score per $ / 1M tokens", font=dict(color=_AXIS, size=12), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
            range=[0, ranked["value_score"].max() * 1.28],
        ),
        yaxis=dict(
            tickfont=dict(color="#999999", size=11, family=_FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        ),
        barmode="overlay",
        margin=dict(l=20, r=130, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )

    return fig
