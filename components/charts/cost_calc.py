"""
Budget calculator chart.
Given a monthly token volume, shows projected monthly cost per model.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)


def build_cost_calc(df: pd.DataFrame, monthly_tokens_m: float = 1.0, top_n: int = 40) -> go.Figure:
    """Horizontal bar chart: monthly cost = monthly_tokens_m × price/M tokens."""
    plot_df = df[
        (df["price"] > 0) &
        (df["quality"] > 0) &
        df["price"].notna()
    ].copy()

    plot_df["monthly_cost"] = monthly_tokens_m * plot_df["price"]
    plot_df = plot_df.sort_values("monthly_cost").head(top_n).reset_index(drop=True)

    colors = [PROVIDER_COLORS.get(p, DEFAULT_COLOR) for p in plot_df["provider"]]
    short_names = plot_df["model"].apply(lambda m: m[:30] + "…" if len(m) > 30 else m)

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "Provider: %{customdata[1]}<br>"
        "Monthly cost: $%{x:,.2f}<br>"
        "Price: $%{customdata[2]:.4f}/M tokens<br>"
        "Intelligence: %{customdata[3]:.0f}<br>"
        "<extra></extra>"
    )

    fig = go.Figure()

    # Background track
    fig.add_trace(go.Bar(
        y=short_names,
        x=[plot_df["monthly_cost"].max() * 1.05] * len(plot_df),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Cost bars
    fig.add_trace(go.Bar(
        y=short_names,
        x=plot_df["monthly_cost"],
        orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        customdata=plot_df[["model", "provider", "price", "quality"]].values,
        hovertemplate=hover,
        showlegend=False,
        text=plot_df["monthly_cost"].apply(
            lambda c: f"${c:,.2f}" if c >= 1 else f"${c:.3f}"
        ),
        textposition="inside",
        textfont=dict(color="rgba(255,255,255,0.6)", size=10, family=_FONT),
    ))

    # Provider label on right
    for i, row in plot_df.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        fig.add_annotation(
            x=plot_df["monthly_cost"].max() * 1.06,
            y=short_names[i],
            text=row["provider"],
            showarrow=False,
            xanchor="left",
            font=dict(size=9, family=_FONT, color=color),
            xref="x", yref="y",
        )

    height = max(500, top_n * 26)

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Monthly API Cost"
                "  <span style='font-size:11px;color:#666666;font-weight:400'>"
                "  ·  cheapest models for your token budget</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Estimated Monthly Cost  (USD)", font=dict(color=_AXIS, size=11), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            tickprefix="$",
            showgrid=True, showline=False, ticks="",
            range=[0, plot_df["monthly_cost"].max() * 1.3],
        ),
        yaxis=dict(
            tickfont=dict(color="#888888", size=10, family=_FONT),
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
