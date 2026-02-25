"""
Top-N model rankings — horizontal bar chart.
Shows quality, price, and speed as stacked visual context.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)


def build_rankings(df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    """Horizontal bar chart of top-N models by quality score."""
    ranked = (
        df[df["quality"] > 0]
        .sort_values("quality", ascending=False)
        .head(top_n)
        .copy()
    )
    # Reverse for bottom-up bar display
    ranked = ranked.iloc[::-1].reset_index(drop=True)

    colors = [PROVIDER_COLORS.get(p, DEFAULT_COLOR) for p in ranked["provider"]]
    short_names = ranked["model"].apply(lambda m: m[:32] + "…" if len(m) > 32 else m)

    speed_str = ranked["speed"].apply(
        lambda s: f"{s:.0f} tok/s" if pd.notna(s) and s > 0 else "N/A"
    )

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "Provider: %{customdata[1]}<br>"
        "Intelligence: %{x}<br>"
        "Price: $%{customdata[2]:.3f}/M tokens<br>"
        "Speed: %{customdata[3]}<br>"
        "<extra></extra>"
    )

    fig = go.Figure()

    # Background track
    fig.add_trace(go.Bar(
        y=short_names,
        x=[ranked["quality"].max() * 1.05] * len(ranked),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.03)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Actual bars
    fig.add_trace(go.Bar(
        y=short_names,
        x=ranked["quality"],
        orientation="h",
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(width=0),
        ),
        customdata=list(zip(ranked["model"], ranked["provider"], ranked["price"], speed_str)),
        hovertemplate=hover,
        showlegend=False,
        text=ranked["quality"].apply(lambda q: f"{q:.0f}"),
        textposition="inside",
        textfont=dict(color="rgba(255,255,255,0.6)", size=10, family=_FONT),
    ))

    # Provider legend annotation on right — use category string for reliable positioning
    for i, row in ranked.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        fig.add_annotation(
            x=ranked["quality"].max() * 1.06,
            y=short_names[i],
            text=f"<span style='color:{color}'>{row['provider']}</span>",
            showarrow=False,
            xanchor="left",
            font=dict(size=9, family=_FONT, color=color),
            xref="x", yref="y",
        )

    height = max(400, top_n * 26)

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                f"Top {top_n} Models by Intelligence"
                "  <span style='font-size:11px;color:#3d3d3d;font-weight:400'>"
                "  ·  AA Intelligence Index</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=11), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
            range=[0, ranked["quality"].max() * 1.3],
        ),
        yaxis=dict(
            tickfont=dict(color="#666666", size=10, family=_FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        ),
        barmode="overlay",
        margin=dict(l=20, r=120, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )

    return fig
