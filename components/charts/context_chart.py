"""
Context window chart.

Scatter plot: context window size (K tokens) vs intelligence score.
Lets users find models that support long documents/conversations.

X-axis: context window in K tokens (log scale)
Y-axis: AA Intelligence Index
Bubble size: inversely proportional to price (larger = cheaper)
Color: provider
"""
import math
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)


def _parse_context_k(val) -> float:
    """Parse context string to K tokens: '128K' → 128.0, '32768' → 32.0."""
    s = str(val).upper().replace(",", "").strip()
    if not s or s in ("NAN", "NONE", "--", ""):
        return float("nan")
    if "K" in s:
        try:
            return float(s.replace("K", ""))
        except ValueError:
            return float("nan")
    try:
        raw = float(s)
        return round(raw / 1000, 1)   # raw tokens → K
    except ValueError:
        return float("nan")


def build_context_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter: context window vs quality."""
    plot_df = df.copy()
    plot_df["context_k"] = plot_df["context"].apply(_parse_context_k)
    plot_df = plot_df[
        plot_df["context_k"].notna() &
        (plot_df["context_k"] > 0) &
        (plot_df["quality"] > 0)
    ].copy()

    if plot_df.empty:
        return _empty("No context window data available.")

    # Bubble size: bigger = cheaper (inverse of price, capped)
    plot_df["inv_price"] = 1.0 / plot_df["price"].clip(lower=0.001)
    max_inv = plot_df["inv_price"].max()
    plot_df["size"] = (plot_df["inv_price"] / max_inv * 28 + 6).clip(upper=34)

    fig = go.Figure()

    for provider, pdf in plot_df.groupby("provider"):
        color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
        fig.add_trace(go.Scatter(
            x=pdf["context_k"],
            y=pdf["quality"],
            mode="markers",
            name=provider,
            marker=dict(
                color=color, opacity=0.82,
                size=pdf["size"],
                sizemode="diameter",
                line=dict(width=0.5, color="rgba(255,255,255,0.12)"),
            ),
            customdata=pdf[["model", "provider", "context_k", "price", "quality"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Provider: %{customdata[1]}<br>"
                "Context: %{customdata[2]:.0f}K tokens<br>"
                "Price: $%{customdata[3]:.4f}/M<br>"
                "Intelligence: %{customdata[4]:.0f}<br>"
                "<extra></extra>"
            ),
        ))

    # Reference lines at common context boundaries
    for ctx_k, label in [(8, "8K"), (32, "32K"), (128, "128K"), (1000, "1M")]:
        if plot_df["context_k"].max() >= ctx_k * 0.9:
            fig.add_vline(
                x=ctx_k,
                line=dict(color="rgba(255,255,255,0.05)", width=1, dash="dot"),
                annotation_text=label,
                annotation_font=dict(color="#666666", size=9, family=_FONT),
                annotation_position="top right",
            )

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Context Window vs Intelligence"
                "  <span style='font-size:11px;color:#666666;font-weight:400'>"
                "  ·  bubble size = cheaper  ·  click legend to isolate providers</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Context Window (K tokens, log scale)", font=dict(color=_AXIS, size=11), standoff=12),
            type="log",
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
            tickvals=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1000],
            ticktext=["2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M"],
        ),
        yaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=11), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
            font=dict(color="#888888", size=9, family=_FONT),
            x=1.01, y=1, xanchor="left",
        ),
        margin=dict(l=56, r=160, t=52, b=52),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
        annotations=[dict(
            x=0.5, y=-0.12, xref="paper", yref="paper", xanchor="center",
            text="Bubble size = cheaper",
            showarrow=False,
            font=dict(color="#555555", size=9, family=_FONT),
        )],
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
        margin=dict(l=40, r=40, t=60, b=40), height=500,
    )
    return fig
