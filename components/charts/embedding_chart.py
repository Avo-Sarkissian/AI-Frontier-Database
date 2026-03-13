"""
Embedding model charts.
Primary: MTEB score vs price scatter (log X), bubble size = context length.
Secondary: horizontal bar ranked by MTEB with dimension + price annotations.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import BG, GRID, TICK, AXIS, FONT
from data.embedding_models import PROVIDER_COLORS, DEFAULT_COLOR


def build_embedding_scatter(df: pd.DataFrame) -> go.Figure:
    """MTEB score vs price scatter. Bubble size = context length. Free models use placeholder X."""
    plot_df = df.copy()

    max_ctx = plot_df["max_tokens"].replace(0, np.nan).max() or 1
    plot_df["size"] = plot_df["max_tokens"].apply(
        lambda t: 8 + (t / max_ctx) * 28 if pd.notna(t) and t > 0 else 8
    )

    fig = go.Figure()

    for provider in sorted(plot_df["provider"].unique()):
        pdf = plot_df[plot_df["provider"] == provider]
        color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)

        price_label = pdf["price_per_1m"].apply(
            lambda p: "Free (open weights)" if p == 0 else f"${p:.3f}/M tok"
        )

        fig.add_trace(go.Scatter(
            x=pdf["price_plot"],
            y=pdf["mteb"],
            mode="markers",
            name=provider,
            marker=dict(
                color=color,
                size=pdf["size"],
                opacity=pdf["is_free"].apply(lambda f: 0.55 if f else 0.85),
                line=dict(
                    width=pdf["is_free"].apply(lambda f: 1.5 if f else 0),
                    color=pdf["is_free"].apply(lambda f: color if f else "rgba(0,0,0,0)"),
                ),
                symbol=pdf["is_free"].apply(lambda f: "circle-open-dot" if f else "circle"),
            ),
            customdata=list(zip(
                pdf["model"], pdf["provider"],
                pdf["mteb"], price_label,
                pdf["dimensions"], pdf["max_tokens"],
                pdf["tags_str"],
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Provider: %{customdata[1]}<br>"
                "MTEB Score: %{customdata[2]:.1f}<br>"
                "Price: %{customdata[3]}<br>"
                "Dimensions: %{customdata[4]}<br>"
                "Max tokens: %{customdata[5]:,}<br>"
                "Tags: %{customdata[6]}<br>"
                "<extra></extra>"
            ),
        ))

    # Vertical dashed line separating free from paid zone
    fig.add_vline(
        x=0.008, line_dash="dot", line_color="rgba(255,255,255,0.12)", line_width=1,
    )
    fig.add_annotation(
        x=math.log10(0.008), y=plot_df["mteb"].min() - 0.5,
        text="← free (open weights)   paid →",
        showarrow=False, xanchor="center",
        font=dict(size=8, family=FONT, color="#444"),
        xref="x", yref="y",
    )

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#888888", size=12),
        title=dict(
            text=(
                "MTEB Score vs. Price"
                "  <span style='font-size:11px;color:#666666;font-weight:400'>"
                "  ·  bubble size = context length  ·  open rings = open weights (free)</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Price  (USD / 1M tokens)  —  open weights plotted at $0.008 for scale",
                       font=dict(color=AXIS, size=11), standoff=12),
            type="log",
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=TICK, size=10, family=FONT),
            tickformat="$~g",
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            title=dict(text="MTEB Score", font=dict(color=AXIS, size=11), standoff=12),
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=TICK, size=10, family=FONT),
            showgrid=True, showline=False, ticks="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1,
            font=dict(color="#888888", size=11, family=FONT),
            x=1.01, y=1, xanchor="left",
        ),
        margin=dict(l=20, r=160, t=52, b=36),
        height=520,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
        ),
    )

    return fig


def build_embedding_rankings(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar ranked by MTEB, annotated with dimensions and price."""
    plot_df = df.sort_values("mteb", ascending=True).reset_index(drop=True)

    short_name = plot_df["model"].apply(lambda n: n[:36] + "…" if len(n) > 36 else n)
    colors = plot_df["provider"].map(PROVIDER_COLORS).fillna(DEFAULT_COLOR).tolist()
    max_mteb = plot_df["mteb"].max() or 1

    fig = go.Figure()

    # Ghost track
    fig.add_trace(go.Bar(
        y=short_name, x=[max_mteb] * len(plot_df),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))

    opacity = plot_df["is_free"].apply(lambda f: 0.55 if f else 0.82).tolist()

    fig.add_trace(go.Bar(
        y=short_name,
        x=plot_df["mteb"],
        orientation="h",
        marker=dict(color=colors, opacity=opacity, line=dict(width=0)),
        customdata=list(zip(
            plot_df["model"], plot_df["provider"],
            plot_df["mteb"], plot_df["price_per_1m"],
            plot_df["dimensions"], plot_df["max_tokens"],
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "MTEB: %{customdata[2]:.1f}<br>"
            "Price: %{customdata[3]}/M tok<br>"
            "Dimensions: %{customdata[4]}<br>"
            "Max tokens: %{customdata[5]:,}<br>"
            "<extra></extra>"
        ),
        showlegend=False,
        text=plot_df["mteb"].apply(lambda s: f"{s:.1f}"),
        textposition="inside",
        textfont=dict(color="rgba(255,255,255,0.55)", size=9, family=FONT),
    ))

    # Right-side annotations
    for i, row in plot_df.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        price_str = "free" if row["price_per_1m"] == 0 else f"${row['price_per_1m']:.3f}/M"
        fig.add_annotation(
            x=max_mteb + 0.1,
            y=short_name[i],
            text=f"{row['dimensions']}d  ·  {price_str}",
            showarrow=False, xanchor="left",
            font=dict(size=9, family=FONT, color=color),
            xref="x", yref="y",
        )

    height = max(400, len(plot_df) * 22 + 80)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Embedding Models — Ranked by MTEB"
                "  <span style='font-size:11px;color:#666666;font-weight:400'>"
                "  ·  faded bars = open weights (free)  ·  annotations: dimensions · price</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="MTEB Score", font=dict(color=AXIS, size=11), standoff=12),
            range=[min(plot_df["mteb"]) - 2, max_mteb * 1.18],
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=TICK, size=10, family=FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            tickfont=dict(color="#888888", size=10, family=FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        ),
        barmode="overlay",
        margin=dict(l=20, r=180, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
        ),
    )

    return fig
