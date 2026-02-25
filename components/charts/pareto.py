"""
Pareto Frontier scatter: Cost (x) vs Quality (y), sized by speed.
Highlights the Pareto-optimal models (best quality for their price tier).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR, BG, GRID, TICK, AXIS, FONT


def _pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows on the Pareto frontier: cheapest model for each quality tier.
    A model is Pareto-optimal if no other model has both higher quality AND lower price.
    """
    pareto = []
    for _, row in df.iterrows():
        dominated = df[
            (df["quality"] >= row["quality"]) &
            (df["price"] <= row["price"]) &
            ~((df["quality"] == row["quality"]) & (df["price"] == row["price"]))
        ]
        if dominated.empty:
            pareto.append(row)
    return pd.DataFrame(pareto).sort_values("price")


def build_pareto_scatter(df: pd.DataFrame) -> go.Figure:
    """Build the Cost vs Quality Pareto scatter figure."""
    # Filter to models with valid price and quality
    plot_df = df[
        (df["price"] > 0) &
        (df["quality"] > 0) &
        df["price"].notna() &
        df["quality"].notna()
    ].copy()

    # Normalize speed for bubble size (5–40px range)
    max_speed = plot_df["speed"].replace(0, np.nan).max()
    if pd.isna(max_speed) or max_speed == 0:
        max_speed = 1  # guard: no valid speed data → uniform bubble size
    plot_df["size"] = plot_df["speed"].apply(
        lambda s: 8 + (s / max_speed) * 28 if pd.notna(s) and s > 0 else 8
    )

    fig = go.Figure()

    # --- Per-provider scatter traces ---
    providers = sorted(plot_df["provider"].unique())
    for provider in providers:
        pdf = plot_df[plot_df["provider"] == provider]
        color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "Quality: %{y}<br>"
            "Price: $%{x:.3f}/M tokens<br>"
            "Speed: %{customdata[2]}<br>"
            "<extra></extra>"
        )

        speed_str = pdf["speed"].apply(
            lambda s: f"{s:.0f} tok/s" if pd.notna(s) and s > 0 else "N/A"
        )

        fig.add_trace(go.Scatter(
            x=pdf["price"],
            y=pdf["quality"],
            mode="markers",
            name=provider,
            marker=dict(
                color=color,
                size=pdf["size"],
                opacity=0.75,
                line=dict(width=0, color="rgba(0,0,0,0)"),
            ),
            customdata=list(zip(pdf["model"], pdf["provider"], speed_str)),
            hovertemplate=hover,
        ))

    # --- Pareto frontier line ---
    pareto_df = _pareto_frontier(plot_df)
    if not pareto_df.empty:
        fig.add_trace(go.Scatter(
            x=pareto_df["price"],
            y=pareto_df["quality"],
            mode="lines",
            name="Pareto Frontier",
            line=dict(color="rgba(0,212,255,0.5)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=True,
        ))

        # Label Pareto-optimal models
        fig.add_trace(go.Scatter(
            x=pareto_df["price"],
            y=pareto_df["quality"],
            mode="text",
            text=pareto_df["model"].apply(lambda m: m[:18] + "…" if len(m) > 18 else m),
            textposition="top center",
            textfont=dict(color="rgba(0,212,255,0.6)", size=8, family="Inter, -apple-system, sans-serif"),
            hoverinfo="skip",
            showlegend=False,
        ))

    _zero = "rgba(255,255,255,0.06)"

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Cost vs. Intelligence"
                "  <span style='font-size:11px;color:#3d3d3d;font-weight:400'>"
                "  ·  bubble size = speed (tok/s)</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=FONT, weight=600),
            x=0.0,
            xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(
                text="Price  (USD / 1M tokens)",
                font=dict(color=AXIS, size=11),
                standoff=12,
            ),
            type="log",
            gridcolor=GRID,
            zerolinecolor=_zero,
            zerolinewidth=1,
            tickfont=dict(color=TICK, size=10, family=FONT),
            showgrid=True,
            showline=False,
            ticks="",
        ),
        yaxis=dict(
            title=dict(
                text="AA Intelligence Index",
                font=dict(color=AXIS, size=11),
                standoff=12,
            ),
            gridcolor=GRID,
            zerolinecolor=_zero,
            zerolinewidth=1,
            tickfont=dict(color=TICK, size=10, family=FONT),
            showgrid=True,
            showline=False,
            ticks="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(color="#555555", size=10, family=_FONT),
            itemsizing="constant",
            orientation="v",
            x=1.01,
            y=1,
            xanchor="left",
            tracegroupgap=2,
        ),
        margin=dict(l=56, r=172, t=52, b=52),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT),
            namelength=-1,
        ),
    )

    return fig
