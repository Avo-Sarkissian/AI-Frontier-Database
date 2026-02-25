"""
Pareto Frontier scatter: Cost (x) vs Quality (y), sized by speed.
Highlights the Pareto-optimal models (best quality for their price tier).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Provider → color mapping (consistent across all charts)
PROVIDER_COLORS: dict[str, str] = {
    "Anthropic":              "#c084fc",  # purple
    "OpenAI":                 "#34d399",  # emerald
    "Google":                 "#60a5fa",  # blue
    "Meta":                   "#fb923c",  # orange
    "DeepSeek":               "#f472b6",  # pink
    "Mistral":                "#facc15",  # yellow
    "xAI":                    "#a3e635",  # lime
    "Alibaba":                "#38bdf8",  # sky
    "Amazon":                 "#ff9900",  # aws orange
    "NVIDIA":                 "#22d3ee",  # cyan
    "Microsoft Azure":        "#818cf8",  # indigo
    "Cohere":                 "#f87171",  # red
    "Kimi":                   "#d4a1f5",  # lavender
    "Z AI":                   "#7dd3fc",  # light blue
    "MiniMax":                "#86efac",  # light green
    "InclusionAI":            "#fca5a5",  # light red
    "Xiaomi":                 "#6ee7b7",  # teal
    "Baidu":                  "#fde68a",  # amber
    "IBM":                    "#93c5fd",  # periwinkle
    "LG AI Research":         "#c4b5fd",  # violet
    "Nous Research":          "#f9a8d4",  # rose
    "Reka AI":                "#a78bfa",  # purple-blue
    "AI21 Labs":              "#34d399",  # green
    "Allen Institute for AI": "#67e8f9",  # light cyan
    "Inception":              "#fb7185",  # hot pink
    "Upstage":                "#fbbf24",  # amber
    "Perplexity":             "#a3a3a3",  # neutral
}
DEFAULT_COLOR = "#6b7280"


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
            "Speed: %{customdata[2]:.0f} tok/s<br>"
            "<extra></extra>"
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
            customdata=pdf[["model", "provider", "speed"]].values,
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

    # ── Layout — matches Linear/Vercel dark aesthetic ──
    _bg     = "#111111"       # card bg
    _grid   = "rgba(255,255,255,0.04)"   # near-invisible grid
    _zero   = "rgba(255,255,255,0.06)"
    _tick   = "#444444"
    _axis   = "#444444"
    _font   = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

    fig.update_layout(
        paper_bgcolor=_bg,
        plot_bgcolor=_bg,
        font=dict(family=_font, color="#888888", size=12),
        title=dict(
            text=(
                "Cost vs. Intelligence"
                "  <span style='font-size:11px;color:#3d3d3d;font-weight:400'>"
                "  ·  bubble size = speed (tok/s)</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_font, weight=600),
            x=0.0,
            xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(
                text="Price  (USD / 1M tokens)",
                font=dict(color=_axis, size=11),
                standoff=12,
            ),
            type="log",
            gridcolor=_grid,
            zerolinecolor=_zero,
            zerolinewidth=1,
            tickfont=dict(color=_tick, size=10, family=_font),
            showgrid=True,
            showline=False,
            ticks="",
        ),
        yaxis=dict(
            title=dict(
                text="AA Intelligence Index",
                font=dict(color=_axis, size=11),
                standoff=12,
            ),
            gridcolor=_grid,
            zerolinecolor=_zero,
            zerolinewidth=1,
            tickfont=dict(color=_tick, size=10, family=_font),
            showgrid=True,
            showline=False,
            ticks="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(color="#555555", size=10, family=_font),
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
            font=dict(color="#f2f2f2", size=12, family=_font),
            namelength=-1,
        ),
    )

    return fig
