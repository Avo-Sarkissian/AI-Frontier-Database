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
                opacity=0.85,
                line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
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
            line=dict(color="#00d4ff", width=1.5, dash="dot"),
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
            textfont=dict(color="#00d4ff", size=9, family="Inter, sans-serif"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # --- Layout ---
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(family="Inter, ui-sans-serif, sans-serif", color="#e2e8f0", size=12),
        title=dict(
            text="Cost vs. Intelligence  <span style='font-size:13px;color:#64748b'>bubble size = speed (tok/s)</span>",
            font=dict(size=18, color="#f8fafc", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(
            title=dict(text="Price (USD / 1M tokens)", font=dict(color="#94a3b8", size=12)),
            type="log",
            gridcolor="#1e293b",
            zerolinecolor="#1e293b",
            tickfont=dict(color="#94a3b8", size=11),
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color="#94a3b8", size=12)),
            gridcolor="#1e293b",
            zerolinecolor="#1e293b",
            tickfont=dict(color="#94a3b8", size=11),
            showgrid=True,
        ),
        legend=dict(
            bgcolor="rgba(15,17,23,0.8)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(color="#94a3b8", size=11),
            itemsizing="constant",
            orientation="v",
            x=1.01,
            y=1,
            xanchor="left",
        ),
        margin=dict(l=60, r=180, t=60, b=60),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            font=dict(color="#f8fafc", size=12, family="Inter, sans-serif"),
        ),
    )

    return fig
