"""
Speed vs Quality quadrant chart.
Divides models into 4 zones based on median speed and quality.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROVIDER_COLORS: dict[str, str] = {
    "Anthropic":              "#c084fc",
    "OpenAI":                 "#34d399",
    "Google":                 "#60a5fa",
    "Meta":                   "#fb923c",
    "DeepSeek":               "#f472b6",
    "Mistral":                "#facc15",
    "xAI":                    "#a3e635",
    "Alibaba":                "#38bdf8",
    "Amazon":                 "#ff9900",
    "NVIDIA":                 "#22d3ee",
    "Microsoft Azure":        "#818cf8",
    "Cohere":                 "#f87171",
    "Kimi":                   "#d4a1f5",
    "Z AI":                   "#7dd3fc",
    "MiniMax":                "#86efac",
    "InclusionAI":            "#fca5a5",
    "Xiaomi":                 "#6ee7b7",
    "Baidu":                  "#fde68a",
    "IBM":                    "#93c5fd",
    "LG AI Research":         "#c4b5fd",
    "Nous Research":          "#f9a8d4",
    "Reka AI":                "#a78bfa",
    "AI21 Labs":              "#34d399",
    "Allen Institute for AI": "#67e8f9",
    "Inception":              "#fb7185",
    "Upstage":                "#fbbf24",
    "Perplexity":             "#a3a3a3",
}
DEFAULT_COLOR = "#6b7280"

_BG    = "#111111"
_GRID  = "rgba(255,255,255,0.04)"
_ZONE  = "rgba(255,255,255,0.02)"
_TICK  = "#444444"
_AXIS  = "#444444"
_FONT  = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"


def build_quadrant(df: pd.DataFrame) -> go.Figure:
    plot_df = df[
        (df["speed"] > 0) &
        (df["quality"] > 0) &
        df["speed"].notna() &
        df["quality"].notna()
    ].copy()

    if plot_df.empty:
        return go.Figure()

    med_speed   = plot_df["speed"].median()
    med_quality = plot_df["quality"].median()

    # Bubble size normalized on price (inverted: cheaper = bigger)
    max_price = plot_df["price"].replace(0, np.nan).max()
    plot_df["size"] = plot_df["price"].apply(
        lambda p: 8 + (1 - min(p / max_price, 1)) * 22 if pd.notna(p) and p > 0 else 8
    )

    fig = go.Figure()

    # Quadrant shading
    x_max = plot_df["speed"].max() * 1.15
    y_max = plot_df["quality"].max() * 1.15

    zone_labels = [
        (0,         med_speed,   med_quality, y_max,   "Slow · Smart",  0.25, 0.75),
        (med_speed, x_max,       med_quality, y_max,   "Fast · Smart",  0.75, 0.75),
        (0,         med_speed,   0,           med_quality, "Slow · Weak", 0.25, 0.25),
        (med_speed, x_max,       0,           med_quality, "Fast · Weak", 0.75, 0.25),
    ]

    for x0, x1, y0, y1, label, rx, ry in zone_labels:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=_ZONE, line=dict(width=0), layer="below")

    # Median crosshair lines
    fig.add_hline(y=med_quality, line=dict(color="rgba(255,255,255,0.06)", width=1, dash="dot"))
    fig.add_vline(x=med_speed,   line=dict(color="rgba(255,255,255,0.06)", width=1, dash="dot"))

    # Per-provider scatter
    providers = sorted(plot_df["provider"].unique())
    for provider in providers:
        pdf = plot_df[plot_df["provider"] == provider]
        color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "Quality: %{y}<br>"
            "Speed: %{x:.0f} tok/s<br>"
            "Price: $%{customdata[2]:.3f}/M tokens<br>"
            "<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=pdf["speed"],
            y=pdf["quality"],
            mode="markers",
            name=provider,
            marker=dict(
                color=color,
                size=pdf["size"],
                opacity=0.75,
                line=dict(width=0),
            ),
            customdata=pdf[["model", "provider", "price"]].values,
            hovertemplate=hover,
        ))

    # Zone annotation text
    annotations = []
    for x0, x1, y0, y1, label, rx, ry in zone_labels:
        annotations.append(dict(
            x=x0 + (x1 - x0) * rx,
            y=y0 + (y1 - y0) * ry,
            text=label,
            showarrow=False,
            font=dict(color="rgba(255,255,255,0.08)", size=11, family=_FONT),
            xref="x", yref="y",
        ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Speed vs. Intelligence"
                "  <span style='font-size:11px;color:#3d3d3d;font-weight:400'>"
                "  ·  bubble size ∝ affordability</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Speed  (tokens / second)", font=dict(color=_AXIS, size=11), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
            range=[0, x_max],
        ),
        yaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=11), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
            range=[0, y_max],
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
            font=dict(color="#555555", size=10, family=_FONT),
            itemsizing="constant", orientation="v",
            x=1.01, y=1, xanchor="left", tracegroupgap=2,
        ),
        margin=dict(l=56, r=172, t=52, b=52),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
        annotations=annotations,
    )

    return fig
