"""
Model comparison radar chart.
Normalizes quality, speed, affordability, and context window
to 0–1 scale and overlays up to 5 selected models.
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

_PALETTE = [
    "#00d4ff", "#c084fc", "#34d399", "#fb923c", "#f472b6",
]

_BG   = "#111111"
_FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

DIMS = ["Intelligence", "Speed", "Affordability", "Context"]


def _context_k(ctx_str: str) -> float:
    """Convert '200k', '1m', '128k' etc. to numeric thousands."""
    if not ctx_str or pd.isna(ctx_str):
        return 0.0
    s = str(ctx_str).lower().strip()
    try:
        if s.endswith("m"):
            return float(s[:-1]) * 1000
        elif s.endswith("k"):
            return float(s[:-1])
        else:
            return float(s)
    except ValueError:
        return 0.0


def build_radar(df: pd.DataFrame, selected_models: list[str] | None = None) -> go.Figure:
    if not selected_models:
        # Default: top-5 by quality
        selected_models = (
            df[df["quality"] > 0]
            .sort_values("quality", ascending=False)
            .head(5)["model"]
            .tolist()
        )

    plot_df = df[df["model"].isin(selected_models)].copy()
    if plot_df.empty:
        return go.Figure()

    # Normalize each dimension to 0–1 across the FULL dataset (not just selected)
    q_max  = df["quality"].max()
    s_max  = df["speed"].replace(0, np.nan).max()
    p_max  = df["price"].max()

    df_ctx = df["context"].apply(_context_k)
    c_max  = df_ctx.max()

    fig = go.Figure()

    for idx, model_name in enumerate(selected_models):
        rows = plot_df[plot_df["model"] == model_name]
        if rows.empty:
            continue
        row = rows.iloc[0]

        q_norm = row["quality"] / q_max if q_max else 0
        s_norm = (row["speed"] / s_max) if (s_max and row["speed"] > 0) else 0
        # Affordability: inverse of price, normalized
        p_norm = 1 - (row["price"] / p_max) if (p_max and row["price"] > 0) else 0
        ctx_k  = _context_k(row.get("context", ""))
        c_norm = (ctx_k / c_max) if c_max else 0

        values = [q_norm, s_norm, p_norm, c_norm]
        values_pct = [round(v * 100) for v in values]

        color = _PALETTE[idx % len(_PALETTE)]
        provider_color = PROVIDER_COLORS.get(row.get("provider", ""), color)

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=DIMS + [DIMS[0]],
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
            line=dict(color=color, width=1.5),
            name=model_name[:28] + ("…" if len(model_name) > 28 else ""),
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                f"Intelligence: {values_pct[0]}%<br>"
                f"Speed: {values_pct[1]}%<br>"
                f"Affordability: {values_pct[2]}%<br>"
                f"Context: {values_pct[3]}%<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Model Comparison"
                "  <span style='font-size:11px;color:#3d3d3d;font-weight:400'>"
                "  ·  normalized 0–100 across all models</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        polar=dict(
            bgcolor=_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
                tickfont=dict(color="#333333", size=8, family=_FONT),
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.05)",
                showline=True,
            ),
            angularaxis=dict(
                tickfont=dict(color="#666666", size=11, family=_FONT),
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.06)",
            ),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(color="#555555", size=10, family=_FONT),
            x=1.05, y=1, xanchor="left",
        ),
        margin=dict(l=60, r=160, t=52, b=60),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )

    return fig
