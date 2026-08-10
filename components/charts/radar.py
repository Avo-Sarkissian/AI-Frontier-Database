"""
Model comparison radar chart.
Normalizes quality, speed, affordability, and context window
to 0–1 scale and overlays up to 5 selected models.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, FONT as _FONT, empty_figure, QUALITY_INDEX_MAX, legend_below,
    RADAR_SPEED_MAX, RADAR_PRICE_MAX, RADAR_LATENCY_MAX, RADAR_CONTEXT_K_MAX,
)

_PALETTE = [
    "#00d4ff", "#c084fc", "#34d399", "#fb923c", "#f472b6",
]

DIMS = ["Intelligence", "Speed", "Affordability", "Context", "Latency"]


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
        # A bare go.Figure() renders with Plotly's default WHITE canvas, which
        # flashed a blank white card inside the dark dashboard whenever the
        # selection was cleared.
        return empty_figure("Select up to 5 models to compare")

    # Fixed reference per axis. The comment here used to claim these were taken
    # "across the FULL dataset", but both render paths hand this function an
    # already-filtered frame, so the axes rescaled with the provider/search
    # filters — the same model's profile changed shape, and the subtitle's
    # "normalized 0-100 across all models" was untrue. Now a shape means the
    # same thing in every filter state.
    q_max = QUALITY_INDEX_MAX
    s_max = RADAR_SPEED_MAX
    p_max = RADAR_PRICE_MAX
    l_max = RADAR_LATENCY_MAX
    c_max = RADAR_CONTEXT_K_MAX

    df_ctx = df["context"].apply(_context_k)

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
        # Latency: lower is better — inverted
        lat = row.get("latency", 0)
        l_norm = 1 - (lat / l_max) if (l_max and pd.notna(lat) and lat > 0) else 0

        # Clamp to [0, 1]. With a fixed ceiling a model can exceed it — a 70s
        # TTFT against a 30s reference produced a NEGATIVE radius, which Plotly
        # draws through the centre of the polar plot.
        values = [min(1.0, max(0.0, v)) for v in (q_norm, s_norm, p_norm, c_norm, l_norm)]
        values_pct = [round(v * 100) for v in values]

        # One trace per MODEL, so colour must vary per model. Keying it on the
        # provider drew two Anthropic models in the identical hue and fill,
        # making the one chart whose whole job is comparison unable to tell
        # them apart. _PALETTE is indexed by trace instead, which is also what
        # its per-index design was always for.
        color = _PALETTE[idx % len(_PALETTE)]

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=DIMS + [DIMS[0]],
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
            line=dict(color=color, width=1.5),
            name=model_name[:28] + ("…" if len(model_name) > 28 else ""),
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                f"Intelligence: {values_pct[0]}%<br>"
                f"Speed: {values_pct[1]}%<br>"
                f"Affordability: {values_pct[2]}%<br>"
                f"Context: {values_pct[3]}%<br>"
                f"Latency: {values_pct[4]}%<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Model Comparison"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  normalized 0–100 across all models</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        polar=dict(
            bgcolor=_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
                tickfont=dict(color="#777777", size=9, family=_FONT),
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.05)",
                showline=True,
            ),
            angularaxis=dict(
                tickfont=dict(color="#999999", size=12, family=_FONT),
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.06)",
            ),
        ),
        # Model names are long, so the vertical legend was wider than its own
        # 160px gutter and overlapped the polar plot on narrow viewports.
        legend=legend_below(y=-0.08),
        margin=dict(l=60, r=40, t=52, b=110),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )

    return fig
