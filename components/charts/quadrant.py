"""
Speed vs Quality quadrant chart.
Divides models into 4 zones based on median speed and quality.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, PROVIDER_SHAPES, DEFAULT_COLOR, DEFAULT_SHAPE,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)

_ZONE = "rgba(255,255,255,0.02)"


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
    if pd.isna(max_price) or max_price == 0:
        max_price = 1
    plot_df["size"] = plot_df["price"].apply(
        lambda p: 8 + (1 - min(p / max_price, 1)) * 22 if pd.notna(p) and p > 0 else 8
    )

    # Pearson r between speed and quality
    _corr_r = None
    try:
        _corr_r = np.corrcoef(plot_df["speed"], plot_df["quality"])[0, 1]
    except Exception:
        pass

    fig = go.Figure()

    # Quadrant shading
    x_max = plot_df["speed"].max() * 1.15
    y_max = plot_df["quality"].max() * 1.15

    zone_labels = [
        (0,         med_speed,   med_quality, y_max,       "Slow · Smart",  0.25, 0.75),
        (med_speed, x_max,       med_quality, y_max,       "Fast · Smart",  0.75, 0.75),
        (0,         med_speed,   0,           med_quality, "Slow · Weak",   0.25, 0.25),
        (med_speed, x_max,       0,           med_quality, "Fast · Weak",   0.75, 0.25),
    ]

    for x0, x1, y0, y1, label, rx, ry in zone_labels:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=_ZONE, line=dict(width=0), layer="below")

    # Median crosshair lines
    fig.add_hline(y=med_quality, line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"))
    fig.add_vline(x=med_speed,   line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"))

    # Group small providers into "Other"
    _counts = plot_df["provider"].value_counts()
    _top_provs = set(_counts.head(10).index)
    plot_df["display_provider"] = plot_df["provider"].apply(
        lambda p: p if p in _top_provs else "Other"
    )

    # Per-provider scatter
    providers = sorted(plot_df["display_provider"].unique(), key=lambda p: (p == "Other", p))
    for provider in providers:
        pdf = plot_df[plot_df["display_provider"] == provider]
        color  = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
        symbol = PROVIDER_SHAPES.get(provider, DEFAULT_SHAPE)

        latency_str = pdf["latency"].apply(
            lambda l: f"{l:.2f}s" if pd.notna(l) and l > 0 else "N/A"
        )

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "Quality: %{y}<br>"
            "Speed: %{x:.0f} tok/s<br>"
            "Price: $%{customdata[2]:.3f}/M tokens<br>"
            "Latency (TTFT): %{customdata[3]}<br>"
            "<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=pdf["speed"],
            y=pdf["quality"],
            mode="markers",
            name=provider,
            marker=dict(
                color=color,
                symbol=symbol,
                size=pdf["size"],
                opacity=0.75,
                line=dict(width=0.5, color="rgba(255,255,255,0.10)"),
            ),
            customdata=list(zip(pdf["model"], pdf["provider"], pdf["price"], latency_str)),
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
            font=dict(color="rgba(255,255,255,0.28)", size=13, family=_FONT),
            xref="x", yref="y",
        ))

    # Correlation annotation
    if _corr_r is not None:
        annotations.append(dict(
            x=0.01, y=0.99,
            xref="paper", yref="paper",
            text=f"r = {_corr_r:.2f}  (speed vs quality)",
            showarrow=False,
            xanchor="left", yanchor="top",
            font=dict(color="#666666", size=10, family=_FONT),
        ))

    # Label top-right quadrant outliers (Fast · Smart).
    # Use 75th-percentile speed as threshold so labels stay in the genuinely
    # fast region, and cap at 5 to avoid label pile-ups.
    speed_p75 = plot_df["speed"].quantile(0.75)
    fast_smart = plot_df[
        (plot_df["speed"] > speed_p75) & (plot_df["quality"] > med_quality)
    ].sort_values("quality", ascending=False).head(5)
    if not fast_smart.empty:
        fig.add_trace(go.Scatter(
            x=fast_smart["speed"],
            y=fast_smart["quality"],
            mode="text",
            text=fast_smart["model"].apply(lambda m: m[:20] + "…" if len(m) > 20 else m),
            textposition="top center",
            textfont=dict(color="rgba(255,255,255,0.45)", size=10, family=_FONT),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Speed vs. Intelligence"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  bubble size ∝ affordability  ·  shape = provider family</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Speed  (tokens / second)", font=dict(color=_AXIS, size=12), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
            range=[0, x_max],
        ),
        yaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=12), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
            range=[0, y_max],
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
            font=dict(color="#999999", size=11, family=_FONT),
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
