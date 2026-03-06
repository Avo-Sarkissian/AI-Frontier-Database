"""
Local models — VRAM vs Quality scatter chart.

X-axis : VRAM required (GB) at the selected quantization — log scale
Y-axis : AA Intelligence Index (raw, calibrated to AA scale)
Size   : Estimated tokens/second on the selected hardware
Color  : Model family
Shape  : ● dense  ◆ MoE (mixture-of-experts)

A vertical dashed line marks the user's available VRAM.
Models to the left of the line are runnable; those to the right are not.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT
from data.local_models import FAMILY_COLORS, DEFAULT_FAMILY_COLOR

_FIT_ALPHA  = 1.00   # fully runnable
_TIGHT_ALPHA = 0.75  # fits but < 1 GB headroom
_NO_ALPHA   = 0.25   # won't fit — greyed out


def build_local_scatter(
    df: pd.DataFrame,
    vram_gb: float,
    quant: str,
) -> go.Figure:
    """
    Scatter: VRAM required vs quality score.
    df is the output of data.local_models.get_local_df().
    """
    if df.empty:
        return _empty("No models found.")

    fig = go.Figure()

    # Draw each family as its own trace so the legend is grouped by family
    for family, fdf in df.groupby("family", sort=False):
        color = FAMILY_COLORS.get(family, DEFAULT_FAMILY_COLOR)

        for fits_val, opacity in [("yes", _FIT_ALPHA), ("tight", _TIGHT_ALPHA), ("no", _NO_ALPHA)]:
            sub = fdf[fdf["fits"] == fits_val]
            if sub.empty:
                continue

            # Scale bubble size: sqrt of tok/s so fast models aren't enormous
            sizes = (sub["speed_tps"].clip(lower=1) ** 0.45 * 4).clip(upper=40)

            hover = (
                "<b>%{customdata[0]}</b><br>"
                f"Family: {family}<br>"
                "VRAM required: %{x:.1f} GB<br>"
                "Intelligence: %{y:.0f}<br>"
                "Speed: %{customdata[1]:.0f} tok/s<br>"
                "License: %{customdata[2]}<br>"
                "Context: %{customdata[3]}k tokens<br>"
                "Tags: %{customdata[4]}<br>"
                "<extra></extra>"
            )

            fig.add_trace(go.Scatter(
                x=sub["vram_req_gb"],
                y=sub["quality"],
                mode="markers",
                name=family,
                showlegend=(fits_val == "yes"),   # one legend entry per family
                legendgroup=family,
                marker=dict(
                    color=color,
                    opacity=opacity,
                    size=sizes,
                    sizemode="diameter",
                    symbol="diamond" if sub["moe"].any() else "circle",
                    line=dict(width=0.5, color="rgba(255,255,255,0.15)"),
                ),
                customdata=sub[["name", "speed_tps", "license", "context_k", "tags_str"]].values,
                hovertemplate=hover,
            ))

    # Vertical threshold line at user's VRAM
    fig.add_vline(
        x=vram_gb,
        line=dict(color="rgba(0,212,255,0.55)", width=1.5, dash="dot"),
        annotation_text=f"  {vram_gb:.0f} GB",
        annotation_font=dict(color="#00d4ff", size=10, family=_FONT),
        annotation_position="top right",
    )

    # Shaded "runnable" region (left of threshold)
    fig.add_vrect(
        x0=0, x1=vram_gb,
        fillcolor="rgba(34,197,94,0.025)",
        line_width=0,
        layer="below",
    )

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                "VRAM Requirement vs Intelligence"
                "  <span style='font-size:11px;color:#666666;font-weight:400'>"
                f"  ·  {quant} quantization  ·  left of line = runnable  ·  bubble = speed</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="VRAM Required (GB)", font=dict(color=_AXIS, size=11), standoff=12),
            type="log",
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
            tickvals=[0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            ticktext=["0.1", "0.25", "0.5", "1", "2", "4", "8", "16", "32", "64", "128", "256", "512"],
        ),
        yaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=11), standoff=12),
            range=[15, 100],
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
            font=dict(color="#888888", size=9, family=_FONT),
            x=1.01, y=1, xanchor="left",
            tracegroupgap=2,
            title=dict(text="FAMILY", font=dict(color="#666", size=9)),
        ),
        margin=dict(l=56, r=160, t=52, b=52),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
        # Legend annotation for size = speed
        annotations=[
            dict(
                x=1.01, y=0.02, xref="paper", yref="paper",
                xanchor="left",
                text="Bubble size = tokens/s<br>◆ = MoE architecture",
                showarrow=False,
                font=dict(color="#666666", size=9, family=_FONT),
                align="left",
            ),
        ],
    )

    return fig


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        annotations=[dict(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=msg, showarrow=False,
            font=dict(color="#777777", size=13, family=_FONT),
        )],
        margin=dict(l=40, r=40, t=60, b=40), height=500,
    )
    return fig
