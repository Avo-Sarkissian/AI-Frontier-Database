"""
Local models — VRAM vs Quality scatter chart.

X-axis : VRAM required (GiB) at the selected quantization AND context —
         weights + KV cache + runtime overhead — log scale
Y-axis : AA Intelligence Index (raw, calibrated to AA scale)
Size   : Estimated SINGLE-STREAM tokens/second on the selected hardware
Color  : Model family
Shape  : ● dense  ◆ MoE (mixture-of-experts)

A vertical dashed line marks the user's available VRAM.
Models to the left of the line are runnable; those to the right are not.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    plot_text,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
    bubble_size, legend_below, QUALITY_INDEX_MAX, LOCAL_SPEED_REF,
)
from data.local_models import FAMILY_COLORS, DEFAULT_FAMILY_COLOR

_FIT_ALPHA  = 1.00   # fully runnable
_TIGHT_ALPHA = 0.75  # fits but < 1 GB headroom
_NO_ALPHA   = 0.25   # won't fit — greyed out


def _ctx_label(ctx_tokens) -> str:
    """"32k", not "32768"."""
    try:
        n = int(ctx_tokens or 0)
    except (TypeError, ValueError):
        return "?"
    if n <= 0:
        return "0"
    return f"{n // 1024}k" if n >= 1024 else str(n)


def build_local_scatter(
    df: pd.DataFrame,
    vram_gb: float,
    quant: str,
    ctx_tokens=None,
) -> go.Figure:
    """
    Scatter: VRAM required vs quality score.
    df is the output of data.local_models.get_local_df().
    """
    if df.empty:
        return _empty("No models found.")

    # This chart's Y axis IS the quality score, so a model nobody has scored
    # cannot be placed on it. Dropping it silently would be the same lie in the
    # other direction, so the count is surfaced in the subtitle and the model is
    # still listed — with an outlined bar — in the compatibility chart below.
    _pending_n = 0
    if "pending" in df.columns:
        _mask = df["pending"].fillna(False) & df["quality"].isna()
        _pending_n = int(_mask.sum())
        df = df[~_mask]
    if df.empty:
        return _empty("No scored models found.")

    # Whether the KV figure in the hover came from a published config or the
    # fitted estimator. The reader has to be able to tell — see the same note
    # in local_compat.py.
    _KV_NOTE = {"config": "published architecture",
                "estimated": "architecture estimated, ±30%",
                "none": "no context priced"}
    df = df.copy()
    df["kv_note"] = (df["kv_source"] if "kv_source" in df else "none") \
        .map(_KV_NOTE).fillna("architecture estimated, ±30%")
    df["ctx_label"] = _ctx_label(ctx_tokens)
    for _c in ("weights_gb", "kv_gb", "sessions", "total_tps"):
        if _c not in df:
            df[_c] = 0

    fig = go.Figure()

    # Draw each family as its own trace so the legend is grouped by family
    for family, fdf in df.groupby("family", sort=False):
        color = FAMILY_COLORS.get(family, DEFAULT_FAMILY_COLOR)

        for fits_val, opacity in [("yes", _FIT_ALPHA), ("tight", _TIGHT_ALPHA), ("no", _NO_ALPHA)]:
            sub = fdf[fdf["fits"] == fits_val]
            if sub.empty:
                continue

            # Scale bubble size: sqrt of tok/s so fast models aren't enormous
            sizes = bubble_size(sub["speed_tps"], LOCAL_SPEED_REF).values

            hover = (
                "<b>%{customdata[0]}</b><br>"
                f"Family: {plot_text(family)}<br>"
                "VRAM required: %{x:.1f} GB at %{customdata[8]} context<br>"
                "  ↳ weights %{customdata[5]:.1f} + KV cache %{customdata[6]:.1f}"
                " + runtime 1.5 GB  ·  %{customdata[7]}<br>"
                "Intelligence: %{y:.0f}<br>"
                "Speed: %{customdata[1]:.0f} tok/s single stream<br>"
                "Sessions: ×%{customdata[9]} concurrent → %{customdata[10]:,.0f} tok/s total<br>"
                "License: %{customdata[2]}<br>"
                "Max context: %{customdata[3]}k tokens<br>"
                "Tags: %{customdata[4]}<br>"
                "<extra></extra>"
            )

            fig.add_trace(go.Scatter(
                x=sub["vram_req_gb"],
                y=sub["quality"],
                mode="markers",
                name=plot_text(family),
                showlegend=(fits_val == "yes"),   # one legend entry per family
                legendgroup=plot_text(family),
                marker=dict(
                    color=color,
                    opacity=opacity,
                    size=sizes,
                    sizemode="diameter",
                    # Per POINT, not per subgroup: `sub["moe"].any()` drew every
                    # dense model in a mixed family as a diamond, making the
                    # on-chart "◆ = MoE" key false for 24 of 38 dense models.
                    symbol=["diamond" if m else "circle" for m in sub["moe"]],
                    line=dict(width=0.5, color="rgba(255,255,255,0.15)"),
                ),
                # customdata is POSITIONAL — every %{customdata[N]} above indexes
                # into this list, so a column added here without updating the
                # template silently shifts every later field.
                customdata=sub.assign(
                    name=sub["name"].map(plot_text),
                    license=sub["license"].map(plot_text),
                    tags_str=sub["tags_str"].map(plot_text),
                )[["name", "speed_tps", "license", "context_k", "tags_str",
                   "weights_gb", "kv_gb", "kv_note", "ctx_label",
                   "sessions", "total_tps"]].values,
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

    import math

    # The axis is logarithmic, so including the 600B-class models costs a
    # little width rather than compressing the runnable region — and dropping
    # them silently removed a third of the catalogue from the chart.
    x_max = max(vram_gb * 2.5, float(df["vram_req_gb"].max()) * 1.15, 8)
    x_log_max = math.log10(x_max)
    x_log_min = math.log10(0.08)   # ~0.08 GB minimum so tiny models show

    # Build sensible tick positions within the visible range
    all_tick_vals  = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    all_tick_texts = ["0.1", "0.25", "0.5", "1", "2", "4", "8", "16", "32", "64", "128", "256", "512"]
    tick_pairs = [(v, t) for v, t in zip(all_tick_vals, all_tick_texts) if v <= x_max * 1.05]
    tick_vals  = [p[0] for p in tick_pairs]
    tick_texts = [p[1] for p in tick_pairs]

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "VRAM Requirement vs Intelligence"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                f"  ·  {quant} at {_ctx_label(ctx_tokens)} context"
                f"  ·  left of line = runnable  ·  bubble = single-stream speed"
                + (f"  ·  {_pending_n} newer model{'s' if _pending_n != 1 else ''} "
                   f"not yet scored — see the ranking below" if _pending_n else "")
                + "</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="VRAM Required (GB)", font=dict(color=_AXIS, size=12), standoff=12),
            type="log",
            range=[x_log_min, x_log_max],
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
            tickvals=tick_vals,
            ticktext=tick_texts,
        ),
        yaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=12), standoff=12),
            range=[0, QUALITY_INDEX_MAX],
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        # Up to 26 family entries in a vertical side legend meant ~10 were
        # hidden behind a Plotly scrollbar, while its fixed 160px gutter left
        # the plot 35% of the card on a phone.
        legend={**legend_below(y=-0.14),
                "font": dict(color="#999999", size=10, family=_FONT),
                "title": dict(text="FAMILY", font=dict(color="#777", size=10))},
        margin=dict(l=56, r=28, t=52, b=132),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
        # Legend annotation for size = speed
        annotations=[
            dict(
                x=1.0, y=1.04, xref="paper", yref="paper",
                xanchor="right",
                text="Bubble size = single-stream tok/s<br>◆ = MoE architecture",
                showarrow=False,
                font=dict(color="#666666", size=10, family=_FONT),
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
