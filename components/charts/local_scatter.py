"""
Local models — VRAM vs Quality scatter chart.

X-axis : VRAM required (GiB) at the selected quantization AND context —
         weights + KV cache + runtime overhead — log scale
Y-axis : AA Intelligence Index (raw, calibrated to AA scale)
Size   : tokens/second — single-stream or aggregate at optimal concurrency,
         whichever the SPEED control selects
Color  : Model family (the legend encodes colour ONLY)
Shape  : ● dense  ◆ MoE (mixture-of-experts) — never the lab

A vertical dashed line marks the user's available VRAM.
Models to the left of the line are runnable; those to the right are not.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    plot_text,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
    bubble_size, legend_below, QUALITY_INDEX_MAX, LOCAL_SPEED_REF,
    LOCAL_THROUGHPUT_REF,
)
from components.charts.local_compat import (
    KV_NOTE, ESTIMATED_KV_NOTE, ctx_labels, speed_notes, cost_notes,
    one_per_family, empty_state,
)
from data.local_models import (
    DEFAULT_FAMILY_COLOR, DEFAULT_SPEED_MODE, speed_columns,
    tab_family_colors, catalogue_quality_max, OTHER_FAMILY,
)

_TITLE = "VRAM Requirement vs Intelligence"


def _x_bounds(vram_req, vram_gb: float) -> tuple[float, float]:
    """Log-axis ends from the data and the VRAM line, with a little padding.

    The axis used to start at a fixed 0.08 GB while the smallest model needs
    ~1.8 GB, so the first two decades of every chart were empty. It now spans
    the frame's own requirements and the user's VRAM, whichever reach further,
    padded by ~1/3 of a doubling-step at each end so no mark sits on an edge."""
    vals = [float(v) for v in vram_req if v == v and v > 0]
    lo = min(vals + [float(vram_gb)]) if vram_gb and vram_gb > 0 else min(vals or [1.0])
    hi = max(vals + [float(vram_gb) if vram_gb and vram_gb > 0 else 0.0, 1.0])
    return lo / 1.45, hi * 1.3


def _y_top(df_quality) -> float:
    """Top of the score axis: just above the open-weight catalogue's best
    score, rounded up to a multiple of 5. Not the hosted scale's fixed 70 —
    the open-weight peak is 47 on v4 and 33 on v5, which left a third of the
    plot empty — and not the frame's own max, so a tag filter cannot move a
    model up or down the axis."""
    import math
    ref = catalogue_quality_max()
    here = pd.to_numeric(df_quality, errors="coerce").max()
    peak = max([v for v in (ref, here) if v is not None and v == v] or [QUALITY_INDEX_MAX])
    return float(math.ceil(peak * 1.08 / 5.0) * 5)

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
    speed_mode: str = DEFAULT_SPEED_MODE,
) -> go.Figure:
    """
    Scatter: VRAM required vs quality score.
    df is the output of data.local_models.get_local_df().
    """
    if df.empty:
        return empty_state(_TITLE, "No model matches these tags",
                           "Clear a tag filter to see the catalogue", height=440)

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
        return empty_state(_TITLE, "No scored model to place",
                           "Every model here is still awaiting an AA score",
                           height=440)
    # One mark per model: AA's reasoning-effort tiers share weights, VRAM and
    # speed, so "GLM-5.3 (Max)" and "(Low)" were two marks at one x. The best
    # variant speaks for the model, as on Overview and the ranked chart below.
    df = one_per_family(df)
    _named = tab_family_colors()
    df = df.assign(_tab_family=[f if f in _named else OTHER_FAMILY for f in df["family"]])

    _speed_col, _speed_label = speed_columns(speed_mode)
    _speed_ref = (LOCAL_THROUGHPUT_REF if speed_mode == "throughput"
                  else LOCAL_SPEED_REF)

    # Whether the KV figure in the hover came from a published config or the
    # fitted estimator, the per-row context actually priced, and the speed
    # line. All three are shared with local_compat.py so the two hovers say the
    # same thing about the same row — see the notes there.
    df = df.copy()
    df["kv_note"] = (df["kv_source"] if "kv_source" in df else "none") \
        .map(KV_NOTE).fillna(ESTIMATED_KV_NOTE)
    df["ctx_label"], _n_capped = ctx_labels(df, ctx_tokens)
    df["speed_note"] = [s + c for s, c in zip(speed_notes(df, speed_mode), cost_notes(df))]
    for _c in ("weights_gb", "kv_gb", "sessions", "total_tps"):
        if _c not in df:
            df[_c] = 0

    fig = go.Figure()

    # Draw each family as its own trace so the legend is grouped by family
    # Families that actually got a runnable mark. The old rule was
    # showlegend=(fits_val == "yes"), so a family with nothing that fits never
    # appeared; the proxies have to reproduce that or the legend grows by the
    # 18 families whose models are all too big for the selected hardware.
    _legend_families: list = []
    for family, fdf in df.groupby("_tab_family", sort=False):
        color = _named.get(family, DEFAULT_FAMILY_COLOR)
        if not fdf[fdf["fits"] == "yes"].empty:
            _legend_families.append(family)

        for fits_val, opacity in [("yes", _FIT_ALPHA), ("tight", _TIGHT_ALPHA), ("no", _NO_ALPHA)]:
            sub = fdf[fdf["fits"] == fits_val]
            if sub.empty:
                continue

            # Scale bubble size: sqrt of tok/s so fast models aren't enormous.
            # The reference is per-metric — aggregate throughput runs 2-4x
            # single-stream, so sharing one would clamp every bubble in
            # throughput mode and the size channel would stop encoding anything.
            sizes = bubble_size(sub[_speed_col], _speed_ref).values

            hover = (
                "<b>%{customdata[0]}</b><br>"
                "Family: %{customdata[13]}<br>"
                "VRAM required: %{x:.1f} GB at %{customdata[8]} context<br>"
                "  ↳ weights %{customdata[5]:.1f} + KV cache %{customdata[6]:.1f}"
                " + runtime %{customdata[11]:.1f} GB  ·  %{customdata[7]}<br>"
                "Intelligence: %{y:.0f}<br>"
                "%{customdata[12]}<br>"
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
                # NEVER the legend entry. The symbol is a per-point array, so
                # Plotly draws the swatch with whichever model happens to sort
                # first in the family — Alibaba came out a circle and NVIDIA a
                # diamond purely by accident of ordering, which reads as though
                # SHAPE encoded the lab. It encodes dense vs MoE. The legend is
                # drawn from fixed-symbol proxies below so it carries colour and
                # nothing else.
                showlegend=False,
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
                    family=sub["family"].map(plot_text),
                )[["name", "speed_tps", "license", "context_k", "tags_str",
                   "weights_gb", "kv_gb", "kv_note", "ctx_label",
                   "sessions", "total_tps", "overhead_gb", "speed_note",
                   "family"]].values,
                hovertemplate=hover,
            ))

    # Legend proxies: one per family, fixed circle, no data. They exist so the
    # FAMILY legend encodes colour only. The dense/MoE key is the annotation in
    # the corner, which is the one place shape is explained.
    # Named families by catalogue order of the palette, "Other" last.
    _rank = {f: i for i, f in enumerate(_named)}
    _legend_families.sort(key=lambda f: (f == OTHER_FAMILY, _rank.get(f, 99), f))
    for family in _legend_families:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=plot_text(family), legendgroup=plot_text(family),
            marker=dict(color=_named.get(family, DEFAULT_FAMILY_COLOR),
                        size=9, symbol="circle",
                        line=dict(width=0.5, color="rgba(255,255,255,0.15)")),
            hoverinfo="skip", showlegend=True,
        ))

    # Vertical threshold line at user's VRAM
    # The dashed line only; its label is placed below, once the axis is known.
    fig.add_vline(
        x=vram_gb,
        line=dict(color="rgba(0,212,255,0.55)", width=1.5, dash="dot"),
    )

    import math

    x_lo, x_hi = _x_bounds(df["vram_req_gb"], vram_gb)

    # Shaded "runnable" region (left of threshold)
    fig.add_vrect(
        x0=x_lo, x1=vram_gb,
        fillcolor="rgba(34,197,94,0.025)",
        line_width=0,
        layer="below",
    )

    # 1-2-4 doubling ladder (with 0.1/0.25/0.5 under 1 GB) clipped to the
    # visible span, so the last gap is never wider than the others.
    ladder = [0.1, 0.25, 0.5]
    _v = 1
    while _v <= x_hi:
        ladder.append(_v)
        _v *= 2
    tick_vals = [v for v in ladder if x_lo <= v <= x_hi]
    tick_texts = [f"{v:g}" for v in tick_vals]
    x_log_min, x_log_max = math.log10(x_lo), math.log10(x_hi)
    y_top = _y_top(df["quality"])

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                _TITLE
                + "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                f"  ·  {quant} at {_ctx_label(ctx_tokens)} context"
                + (f" ({_n_capped} priced at their own shorter max)" if _n_capped else "")
                + "  ·  left of line = runnable  ·  bubble = "
                f"{_speed_label} tok/s"
                + (f"  ·  {_pending_n} not yet scored (listed below)" if _pending_n else "")
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
            range=[0, y_top],
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        # Up to 26 family entries in a vertical side legend meant ~10 were
        # hidden behind a Plotly scrollbar, while its fixed 160px gutter left
        # the plot 35% of the card on a phone.
        # The key rides in the legend title, above the entries: as an
        # annotation in the plot's top-right corner it sat on the
        # highest-scoring marks and on the VRAM line whenever the line ran
        # near the right edge.
        legend={**legend_below(y=-0.14),
                "font": dict(color="#999999", size=10, family=_FONT),
                "title": dict(text=(f"colour = lab (grey = other labs)  ·  "
                                    f"● dense  ◆ mixture-of-experts  ·  "
                                    f"bubble size = {_speed_label} tok/s"),
                              font=dict(color="#777", size=10), side="top")},
        margin=dict(l=56, r=28, t=52, b=148),
        # docs/app.js narrowFigure, below max_width (a phone): the key in
        # short form, so the legend title fits the plot's width.
        meta={"narrow": {"max_width": 500, "data": {}, "layout": {
            "legend": {"title": dict(text="● dense  ◆ MoE  ·  size = tok/s",
                                     font=dict(color="#777", size=10), side="top")},
        }}},
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )
    # The VRAM line's value. add_vline's own label took x in DATA units, and on
    # a log axis Plotly reads an annotation's x as log10(value): "32 GB" was
    # placed at 10^32 and never drawn, and "1 GB" sat over 10 GB, a decade
    # right of its line. Placed explicitly in the axis's own (log) coordinate,
    # on the side of the line with room.
    _at = math.log10(vram_gb) if vram_gb and vram_gb > 0 else x_log_min
    _right_side = (_at - x_log_min) / ((x_log_max - x_log_min) or 1) > 0.85
    fig.add_annotation(
        x=_at, xref="x", y=1, yref="y domain", yanchor="top",
        xanchor="right" if _right_side else "left",
        text=f"{vram_gb:.0f} GB  " if _right_side else f"  {vram_gb:.0f} GB",
        showarrow=False,
        font=dict(color="#00d4ff", size=10, family=_FONT),
    )

    return fig
