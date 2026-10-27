"""Economics › Serve: GPU serving cost vs API output price, log–log.

One point per open-weights model family (data/serving_cost.serving_data).
x = Artificial Analysis output price, y = GPU cost per 1M output tokens, both
in $ per 1M and on ONE shared log range, so the dashed y = x line is the
break-even: below it the API price exceeds the GPU cost.

Inside Economics fill strength means strength of evidence (economics plan
0.6): a diamond is an InferenceX measurement, a 0.55-opacity circle is the
calibrated roofline. Hollow-vs-filled is NOT used: outside Economics it means
open vs closed weights. Two grey legend-only traces key the symbols, and each
provider's legend entry is its own legend-only circle (legendgroup), because
Plotly draws a trace's swatch from its FIRST point: a provider whose first
point was measured showed the "measured" diamond in the colour legend.

Service level: a measured point's 35 tok/s floor is on InferenceX's MEDIAN
per-user interactivity; a modelled point's is on every session (the roofline
gives every session the same rate). The hover says which.

The axis range comes from serving_cost.axis_range(), which depends on the data
only, so the axes never move when a control does.
"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    plot_text, empty_figure, PROVIDER_COLORS, DEFAULT_COLOR,
    BG, GRID, TICK, AXIS, FONT, legend_below, CHART_MARGIN, spotlight_split,
    log_ticks,
)

ACCENT = "#00d4ff"
KEY_GREY = "#9a9a9a"
SUB = "<span style='font-size:12px;color:#777777;font-weight:400'>"
MEASURED_SYMBOL = "diamond"
MODELLED_SYMBOL = "circle"
MODELLED_OPACITY = 0.55
MEASURED_OPACITY = 0.95

X_TITLE = "API output price, $ / 1M tokens  (Artificial Analysis, log)"
Y_TITLE = "GPU cost, $ / 1M output tokens  (8k-in / 1k-out, log)"


def _money(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    if v >= 1:
        return f"${v:,.2f}"
    if v >= 0.01:
        return f"${v:.3f}"
    return f"${v:.2g}"


def _tick(v: float) -> str:
    return f"${v:g}"


def _times(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v:.1f}×" if v < 10 else f"{v:.0f}×"


def _evidence_line(r) -> str:
    if r["evidence"] == "measured":
        sib = " (sibling checkpoint)" if r.get("identity") == "sibling" else ""
        tp = r.get("tp")
        ep = r.get("ep")
        par = []
        if pd.notna(tp) and tp:
            par.append(f"TP{int(tp)}")
        if pd.notna(ep) and ep and int(ep) > 1:
            par.append(f"EP{int(ep)}")
        spec = str(r.get("spec_method") or "")
        bits = [str(r.get("precision") or "").upper(), str(r.get("framework") or ""),
                " ".join(par), str(r.get("deployment") or "")]
        if spec and spec != "none":
            bits.append(spec.upper())
        bits = [b for b in bits if b]
        link = "click to open the run" if r.get("run_url") else "run link unavailable"
        return (f"Measured · InferenceX {plot_text(r['ix_model'])}{sib}<br>"
                f"{plot_text(' · '.join(bits))} · {plot_text(r.get('run_date', ''))}"
                f" · {link}")
    gsd = r.get("calib_gsd")
    if r.get("calib_fallback") or not (isinstance(gsd, float) and math.isfinite(gsd)):
        fit = f"k = {float(r['calib_k']):.2f}, vendor median (no InferenceX run on this GPU)"
    else:
        fit = (f"k = {float(r['calib_k']):.2f} fitted on {int(r['calib_models'])} "
               f"InferenceX models, residual ×{gsd:.1f}")
    sess = r.get("sessions")
    per = r.get("per_session_tps")
    tail = ""
    if pd.notna(sess) and pd.notna(per):
        tail = f"<br>FP8, {int(sess)} sessions at {float(per):.0f} tok/s each"
        if r.get("binding") == "session_cap":
            tail += " (InferenceX's largest swept concurrency)"
    return f"Modelled · roofline, {fit}{tail}"


def _service_line(r) -> str:
    """The 35 tok/s floor means different things for the two kinds of point."""
    if r["evidence"] == "measured":
        return "median user ≥ 35 tok/s (InferenceX median interactivity)"
    return "every session ≥ 35 tok/s"


def _hover(r) -> str:
    n = r.get("gpu_count")
    n_txt = f" ×{int(n)}" if pd.notna(n) and n else ""
    from data.serving_cost import preset_label, BASIS_LABELS
    basis = BASIS_LABELS.get(r["basis"], r["basis"])
    return (
        f"<b>{plot_text(r['base_model'])}</b>  ({plot_text(r['provider'])}, open)<br>"
        f"{plot_text(preset_label(r['gpu']))}{n_txt} · {basis} {_money(r['usd_per_gpu_hr'])}"
        f"/GPU-hr (InferenceX hyperscaler volume, {plot_text(r.get('rate_observed_on', ''))})"
        f" · {float(r['utilisation_pct']):g}% use<br>"
        f"{float(r['tok_s_per_gpu']):,.0f} output tok/s per GPU, {_service_line(r)}<br>"
        f"GPU cost {_money(r['usd_per_m_out'])} / 1M output tokens<br>"
        f"AA price {_money(r['price_in'])} in · {_money(r['price_out'])} out<br>"
        f"Markup ≥ {_times(r['markup_out'])} on output price alone<br>"
        f"Same-mix markup (8k in : 1k out): {_times(r['markup_mix'])}<br>"
        + _evidence_line(r)
    )


# Label placement geometry: the plot area at a 1366 px desktop (about 1,100 x
# 420 px inside the margins), the 10 px label font (~5.6 px per character,
# 12 px line) and the measured marker (12 px, radius 6). Approximate on purpose:
# it only has to keep a label off its near neighbours.
_PLOT_W_PX, _PLOT_H_PX = 1100.0, 420.0
_CHAR_PX, _LINE_PX, _MARK_R_PX, _GAP_PX = 5.6, 12.0, 6.0, 2.0
LABEL_POSITIONS = ("top center", "bottom center", "middle right", "middle left")


def _label_box(px, py, pos, n_chars):
    """(x0, x1, y0, y1) of a label in plot pixels (y up)."""
    w, h, off = n_chars * _CHAR_PX, _LINE_PX, _MARK_R_PX + _GAP_PX
    if pos == "top center":
        return (px - w / 2, px + w / 2, py + off, py + off + h)
    if pos == "bottom center":
        return (px - w / 2, px + w / 2, py - off - h, py - off)
    if pos == "middle right":
        return (px + off, px + off + w, py - h / 2, py + h / 2)
    return (px - off - w, px - off, py - h / 2, py + h / 2)


def _overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def label_positions(x, y, names, axis_range) -> dict:
    """{index: textposition} for the labelled (measured) points.

    Labels used to alternate top/bottom by price order alone, so two points
    close in price but apart in cost could both print below, and one name sat
    on the other's marker (Kimi K2.7 Code over Qwen3.5 397B). Here, highest
    point first, each label takes the first of top / bottom / right / left that
    covers no other labelled marker and no label already placed; if none is
    clear, the one with the least overlap."""
    lo, hi = axis_range
    span = max(hi - lo, 1e-9)
    pts = []
    for ix, xv, yv, nm in zip(getattr(x, "index", range(len(x))), x, y, names):
        try:
            xv, yv = float(xv), float(yv)
        except (TypeError, ValueError):
            continue
        if not (xv > 0 and yv > 0 and math.isfinite(xv) and math.isfinite(yv)):
            continue
        pts.append((ix, (math.log10(xv) - lo) / span * _PLOT_W_PX,
                    (math.log10(yv) - lo) / span * _PLOT_H_PX, len(str(nm))))
    r = _MARK_R_PX
    marks = {ix: (px - r, px + r, py - r, py + r) for ix, px, py, _ in pts}
    placed: list[tuple] = []
    out = {}
    for ix, px, py, n in sorted(pts, key=lambda p: (-p[2], p[1])):
        others = [m for j, m in marks.items() if j != ix] + placed
        best, best_hits = None, None
        for pos in LABEL_POSITIONS:
            box = _label_box(px, py, pos, n)
            hits = sum(_overlap(box, o) for o in others)
            if best_hits is None or hits < best_hits:
                best, best_hits = pos, hits
            if hits == 0:
                break
        out[ix] = best
        placed.append(_label_box(px, py, best, n))
    return out


def build_econ_serve(frame: pd.DataFrame, meta: dict | None = None,
                     axis_range: tuple[float, float] | None = None) -> go.Figure:
    meta = meta or {}
    if frame is None or frame.empty:
        return empty_figure("No open model can be priced on this GPU at these settings")
    lo, hi = axis_range or (-2.0, 1.0)
    df = frame.copy()
    df, providers = spotlight_split(df)
    # Measured points carry a name, placed so it covers no other measured
    # marker or label (label_positions).
    df["_label_pos"] = "top center"
    meas = df[df["evidence"] == "measured"]
    for ix, pos in label_positions(meas["price_out"], meas["usd_per_m_out"],
                                   meas["base_model"], (lo, hi)).items():
        df.loc[ix, "_label_pos"] = pos
    fig = go.Figure()

    # The break-even line first, so every point draws over it.
    fig.add_trace(go.Scatter(
        x=[10 ** lo, 10 ** hi], y=[10 ** lo, 10 ** hi], mode="lines",
        line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dash"),
        hoverinfo="skip", showlegend=False, name="1× · price = cost",
    ))

    for prov in providers:
        part = df[df["display_provider"] == prov]
        measured = (part["evidence"] == "measured").tolist()
        color = PROVIDER_COLORS.get(prov, DEFAULT_COLOR)
        # The colour key: one circle per provider whatever its points are.
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=prov, legendgroup=prov,
            marker=dict(color=color, symbol=MODELLED_SYMBOL, size=9),
            hoverinfo="skip", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=part["price_out"], y=part["usd_per_m_out"],
            mode="markers+text", name=prov, legendgroup=prov, showlegend=False,
            text=[plot_text(b) if m else "" for b, m in zip(part["base_model"], measured)],
            textposition=part["_label_pos"].tolist(),
            textfont=dict(color="#bbbbbb", size=10, family=FONT),
            marker=dict(
                color=color,
                symbol=[MEASURED_SYMBOL if m else MODELLED_SYMBOL for m in measured],
                size=[12 if m else 8 for m in measured],
                opacity=[MEASURED_OPACITY if m else MODELLED_OPACITY for m in measured],
                line=dict(width=0.8, color=BG),
            ),
            customdata=[[_hover(r), str(r.get("run_url") or "")]
                        for _, r in part.iterrows()],
            hovertemplate="%{customdata[0]}<extra></extra>",
        ))

    # Legend keys for the evidence encoding (neutral grey, legend only).
    for name, symbol, op in (("measured (InferenceX run)", MEASURED_SYMBOL, MEASURED_OPACITY),
                             ("modelled (calibrated roofline)", MODELLED_SYMBOL,
                              MODELLED_OPACITY)):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=name,
            marker=dict(color=KEY_GREY, symbol=symbol, size=10, opacity=op),
            hoverinfo="skip", showlegend=True,
        ))

    n_meas = int((frame["evidence"] == "measured").sum())
    n_mod = len(frame) - n_meas
    med = float(pd.to_numeric(frame["markup_out"], errors="coerce").median())
    med_mix = float(pd.to_numeric(frame["markup_mix"], errors="coerce").median())
    # Both shares, each named by its basis: on output price alone (the floor,
    # and what the dashed 1x line shows) and at the same 8k:1k mix.
    above_out = float((pd.to_numeric(frame["markup_out"], errors="coerce") > 1).mean())
    above_mix = float((pd.to_numeric(frame["markup_mix"], errors="coerce") > 1).mean())
    from data.serving_cost import preset_label, BASIS_LABELS, DEFAULT_SERVE_GPU
    gpu = meta.get("gpu", DEFAULT_SERVE_GPU)
    gpu_txt = "cheapest GPU per model" if gpu == DEFAULT_SERVE_GPU else preset_label(gpu)
    basis = BASIS_LABELS.get(meta.get("basis", ""), meta.get("basis", "")).lower()
    util = meta.get("util_pct")
    unserved = int(meta.get("n_unserved") or 0)
    title = (
        "Serving Cost vs API Price  " + SUB
        + f"{len(frame)} open model families · {n_meas} measured, {n_mod} modelled"
        + (f" · {unserved} not servable on ≤ 8 GPUs" if unserved else "")
        + "<br>"
        + f"{plot_text(gpu_txt)} · InferenceX hyperscaler-volume {basis}ed $/GPU-hr"
        + (f" at {float(util):g}% use" if util is not None else "")
        + "<br>"
        + f"priced above GPU cost: {above_out:.0%} on output price alone (1× line),"
        + f" {above_mix:.0%} at the same 8k:1k mix · median markup ≥ {_times(med)},"
        + f" {_times(med_mix)} same-mix"
        + "</span>"
    )
    tickvals, ticktext = log_ticks(10 ** lo, 10 ** hi, fmt=_tick)
    axis = dict(type="log", range=[lo, hi], tickmode="array", tickvals=tickvals,
                ticktext=ticktext, gridcolor=GRID, zeroline=False,
                tickfont=dict(color=TICK, size=11, family=FONT), showgrid=True,
                showline=False, ticks="", automargin=True)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(text=title, font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
                   x=0.0, xanchor="left", pad=dict(l=20, t=16)),
        xaxis={**axis, "title": dict(text=X_TITLE, font=dict(color=AXIS, size=12),
                                     standoff=12)},
        yaxis={**axis, "title": dict(text=Y_TITLE, font=dict(color=AXIS, size=12),
                                     standoff=12)},
        legend={**legend_below(), "yref": "container", "y": 0, "yanchor": "bottom"},
        margin={**CHART_MARGIN, "t": 100, "b": 120},
        height=640,
        hovermode="closest",
        hoverlabel=dict(bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
                        font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1),
        annotations=[
            dict(x=hi, y=hi, xref="x", yref="y", text="1× · output price = cost ",
                 showarrow=False, xanchor="right", yanchor="bottom",
                 font=dict(color="#777777", size=10, family=FONT)),
            dict(x=1, y=0, xref="paper", yref="paper", text="price above GPU cost ↘",
                 showarrow=False, xanchor="right", yanchor="bottom",
                 font=dict(color="#666666", size=10, family=FONT)),
            dict(x=0, y=1, xref="paper", yref="paper", text="↖ GPU cost above price",
                 showarrow=False, xanchor="left", yanchor="top",
                 font=dict(color="#666666", size=10, family=FONT)),
        ],
    )
    return fig
