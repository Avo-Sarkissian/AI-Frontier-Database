"""Economics › Serve: GPU serving cost vs API output price, log–log.

One point per open-weights model family (data/serving_cost.serving_data).
x = Artificial Analysis output price, y = GPU cost per 1M output tokens, both
in $ per 1M and on ONE shared log range, so the dashed y = x line is the
break-even: below it the API price exceeds the GPU cost.

Inside Economics fill strength means strength of evidence (economics plan
0.6): a diamond is an InferenceX measurement, a 0.55-opacity circle is the
calibrated roofline. Hollow-vs-filled is NOT used: outside Economics it means
open vs closed weights. Two grey legend-only traces key the symbols.

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
    return f"Modelled · roofline, {fit}{tail}"


def _hover(r) -> str:
    n = r.get("gpu_count")
    n_txt = f" ×{int(n)}" if pd.notna(n) and n else ""
    from data.serving_cost import preset_label, BASIS_LABELS
    basis = BASIS_LABELS.get(r["basis"], r["basis"])
    return (
        f"<b>{plot_text(r['base_model'])}</b>  ({plot_text(r['provider'])}, open)<br>"
        f"{plot_text(preset_label(r['gpu']))}{n_txt} · {basis} {_money(r['usd_per_gpu_hr'])}"
        f"/GPU-hr (InferenceX, {plot_text(r.get('rate_observed_on', ''))})"
        f" · {float(r['utilisation_pct']):g}% use<br>"
        f"{float(r['tok_s_per_gpu']):,.0f} output tok/s per GPU, every user ≥ 35 tok/s<br>"
        f"GPU cost {_money(r['usd_per_m_out'])} / 1M output tokens<br>"
        f"AA price {_money(r['price_in'])} in · {_money(r['price_out'])} out<br>"
        f"Markup ≥ {_times(r['markup_out'])} on output price alone<br>"
        f"Same-mix markup (8k in : 1k out): {_times(r['markup_mix'])}<br>"
        + _evidence_line(r)
    )


def build_econ_serve(frame: pd.DataFrame, meta: dict | None = None,
                     axis_range: tuple[float, float] | None = None) -> go.Figure:
    meta = meta or {}
    if frame is None or frame.empty:
        return empty_figure("No open model can be priced on this GPU at these settings")
    lo, hi = axis_range or (-2.0, 1.0)
    df = frame.copy()
    df, providers = spotlight_split(df)
    # Measured points carry a name. Neighbours in x alternate above and below
    # their marker so labels of points close together do not print over each
    # other.
    df["_label_pos"] = "top center"
    meas_idx = df.index[df["evidence"] == "measured"]
    order = df.loc[meas_idx].sort_values(["price_out", "usd_per_m_out"]).index
    for i, ix in enumerate(order):
        df.loc[ix, "_label_pos"] = "top center" if i % 2 == 0 else "bottom center"
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
        fig.add_trace(go.Scatter(
            x=part["price_out"], y=part["usd_per_m_out"],
            mode="markers+text", name=prov,
            text=[plot_text(b) if m else "" for b, m in zip(part["base_model"], measured)],
            textposition=part["_label_pos"].tolist(),
            textfont=dict(color="#bbbbbb", size=10, family=FONT),
            marker=dict(
                color=PROVIDER_COLORS.get(prov, DEFAULT_COLOR),
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
    below = float((frame["usd_per_m_out"] < frame["price_out"]).mean())
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
        + f"{plot_text(gpu_txt)} · {basis}ed $/GPU-hr (InferenceX)"
        + (f" at {float(util):g}% use" if util is not None else "")
        + f" · {below:.0%} priced above GPU cost<br>median markup ≥ {_times(med)} on"
        + f" output price alone, {_times(med_mix)} at the same 8k:1k mix"
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
            dict(x=hi, y=hi, xref="x", yref="y", text="1× · price = cost ",
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
