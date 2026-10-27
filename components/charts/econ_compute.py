"""Economics › Compute: three figures from one data/gpu_tco.compute() result.

econ_tco            owned $ per busy GPU-hour per GPU as a stacked bar (capital |
                    power | colo), at the shared USE. The selected GPU is
                    solid, the rest muted; each bar names its capex basis
                    (they differ by source); the subtitle gives the total and
                    the three shares (they sum to 100%: the units check).
econ_ladder         the selected GPU's spread, cheapest to dearest on a log
                    $ per paid GPU-hour axis (owned at full use, like every
                    rental): one row per rung, one trace per mark, so no two
                    sources are ever joined by a line.
econ_rental_series  GetDeploying's four weekly medians for the GPU, one line
                    per billing series (never averaged together), with a gap
                    where a week is missing, and the owned cost per paid hour
                    as a dotted accent line.

Neutral greys plus the site accent: these are sources and cost components,
not model providers, so provider colours are not used. All marks are filled
(Economics' fill = evidence convention); nothing here uses hollow marks.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    plot_text, empty_figure, BG, GRID, TICK, AXIS, FONT, legend_below, CHART_MARGIN,
    log_ticks,
)
from data import gpu_tco as T

ACCENT = "#00d4ff"
SUB = "<span style='font-size:12px;color:#777777;font-weight:400'>"
COMPONENTS = (("capital", "capital", ACCENT), ("power", "power", "#9a9a9a"),
              ("colo", "colocation", "#5c5c5c"))
MUTED_OPACITY = 0.32
RUNG_COLORS = {"owned": ACCENT, "inferencex": "#8c8c8c", "contract_index": "#b3b3b3",
               "marketplace": "#cfcfcf", "neocloud_list": "#e0e0e0",
               "capacity_block": "#a6a6a6", "hyperscaler": "#f2f2f2"}
SERIES_STYLE = {"on_demand": ("#e6e6e6", "solid"), "spot": ("#6f6f6f", "dashdot"),
                "reserved_12m": ("#b3b3b3", "dash"), "reserved_36m": ("#8c8c8c", "longdash")}
GD_FOOTER = ("Weekly medians: GetDeploying (CC BY 4.0), getdeploying.com/dataset/gpu-prices, "
             "archived weekly by this site · newest week provisional")

_HOVERLABEL = dict(bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
                   font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
                   align="left")


def _usd(v, digits: int = 2) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"${v:,.{digits}f}"


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _layout(fig: go.Figure, title: str, height: int, **kw) -> None:
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(text=title, font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
                   x=0.0, xanchor="left", pad=dict(l=20, t=16)),
        height=height, hoverlabel=_HOVERLABEL, **kw)


def _axis(**kw) -> dict:
    return dict(gridcolor=GRID, zeroline=False, showline=False, ticks="",
                tickfont=dict(color=TICK, size=11, family=FONT), automargin=True, **kw)


def _prov_line(name: str, p: dict, fmt: str) -> str:
    v = p.get("value")
    val = "—" if v is None else fmt.format(v)
    tail = " · ".join(b for b in (p.get("status", ""), p.get("as_of", "")) if b)
    return f"{name} {plot_text(val)}" + (f"  <i>({plot_text(tail)})</i>" if tail else "")


def _basis(row: dict) -> str:
    return row["inputs"]["capex_usd"].get("basis", "")


def _inputs_hover(result: dict, row: dict) -> str:
    fin, fac, inp = result["financing"], result["facility"], row["inputs"]
    rate = fin["rate"]
    b = _basis(row)
    return "<br>".join([
        _prov_line("capex", inp["capex_usd"], "${:,.0f}/GPU"),
        *([f"capex basis: {plot_text(T.CAPEX_BASES[b][1])}"] if b else []),
        _prov_line("IT power", inp["it_kw"], "{:.3f} kW/GPU"),
        _prov_line("PUE", fac["pue"], "{:g}"),
        _prov_line("electricity", fac["usd_per_kwh"], "${:.4f}/kWh"),
        _prov_line("colocation", fac["colo_usd_kw_month"], "${:,.2f}/kW-month"),
        _prov_line("life", fin["life"], "{:g} yr"),
        f"rate {plot_text(rate.get('text', '—'))}",
    ])


# ── econ_tco ──────────────────────────────────────────────────────────────────

def _fmt(v, fmt: str) -> str:
    return "—" if v is None else fmt.format(v)


def _typed_tag(p: dict, text: str) -> str:
    """A value as the subtitle shows it: '1 yr typed (clamped)' when a typed
    box was outside its bounds and the bound was used instead."""
    if not p.get("typed"):
        return plot_text(text)
    return plot_text(text) + " typed" + (" (clamped)" if p.get("clamped") else "")


def _tick(row: dict) -> str:
    """The GPU's axis label with its capex basis underneath: the bars are not
    on one price basis, so each says which it is on."""
    b = _basis(row)
    if not b:
        return plot_text(row["spec"].label)
    color = "#999999" if row["selected"] else "#666666"
    return (f"{plot_text(row['spec'].label)}<br><span style='font-size:9px;color:{color}'>"
            f"{plot_text(T.CAPEX_BASES[b][0])}</span>")


def build_tco_stack(result: dict) -> go.Figure:
    rows = result["rows"]
    if not any(r["breakdown"] for r in rows):
        miss = ", ".join(result["selected"]["missing"]) or "inputs"
        return empty_figure(f"Nothing can be priced: {miss} unavailable")
    util = result["util_pct"]
    u = util / 100.0
    labels = [r["spec"].label for r in rows]
    ticktext = [_tick(r) for r in rows]
    opac = [1.0 if r["selected"] else MUTED_OPACITY for r in rows]
    fig = go.Figure()
    for key, name, color in COMPONENTS:
        xs, hov = [], []
        for r in rows:
            bd = r["breakdown"]
            if bd is None:
                xs.append(None)
                hov.append("")
                continue
            xs.append(bd[key] / u)
            hov.append(
                f"<b>{plot_text(r['spec'].label)}</b> · {name} {_usd(bd[key] / u)}/GPU-hr "
                f"at {util:g}% use ({_pct(bd['shares'][key])} of the total)<br>"
                f"total {_usd(bd['total_at_util'])}/GPU-hr at {util:g}% use, "
                f"{_usd(bd['total_100'])} at full use · CRF {bd['crf']:.4f}/yr<br>"
                + _inputs_hover(result, r))
        fig.add_trace(go.Bar(
            y=labels, x=xs, orientation="h", name=name, legendgroup=key,
            marker=dict(color=color, opacity=opac, line=dict(width=0)),
            customdata=hov, hovertemplate="%{customdata}<extra></extra>",
        ))
    annotations = []
    priced = [r["breakdown"]["total_at_util"] for r in rows if r["breakdown"]]
    xmax = max(priced) if priced else 1.0
    for r in rows:
        bd = r["breakdown"]
        if bd:
            annotations.append(dict(
                x=bd["total_at_util"], y=r["spec"].label, xref="x", yref="y",
                text=" " + _usd(bd["total_at_util"]), showarrow=False, xanchor="left",
                font=dict(color="#f2f2f2" if r["selected"] else "#777777", size=11,
                          family=FONT)))
        else:
            why = ("no public all-in price · type CAPEX" if r["missing"] == ["CAPEX"]
                   else f"{', '.join(r['missing'])} unavailable")
            annotations.append(dict(
                x=0, y=r["spec"].label, xref="x", yref="y", text=why, showarrow=False,
                xanchor="left", font=dict(color="#666666" if not r["selected"] else "#aaaaaa",
                                          size=10, family=FONT)))
    sel = result["selected"]
    fin, fac = result["financing"], result["facility"]
    if sel["breakdown"]:
        bd = sel["breakdown"]
        line1 = (f"<b>{plot_text(sel['spec'].label)}</b> {_usd(bd['total_at_util'])}/GPU-hr at "
                 f"{util:g}% use ({_usd(bd['total_100'])} at full use) · capital "
                 f"{_pct(bd['shares']['capital'])} · power {_pct(bd['shares']['power'])} · "
                 f"colo {_pct(bd['shares']['colo'])}")
    elif sel["missing"] == ["CAPEX"]:
        line1 = (f"<b>{plot_text(sel['spec'].label)}</b>: no public all-in system price; "
                 f"type CAPEX ($K per GPU) to price it")
    else:
        line1 = (f"<b>{plot_text(sel['spec'].label)}</b>: "
                 f"{plot_text(', '.join(sel['missing']))} unavailable")
    life_txt = _typed_tag(fin["life"], f"{fin['life']['value']:g} yr")
    pue, kwh, colo = fac["pue"], fac["usd_per_kwh"], fac["colo_usd_kw_month"]
    line2 = (f"{plot_text(fin['label'].split(' ')[0])} preset: {life_txt}, "
             f"{plot_text(fin['rate'].get('text', ''))}"
             f" · PUE {_typed_tag(pue, _fmt(pue['value'], '{:g}'))}"
             f" · {_typed_tag(kwh, _fmt(kwh['value'], '${:.4f}'))}/kWh"
             f" · colo {_typed_tag(colo, _fmt(colo['value'], '${:,.2f}'))}/kW-month")
    # A typed CAPEX or kW applies to the selected GPU only; say what was used.
    for name, key, fmt in (("capex", "capex_usd", "${:,.0f}"), ("kW", "it_kw", "{:.3f}")):
        p = sel["inputs"][key]
        if p.get("typed"):
            line2 += f" · {name} {_typed_tag(p, _fmt(p['value'], fmt))}"
    title = ("Owned Cost per GPU-hour  " + SUB + line1 + "<br>" + line2 + "</span>")
    _layout(fig, title, 520,
            barmode="stack", bargap=0.35,
            xaxis=_axis(title=dict(text=f"owned $ per GPU-hour at {util:g}% use",
                                   font=dict(color=AXIS, size=12), standoff=10),
                        range=[0, xmax * 1.18], tickprefix="$", showgrid=True),
            yaxis=_axis(autorange="reversed", showgrid=False, categoryorder="array",
                        categoryarray=labels, tickmode="array", tickvals=labels,
                        ticktext=ticktext),
            legend={**legend_below(), "yref": "container", "y": 0, "yanchor": "bottom",
                    "traceorder": "normal"},
            margin={**CHART_MARGIN, "l": 130, "t": 96, "b": 96},
            annotations=annotations, hovermode="closest")
    return fig


# ── econ_ladder ───────────────────────────────────────────────────────────────

# Label layout geometry. Plotly cannot avoid text collisions, so the ladder
# lays its labels out itself against a nominal plot: LABEL_PLOT_PX is narrower
# than the ~1,000-1,100 px plot a 1,280-1,366 px desktop gives (narrower is the
# safe side), and LABEL_CHAR_PX is a 10 px Inter advance rounded up.
LABEL_PLOT_PX = 900.0
LABEL_CHAR_PX = 6.0
LABEL_PAD_PX = 8.0
# A mark in the outer 12% of the axis is labelled toward the inside, so its
# label stays in the plot even on a phone, where the plot is ~200 px wide.
LABEL_EDGE = 0.12


def _extent(x: float, w: float, rightmost: bool = False) -> tuple[str, float, float]:
    """(side, x0, x1) for a label of width w at marker x: centred, or toward
    the inside of the plot near an edge, when centring would cross one, or for
    the chart's dearest mark (the one nearest the right edge at any width)."""
    if (rightmost or x + w / 2 > LABEL_PLOT_PX
            or x > LABEL_PLOT_PX * (1 - LABEL_EDGE)):
        return "left", x - w - 6, x
    if x - w / 2 < 0 or x < LABEL_PLOT_PX * LABEL_EDGE:
        return "right", x, x + w + 6
    return "center", x - w / 2, x + w / 2


def label_layout(df: pd.DataFrame, lo: float, hi: float) -> list[dict]:
    """Per mark: {text, pos, slot, x0, x1} (x in nominal px; text "" for a mark
    whose price is named in another mark's label). Marks at the same price in
    a rung (Lambda and Together at $3.99) share ONE label. Along a rung the
    price groups are taken cheapest first and put above or below the line,
    alternating, in the first slot where the label clears every label already
    there. When one fits in neither, it is folded into its nearer neighbour
    ("spot $5.27 · 12-month reserved $5.50" becomes one label) and the rung is
    laid out again; one label per rung always fits, so this ends. No two
    labels in a rung and slot ever overlap at the nominal geometry, and every
    price is named."""
    span = (hi - lo) or 1.0
    px = lambda v: (math.log10(v) - lo) / span * LABEL_PLOT_PX
    out = [{"text": "", "pos": "top center", "slot": "", "x0": 0.0, "x1": 0.0}
           for _ in range(len(df))]
    x_max = px(float(df["usd_per_gpu_hr"].astype(float).max())) if len(df) else 0.0
    for rung in dict.fromkeys(df["rung"]):
        groups: dict[float, list[int]] = {}
        for i in [i for i in range(len(df)) if df["rung"].iloc[i] == rung]:
            groups.setdefault(round(float(df["usd_per_gpu_hr"].iloc[i]), 2), []).append(i)
        items = []                       # (x px, first row index, label part)
        for v, idx in sorted(groups.items()):
            names = " · ".join(plot_text(df["label"].iloc[i]) for i in idx)
            items.append((px(max(float(df["usd_per_gpu_hr"].iloc[i]) for i in idx)), idx[0],
                          f"{names} {_usd(v)}"))
        clusters = [[it] for it in items]
        while True:
            placed, failed = _place_rung(clusters, x_max)
            if failed is None:
                break
            k = failed
            if k == 0:
                j = 1
            elif k == len(clusters) - 1:
                j = k - 1
            else:                        # fold into the nearer neighbour
                gap = lambda a, b: abs(clusters[a][0][0] - clusters[b][-1][0]) \
                    if a > b else abs(clusters[b][0][0] - clusters[a][-1][0])
                j = k - 1 if gap(k, k - 1) <= gap(k, k + 1) else k + 1
            a, b = min(j, k), max(j, k)
            clusters[a:b + 1] = [clusters[a] + clusters[b]]
        for q in placed:
            out[q["anchor"]] = {"text": q["text"], "pos": f"{q['slot']} {q['side']}",
                                "slot": q["slot"], "x0": q["x0"], "x1": q["x1"]}
    return out


def _place_rung(clusters: list[list[tuple]], x_max: float) -> tuple[list[dict], int | None]:
    """Greedy alternate-slot placement of one rung's clusters, cheapest first.
    Returns (placed labels, None) or (partial, index of the first cluster that
    fits in neither slot)."""
    placed: list[dict] = []
    for k, cl in enumerate(clusters):
        text = " · ".join(part for _x, _i, part in cl)
        mean = sum(x for x, _i, _p in cl) / len(cl)
        x, anchor, _p = min(cl, key=lambda it: abs(it[0] - mean))
        side, x0, x1 = _extent(x, len(text) * LABEL_CHAR_PX, x >= x_max - 1e-9)
        last = placed[-1]["slot"] if placed else "bottom"
        for slot in (("bottom", "top") if last == "top" else ("top", "bottom")):
            if all(q["slot"] != slot or x1 + LABEL_PAD_PX <= q["x0"]
                   or q["x1"] + LABEL_PAD_PX <= x0 for q in placed):
                placed.append({"anchor": anchor, "text": text, "slot": slot, "side": side,
                               "x0": x0, "x1": x1})
                break
        else:
            return placed, k
    return placed, None


def _labels(df: pd.DataFrame, lo: float, hi: float) -> tuple[list[str], list[str]]:
    """(text, textposition) per mark, from label_layout."""
    lay = label_layout(df, lo, hi)
    return [x["text"] for x in lay], [x["pos"] for x in lay]


def build_spread_ladder(result: dict, lad: pd.DataFrame) -> go.Figure:
    spec = result["selected"]["spec"]
    if lad is None or lad.empty:
        return empty_figure(f"No rental marks for {spec.label}")
    rung_labels = [lab for key, lab in T.LADDER_RUNGS if (lad["rung"] == key).any()]
    df = lad.reset_index(drop=True).copy()
    vals = df["usd_per_gpu_hr"].astype(float)
    lo, hi = math.log10(vals.min() * 0.75), math.log10(vals.max() * 1.35)
    df["_text"], df["_pos"] = _labels(df, lo, hi)
    fig = go.Figure()
    for i, r in df.iterrows():
        v = float(r["usd_per_gpu_hr"])
        hover = (f"<b>{plot_text(r['series_name'])}</b><br>{_usd(v)} per GPU-hour<br>"
                 f"{plot_text(r['basis'])}<br>"
                 f"{plot_text(r['period'])} · {plot_text(r['status'])}"
                 + (f" · as of {plot_text(r['as_of'])}" if r["as_of"] != r["period"] else "")
                 + f"<br>{plot_text(r['source_url'])}")
        fig.add_trace(go.Scatter(
            x=[v], y=[r["rung_label"]], mode="markers+text",
            name=r["series_name"], showlegend=False,
            text=[r["_text"]], textposition=r["_pos"],
            textfont=dict(color="#bbbbbb", size=10, family=FONT),
            marker=dict(color=RUNG_COLORS.get(r["rung"], "#cccccc"),
                        size=12 if r["rung"] == "owned" else 10,
                        symbol="diamond" if r["rung"] == "owned" else "circle",
                        line=dict(width=0.8, color=BG)),
            customdata=[[hover, r["source_url"]]],
            hovertemplate="%{customdata[0]}<extra></extra>",
            meta=r["rung"],
            # Plotly.react never draws a label on a point whose text was ""
            # in the previous figure (reproduced in the browser: CoreWeave's
            # label vanished after a GPU round trip). A point id that names
            # the label makes any change a new point, drawn fresh. (uid would
            # not do: plotly.py drops it from to_json.)
            ids=[f"{spec.key}:{i}:{r['_text']}:{r['_pos']}"],
        ))
    tickvals, ticktext = log_ticks(10 ** lo, 10 ** hi, fmt=lambda v: f"${v:g}")
    shapes, notes = [], []
    owned = df[df["rung"] == "owned"]
    hyp = df[df["rung"] == "hyperscaler"]
    sub2 = ""
    if not owned.empty:
        o = float(owned.iloc[0]["usd_per_gpu_hr"])
        shapes.append(dict(type="line", x0=o, x1=o, xref="x", y0=0, y1=1, yref="paper",
                           line=dict(color=ACCENT, width=1, dash="dash"), layer="below"))
        if not hyp.empty:
            m = float(hyp["usd_per_gpu_hr"].max()) / o
            sub2 = f"the dearest hyperscaler on-demand list price is {m:.1f}× the owned cost"
    else:
        sub2 = f"{plot_text(spec.label)} has no owned cost until CAPEX is typed"
    u = result["util_pct"]
    bits = [sub2] if sub2 else []
    if u < 100:
        bits.append(f"at {u:g}% use each mark ÷ {u / 100:g} is its cost per busy hour, "
                    "so the ratios hold")
    title = (f"Rental Spread · {plot_text(spec.label)}  " + SUB
             + "$ per paid GPU-hour, busy or idle (log) · owned and rented on one basis<br>"
             + " · ".join(bits) + "</span>")
    _layout(fig, title, 120 + 62 * len(rung_labels) + 70,
            xaxis=_axis(type="log", range=[lo, hi], tickmode="array", tickvals=tickvals,
                        ticktext=ticktext, showgrid=True,
                        title=dict(text="$ per paid GPU-hour (log)",
                                   font=dict(color=AXIS, size=12),
                                   standoff=10)),
            yaxis=_axis(categoryorder="array", categoryarray=rung_labels, autorange="reversed",
                        showgrid=False),
            margin={**CHART_MARGIN, "l": 150, "t": 96, "b": 60},
            shapes=shapes, annotations=notes, hovermode="closest", showlegend=False)
    return fig


# ── econ_rental_series ────────────────────────────────────────────────────────

def _weekly_grid(dates: list[str]) -> list[str]:
    """Every week from the first to the last listed date, so a missing week is
    a None (a gap), never a straight line drawn across it."""
    ds = sorted({date.fromisoformat(d[:10]) for d in dates})
    if not ds:
        return []
    out, d = [], ds[0]
    while d <= ds[-1]:
        out.append(d.isoformat())
        d += timedelta(days=7)
    extra = [x.isoformat() for x in ds if x.isoformat() not in set(out)]
    return sorted(set(out) | set(extra))


def build_rental_series(result: dict, weekly: pd.DataFrame) -> go.Figure:
    spec = result["selected"]["spec"]
    label = T.gd_label(spec)
    if weekly is None or weekly.empty:
        return empty_figure(f"GetDeploying lists no weekly prices for {label}")
    grid = _weekly_grid(weekly["date"].tolist())
    fig = go.Figure()
    ymax = 0.0
    for s in T.SERIES:
        part = weekly[weekly["series"] == s]
        if part.empty:
            continue
        by = part.set_index("date")
        ys = [float(by.at[d, "median_usd"]) if d in by.index and pd.notna(by.at[d, "median_usd"])
              else None for d in grid]
        ns = [T.offerings_phrase(by.at[d, "offering_count"] if "offering_count" in by.columns
                                 else None, by.at[d, "provider_count"])
              if d in by.index else None for d in grid]
        ymax = max([ymax] + [y for y in ys if y is not None])
        color, dash = SERIES_STYLE[s]
        fig.add_trace(go.Scatter(
            x=grid, y=ys, mode="lines+markers", name=T.SERIES_LABELS[s], meta=s,
            connectgaps=False, line=dict(color=color, width=1.8, dash=dash),
            marker=dict(color=color, size=3),
            customdata=[n if n is not None else "" for n in ns],
            hovertemplate=(f"{T.SERIES_LABELS[s]} %{{y:$.2f}}/GPU-hr · week of "
                           "%{x|%Y-%m-%d} · %{customdata}<extra></extra>"),
        ))
    shapes = []
    bd = result["selected"]["breakdown"]
    if bd:
        o = bd["total_100"]          # per paid hour, the basis of every median here
        ymax = max(ymax, o)
        shapes.append(dict(type="line", xref="paper", x0=0, x1=1, y0=o, y1=o, yref="y",
                           line=dict(color=ACCENT, width=1.4, dash="dot")))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", showlegend=True,
            name=f"owned {_usd(o)} per GPU-hour owned, busy or idle, today's assumptions",
            line=dict(color=ACCENT, width=1.4, dash="dot"), hoverinfo="skip"))
    n_weeks = len(grid)
    title = (f"Weekly Rental Medians · {plot_text(label)}  " + SUB
             + f"{n_weeks} weeks to {grid[-1]} · each billing series is its own line, "
               "never averaged together; a gap is a week with no listing</span>")
    _layout(fig, title, 470,
            xaxis=_axis(type="date", showgrid=False),
            yaxis=_axis(range=[0, ymax * 1.12], tickprefix="$", showgrid=True,
                        title=dict(text="median $ per GPU-hour", font=dict(color=AXIS, size=12),
                                   standoff=10)),
            legend={**legend_below(), "yref": "container", "y": 0, "yanchor": "bottom"},
            margin={**CHART_MARGIN, "t": 80, "b": 110},
            shapes=shapes, hovermode="x unified",
            annotations=[dict(x=0, y=1.0, xref="paper", yref="paper", yanchor="bottom",
                              xanchor="left", showarrow=False, text=GD_FOOTER,
                              font=dict(color="#666666", size=10, family=FONT))])
    return fig


def build_econ_compute(result: dict) -> dict[str, go.Figure]:
    """{figure id: figure} for the three Compute charts."""
    return {
        "econ_tco": build_tco_stack(result),
        "econ_ladder": build_spread_ladder(result, T.ladder_rows(result)),
        "econ_rental_series": build_rental_series(result, T.weekly_for(result["gpu"])),
    }
