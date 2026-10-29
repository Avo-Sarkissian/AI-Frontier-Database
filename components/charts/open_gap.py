"""Economics › Labs: the open-weights vs closed-weights gap, three ways.

  build_gap_history  — score gap per daily snapshot, broken at AA rescales;
  build_release_lag  — closed-weights records and how long open weights took
                       to match each (from release dates);
  build_iso_price    — per open family, the cheapest closed family scoring at
                       least as well, as a price ratio.

The data comes from data/open_gap.py. Inside the Economics tab fill strength
means strength of evidence, so these charts never use hollow-vs-filled for
open vs closed: the lag chart separates them by colour (closed #f2f2f2, open
accent) with filled dots for both, and the iso-price strip is all open models,
filled, coloured by provider.
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
CLOSED = "#f2f2f2"
RULE = "rgba(255,255,255,0.15)"
SUB = "<span style='font-size:12px;color:#777777;font-weight:400'>"
_ZERO = "rgba(255,255,255,0.06)"


def _layout(fig: go.Figure, title: str, x_title: str, y_title: str, **extra) -> None:
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(text=title, font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
                   x=0.0, xanchor="left", pad=dict(l=20, t=16)),
        xaxis=dict(title=dict(text=x_title, font=dict(color=AXIS, size=12), standoff=12),
                   gridcolor=GRID, zerolinecolor=_ZERO, tickfont=dict(color=TICK, size=11,
                   family=FONT), showgrid=True, showline=False, ticks="",
                   automargin=True),
        yaxis=dict(title=dict(text=y_title, font=dict(color=AXIS, size=12), standoff=12),
                   gridcolor=GRID, zerolinecolor=_ZERO, tickfont=dict(color=TICK, size=11,
                   family=FONT), showgrid=True, showline=False, ticks="",
                   automargin=True),
        legend=legend_below(),
        margin=CHART_MARGIN,
        hovermode="closest",
        hoverlabel=dict(bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
                        font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1),
        **extra,
    )


NARROW_PX = 500        # docs/app.js applies layout.meta.narrow below this width


def _narrow(layout: dict, data: dict | None = None) -> dict:
    """layout.meta for docs/app.js narrowFigure: the patch for a phone."""
    return {"narrow": {"max_width": NARROW_PX,
                       "data": {str(i): u for i, u in (data or {}).items()},
                       "layout": layout}}


def _fmt_q(v) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}"


# ── 1. Score gap over time ────────────────────────────────────────────────────

RESCALE_MERGE_DAYS = 14        # rescales closer than this share one label
RESCALE_EDGE = 0.15            # a label this near the right edge points left


def rescale_labels(breaks, first, last, min_gap_frac: float = 0.06) -> list[dict]:
    """One small horizontal label above the plot per cluster of rescales.

    They used to be vertical "AA rescale" strings inside the plot, one per
    rule: two rescales three days apart printed side by side, and the v5 one
    on the newest snapshot sat on the right edge beside the newest point.
    Now rescales within RESCALE_MERGE_DAYS share one label ("AA rescales"),
    labels sit in the top margin where no data is, and one near the right
    edge reads leftward from its rule. `min_gap_frac` also merges rescales
    closer than that share of the x span (a label is ~60 px: 0.06 of a desktop
    plot, a quarter of a phone's)."""
    ds = sorted(pd.Timestamp(b) for b in breaks)
    if not ds:
        return []
    lo, hi = pd.Timestamp(first), pd.Timestamp(last)
    span = max((hi - lo).days, 1)
    groups: list[list[pd.Timestamp]] = []
    for d in ds:
        gap = (d - groups[-1][-1]).days if groups else None
        if gap is not None and (gap < RESCALE_MERGE_DAYS or gap / span < min_gap_frac):
            groups[-1].append(d)
        else:
            groups.append([d])
    out = []
    for g in groups:
        right = (g[-1] - lo).days / span > 1 - RESCALE_EDGE
        at = g[-1] if right else g[0]
        out.append(dict(
            x=at.strftime("%Y-%m-%d"), y=1.0, xref="x", yref="paper",
            text="AA rescale" + ("s" if len(g) > 1 else ""), showarrow=False,
            xanchor="right" if right else "left", yanchor="bottom",
            xshift=-3 if right else 3, yshift=2,
            font=dict(color="#777777", size=10, family=FONT)))
    return out

def build_gap_history(gh: pd.DataFrame, seg: dict) -> go.Figure:
    """One accent line of (best closed − best open) per kept snapshot.

    The y array holds None at every segment boundary, so no line segment ever
    joins two index scales; a dotted rule labelled "AA rescale" marks each
    break. `seg` is data/open_gap.index_segments' result.
    """
    data = gh[gh["gap"].notna()].sort_values("date")
    if data.empty:
        return empty_figure("No snapshot has both an open and a closed model")

    xs, ys, cds = [], [], []
    prev_seg = None
    for _, r in data.iterrows():
        if prev_seg is not None and r["segment"] != prev_seg:
            xs.append(None)
            ys.append(None)
            cds.append(None)
        prev_seg = r["segment"]
        xs.append(r["date"])
        ys.append(round(float(r["gap"]), 2))
        cds.append([
            plot_text(r["closed_model"]), _fmt_q(r["closed_q"]),
            plot_text(r["open_model"]), _fmt_q(r["open_q"]),
            f"{r['gap_pct']:.1f}%", int(r["segment"]) + 1,
            int(r["n_open"]), int(r["n_closed"]),
            int(r["src_aa"]), int(r["src_catalogue"]), int(r["src_inferred"]),
        ])

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name="Score gap",
        line=dict(color=ACCENT, width=1.6),
        marker=dict(color=ACCENT, size=4),
        connectgaps=False,
        customdata=cds,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Gap: %{y:.1f} pts  (%{customdata[4]} of the leader)<br>"
            "Best closed: %{customdata[0]}  ·  %{customdata[1]}<br>"
            "Best open: %{customdata[2]}  ·  %{customdata[3]}<br>"
            "Index version %{customdata[5]} (scale changes at each rescale)<br>"
            "%{customdata[6]} open · %{customdata[7]} closed models; weights from "
            "AA flag %{customdata[8]} · name match %{customdata[9]} · "
            "inferred closed %{customdata[10]}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    shapes = [dict(type="line", xref="x", yref="paper", x0=d, x1=d, y0=0, y1=1,
                   line=dict(color=RULE, width=1, dash="dot"))
              for d in seg.get("breaks", [])]
    annotations = rescale_labels(seg.get("breaks", []), data["date"].min(),
                                 data["date"].max())

    last = data.iloc[-1]
    excl = seg.get("excluded", [])
    title = (
        "Open vs. Closed: Score Gap" + "  " + SUB
        + f"best closed − best open, AA Intelligence Index, {len(data)} daily snapshots"
        + f"  ·  latest {last['gap']:.1f} pts ({last['gap_pct']:.0f}%)"
        + "  ·  line breaks at AA rescales"
        + (f"  ·  {len(excl)} glitch snapshot{'s' if len(excl) != 1 else ''} excluded"
           if excl else "")
        + "</span>"
    )
    _layout(fig, title, "Snapshot date",
            "Gap  (index points, within one index version)",
            shapes=shapes, annotations=annotations)
    fig.update_layout(margin={**CHART_MARGIN, "t": 64}, meta=_narrow({
        "annotations": rescale_labels(seg.get("breaks", []), data["date"].min(),
                                      data["date"].max(), min_gap_frac=0.25)}))
    fig.update_yaxes(rangemode="tozero")
    return fig


# ── 2. Release lag ────────────────────────────────────────────────────────────

def _staircase(recs: pd.DataFrame, asof: pd.Timestamp):
    """hv step x/y through the records, held flat to `asof`."""
    if recs is None or recs.empty:
        return [], []
    xs = [d.date().isoformat() for d in recs["d"]]
    ys = [float(q) for q in recs["quality"]]
    if pd.Timestamp(xs[-1]) < asof:
        xs.append(asof.date().isoformat())
        ys.append(ys[-1])
    return xs, ys


def _iso(d) -> str:
    return pd.Timestamp(d).date().isoformat()


def _dated_note(model, date_, release) -> str:
    """Hover line for a model dated by its score's first snapshot."""
    if release is None or pd.isna(release) or pd.Timestamp(release) == pd.Timestamp(date_):
        return ""
    return (f"<br>dated {_iso(date_)}: first AA snapshot with this score; AA's "
            f"release date {_iso(release)} is the base model's")


def build_release_lag(recs: pd.DataFrame, summary: dict) -> go.Figure:
    """Two record staircases by date and the lag between them.

    Closed records (#f2f2f2) and open records (accent), both filled. From each
    counted closed record an ACCENT connector runs at its score to the date of
    the first open model to reach it (dashed, ending in a tick, labelled in
    months); one nothing open has reached yet runs dotted to the snapshot date
    and is censored, not a lag. The connectors are accent and drawn last
    because each lies along the closed staircase's own step: in #f2f2f2 they
    were invisible. A record the open side had already passed (lag 0) has no
    connector; its marker's hover says so.
    """
    rec_c, rec_o = summary.get("closed_records"), summary.get("open_records")
    if rec_c is None or getattr(rec_c, "empty", True):
        return empty_figure("No dated closed-weights models in this snapshot")
    asof = pd.Timestamp(summary["asof"])
    fig = go.Figure()

    lag_of = {}
    for _, r in recs.iterrows():
        if bool(r["censored"]):
            lag_of[r["closed_model"]] = (f"no open model scores ≥ {r['closed_q']:.1f} yet: "
                                         f"lag ≥ {r['lag_months']:.1f} months")
        elif r["lag_months"] <= 0:
            lag_of[r["closed_model"]] = (f"already matched: {plot_text(r['open_model'])} "
                                         f"({r['open_q']:.1f}) was out on {_iso(r['open_date'])}")
        else:
            lag_of[r["closed_model"]] = (f"matched by {plot_text(r['open_model'])} "
                                         f"({r['open_q']:.1f}) on {_iso(r['open_date'])}: "
                                         f"lag {r['lag_months']:.1f} months")
    uncounted = set(summary.get("uncounted") or [])

    for side, color, name, is_closed in ((rec_c, CLOSED, "Closed-weights record", True),
                                         (rec_o, ACCENT, "Open-weights record", False)):
        if side is None or side.empty:
            continue
        sx, sy = _staircase(side, asof)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=color, width=1.4,
                                 shape="hv"), name=name, legendgroup=name,
                                 hoverinfo="skip", showlegend=False))
        notes = []
        for _, r in side.iterrows():
            note = _dated_note(r["model"], r["d"], r.get("release"))
            if is_closed:
                m = str(r["model"])
                note += "<br>" + (lag_of.get(m) if m not in uncounted else
                                  "oldest closed model AA still lists: a record only "
                                  "because stronger ones were dropped, so not counted")
            notes.append(note)
        fig.add_trace(go.Scatter(
            x=[_iso(d) for d in side["d"]], y=side["quality"].astype(float),
            mode="markers", name=name, legendgroup=name,
            marker=dict(color=color, size=8, line=dict(width=0.8, color=BG)),
            customdata=list(zip(side["model"].map(plot_text), side["provider"].map(plot_text),
                                notes)),
            hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                           "%{x}  ·  AA index %{y:.1f}<br>"
                           f"New {name.split('-')[0].lower()}-weights high"
                           "%{customdata[2]}<extra></extra>"),
        ))

    lag_shown = cens_shown = False
    labels = []
    for _, r in recs.iterrows():
        d0 = _iso(r["closed_date"])
        if bool(r["censored"]):
            end, dash, label = asof.date().isoformat(), "dot", "Not matched yet (≥ months)"
            text = f"≥{r['lag_months']:.1f} mo"
            show = not cens_shown
            cens_shown = True
        elif r["lag_months"] > 0:
            end, dash, label = _iso(r["open_date"]), "dash", \
                "Lag to first open model ≥ that score"
            text = f"{r['lag_months']:.1f} mo"
            show = not lag_shown
            lag_shown = True
        else:
            continue
        fig.add_trace(go.Scatter(
            x=[d0, end], y=[r["closed_q"], r["closed_q"]], mode="lines+markers",
            line=dict(color=ACCENT, width=1.6, dash=dash),
            marker=dict(symbol="line-ns", size=[0, 9], color=ACCENT,
                        line=dict(color=ACCENT, width=1.6)),
            name=label, legendgroup=label, showlegend=show,
            hovertemplate=(f"<b>{plot_text(r['closed_model'])}</b> ({r['closed_q']:.1f})<br>"
                           + lag_of[r["closed_model"]] + "<extra></extra>"),
        ))
        mid = pd.Timestamp(d0) + (pd.Timestamp(end) - pd.Timestamp(d0)) / 2
        labels.append(dict(x=mid.date().isoformat(), y=r["closed_q"], xref="x", yref="y",
                           text=text, showarrow=False, yanchor="bottom", yshift=2,
                           font=dict(color=ACCENT, size=10, family=FONT)))

    med = summary.get("median_lag_months")
    m_n, c_n = summary["matched"], summary["censored"]
    note = (f"median lag {med:.1f} mo (Kaplan–Meier: {m_n} matched + {c_n} not yet)"
            if med is not None else
            f"median not identifiable yet: {m_n} matched, {c_n} not yet matched")
    title = (
        "Open vs. Closed: Release Lag" + "  " + SUB
        + note + f"  ·  only models AA still lists on {summary['asof']}"
        + "</span>"
    )
    labels = _clear_lag_labels(labels, fig)
    _layout(fig, title, "Release date (or first snapshot with the score)",
            "AA Intelligence Index (current scale)", annotations=labels)
    # A phone has room for the lines, not their labels: the hover names each.
    fig.update_layout(meta=_narrow({"annotations": []}))
    return fig


# Nominal desktop plot for the lag labels' collision test (approximate on
# purpose; it only has to keep two labels and a label and a line apart).
_LAG_PLOT_W, _LAG_PLOT_H, _LAG_CHAR_PX, _LAG_LINE_PX = 1100.0, 300.0, 5.6, 12.0


def _clear_lag_labels(labels: list[dict], fig: go.Figure) -> list[dict]:
    """Drop a lag label that would print on another label, across another lag
    line, or past the ends of its own: "≥1.8 mo" sat on the "≥1.2 mo" line 1.4
    points above it, and three labels overlapped at the newest record. Longer
    texts go first; the hover keeps every value."""
    seq = lambda v: [] if v is None else list(v)
    xs = [pd.Timestamp(x) for t in fig.data for x in seq(t.x) if x is not None]
    ys = [float(y) for t in fig.data for y in seq(t.y) if y is not None]
    if not labels or not xs or not ys:
        return labels
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    sx = _LAG_PLOT_W / max((x1 - x0).days, 1)
    sy = _LAG_PLOT_H / max(y1 - y0, 1e-9)
    lines = []
    for t in fig.data:
        tx, ty = seq(t.x), seq(t.y)
        if t.mode == "lines+markers" and len(tx) == 2 and ty[0] == ty[1]:
            t = type("T", (), {"x": tx, "y": ty})
            lines.append(((pd.Timestamp(t.x[0]) - x0).days * sx,
                          (pd.Timestamp(t.x[1]) - x0).days * sx, (t.y[0] - y0) * sy))
    kept, boxes = [], []
    for a in sorted(labels, key=lambda a: -len(a["text"])):
        cx = (pd.Timestamp(a["x"]) - x0).days * sx
        by = (float(a["y"]) - y0) * sy + 2
        w = len(a["text"]) * _LAG_CHAR_PX
        box = (cx - w / 2, cx + w / 2, by, by + _LAG_LINE_PX)
        hit = any(box[0] < b[1] and b[0] < box[1] and box[2] < b[3] and b[2] < box[3]
                  for b in boxes)
        hit = hit or any(lx0 < box[1] and box[0] < lx1 and box[2] < ly < box[3]
                         for lx0, lx1, ly in lines)
        # A line shorter than its label ("≥0.0 mo" on a record set yesterday)
        # cannot carry it: the text would sit on the record's own dot.
        own = [lx1 - lx0 for lx0, lx1, ly in lines
               if abs(ly - (by - 2)) < 0.5 and lx0 <= cx <= lx1]
        hit = hit or (bool(own) and min(own) < w)
        if not hit:
            kept.append(a)
            boxes.append(box)
    order = {id(a): i for i, a in enumerate(labels)}
    return sorted(kept, key=lambda a: order[id(a)])


# ── 3. Iso-quality price ratio ────────────────────────────────────────────────

def _ratio_text(v: float) -> str:
    return f"{v:.2f}×" if v < 10 else f"{v:.0f}×"


def build_iso_price(ratios: pd.DataFrame, summary: dict) -> go.Figure:
    """Log-x strip: one filled dot per open family at (price ratio, its score).

    x = price of the cheapest closed family scoring at least as well ÷ the
    open family's price, both at the same workload mix. Right of the 1× rule
    the open model is cheaper; left of it a closed model that scores at least
    as well costs less. The median is marked; no headline is asserted.
    """
    if ratios is None or ratios.empty:
        return empty_figure("No open model has a closed peer at or above its score")
    df = ratios.rename(columns={"open_provider": "provider"}).copy()
    df, providers = spotlight_split(df)
    fig = go.Figure()

    def _price(v, filled):
        from data.workload_mix import mark_price
        return mark_price(v, filled)

    for prov in providers:
        part = df[df["display_provider"] == prov]
        cd = [[plot_text(r["open_model"]), plot_text(r["provider"]), f"{r['open_q']:.1f}",
               _price(r["open_price"], r["open_filled"]),
               plot_text(r["closed_model"]), plot_text(r["closed_provider"]),
               f"{r['closed_q']:.1f}", _price(r["closed_price"], r["closed_filled"]),
               _ratio_text(r["ratio"]),
               (f"{_ratio_text(r['task_ratio'])} per AA task" if pd.notna(r["task_ratio"])
                else "no AA cost per task for both")]
              for _, r in part.iterrows()]
        fig.add_trace(go.Scatter(
            x=part["ratio"], y=part["open_q"], mode="markers", name=prov,
            marker=dict(color=PROVIDER_COLORS.get(prov, DEFAULT_COLOR), size=9, opacity=0.9,
                        line=dict(width=0.8, color=BG)),
            customdata=cd,
            hovertemplate=(
                "<b>%{customdata[0]}</b>  (%{customdata[1]}, open)<br>"
                "AA index %{customdata[2]}  ·  %{customdata[3]}/M<br>"
                "Cheapest closed at ≥ that score: %{customdata[4]} (%{customdata[5]})<br>"
                "AA index %{customdata[6]}  ·  %{customdata[7]}/M<br>"
                "Closed ÷ open price: %{customdata[8]}  ·  %{customdata[9]}"
                "<extra></extra>"
            ),
        ))

    lo, hi = float(df["ratio"].min()), float(df["ratio"].max())
    lo, hi = min(lo, 1.0), max(hi, 1.0)
    tickvals, ticktext = log_ticks(lo, hi, fmt=lambda v: f"{v:g}×")
    med = summary.get("median")
    shapes = [dict(type="line", xref="x", yref="paper", x0=1, x1=1, y0=0, y1=1,
                   line=dict(color="rgba(255,255,255,0.35)", width=1))]
    annotations = [
        dict(x=0, y=1.0, xref="x", yref="paper", text="← a closed peer is cheaper ",
             showarrow=False, xanchor="right", yanchor="bottom",
             font=dict(color="#777777", size=10, family=FONT)),
        dict(x=0, y=1.0, xref="x", yref="paper", text=" open is cheaper →",
             showarrow=False, xanchor="left", yanchor="bottom",
             font=dict(color="#777777", size=10, family=FONT)),
    ]
    if med:
        shapes.append(dict(type="line", xref="x", yref="paper", x0=med, x1=med, y0=0, y1=1,
                           line=dict(color=ACCENT, width=1, dash="dot")))
        annotations.append(dict(x=math.log10(med), y=0.0, xref="x", yref="paper",
                                text=f"median {_ratio_text(med)}", showarrow=False,
                                xanchor="left", yanchor="bottom", xshift=3,
                                font=dict(color=ACCENT, size=10, family=FONT)))
    below = summary.get("share_below_1")
    basis = ("AA-measured scores only"
             + (f" ({summary['n_estimated_rows']} estimated rows left out)"
                if summary.get("n_estimated_rows") else "")
             if summary.get("measured_only") else "incl. AA-estimated scores")
    title = (
        "Open Weights at Equal Score: Price Ratio" + "  " + SUB
        + f"{summary['n']} open families, {basis}"
        + (f"  ·  {below:.0%} have a cheaper closed peer" if below is not None else "")
        + (f"  ·  {summary['n_excluded']} with no closed peer at their score"
           if summary.get("n_excluded") else "")
        + f"  ·  priced at {summary['mix_short']}"
        + "</span>"
    )
    _layout(fig,
            title,
            "Cheapest closed model scoring ≥ it ÷ open model's price  (log)"
            "<br>every point is an open-weights model  ·  AA list prices",
            "Open model's AA Intelligence Index",
            shapes=shapes, annotations=annotations)
    fig.update_xaxes(type="log", tickmode="array", tickvals=tickvals, ticktext=ticktext,
                     range=[math.log10(lo) - 0.12, math.log10(hi) + 0.12])
    narrow_anns = [dict(a, text=t) for a, t in zip(
        annotations, ["← closed cheaper ", " open cheaper →"])] + annotations[2:]
    fig.update_layout(legend={**legend_below(), "yref": "container", "y": 0,
                              "yanchor": "bottom"},
                      meta=_narrow({"annotations": narrow_anns,
                                    "xaxis": {"title": {"text": "closed peer ÷ open price (log)",
                                                        "font": {"color": AXIS, "size": 11}}}}))
    return fig
