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


def _fmt_q(v) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}"


# ── 1. Score gap over time ────────────────────────────────────────────────────

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

    shapes, annotations = [], []
    for d in seg.get("breaks", []):
        shapes.append(dict(type="line", xref="x", yref="paper", x0=d, x1=d, y0=0, y1=1,
                           line=dict(color=RULE, width=1, dash="dot")))
        annotations.append(dict(x=d, y=1.0, xref="x", yref="paper", text="AA rescale",
                                showarrow=False, xanchor="left", yanchor="top",
                                textangle=-90, xshift=2,
                                font=dict(color="#777777", size=10, family=FONT)))

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


def build_release_lag(recs: pd.DataFrame, summary: dict) -> go.Figure:
    """Two record staircases by release date and the lag between them.

    Closed records (#f2f2f2) and open records (accent), both filled. From each
    closed record a dashed connector runs at its score to the release of the
    first open model to reach it; a record nothing open has reached yet runs
    dotted to the snapshot date and is counted as censored, not as a lag.
    """
    rec_c, rec_o = summary.get("closed_records"), summary.get("open_records")
    if rec_c is None or getattr(rec_c, "empty", True):
        return empty_figure("No dated closed-weights models in this snapshot")
    asof = pd.Timestamp(summary["asof"])
    fig = go.Figure()

    for side, color, name in ((rec_c, CLOSED, "Closed-weights record"),
                              (rec_o, ACCENT, "Open-weights record")):
        if side is None or side.empty:
            continue
        sx, sy = _staircase(side, asof)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=color, width=1.4,
                                 shape="hv"), name=name, legendgroup=name,
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[d.date().isoformat() for d in side["d"]], y=side["quality"].astype(float),
            mode="markers", name=name, legendgroup=name,
            marker=dict(color=color, size=8, line=dict(width=0.8, color=BG)),
            customdata=list(zip(side["model"].map(plot_text), side["provider"].map(plot_text))),
            hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                           "Released %{x}<br>AA index %{y:.1f}<br>"
                           f"New {name.split('-')[0].lower()}-weights high when released"
                           "<extra></extra>"),
        ))

    lag_shown = cens_shown = False
    for _, r in recs.iterrows():
        d0 = pd.Timestamp(r["closed_date"]).date().isoformat()
        if bool(r["censored"]):
            end, dash, label = asof.date().isoformat(), "dot", "Not matched yet"
            hover = (f"<b>{plot_text(r['closed_model'])}</b><br>No open model scores ≥ "
                     f"{r['closed_q']:.1f} yet: lag ≥ {r['lag_months']:.1f} months")
            show = not cens_shown
            cens_shown = True
        else:
            end, dash, label = pd.Timestamp(r["open_date"]).date().isoformat(), "dash", \
                "Lag to first open model ≥ that score"
            hover = (f"<b>{plot_text(r['closed_model'])}</b> ({r['closed_q']:.1f})<br>"
                     f"matched by {plot_text(r['open_model'])} ({r['open_q']:.1f}) "
                     f"on {end}<br>lag {r['lag_months']:.1f} months")
            show = not lag_shown
            lag_shown = True
        fig.add_trace(go.Scatter(
            x=[d0, end], y=[r["closed_q"], r["closed_q"]], mode="lines",
            line=dict(color="rgba(242,242,242,0.45)", width=1, dash=dash),
            name=label, legendgroup=label, showlegend=show,
            hovertemplate=hover + "<extra></extra>",
        ))

    med = summary.get("median_lag_months")
    note = (f"median lag {med:.1f} mo  ·  {summary['matched']} matched"
            if med is not None else f"{summary['matched']} matched")
    note += f"  ·  {summary['censored']} not yet matched"
    title = (
        "Open vs. Closed: Release Lag" + "  " + SUB
        + note + f"  ·  models still listed by AA on {summary['asof']} only"
        + "</span>"
    )
    _layout(fig, title, "Release date", "AA Intelligence Index (current scale)")
    return fig


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
    title = (
        "Open Weights at Equal Score: Price Ratio" + "  " + SUB
        + f"{summary['n']} open families"
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
    fig.update_layout(legend={**legend_below(), "yref": "container", "y": 0,
                              "yanchor": "bottom"})
    return fig
