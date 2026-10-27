"""Economics › Labs: four figures and an HTML ledger (data/lab_economics.py).

  build_labs_runrate      — Epoch AI's annualised revenue per lab, log scale
  build_labs_margins      — one dot row per (entity × definition), never mixed
  build_labs_commitments  — compute commitments ÷ revenue run-rate (×)
  build_labs_tokens       — disclosed token volumes, normalised to per day
  labs_ledger_html        — every curated row with its badge and source link

Inside Economics, fill strength is strength of evidence: disclosed = filled,
reported = open, estimated = open diamond (on the token chart, whose symbol
already encodes scope, estimated = open with a centre dot). Every chart carries
legend-only grey key traces that say so. Colours are the provider palette.
Deterministic: no clock is read here, every range comes from the data and
every sort is a stable mergesort, so the pre-rendered figure equals the live
one.
"""
from __future__ import annotations

import html
import math
import re

import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    plot_text, empty_figure, PROVIDER_COLORS, DEFAULT_COLOR, BG, FONT,
)
from components.charts.open_gap import _layout, SUB, RULE
from data import lab_economics as L

KEY_GREY = "#999999"
MARGIN_RANGE = (-120.0, 100.0)          # values are clipped to this
MARGIN_AXIS = (-134.0, 104.0)            # the axis leaves room for the edge marks
BADGE_TEXT = {
    "disclosed": "disclosed (filing or company on record)",
    "reported": "reported (press citing people or documents)",
    "estimated": "estimated (analyst model or Epoch interpolation)",
}
EPOCH_SUB = "Epoch AI, CC BY 4.0"


def _color(palette_name) -> str:
    return PROVIDER_COLORS.get(str(palette_name or ""), DEFAULT_COLOR)


def badge_symbol(badge: str, base: str = "circle") -> str:
    """disclosed = filled, reported = open, estimated = open diamond (or, when
    `base` already encodes something else, open with a centre dot)."""
    if badge == "disclosed":
        return base
    if badge == "reported":
        return f"{base}-open"
    return "diamond-open" if base == "circle" else f"{base}-open-dot"


def _badge_keys(badges=L.BADGES, base: str = "circle") -> list[go.Scatter]:
    return [go.Scatter(x=[None], y=[None], mode="markers", name=BADGE_TEXT[b],
                       marker=dict(symbol=badge_symbol(b, base), color=KEY_GREY, size=8,
                                   line=dict(color=KEY_GREY, width=1.5)),
                       legendgroup="badge", showlegend=True, hoverinfo="skip")
            for b in badges]


def _usd_tick(v: float) -> str:
    for base, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if v >= base:
            return f"${v / base:g}{suf}"
    return f"${v:,.0f}"


def _decades(lo: float, hi: float) -> list[float]:
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    return [10.0 ** k for k in range(a, b + 1)]


def _esc(v) -> str:
    return html.escape("" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v),
                       quote=True)


# ── 1. Revenue run-rate ─────────────────────────────────────────────────────

RUNRATE_FOOTNOTE = (
    "Not like for like. Anthropic's run-rate = trailing-28-day consumption ×13 + "
    "subscriptions ×12, and it books cloud-marketplace sales gross<br>"
    "(47% of FY2025, 48% of H1 2026 revenue; S-1). OpenAI's ~$50B (FT, Sep 2026) "
    "excludes cloud partners' resale, per Epoch's note."
)


def build_labs_runrate(rr: pd.DataFrame | None = None) -> go.Figure:
    rr = L.runrate_frame() if rr is None else rr
    if rr is None or rr.empty:
        return empty_figure("No Epoch AI revenue rows available")
    fig = go.Figure()
    companies = list(dict.fromkeys(rr.sort_values(["company"], kind="mergesort")["company"]))
    for co in companies:
        d = rr[rr["company"] == co].sort_values(["date", "epoch_id"], kind="mergesort")
        col = _color(d["palette_name"].iloc[0])
        cds = [[plot_text(co), L._usd(v), plot_text(t or "—"), plot_text(c or "—"),
                plot_text(s or "—"), plot_text(i), L.source_domain(u) or "—", b]
               for v, t, c, s, i, u, b in zip(d["annualized_usd"], d["revenue_type"],
                                              d["confidence"], d["source_type"],
                                              d["epoch_id"], d["source_url"], d["badge"])]
        fig.add_trace(go.Scatter(
            x=list(d["date"]), y=[float(v) for v in d["annualized_usd"]],
            mode="lines+markers", name=plot_text(co), legendgroup=co,
            line=dict(color=col, width=1.3),
            marker=dict(color=col, size=7, symbol=[badge_symbol(b) for b in d["badge"]],
                        line=dict(color=col, width=1.5)),
            customdata=cds, meta={"company": co},
            hovertemplate=("<b>%{customdata[0]}</b>  %{x}<br>%{customdata[1]}/yr  "
                           "(%{customdata[2]})<br>Badge: %{customdata[7]} · Epoch "
                           "confidence: %{customdata[3]}<br>Source type: %{customdata[4]} "
                           "· %{customdata[6]}<br>Epoch row: %{customdata[5]}<extra></extra>"),
        ))
    for k in _badge_keys():
        fig.add_trace(k)
    lo, hi = float(rr["annualized_usd"].min()), float(rr["annualized_usd"].max())
    ticks = _decades(lo, hi)
    title = ("Lab Revenue Run-Rate" + "  " + SUB
             + f"annualised revenue, USD/yr (log) · {len(rr)} reports, "
               f"{len(companies)} companies · {EPOCH_SUB}</span>")
    _layout(fig, title, "Date of the figure", "Annualised revenue (USD/yr, log)",
            height=520,
            annotations=[dict(x=0.0, y=1.0, xref="paper", yref="paper", xanchor="left",
                              yanchor="top", align="left", showarrow=False,
                              text=RUNRATE_FOOTNOTE,
                              font=dict(color="#888888", size=11, family=FONT))])
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[_usd_tick(t) for t in ticks],
                     range=[math.log10(lo) - 0.15, math.log10(hi) + 0.9])
    return fig


# ── 2. Margins ──────────────────────────────────────────────────────────────

def _period_labels(xs: list[float], periods: list[str], gap: float = 4.0):
    """Text per point: points within `gap` pp of each other share one label
    (on the first), and neighbouring label groups alternate above/below."""
    order = sorted(range(len(xs)), key=lambda i: (xs[i], i))
    groups: list[list[int]] = []
    for i in order:
        if groups and abs(xs[i] - xs[groups[-1][-1]]) <= gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    text = [""] * len(xs)
    pos = ["top center"] * len(xs)
    prev_x, prev_top = None, False
    for g in groups:
        x0 = xs[g[0]]
        top = not (prev_top and prev_x is not None and abs(x0 - prev_x) < 22)
        lead = min(g)
        text[lead] = " · ".join(periods[i] for i in sorted(g))
        for i in g:
            pos[i] = "top center" if top else "bottom center"
        prev_x, prev_top = xs[g[-1]], top
    return text, pos


def build_labs_margins(rows: pd.DataFrame | None = None) -> go.Figure:
    rows = L.margin_rows() if rows is None else rows
    if rows is None or rows.empty:
        return empty_figure("No margin rows in the ledger")
    lo, hi = MARGIN_RANGE
    labels = list(dict.fromkeys(rows.sort_values("row_order", kind="mergesort")["label"]))
    fig = go.Figure()
    notes = []
    for lab in labels:
        d = rows[rows["label"] == lab]
        col = _color(d["palette_name"].iloc[0])
        vals = [float(v) for v in d["value"]]
        xs = [min(max(v, lo), hi) for v in vals]
        off = [v < lo or v > hi for v in vals]
        periods = [str(p) for p in d["period_label"]]
        text, pos = _period_labels([x for x in xs], periods)
        for i, o in enumerate(off):
            if o:
                text[i] = ""
                notes.append(dict(
                    x=xs[i], y=lab, xref="x", yref="y", xanchor="left", yanchor="bottom",
                    showarrow=False, yshift=6,
                    text=f"← {periods[i]} {vals[i]:.0f}% (off scale)".replace("-", "−"),
                    font=dict(color=col, size=10, family=FONT)))
        syms = [("triangle-left" if o else badge_symbol(b)) for o, b in zip(off, d["badge"])]
        q = [{">": ">", "~": "~"}.get(str(v), "") for v in d["value_qualifier"]]
        nd = []
        for n, den, u in zip(d["numerator"], d["denominator"], d["nd_unit"]):
            nd.append(f"{n:,.0f} ÷ {den:,.0f} {u}" if (n is not None and den is not None
                                                       and not pd.isna(n) and not pd.isna(den))
                      else "not filed")
        cds = [[plot_text(lab), p, f"{qq}{v:.1f}%", plot_text(df_), b, fs, ndx,
                L.source_domain(u) or "—", plot_text(sn)]
               for p, qq, v, df_, b, fs, ndx, u, sn in zip(
                   periods, q, vals, d["definition"], d["badge"], d["filing_status"], nd,
                   d["source_url"], d["source_name"])]
        entity = str(d["entity"].iloc[0])
        definition = str(d["definition"].iloc[0])
        fig.add_trace(go.Scatter(
            x=xs, y=[lab] * len(xs), mode="lines+markers+text", name=plot_text(lab),
            showlegend=False, cliponaxis=False,
            line=dict(color=col, width=1), opacity=1.0,
            marker=dict(color=col, size=9, symbol=syms, line=dict(color=col, width=1.6)),
            text=text, textposition=pos, textfont=dict(color="#aaaaaa", size=10, family=FONT),
            customdata=cds,
            meta={"entity": entity, "definition": definition, "label": lab},
            hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                           "%{customdata[2]}  (%{customdata[4]}, %{customdata[5]})<br>"
                           "%{customdata[3]}<br>Inputs: %{customdata[6]}<br>"
                           "%{customdata[8]} · %{customdata[7]}<extra></extra>"),
        ))
    for k in _badge_keys():
        fig.add_trace(k)
    title = ("Lab Margins, One Definition per Row" + "  " + SUB
             + "rows are not comparable to each other · filed gross margins, a derived "
               "S-1 ratio, press and analyst figures · x clipped to −120…100%</span>")
    _layout(fig, title, "Margin (% of revenue)", "",
            height=150 + 52 * len(labels),
            shapes=[dict(type="line", xref="x", yref="paper", x0=0, x1=0, y0=0, y1=1,
                         line=dict(color=RULE, width=1))],
            annotations=notes)
    ticks = [-100, -50, 0, 50, 100]
    fig.update_xaxes(range=list(MARGIN_AXIS), zeroline=False, tickvals=ticks,
                     ticktext=[f"{t}%".replace("-", "−") for t in ticks])
    # Two-line row labels (entity, then definition) keep the label column
    # narrow, so the plot keeps its width on a phone.
    fig.update_yaxes(categoryorder="array", categoryarray=labels, autorange="reversed",
                     showgrid=False, tickvals=labels,
                     ticktext=[plot_text(lab).replace(" · ", "<br>", 1) for lab in labels])
    return fig


# ── 3. Commitments ÷ run-rate ───────────────────────────────────────────────

def _short_date(iso: str) -> str:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(iso or ""))
    if not m:
        return str(iso or "")
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"{int(m.group(3))} {months[int(m.group(2)) - 1]} '{m.group(1)[2:]}"


def build_labs_commitments(rows: pd.DataFrame | None = None) -> go.Figure:
    rows = L.commitment_rows() if rows is None else rows
    if rows is None or rows.empty or not rows["charted"].any():
        return empty_figure("No commitment row has a matching run-rate")
    d = rows[rows["charted"]].sort_values("order", kind="mergesort")
    labels = list(d["label"])
    fig = go.Figure()
    for _, r in d.iterrows():
        col = _color(r["palette_name"])
        q = {">": ">", "~": "~"}.get(str(r["value_qualifier"]), "")
        txt = (f"{r['multiple']:.1f}×  {q}{L._usd(r['value'])} ({_short_date(r['as_of'])}) "
               f"÷ {r['den_label']}")
        fig.add_trace(go.Scatter(
            x=[float(r["multiple"])], y=[r["label"]], mode="markers+text", showlegend=False,
            name=plot_text(r["label"]), cliponaxis=False,
            marker=dict(color=col, size=10, symbol=badge_symbol(r["badge"]),
                        line=dict(color=col, width=1.6)),
            text=[plot_text(txt)], textposition="middle right",
            textfont=dict(color="#aaaaaa", size=11, family=FONT),
            customdata=[[plot_text(r["label"]), plot_text(r["definition"]),
                         f"{q}{L._usd(r['value'])}", r["as_of"], r["badge"],
                         r["filing_status"], plot_text(r["den_label"]),
                         f"{r['multiple']:.2f}", L.source_domain(r["source_url"])]],
            meta={"row_id": r["row_id"], "entity": r["entity"]},
            hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[2]} at %{customdata[3]} "
                           "(%{customdata[4]}, %{customdata[5]})<br>%{customdata[1]}<br>"
                           "÷ %{customdata[6]} = %{customdata[7]}×<br>%{customdata[8]}"
                           "<extra></extra>"),
        ))
    for k in _badge_keys():
        fig.add_trace(k)
    flagged = int((~rows["charted"]).sum())
    title = ("Compute Commitments ÷ Revenue Run-Rate" + "  " + SUB
             + "labs buy compute; CoreWeave's RPO is the sell side · the leaked $518B and "
               "the projected $665B are excluded (ledger)"
             + (f" · {flagged} row{'s' if flagged != 1 else ''} without a run-rate "
                f"within {L.COMMITMENT_MAX_GAP_DAYS} days" if flagged else "")
             + "</span>")
    hi = float(d["multiple"].max())
    _layout(fig, title, "Multiple of annualised revenue (×)", "", height=170 + 48 * len(labels))
    fig.update_xaxes(range=[0, hi * 2.8], ticksuffix="×")
    fig.update_yaxes(categoryorder="array", categoryarray=labels, autorange="reversed",
                     showgrid=False, tickvals=labels,
                     ticktext=[plot_text(lab).replace(" · ", "<br>", 1) for lab in labels])
    return fig


# ── 4. Token volumes ────────────────────────────────────────────────────────

def _tok_tick(v: float) -> str:
    for base, suf in ((1e15, "Q"), (1e12, "T"), (1e9, "B"), (1e6, "M")):
        if v >= base:
            return f"{v / base:g}{suf}"
    return f"{v:g}"


def build_labs_tokens(rows: pd.DataFrame | None = None) -> go.Figure:
    rows = L.token_rows() if rows is None else rows
    if rows is None or rows.empty:
        return empty_figure("No token-volume disclosures available")
    d0 = rows[~rows["graph_excluded"].astype(bool)]
    if d0.empty:
        return empty_figure("No token-volume disclosures available")
    fig = go.Figure()
    seen = set()
    for (ent, scope), d in d0.groupby(["entity", "scope"], sort=True):
        d = d.sort_values("date", kind="mergesort")
        col = _color(d["palette_name"].iloc[0])
        base = L.SCOPE_SYMBOLS.get(scope, "circle")
        cds = [[plot_text(ent), plot_text(p or "—"), L.SCOPE_LABELS.get(scope, scope),
                plot_text(s), L.human_tokens(float(t)), b,
                ("Epoch AI usage_reports" if o == "epoch" else plot_text(sn)),
                L.source_domain(u) or "—"]
               for p, s, t, b, o, sn, u in zip(d["product"], d["statement"],
                                               d["tokens_per_day"], d["badge"], d["origin"],
                                               d["source_name"], d["source_url"])]
        fig.add_trace(go.Scatter(
            x=list(d["date"]), y=[float(v) for v in d["tokens_per_day"]],
            mode="lines+markers", name=plot_text(ent), legendgroup=ent,
            showlegend=ent not in seen,
            line=dict(color=col, width=1.2),
            marker=dict(color=col, size=9, symbol=[badge_symbol(b, base) for b in d["badge"]],
                        line=dict(color=col, width=1.6)),
            customdata=cds, meta={"entity": ent, "scope": scope},
            hovertemplate=("<b>%{customdata[0]}</b>  %{x}<br>%{customdata[1]} "
                           "(%{customdata[2]})<br>Stated: %{customdata[3]}<br>"
                           "≈ %{customdata[4]} tokens/day<br>Badge: %{customdata[5]} · "
                           "%{customdata[6]} · %{customdata[7]}<extra></extra>"),
        ))
        seen.add(ent)
    for scope in ("all", "api", "product"):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=L.SCOPE_LABELS[scope],
            marker=dict(symbol=L.SCOPE_SYMBOLS[scope], color=KEY_GREY, size=8),
            legendgroup="scope", showlegend=True, hoverinfo="skip"))
    present = [b for b in L.BADGES if b in set(d0["badge"])] or ["disclosed"]
    for k in _badge_keys(present, base="square"):
        fig.add_trace(k)
    lo, hi = float(d0["tokens_per_day"].min()), float(d0["tokens_per_day"].max())
    ticks = _decades(lo, hi)
    title = ("Disclosed Token Volumes" + "  " + SUB
             + "tokens per day (log) · monthly ÷ 30, weekly ÷ 7, per-minute × 1,440, Epoch "
               "AI's convention · symbol = what is counted · lines join one scope only · "
             + EPOCH_SUB + " + cited disclosures</span>")
    _layout(fig, title, "Date of the figure", "Tokens per day (log)", height=480)
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[_tok_tick(t) for t in ticks],
                     range=[math.log10(lo) - 0.3, math.log10(hi) + 0.3])
    return fig


# ── 5. The ledger ───────────────────────────────────────────────────────────

_VIEW_ORDER = {"margin": 0, "commitment": 1, "token": 2, "revenue": 3, "footnote": 4,
               "ledger": 5}
_SAFE_URL = re.compile(r"^https://[A-Za-z0-9.\-]+(:\d+)?(/[^\s\"'<>`]*)?$")


def _link(url: str, text: str) -> str:
    if not _SAFE_URL.match(str(url or "")):
        return _esc(text)
    return (f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">'
            f"{_esc(text)}</a>")


# The same credit as plain text (Dash renders it as a text node).
LEDGER_CREDIT_TEXT = (
    'Data: Epoch AI, "Data on AI companies" (Josh You, John Croxton, Venkat Somala, Yafah '
    "Edelman), epoch.ai/data/ai-companies, CC BY 4.0; modified: filtered to dated annualised "
    "revenue and tokens-per-day rows. Company filings on SEC EDGAR and HKEXnews; press and "
    "analyst figures as badged. This ledger is hand-curated (data/curated/lab_ledger.csv)."
)
LEDGER_CREDIT = (
    'Data: <a href="https://epoch.ai/data/ai-companies" target="_blank" '
    'rel="noopener noreferrer">Epoch AI, "Data on AI companies"</a> (Josh You, John Croxton, '
    'Venkat Somala, Yafah Edelman), '
    '<a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" '
    'rel="noopener noreferrer">CC BY 4.0</a>; modified: filtered to dated annualised revenue '
    "and tokens-per-day rows. Company filings on SEC EDGAR and HKEXnews; press and analyst "
    "figures as badged. This ledger is hand-curated (data/curated/lab_ledger.csv)."
)


LEDGER_HEADS = ("Entity", "Metric", "Definition", "Value", "Period", "Badge", "Filing",
                "Source")


def ledger_rows(ledger: pd.DataFrame | None = None,
                commitments: pd.DataFrame | None = None) -> list[dict]:
    """The ledger's display rows, raw text (unescaped), shipped rows first by
    view then file order, excluded rows last with their reason. Shared by the
    static HTML (labs_ledger_html) and Dash's html.Table."""
    led = L.load_ledger(include_excluded=True) if ledger is None else ledger
    com = L.commitment_rows() if commitments is None else commitments
    mult = {r["row_id"]: r for _, r in com.iterrows()} if com is not None and len(com) else {}
    if led is None or led.empty:
        return []
    led = led.assign(_v=led["view"].map(_VIEW_ORDER).fillna(9),
                     _x=(led["ledger_state"] != "ship").astype(int),
                     _o=led["order"] if "order" in led.columns else range(len(led)))
    led = led.sort_values(["_x", "_v", "_o"], kind="mergesort")
    out = []
    for _, r in led.iterrows():
        excluded = r["ledger_state"] != "ship"
        val = L.fmt_value(r)
        m = mult.get(r["row_id"])
        if m is not None and bool(m["charted"]):
            val += f" · {float(m['multiple']):.1f}× run-rate"
        elif m is not None and m["flag"]:
            val += f" · {m['flag']}"
        if str(r.get("derived", "")) == "1":
            val += " · derived"
        out.append({
            "excluded": bool(excluded),
            "entity": str(r["entity"]), "metric": str(r["metric"]),
            "definition": str(r["definition"]),
            "reason": str(r["note"]) if excluded else "",
            "value": val, "period": str(r["period_label"] or r["as_of"]),
            "badge": r["badge"] if r["badge"] in L.BADGES else "estimated",
            "filing": (r["filing_status"] if r["filing_status"] in ("audited", "unaudited")
                       else "—"),
            "source_name": str(r["source_name"] or L.source_domain(r["source_url"])),
            "source_url": str(r["source_url"]) if _SAFE_URL.match(str(r["source_url"] or ""))
            else "",
            "source_date": str(r["source_date"]),
        })
    return out


def labs_ledger_html(ledger: pd.DataFrame | None = None,
                     commitments: pd.DataFrame | None = None) -> str:
    """The provenance ledger as an HTML table. Every cell is escaped here;
    links are https only, open in a new tab with rel="noopener noreferrer".
    Excluded rows sit greyed at the bottom with their reason."""
    rows = ledger_rows(ledger, commitments)
    if not rows:
        return '<div class="lab-ledger__empty">No ledger rows.</div>'
    n_x = sum(r["excluded"] for r in rows)
    out = ['<div class="lab-ledger">',
           f'<div class="lab-ledger__title">Ledger · {len(rows) - n_x} rows'
           f' · {n_x} excluded</div>',
           '<table class="lab-ledger__table"><thead><tr>'
           + "".join(f"<th>{h}</th>" for h in LEDGER_HEADS) + "</tr></thead><tbody>"]
    for r in rows:
        definition = _esc(r["definition"])
        if r["reason"]:
            definition += f'<div class="lab-ledger__reason">{_esc(r["reason"])}</div>'
        src = _link(r["source_url"], r["source_name"])
        src += f' <span class="lab-ledger__date">{_esc(r["source_date"])}</span>'
        cls = ' class="lab-ledger__excluded"' if r["excluded"] else ""
        out.append(
            f"<tr{cls}><td>{_esc(r['entity'])}</td><td>{_esc(r['metric'])}</td>"
            f"<td>{definition}</td><td class=\"lab-ledger__num\">{_esc(r['value'])}</td>"
            f"<td>{_esc(r['period'])}</td>"
            f'<td><span class="lab-badge lab-badge--{r["badge"]}">{r["badge"]}</span></td>'
            f"<td>{_esc(r['filing'])}</td><td>{src}</td></tr>")
    out.append("</tbody></table>")
    out.append(f'<div class="lab-ledger__credit">{LEDGER_CREDIT}</div></div>')
    return "".join(out)


def build_labs_figures() -> dict[str, go.Figure]:
    """The four Labs figures at their only state (Labs has no controls)."""
    return {
        "econ_labs_runrate": build_labs_runrate(),
        "econ_labs_margins": build_labs_margins(),
        "econ_labs_commitments": build_labs_commitments(),
        "econ_labs_tokens": build_labs_tokens(),
    }
