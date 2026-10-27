"""The open-weights vs closed-weights gap: score, time and price.

Three measures, each from Artificial Analysis data this project already
scrapes, and each with its own honesty problem stated where it is solved:

1. SCORE GAP OVER TIME (`gap_history`). Within each daily snapshot in
   data/raw/history/, the best closed model's AA Intelligence Index minus the
   best open model's. AA has RESCALED the index more than once, so a gap in
   points is only comparable within one index version: `index_segments`
   finds the rescales from the snapshots themselves and the chart never draws
   a line across one.

2. RELEASE LAG (`release_lag`). From the current snapshot's release dates: for
   each closed model that set a new closed-weights high when it was released,
   how long until an open model scoring at least as much was released. Only
   models still in the catalogue can be counted — AA drops superseded models —
   so this is a survivor's view, and a record no open model has matched yet is
   right-censored, never dropped.

3. ISO-QUALITY PRICE (`iso_quality_ratios`). For each open model family: the
   cheapest closed family scoring at or above it, priced at the same workload
   mix, divided by the open model's own price. >1 = the open model is cheaper.
   AA's price is one model-level figure, not the cheapest third-party host.

Open vs closed per row comes from data/open_weights.py (AA's flag, else the
name registry). Pure functions over DataFrames; history is read only at build
time (build_static) and by Dash, and never shipped to the browser.
"""
from __future__ import annotations

from datetime import date
from statistics import median

import pandas as pd

# A rescale shows up as nearly every model moving by the same factor between
# two consecutive snapshots. Tuned on the 164 committed snapshots (2026-02-25
# .. 2026-10-27) and pinned by tests on both sides: the three real rescales
# move the median ratio by 0.74-0.89 with 50%+ of models moving >3%, while the
# partial re-evaluation of 2026-08-07 (median x1.018, 20% moved) and the
# 2026-09-20 one (13% moved) are not breaks.
BREAK_MIN_COMMON = 20          # models scored in both snapshots
BREAK_MEDIAN_DELTA = 0.05      # |median(q_B / q_A) - 1| at least this
BREAK_MOVED_SHARE = 0.50       # share of common models that moved
BREAK_MOVED_PCT = 0.03         # what "moved" means

DAYS_PER_MONTH = 30.4375       # 365.25 / 12


def _scored(snap: pd.DataFrame) -> pd.Series:
    """model -> quality for one snapshot, scored rows only (max per name)."""
    s = snap[pd.to_numeric(snap["quality"], errors="coerce") > 0]
    return s.groupby(s["model"].astype(str))["quality"].max().astype(float)


def _step(a: pd.Series, b: pd.Series) -> dict:
    common = a.index.intersection(b.index)
    if len(common) < BREAK_MIN_COMMON:
        return {"n": int(len(common)), "rho": None, "moved": None, "break": False}
    r = (b[common] / a[common]).astype(float)
    rho = float(r.median())
    moved = float(((r - 1).abs() > BREAK_MOVED_PCT).mean())
    brk = abs(rho - 1) >= BREAK_MEDIAN_DELTA and moved >= BREAK_MOVED_SHARE
    return {"n": int(len(common)), "rho": rho, "moved": moved, "break": brk}


def snapshot_dates(hist: pd.DataFrame) -> list[str]:
    return sorted(str(d)[:10] for d in hist["scraped_at"].dropna().unique())


def _by_date(hist: pd.DataFrame) -> dict[str, pd.DataFrame]:
    key = hist["scraped_at"].astype(str).str[:10]
    return {d: g for d, g in hist.groupby(key, sort=True)}


def index_segments(hist: pd.DataFrame) -> dict:
    """Where AA's index scale changed, found from the snapshots themselves.

    Returns {"segments": {date: segment id}, "breaks": [dates], "excluded":
    [dates], "steps": [{from, to, n, rho, moved, break}]}.

    A snapshot B is a GLITCH, and excluded, when A->B and B->C both look like
    rescales but A->C does not: the index jumped and came straight back
    (2026-03-07, when AA briefly published it renormalised to 0-100). A BREAK
    is a rescale between consecutive kept snapshots; segment ids count them.
    """
    snaps = _by_date(hist)
    dates = sorted(snaps)
    q = {d: _scored(snaps[d]) for d in dates}
    excluded = []
    for i in range(1, len(dates) - 1):
        a, b, c = dates[i - 1], dates[i], dates[i + 1]
        if _step(q[a], q[b])["break"] and _step(q[b], q[c])["break"] \
                and not _step(q[a], q[c])["break"]:
            excluded.append(b)
    kept = [d for d in dates if d not in set(excluded)]
    segments, breaks, steps, seg = {}, [], [], 0
    for i, d in enumerate(kept):
        if i:
            st = _step(q[kept[i - 1]], q[d])
            steps.append({"from": kept[i - 1], "to": d, **st})
            if st["break"]:
                seg += 1
                breaks.append(d)
        segments[d] = seg
    return {"segments": segments, "breaks": breaks, "excluded": excluded,
            "steps": steps}


def gap_history(hist: pd.DataFrame, annotate=None) -> tuple[pd.DataFrame, dict]:
    """One row per kept snapshot: the best closed and best open model, the gap.

    Columns: date, segment, closed_model, closed_q, open_model, open_q, gap,
    gap_pct, n_open, n_closed, src_aa, src_catalogue, src_inferred.
    gap = closed_q - open_q (AA index points of THAT snapshot's scale);
    gap_pct = 100 * gap / closed_q. A snapshot with no open (or no closed)
    model has NaN gap rather than a fabricated one.

    `annotate` classifies a snapshot (default data/open_weights.
    annotate_open_weights, which reads each row's own flag first, then the
    registry). Returns (frame, index_segments(...)).
    """
    if annotate is None:
        from data.open_weights import annotate_open_weights as annotate
    seg = index_segments(hist)
    rows = []
    for d, snap in _by_date(hist).items():
        if d not in seg["segments"]:
            continue
        s = snap[pd.to_numeric(snap["quality"], errors="coerce") > 0].copy()
        s["quality"] = s["quality"].astype(float)
        s = annotate(s)
        op, cl = s[s["open_weights"]], s[~s["open_weights"]]
        src = s["weights_source"].value_counts()
        row = {"date": d, "segment": seg["segments"][d],
               "n_open": int(op["model"].nunique()), "n_closed": int(cl["model"].nunique()),
               "src_aa": int(src.get("aa", 0)), "src_catalogue": int(src.get("catalogue", 0)),
               "src_inferred": int(src.get("inferred", 0))}
        if not op.empty and not cl.empty:
            top_c = cl.sort_values(["quality", "model"], ascending=[False, True]).iloc[0]
            top_o = op.sort_values(["quality", "model"], ascending=[False, True]).iloc[0]
            gap = float(top_c["quality"]) - float(top_o["quality"])
            row.update(closed_model=str(top_c["model"]), closed_q=float(top_c["quality"]),
                       open_model=str(top_o["model"]), open_q=float(top_o["quality"]),
                       gap=gap, gap_pct=100.0 * gap / float(top_c["quality"]))
        else:
            row.update(closed_model=None, closed_q=float("nan"), open_model=None,
                       open_q=float("nan"), gap=float("nan"), gap_pct=float("nan"))
        rows.append(row)
    cols = ["date", "segment", "closed_model", "closed_q", "open_model", "open_q",
            "gap", "gap_pct", "n_open", "n_closed", "src_aa", "src_catalogue",
            "src_inferred"]
    return pd.DataFrame(rows, columns=cols), seg


# ── Release lag ───────────────────────────────────────────────────────────────

def _dated_families(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model family (best-scoring variant) with a release date."""
    from components.charts.constants import dedupe_to_best_variant

    s = df[(pd.to_numeric(df["quality"], errors="coerce") > 0)].copy()
    if "release_date" not in s.columns:
        return s.iloc[0:0].assign(d=pd.Series(dtype="datetime64[ns]"))
    s["d"] = pd.to_datetime(s["release_date"], errors="coerce", format="%Y-%m-%d")
    s = s[s["d"].notna()]
    if s.empty:
        return s
    if "open_weights" not in s.columns:
        from data.open_weights import annotate_open_weights
        s = annotate_open_weights(s)
    return dedupe_to_best_variant(s)


def frontier_records(fam: pd.DataFrame) -> pd.DataFrame:
    """Models that set a new high for their side when they were released.

    Sorted by release date; on one date only the best counts. A model is a
    record when it beats every model of its side released on or before it."""
    out, best = [], float("-inf")
    for d, g in fam.sort_values("d").groupby("d", sort=True):
        top = g.sort_values(["quality", "model"], ascending=[False, True]).iloc[0]
        if float(top["quality"]) > best:
            out.append(top)
            best = float(top["quality"])
    return pd.DataFrame(out)


def release_lag(df: pd.DataFrame, asof: date | str | None = None) -> tuple[pd.DataFrame, dict]:
    """Closed-weights records and how long open weights took to match each.

    For each closed record C (release date d_C, score q_C): the earliest
    release date of an open family with q >= q_C. lag = max(0, that - d_C) in
    months (days / 30.4375). With no such open model C is right-censored: the
    lag is at least asof - d_C, reported as such, never as a number.

    Returns (records, summary). records columns: closed_model, closed_date,
    closed_q, open_model, open_date, open_q, lag_months, censored. summary:
    median_lag_months (matched only; None when nothing matched), matched,
    censored, asof, open_records, closed_records (the two staircases).
    """
    asof = pd.Timestamp(asof or date.today()).normalize()
    fam = _dated_families(df)
    if fam.empty:
        return pd.DataFrame(columns=["closed_model", "closed_date", "closed_q", "open_model",
                                     "open_date", "open_q", "lag_months", "censored"]), \
            {"median_lag_months": None, "matched": 0, "censored": 0,
             "asof": asof.date().isoformat(), "open_records": pd.DataFrame(),
             "closed_records": pd.DataFrame()}
    op = fam[fam["open_weights"].eq(True)]
    cl = fam[fam["open_weights"].eq(False)]
    rec_c = frontier_records(cl) if not cl.empty else pd.DataFrame()
    rec_o = frontier_records(op) if not op.empty else pd.DataFrame()
    rows = []
    for _, c in rec_c.iterrows():
        q_c, d_c = float(c["quality"]), c["d"]
        cand = op[op["quality"].astype(float) >= q_c]
        if cand.empty:
            rows.append(dict(closed_model=str(c["model"]), closed_date=d_c, closed_q=q_c,
                             open_model=None, open_date=pd.NaT, open_q=float("nan"),
                             lag_months=max(0, (asof - d_c).days) / DAYS_PER_MONTH,
                             censored=True))
            continue
        first = cand.sort_values(["d", "quality", "model"],
                                 ascending=[True, False, True]).iloc[0]
        lag = max(0, (first["d"] - d_c).days) / DAYS_PER_MONTH
        rows.append(dict(closed_model=str(c["model"]), closed_date=d_c, closed_q=q_c,
                         open_model=str(first["model"]), open_date=first["d"],
                         open_q=float(first["quality"]), lag_months=lag, censored=False))
    recs = pd.DataFrame(rows, columns=["closed_model", "closed_date", "closed_q", "open_model",
                                       "open_date", "open_q", "lag_months", "censored"])
    matched = recs[~recs["censored"].astype(bool)] if not recs.empty else recs
    summary = {
        "median_lag_months": float(median(matched["lag_months"])) if len(matched) else None,
        "matched": int(len(matched)),
        "censored": int(len(recs) - len(matched)),
        "asof": asof.date().isoformat(),
        "open_records": rec_o,
        "closed_records": rec_c,
    }
    return recs, summary


# ── Iso-quality price ratio ───────────────────────────────────────────────────

def iso_quality_ratios(df: pd.DataFrame, mix=None) -> tuple[pd.DataFrame, dict]:
    """Per open model family: cheapest closed family scoring >= it, price ratio.

    Both sides are priced at `mix` (data/workload_mix; None = the dashboard's
    default list blend, i.e. the stored `price`) and collapsed to their
    best-scoring variant per family, the variant the Overview draws.
    ratio = price(closed peer) / price(open); > 1 means the open model is
    cheaper than the cheapest closed model that scores at least as well. An
    open family with no closed family at or above its score is excluded and
    counted, not given a ratio.

    Returns (frame, summary). frame columns: open_model, open_provider, open_q,
    open_price, open_filled, closed_model, closed_provider, closed_q,
    closed_price, closed_filled, ratio, task_ratio (closed / open AA cost per
    task where AA publishes both, else NaN). summary: n, median, share_below_1,
    n_excluded, excluded (names), mix_short.
    """
    from components.charts.constants import dedupe_to_best_variant
    from data.workload_mix import apply_mix, resolve_mix, filled_flags
    from data.task_cost import task_costs

    m = resolve_mix(mix)
    priced = apply_mix(df, m)
    priced = priced[(pd.to_numeric(priced["price"], errors="coerce") > 0)
                    & (pd.to_numeric(priced["quality"], errors="coerce") > 0)].copy()
    if "open_weights" not in priced.columns:
        from data.open_weights import annotate_open_weights
        priced = annotate_open_weights(priced)
    priced["_filled"] = filled_flags(priced)
    priced["_cpt"] = task_costs(priced)
    fam = dedupe_to_best_variant(priced)
    op = fam[fam["open_weights"].eq(True)]
    cl = fam[fam["open_weights"].eq(False)]
    rows, excluded = [], []
    for _, o in op.sort_values(["quality", "model"], ascending=[False, True]).iterrows():
        peers = cl[cl["quality"].astype(float) >= float(o["quality"])]
        if peers.empty:
            excluded.append(str(o["model"]))
            continue
        c = peers.sort_values(["price", "quality", "model"],
                              ascending=[True, False, True]).iloc[0]
        cpt_o, cpt_c = o["_cpt"], c["_cpt"]
        rows.append(dict(
            open_model=str(o["model"]), open_provider=str(o["provider"]),
            open_q=float(o["quality"]), open_price=float(o["price"]),
            open_filled=bool(o["_filled"]),
            closed_model=str(c["model"]), closed_provider=str(c["provider"]),
            closed_q=float(c["quality"]), closed_price=float(c["price"]),
            closed_filled=bool(c["_filled"]),
            ratio=float(c["price"]) / float(o["price"]),
            task_ratio=(float(cpt_c) / float(cpt_o)
                        if pd.notna(cpt_o) and pd.notna(cpt_c) and cpt_o > 0 else float("nan")),
        ))
    cols = ["open_model", "open_provider", "open_q", "open_price", "open_filled",
            "closed_model", "closed_provider", "closed_q", "closed_price",
            "closed_filled", "ratio", "task_ratio"]
    out = pd.DataFrame(rows, columns=cols)
    summary = {
        "n": int(len(out)),
        "median": float(out["ratio"].median()) if len(out) else None,
        "share_below_1": float((out["ratio"] < 1).mean()) if len(out) else None,
        "n_excluded": len(excluded),
        "excluded": excluded,
        "mix_short": m.short,
    }
    return out, summary
