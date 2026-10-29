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

# Release lag: a record counts only when it beat at least this many earlier
# models of its side that AA still lists. The oldest surviving model is a
# "record" by default — every stronger model of its day has left the
# catalogue — so its lag measures nothing (Nova Micro, 2024-12-03, "matched"
# 0.1 months later on 2026-10-27).
LAG_MIN_PRIOR = 1

# Score dating: a model whose score belongs to a checkpoint AA published under
# a NEW name for the same base model (e.g. "Qwen3.8 Max" 40.30 on 2026-09-08
# -> "Qwen3.8 Max (0902)" 45.44 on 2026-09-16) is dated by that name's first
# snapshot, not the base model's release date. "New" means a rise above
# re-evaluation noise: more than BREAK_MOVED_PCT, with a margin.
SCORE_DATE_MIN_RISE = 0.05


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


def _usable(hist: pd.DataFrame | None) -> bool:
    """data/ingest.load_history returns a bare pd.DataFrame() when
    data/raw/history is missing or empty; every function here treats that
    (or a frame without the two columns it reads) as no snapshots."""
    return hist is not None and not hist.empty \
        and {"scraped_at", "quality", "model"} <= set(hist.columns)


def snapshot_dates(hist: pd.DataFrame) -> list[str]:
    if not _usable(hist):
        return []
    return sorted(str(d)[:10] for d in hist["scraped_at"].dropna().unique())


def _by_date(hist: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if not _usable(hist):
        return {}
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
        # Per MODEL, like n_open / n_closed: a snapshot can carry one row per
        # host (708 rows for 226 models on 2026-03-06), and the hover sets the
        # source counts beside the model counts, so they must add up to them.
        src = s.drop_duplicates("model")["weights_source"].value_counts()
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

    Sorted by date (`d`); on one date only the best counts. A model is a
    record when it beats every model of its side dated on or before it.
    `n_prior` = how many models of its side are dated strictly before it —
    0 for the oldest one still listed, whose "record" is an artefact of AA
    dropping everything stronger from its day."""
    out, best = [], float("-inf")
    for d, g in fam.sort_values("d").groupby("d", sort=True):
        top = g.sort_values(["quality", "model"], ascending=[False, True]).iloc[0]
        if float(top["quality"]) > best:
            top = top.copy()
            top["n_prior"] = int((fam["d"] < d).sum())
            out.append(top)
            best = float(top["quality"])
    return pd.DataFrame(out)


def score_dates(hist: pd.DataFrame, df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Model name -> the date its CURRENT score first appeared, for models
    whose score belongs to a later checkpoint than AA's release date.

    AA dates a model by its base slug's first release, so a checkpoint
    updated in place keeps the launch date. Caught here when, in the
    snapshots: the model's exact name first appears on F, an earlier name
    with the same base (one trailing parenthetical removed) was listed
    before F, and the score the name carried ON F exceeds that earlier
    name's last score — carried onto F's index scale — by more than
    SCORE_DATE_MIN_RISE. Renames that keep the score (AA adding "(Max)" on
    2026-10-01, ", Default Fallback" on 2026-10-03) do not qualify. Only
    names whose F is after their release date are returned.

    The comparison is made on F's scale, not today's. It used the CURRENT
    score until AA's v5.0 rescale (2026-10-28), which moved models by
    anything from x0.36 to x1.08: the single median factor a rescale is
    carried through (x0.75) left every frontier model ~30% "above" its own
    v4 score, and twenty renamed-in-place checkpoints — Claude Opus 5.5,
    GPT-6 Astra — were redated as new ones. A rescale after F is not a
    checkpoint, so it no longer reaches this test.
    """
    from components.charts.constants import base_model_name

    seg = index_segments(hist)
    kept = sorted(seg["segments"])
    if not kept or df is None or df.empty:
        return {}
    rho = {st["to"]: st["rho"] for st in seg["steps"] if st["break"]}
    factor, f = {}, 1.0
    for d in reversed(kept):            # onto the latest scale
        factor[d] = f
        if d in rho:
            f *= rho[d]
    h = hist.assign(_date=hist["scraped_at"].astype(str).str[:10])
    h = h[h["_date"].isin(factor)
          & (pd.to_numeric(h["quality"], errors="coerce") > 0)]
    h = h.assign(_q=h["quality"].astype(float) * h["_date"].map(factor),
                 _name=h["model"].astype(str))
    h = h.assign(_base=h["_name"].map(base_model_name))
    first = h.groupby("_name")["_date"].min()
    on_first = h[h["_date"] == h["_name"].map(first)].groupby("_name")["_q"].max()
    by_base = {b: g for b, g in h.groupby("_base")}
    cur = df[pd.to_numeric(df["quality"], errors="coerce") > 0]
    cur_q = cur.groupby(cur["model"].astype(str))["quality"].max().astype(float)
    rel = (pd.to_datetime(cur["release_date"], errors="coerce", format="%Y-%m-%d")
           .groupby(cur["model"].astype(str)).min()
           if "release_date" in cur.columns else pd.Series(dtype="datetime64[ns]"))
    out = {}
    for name in cur_q.index:
        if name not in first.index:
            continue
        f0 = first[name]
        q = float(on_first[name])
        g = by_base.get(base_model_name(name))
        if g is None:
            continue
        earlier = g[(g["_name"] != name) & (g["_date"] < f0)]
        if earlier.empty:
            continue
        prev = float(earlier.loc[earlier["_date"] == earlier["_date"].max(), "_q"].max())
        d0 = rel.get(name, pd.NaT)
        if q > (1 + SCORE_DATE_MIN_RISE) * prev and (pd.isna(d0) or pd.Timestamp(f0) > d0):
            out[name] = pd.Timestamp(f0)
    return out


def km_median(times, censored) -> float | None:
    """Kaplan-Meier median of right-censored durations: the first time the
    survival estimate falls to 0.5 or below; None when it never does (too
    many still unmatched to say). At a tie, events count before censorings."""
    pairs = sorted(zip((float(t) for t in times), (bool(c) for c in censored)),
                   key=lambda p: (p[0], p[1]))
    at_risk, surv, i = len(pairs), 1.0, 0
    while i < len(pairs):
        t = pairs[i][0]
        events = cens = 0
        while i < len(pairs) and pairs[i][0] == t:
            cens += pairs[i][1]
            events += not pairs[i][1]
            i += 1
        if events:
            surv *= 1 - events / at_risk
            if surv <= 0.5:
                return t
        at_risk -= events + cens
    return None


_LAG_COLUMNS = ["closed_model", "closed_date", "closed_q", "closed_release", "open_model",
                "open_date", "open_q", "open_release", "lag_months", "censored"]


def release_lag(df: pd.DataFrame, asof: date | str | None = None,
                dated: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Closed-weights records and how long open weights took to match each.

    Each family is dated by AA's release date, or — for a name in `dated`
    (score_dates' result) — by the first snapshot that carried its current
    score, when that is later. For each closed record C (date d_C, score
    q_C) that beat at least LAG_MIN_PRIOR earlier closed models still
    listed: the earliest date of an open family with q >= q_C. lag = max(0,
    that - d_C) in months (days / 30.4375). With no such open model C is
    right-censored: the lag is at least asof - d_C, never a number.

    Returns (records, summary). records columns: closed_model, closed_date,
    closed_q, closed_release, open_model, open_date, open_q, open_release,
    lag_months, censored (*_release = AA's date, kept when *_date differs).
    summary: median_lag_months (Kaplan-Meier over matched AND censored; None
    when it cannot be identified), median_matched_months (matched only),
    matched, censored, asof, open_records, closed_records (the staircases,
    every record incl. the uncounted first), uncounted (closed records left
    out for having no earlier closed model to beat), redated (names dated by
    their score's first snapshot).
    """
    asof = pd.Timestamp(asof or date.today()).normalize()
    fam = _dated_families(df)
    empty = {"median_lag_months": None, "median_matched_months": None, "matched": 0,
             "censored": 0, "asof": asof.date().isoformat(), "open_records": pd.DataFrame(),
             "closed_records": pd.DataFrame(), "uncounted": [], "redated": []}
    if fam.empty:
        return pd.DataFrame(columns=_LAG_COLUMNS), empty
    fam = fam.assign(release=fam["d"])
    redated = []
    for name, when in (dated or {}).items():
        hit = fam["model"].astype(str).eq(name) & (fam["d"] < when)
        if hit.any():
            fam.loc[hit, "d"] = pd.Timestamp(when)
            redated.append(name)
    op = fam[fam["open_weights"].eq(True)]
    cl = fam[fam["open_weights"].eq(False)]
    rec_c = frontier_records(cl) if not cl.empty else pd.DataFrame()
    rec_o = frontier_records(op) if not op.empty else pd.DataFrame()
    counted = rec_c[rec_c["n_prior"] >= LAG_MIN_PRIOR] if not rec_c.empty else rec_c
    uncounted = ([str(m) for m in rec_c.loc[rec_c["n_prior"] < LAG_MIN_PRIOR, "model"]]
                 if not rec_c.empty else [])
    rows = []
    for _, c in counted.iterrows():
        q_c, d_c = float(c["quality"]), c["d"]
        cand = op[op["quality"].astype(float) >= q_c]
        if cand.empty:
            rows.append(dict(closed_model=str(c["model"]), closed_date=d_c, closed_q=q_c,
                             closed_release=c["release"],
                             open_model=None, open_date=pd.NaT, open_q=float("nan"),
                             open_release=pd.NaT,
                             lag_months=max(0, (asof - d_c).days) / DAYS_PER_MONTH,
                             censored=True))
            continue
        first = cand.sort_values(["d", "quality", "model"],
                                 ascending=[True, False, True]).iloc[0]
        lag = max(0, (first["d"] - d_c).days) / DAYS_PER_MONTH
        rows.append(dict(closed_model=str(c["model"]), closed_date=d_c, closed_q=q_c,
                         closed_release=c["release"],
                         open_model=str(first["model"]), open_date=first["d"],
                         open_q=float(first["quality"]), open_release=first["release"],
                         lag_months=lag, censored=False))
    recs = pd.DataFrame(rows, columns=_LAG_COLUMNS)
    matched = recs[~recs["censored"].astype(bool)] if not recs.empty else recs
    summary = {
        **empty,
        "median_lag_months": km_median(recs["lag_months"], recs["censored"]),
        "median_matched_months": (float(median(matched["lag_months"]))
                                  if len(matched) else None),
        "matched": int(len(matched)),
        "censored": int(len(recs) - len(matched)),
        "open_records": rec_o,
        "closed_records": rec_c,
        "uncounted": uncounted,
        "redated": sorted(redated),
    }
    return recs, summary


# ── Iso-quality price ratio ───────────────────────────────────────────────────

def iso_quality_ratios(df: pd.DataFrame, mix=None,
                       measured_only: bool = True) -> tuple[pd.DataFrame, dict]:
    """Per open model family: cheapest closed family scoring >= it, price ratio.

    `measured_only` (the default, and what the chart shows) keeps only rows
    whose AA Intelligence Index AA measured: a row with quality_estimated
    True is left out on BOTH sides before families are formed. On
    2026-10-27 one closed model with an AA-estimated score (Agnes 2.5 Pro
    Beta, 35.24; 49.1 on 2026-08-27) was the cheapest peer for 46 of 65
    open families, and 53 of the 65 peers were estimates, so the headline
    rested on extrapolated scores. The rows left out are counted
    (n_estimated_rows), never silently dropped.

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
    n_excluded, excluded (names), mix_short, measured_only, n_estimated_rows.
    """
    from components.charts.constants import dedupe_to_best_variant
    from data.workload_mix import apply_mix, resolve_mix, filled_flags
    from data.task_cost import task_costs

    m = resolve_mix(mix)
    priced = apply_mix(df, m)
    priced = priced[(pd.to_numeric(priced["price"], errors="coerce") > 0)
                    & (pd.to_numeric(priced["quality"], errors="coerce") > 0)].copy()
    est = (priced["quality_estimated"].eq(True) if "quality_estimated" in priced.columns
           else pd.Series(False, index=priced.index))
    n_est = int(est.sum()) if measured_only else 0
    if measured_only:
        priced = priced[~est]
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
        "measured_only": bool(measured_only),
        "n_estimated_rows": n_est,
    }
    return out, summary


# ── ↓CSV on Economics › Labs ──────────────────────────────────────────────────
# The history this reads never ships to the browser, so the export is a
# build-time slice in pydata.zip (build_static.DERIVED_DATA), read through
# data/econ_common.read_derived; Dash and local runs compute the same text.
DERIVED_EXPORT = "data/derived/econ_open_gap.csv"
EXPORT_CHARTS = ("score_gap", "release_lag", "iso_price")


def export_frame(hist: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Every row behind the three Labs figures in one table, the same calls
    build_static.open_gap_figures makes. `chart` (first column) names the
    figure: score_gap (gap_history), release_lag (release_lag records) or
    iso_price (iso_quality_ratios); columns a figure does not have are blank."""
    dates = snapshot_dates(hist)
    asof = dates[-1] if dates else None
    frames = (gap_history(hist)[0],
              release_lag(df, asof, score_dates(hist, df))[0],
              iso_quality_ratios(df, None)[0])
    parts = [f.assign(chart=name) for name, f in zip(EXPORT_CHARTS, frames)
             if f is not None and len(f)]
    if not parts:
        return pd.DataFrame(columns=["chart"])
    out = pd.concat(parts, ignore_index=True, sort=False)
    return out[["chart"] + [c for c in out.columns if c != "chart"]]


def export_csv_text() -> str:
    """export_frame over the committed history and the current catalogue, as
    CSV text (DERIVED_DATA). Floats to 6 significant digits keep the slice
    small; every chart shows fewer."""
    from data.ingest import get_models, load_history
    return export_frame(load_history(), get_models()).to_csv(index=False,
                                                             float_format="%.6g")
