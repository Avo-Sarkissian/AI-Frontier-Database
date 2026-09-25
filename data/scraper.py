"""
Scrapes live model data from the Artificial Analysis leaderboard into the CSV cache.

Source: https://artificialanalysis.ai/leaderboards/models

THE API IS GONE.
This read /api/data/website/host-models/performance until 2026-08-20, when it
started answering 404. The hourly refresh had been failing every run since,
serving a frozen cache under a badge that (correctly) reported the hosted
dataset as stale. That endpoint is the third AA JSON API this project has
outlived: the image arena went key-gated, video never had one, and the
open-weight catalogue moved off this same URL three days earlier. All four
scrapers now read rendered pages through data/rsc.py.

WHAT CHANGED IN THE NUMBERS, HONESTLY.
The old endpoint returned host x model rows, so this module aggregated across
hosts and kept the cheapest one's whole record. The leaderboard publishes one
row per model, so there is no cheapest-host choice left to make: the price and
speed here are the figures AA puts on the model itself. That is a real change of
meaning, not a like-for-like swap, and it is why the catalogue grew from 155 to
~183 — models nobody sells through a tracked host now appear too.

The 3:1 output-weighted blend below is unchanged and remains OURS, not AA's.

THE PAGE RESTRUCTURED, 2026-09-10.
AA split the single 94-field records array in two and slimmed the half this
module reads to 50 fields. `name` went with the other half — the picker index,
which until then carried nothing worth reading — so every row failed the
empty-name check and the hourly refresh published nothing for two days while
exiting 1. The records are joined back to the picker on `slug` (unique across
all 646 rows) in _extract_models before anything reads them.

`shortName` survives in the metrics array and looks like a drop-in replacement.
It is not: it differs from `name` for 54 of the 198 models published here
("Claude Opus 5 (low)" against "Claude Opus 5 (Adaptive Reasoning, Low
Effort)"), so taking it would rename those rows, orphan their snapshots under
data/raw/history/ and break the spotlight colour mapping — with a healthy row
count the whole way. Hence the hard failure in _extract_models when the picker
index is missing: a dead scrape keeps the cache and turns the run red, which is
recoverable, and a silent mass rename is not.

The same restructure dropped codingIndex and agenticIndex outright, replacing
the composites with their raw component benchmarks rather than renaming them.
See _CARRIED_COLUMNS for why those two are carried — from the committed CSV
and, since 2026-09-24, from the sidecar in data/carried/ (data/carried.py).

AA'S `deprecated` FLAG FLAPS.
Since ~2026-09-17 AA flags superseded-but-still-sold models deprecated, and on
2026-09-22 it toggled the flag hourly for 17 models, so the catalogue went
196 -> 180 -> 196 -> 180 rows. A model this catalogue already publishes is now
dropped only after several consecutive deprecated scrapes (see
data/carried.deprecation_hysteresis), and coverage.json counts the drops.

Falls back to the existing cache on any failure.

Run standalone:  python -m data.scraper
Integrated:      from data.scraper import scrape_and_save; scrape_and_save()
"""

import json
import sys
import threading
import time

import requests

from data.rsc import find_array, payload_from_html

from pathlib import Path

from data.ingest import load_from_raw, save_cache, load_cached
from data import carried, scrape_status

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://artificialanalysis.ai/",
}

_PAGE_URL = "https://artificialanalysis.ai/leaderboards/models"
_TIMEOUT = 45   # seconds — response can be ~20 MB


# ── Parser ────────────────────────────────────────────────────────────────────

def _real(v):
    """RSC encodes JS `undefined` as the literal string "$undefined"."""
    return v is not None and v != "$undefined"


def _index_str(v) -> str:
    """One of AA's sibling indices as a CSV field, or "" when it has no score.

    Empty, never 0. AA scores Coding on 144 of our 198 models and Agentic on
    107, and 0 is a REAL value on these scales — AA-Omniscience is negative for
    most of the catalogue (live range -88.6 to +43.7) precisely because it
    penalises hallucination. Writing 0 for "not scored" would put an unscored
    model above a genuinely bad one.
    """
    if not _real(v) or isinstance(v, bool):
        return ""
    try:
        return str(round(float(v), 2))
    except (TypeError, ValueError):
        return ""


def _names_by_slug(payload: str) -> dict[str, str]:
    """The published name for every slug, read from the picker index.

    AA's 2026-09-10 restructure moved `name` OUT of the metrics records and into
    this array, which until then carried nothing this scrape wanted. It is now
    the only place the published name exists.

    The metrics array kept a `shortName`, and reaching for it is the obvious
    shortcut — but it is a different string for 54 of the 198 models we publish
    ("Claude Opus 5 (low)" against "Claude Opus 5 (Adaptive Reasoning, Low
    Effort)"). Swapping to it would rename those rows, orphan their history
    snapshots under data/raw/history/ and break the spotlight colour mapping,
    all while the row count stayed healthy.
    """
    index = find_array(
        payload, "models",
        where=lambda a: bool(a) and isinstance(a[0], dict)
        and "name" in a[0] and "intelligenceIndex" not in a[0],
    )
    if index is None:
        return {}
    out: dict[str, str] = {}
    for entry in index:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug")
        name = (entry.get("name") or "").strip()
        if slug and name:
            out[slug] = name
    return out


def _extract_models(html: str) -> list[dict]:
    """The leaderboard's metrics records, each carrying the name it publishes under.

    "models" appears TWICE in this payload: the lightweight picker index first,
    then the metrics records. Selecting on a metrics-only field takes the right
    one — without the predicate the scrape publishes a full row count with every
    metric column empty.

    Both halves are needed since 2026-09-10. The metrics array was slimmed from
    94 fields to 50 and `name` went with it, so the records are joined back to
    the picker on `slug` (unique across all 646 rows) before anything reads them.
    """
    payload = payload_from_html(html)
    models = find_array(
        payload, "models",
        where=lambda a: bool(a) and isinstance(a[0], dict) and "intelligenceIndex" in a[0],
    )
    if models is None:
        raise ValueError("no metrics array found for 'models' in RSC payload")
    models = [m for m in models if isinstance(m, dict)]

    names = _names_by_slug(payload)
    if not names:
        # Falling back to shortName for the whole catalogue would rename 54 of
        # 198 models at once. A dead scrape keeps the cache and turns the run
        # red, which is recoverable; a silent mass rename is not.
        raise ValueError(
            "no picker index found for 'models' in RSC payload — "
            "cannot resolve published model names"
        )

    fell_back = 0
    for m in models:
        name = names.get(m.get("slug"), "")
        if not name:
            # One slug the picker does not list is a new model mid-publish, not
            # a schema change. Keep the row under the name AA does give it.
            name = (m.get("shortName") or "").strip()
            if name:
                fell_back += 1
        m["name"] = name
    if fell_back:
        print(f"[scraper] {fell_back} model(s) absent from the picker index — "
              f"published under shortName")
    return models


def _parse_api_response(models: list[dict],
                        hold_deprecated: set[str] | frozenset = frozenset()) -> list[list]:
    """
    Convert leaderboard records into raw_rows format:
      [model, context, provider, quality, price, speed, latency, price_in, price_out]

    provider = the AI lab / model creator (e.g. Google, Anthropic, OpenAI).

    `hold_deprecated` names deprecated models to keep anyway, because their
    deprecated streak is still too short to trust (data/carried.py). It is
    computed by the caller so this stays a pure function of its inputs; the
    default holds nothing, which is the pre-hysteresis behaviour.
    """
    rows = []
    # Models the catalog cannot carry. Counting them is the point: "148 tracked"
    # silently meant "162 upstream minus 14" until this existed, and a silent
    # drop is how this project has been bitten before.
    skipped: dict[str, set] = {"no_score": set(), "no_price": set()}
    kept: set = set()
    deprecated: set = set()
    held: set = set()

    def _skip(reason: str, name: str) -> None:
        # A held model is still a deprecated one. If AA pulls its price or
        # score in the same breath as flagging it, the drop is the flag's, not
        # a gap in coverage: counting it under no_price folded it into
        # distinct_upstream_models and the site called it "not carried —
        # no published price", the misattribution skipped_deprecated exists
        # to prevent.
        (deprecated if name in held else skipped[reason]).add(name)

    for m in models:
        model_name = (m.get("name") or "").strip()
        if not model_name:
            continue
        if m.get("deprecated"):
            if model_name not in hold_deprecated:
                deprecated.add(model_name)
                continue
            held.add(model_name)

        quality = m.get("intelligenceIndex")
        if not _real(quality) or quality <= 0:
            _skip("no_score", model_name)
            continue

        provider = (m.get("modelCreatorName") or "").strip()

        # Context window — the MODEL's own, which is what this field has always
        # meant here; the old endpoint offered a host-specific one too and
        # preferring it published Nemotron 3.5 Lightning at 29k against a real
        # 1,000,000. The leaderboard only publishes the model's, so that whole
        # class of mismatch is gone.
        ctx_tokens = m.get("contextWindowTokens") if _real(m.get("contextWindowTokens")) else 0
        ctx_tokens = int(ctx_tokens or 0)
        if ctx_tokens >= 1_000_000:
            mm = ctx_tokens / 1_000_000
            ctx_str = f"{mm:.0f}m" if abs(mm - round(mm)) < 0.05 else f"{mm:.1f}m"
        elif ctx_tokens >= 1000:
            ctx_str = f"{ctx_tokens // 1000}k"
        else:
            ctx_str = str(ctx_tokens)

        # Price — OUR blend, computed here: 3 parts output to 1 part input, per
        # 1M tokens. Input and output are kept alongside it, because the blend
        # is the right basis for cost estimates but is not the number anyone
        # quotes, so the UI shows all three.
        #
        # AA publishes several of its own blends (price1mBlended0To3To1 is
        # (3*in + out)/4 — the OPPOSITE weighting, a median 1.91x lower). We
        # keep the output-weighted basis deliberately: output-only would
        # overstate cost for input-heavy agentic/RAG workloads, re-rank Value
        # toward models with cheap output and expensive input, and break
        # continuity with the blended-only history snapshots. But nothing may
        # credit AA for it — see the labels in app.py, docs/app.js and README.md.
        p_in  = m.get("price1mInputTokens")
        p_out = m.get("price1mOutputTokens")
        if not _real(p_in) or not _real(p_out) or p_in < 0 or p_out < 0:
            _skip("no_price", model_name)
            continue
        price = (3 * p_out + 1 * p_in) / 4
        if price <= 0:
            _skip("no_price", model_name)
            continue

        speed   = m.get("medianOutputTokensPerSecond")
        latency = m.get("medianTimeToFirstTokenSeconds")
        speed   = float(speed) if _real(speed) else 0
        latency = float(latency) if _real(latency) else 0

        # AA's 2026-09 intelligence overhaul. `intelligenceIndexIsEstimated` is
        # true on 105 of the 198 models this catalogue publishes: AA sets it
        # when it has not run the full new eval suite and is extrapolating the
        # index. Presenting that as a measurement is the error _price_label()
        # exists to prevent for price, one column over.
        #
        # Appended rather than inserted after `quality`: load_from_raw reads
        # this list POSITIONALLY, and the history snapshots under
        # data/raw/history/ were written with the 9-field shape.
        estimated = "1" if m.get("intelligenceIndexIsEstimated") is True else "0"

        kept.add(model_name)
        rows.append([
            model_name,
            ctx_str,
            provider,
            str(round(float(quality), 2)),
            f"${price}",
            str(speed),
            str(latency),
            str(p_in),
            str(p_out),
            estimated,
            _index_str(m.get("codingIndex")),
            _index_str(m.get("agenticIndex")),
            _index_str(m.get("omniscience")),
        ])

    dropped_no_score = sorted(n for n in skipped["no_score"] if n and n not in kept)
    dropped_no_price = sorted(n for n in skipped["no_price"] if n and n not in kept)
    # Deprecated drops are listed on their own and deliberately NOT folded into
    # distinct_upstream_models: that figure is the denominator the site quotes
    # ("N of M models"), and a superseded model is not one it failed to carry.
    # Counting them here is what makes a flapping flag visible at all — before
    # this, 17 rows came and went hourly and coverage.json never moved.
    dropped_deprecated = sorted(n for n in deprecated if n not in kept)
    _last_coverage.clear()
    _last_coverage.update({
        # One record per model now, not per host x model, so this is a real
        # model count rather than the 428 that used to read as one and was the
        # likely origin of the README's retired "300+ models" claim.
        "upstream_host_model_rows": len(models),
        "distinct_upstream_models": len(kept) + len(dropped_no_score) + len(dropped_no_price),
        "kept": len(kept),
        "skipped_no_score": dropped_no_score,
        "skipped_no_price": dropped_no_price,
        "skipped_deprecated": dropped_deprecated,
        # Flagged deprecated upstream but still published, because the streak
        # is shorter than data/carried.DEPRECATED_DROP_AFTER_SCRAPES.
        "held_deprecated": sorted(n for n in held if n in kept),
    })
    return rows


# Populated by the last _parse_api_response call; written beside the cache so the
# site can say what it is not showing.
_last_coverage: dict = {}

COVERAGE_PATH = Path(__file__).parent / "raw" / "coverage.json"


def _save_coverage() -> None:
    if not _last_coverage:
        return
    COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_PATH.write_text(json.dumps(_last_coverage, indent=2) + "\n")


# ── Public entry point ────────────────────────────────────────────────────────

# Columns that must carry real values for the site to be honest, and the share
# of rows that must be populated. Only three upstream keys currently fail loudly
# when renamed (the two price keys and intelligence_index, because a row missing
# either is dropped outright). Everything else was read as `.get(x) or 0`, so a
# rename degraded silently to a constant: renaming `timescaleData` published 148
# rows with speed and latency zero for every one of them, `model_creators`
# published 148 blank providers, and the context keys published '0' everywhere.
#
# The site then looked ~90% healthy — pareto, treemap, leaderboard and the cost
# calculator all render fine without speed — while the Overview speed-quadrant
# drew an empty panel and the Speed ranking said "No models match these
# filters", blaming the user for an upstream schema change.
#
# Thresholds are deliberately loose: they catch a column that has *collapsed*,
# not one with a few genuine gaps. Speed and latency are absent for a handful of
# real models, so 0.80 leaves room without letting a wholesale zeroing through.
# quality_estimated / coding / agentic / omniscience are deliberately NOT here.
# The first is a boolean (a healthy scrape can legitimately be all-False), and
# AA scores Coding on 144 of 198 models and Agentic on 107 — a density floor on
# a column upstream only partially populates is a guard that cries wolf every
# hour. test_the_estimated_flag_is_not_health_checked_like_a_dense_column pins
# this so nobody "completes" the table later.
_COLUMN_HEALTH = {
    "provider": 0.95,
    "quality":  0.95,
    "price":    0.95,
    "speed":    0.80,
    "latency":  0.80,
    "context":  0.90,
}


def column_health_violations(df, thresholds: dict | None = None) -> list[str]:
    """Columns whose populated share has collapsed — empty list means healthy.

    "Populated" means present, non-null, and not the zero/blank value the
    `or <default>` idiom degrades to. A missing column is itself a violation:
    that is the rename case.
    """
    import pandas as pd

    out: list[str] = []
    for col, floor in (thresholds or _COLUMN_HEALTH).items():
        if col not in df.columns:
            out.append(f"column '{col}' is missing entirely — upstream schema changed")
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            live = series.notna() & (series != 0)
        else:
            text = series.astype(str).str.strip()
            live = series.notna() & (text != "") & (text != "0") & (text.str.lower() != "nan")
        share = float(live.mean()) if len(series) else 0.0
        if share < floor:
            out.append(
                f"column '{col}' is only {share:.0%} populated (floor {floor:.0%}) — "
                f"upstream key renamed or dropped"
            )
    return out


# Columns AA's 2026-09-10 restructure removed from the leaderboard. It replaced
# the two composites with the raw component benchmarks (terminalbenchHard, tau2,
# scicode, ifbench, critpt ...) rather than renaming them, so there is no key
# left to re-point at. They are carried the way the image arena's 35 frozen
# columns are: AA has stopped recomputing these scores, but it did measure them,
# and blanking the columns would throw real numbers away.
#
# The committed CSV alone was not enough memory. A model missing from one
# published scrape lost its scores from that CSV, and nothing could bring them
# back — 43 of 143 hosted coding scores went that way. The sidecar in
# data/carried/ only ever gains rows, so it is the second source.
_CARRIED_COLUMNS = carried.FROZEN_COLUMNS

_BY_TAG = object()   # sentinel: resolve the sidecar from `tag`


def _carry_dropped_columns(df, cached, key: str = "model", tag: str = "scraper",
                           sidecar=_BY_TAG):
    """Fill coding/agentic from the committed CSV, then from the sidecar.

    Joins on the model-name column (`model` in the hosted catalogue, `name` in
    the open-weight one) and fills only holes, in order live > cached CSV >
    sidecar: a value AA does publish always wins, and a model absent from both
    keeps an empty cell rather than inheriting the row above it.

    `sidecar` defaults to the file data/carried.py assigns to `tag`; pass a
    path to use another, or None for none. Reading never writes — the caller
    folds the result back with carried.remember() once the scrape publishes.
    """
    import pandas as pd

    if df is None or df.empty or key not in df.columns:
        return df

    sources = []
    if cached is not None and not getattr(cached, "empty", True) \
            and key in cached.columns:
        sources.append(cached.drop_duplicates(subset=key).set_index(key))
    if sidecar is _BY_TAG:
        sidecar = carried.sidecar_path(tag)
    if sidecar is not None:
        side = carried.load_sidecar(sidecar, key)
        if not side.empty:
            sources.append(side.set_index(key))
    if not sources:
        return df

    filled_cols: list[str] = []
    for col in _CARRIED_COLUMNS:
        live = df[col] if col in df.columns else pd.Series(float("nan"),
                                                           index=df.index)
        out = live.copy()
        for lookup in sources:
            if col not in lookup.columns:
                continue
            prior = pd.Series(pd.to_numeric(lookup[col], errors="coerce")
                              .reindex(df[key]).to_numpy(), index=df.index)
            out = out.where(out.notna(), prior)
        # Only a wholly empty column counts as carried; a few genuine gaps in a
        # live column are not an upstream outage worth announcing.
        if not live.notna().any() and out.notna().any():
            filled_cols.append(col)
        df[col] = out

    if filled_cols:
        filled = int(df[filled_cols].notna().any(axis=1).sum())
        print(f"[{tag}] {', '.join(filled_cols)} carried forward from cache for "
              f"{filled}/{len(df)} models — AA stopped publishing them on "
              f"2026-09-10 (see module docstring)")
    return df


MAX_SHRINK_PCT = 20.0


def _shrink_violations(df) -> list[str]:
    """Refuse a scrape that loses an implausible share of the existing cache.

    data_guard.py already does this, but it runs only from the refresh workflow
    and compares *committed* CSVs. app.py starts this scraper on every Dash boot
    and data/ingest.py freezes a history snapshot, so that path overwrote the
    cache — and the day's history file — with no guard of any kind. This is the
    same rule enforced where the write actually happens.
    """
    try:
        existing = load_cached()
    except Exception:
        return []
    if existing is None or existing.empty:
        return []
    before, now = len(existing), len(df)
    if before == 0:
        return []
    drop = (before - now) / before * 100
    if drop > MAX_SHRINK_PCT:
        return [f"row count {before} -> {now}, a {drop:.0f}% drop "
                f"(limit {MAX_SHRINK_PCT:.0f}%)"]
    return []


def _scrape_and_save() -> bool:
    """
    Fetch fresh data from the AA API and update the cache.
    Returns True on success, False on failure (cache unchanged).
    """
    try:
        resp = requests.get(_PAGE_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()

        models = _extract_models(resp.text)

        # Deprecation hysteresis: which flagged models to keep for now. The
        # state is only persisted below, once this scrape actually publishes,
        # so a refused scrape does not advance anyone's streak.
        try:
            prior = load_cached()
            published = set(prior["model"].astype(str)) if prior is not None \
                and "model" in prior.columns else set()
        except Exception:
            published = set()
        flagged = {(m.get("name") or "").strip() for m in models
                   if m.get("deprecated")} - {""}
        hold, dep_state = carried.deprecation_hysteresis(
            carried.load_deprecation_state(), flagged, published)

        rows = _parse_api_response(models, hold_deprecated=hold)
        if not rows:
            print("[scraper] No valid model rows parsed from API response")
            return False

        df = load_from_raw(rows)
        if df.empty:
            print("[scraper] Parsed DataFrame is empty — skipping cache update")
            return False

        df = _carry_dropped_columns(df, load_cached())

        violations = column_health_violations(df)
        violations += _shrink_violations(df)
        if violations:
            for v in violations:
                print(f"[scraper] {v}")
            print("[scraper] Refusing to publish — cache left unchanged")
            return False

        save_cache(df)
        _save_coverage()
        carried.save_deprecation_state(dep_state)
        carried.remember(df, "scraper")
        if hold:
            print(f"[scraper] {len(hold)} deprecated model(s) held until their "
                  f"streak reaches {carried.DEPRECATED_DROP_AFTER_SCRAPES} scrapes "
                  f"/ {carried.DEPRECATED_DROP_AFTER_HOURS:g}h: {', '.join(sorted(hold))}")
        n_skipped = (len(_last_coverage.get("skipped_no_score", []))
                     + len(_last_coverage.get("skipped_no_price", [])))
        print(f"[scraper] Updated cache with {len(df)} models "
              f"({n_skipped} upstream models not carried)")
        return True

    except requests.RequestException as exc:
        print(f"[scraper] Network error: {exc}")
        return False
    except Exception as exc:
        print(f"[scraper] Unexpected error: {exc}")
        return False


def scrape_and_save() -> bool:
    """Records the outcome so the freshness badge reports real fetch times."""
    ok = _scrape_and_save()
    rows = None
    if ok:
        try:
            cached = load_cached()
            rows = None if cached is None else len(cached)
        except Exception:
            rows = None
    scrape_status.record("hosted", ok, rows)
    return ok


def _scraper_loop(interval_s: int = 3600):
    """Background thread: scrape immediately, then every `interval_s` seconds."""
    scrape_and_save()           # ← run once immediately on startup
    while True:
        time.sleep(interval_s)
        scrape_and_save()


def start_background_scraper(interval_s: int = 3600):
    """Start the periodic scraper as a daemon thread (non-blocking)."""
    t = threading.Thread(target=_scraper_loop, args=(interval_s,), daemon=True)
    t.start()
    return t


# ── Standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    success = scrape_and_save()
    if not success:
        print("[scraper] Falling back to existing cache")
        df = load_cached()
        print(f"[scraper] Cache has {len(df)} models")
    # Exit non-zero on failure so .github/workflows/refresh.yml can see it: the
    # workflow records failures with `python -m data.scraper || failed=...`, which
    # is dead code unless this process actually reports the failure.
    sys.exit(0 if success else 1)
