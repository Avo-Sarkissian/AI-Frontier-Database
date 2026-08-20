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
from data import scrape_status

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


def _extract_models(html: str) -> list[dict]:
    """The leaderboard's metrics records.

    "models" appears TWICE in this payload: the lightweight picker index first
    (6 fields), then the metrics records (94). Selecting on a metrics-only field
    takes the right one — without the predicate the scrape publishes a full row
    count with every metric column empty.
    """
    models = find_array(
        payload_from_html(html), "models",
        where=lambda a: bool(a) and isinstance(a[0], dict) and "intelligenceIndex" in a[0],
    )
    if models is None:
        raise ValueError("no metrics array found for 'models' in RSC payload")
    return models


def _parse_api_response(models: list[dict]) -> list[list]:
    """
    Convert leaderboard records into raw_rows format:
      [model, context, provider, quality, price, speed, latency, price_in, price_out]

    provider = the AI lab / model creator (e.g. Google, Anthropic, OpenAI).
    """
    rows = []
    # Models the catalog cannot carry. Counting them is the point: "148 tracked"
    # silently meant "162 upstream minus 14" until this existed, and a silent
    # drop is how this project has been bitten before.
    skipped: dict[str, set] = {"no_score": set(), "no_price": set()}
    kept: set = set()

    for m in models:
        if m.get("deprecated"):
            continue
        model_name = (m.get("name") or "").strip()
        if not model_name:
            continue

        quality = m.get("intelligenceIndex")
        if not _real(quality) or quality <= 0:
            skipped["no_score"].add(model_name)
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
            skipped["no_price"].add(model_name)
            continue
        price = (3 * p_out + 1 * p_in) / 4
        if price <= 0:
            skipped["no_price"].add(model_name)
            continue

        speed   = m.get("medianOutputTokensPerSecond")
        latency = m.get("medianTimeToFirstTokenSeconds")
        speed   = float(speed) if _real(speed) else 0
        latency = float(latency) if _real(latency) else 0

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
        ])

    dropped_no_score = sorted(n for n in skipped["no_score"] if n and n not in kept)
    dropped_no_price = sorted(n for n in skipped["no_price"] if n and n not in kept)
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

        rows = _parse_api_response(_extract_models(resp.text))
        if not rows:
            print("[scraper] No valid model rows parsed from API response")
            return False

        df = load_from_raw(rows)
        if df.empty:
            print("[scraper] Parsed DataFrame is empty — skipping cache update")
            return False

        violations = column_health_violations(df)
        violations += _shrink_violations(df)
        if violations:
            for v in violations:
                print(f"[scraper] {v}")
            print("[scraper] Refusing to publish — cache left unchanged")
            return False

        save_cache(df)
        _save_coverage()
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
