"""
Scrapes live model data from the Artificial Analysis API and updates the local CSV cache.

Strategy:
  1. Call https://artificialanalysis.ai/api/data/website/host-models/performance?prompt_length=1000
     This returns a JSON blob with 800+ host-model records containing intelligence_index,
     price, speed, latency, context window, and provider info.
  2. Map the JSON fields to the schema expected by load_from_raw / the CSV cache.
     provider = the AI lab that created the model (Google, Anthropic, OpenAI, …),
     not the hosting platform. When a model is available through multiple API hosts,
     we keep the cheapest price and fastest speed.
  3. Fall back to the existing cache silently on any failure.

Run standalone:  python -m data.scraper
Integrated:      from data.scraper import scrape_and_save; scrape_and_save()
"""

import json
import sys
import threading
import time

import requests

from pathlib import Path

from data.ingest import load_from_raw, save_cache, load_cached
from data import scrape_status

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://artificialanalysis.ai/models",
}

_API_URL = (
    "https://artificialanalysis.ai/api/data/website/host-models/performance"
    "?prompt_length=1000"
)
_TIMEOUT = 45   # seconds — response can be ~20 MB


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_api_response(data: dict) -> list[list]:
    """
    Convert the hostModels list from the AA API into raw_rows format:
      [model, context, provider, quality, price, speed, latency]

    provider = the AI lab / model creator (e.g. Google, Anthropic, OpenAI).
    When the same model is hosted by multiple API providers, we aggregate to
    keep the cheapest price and fastest speed across all of them.
    """
    host_models = data.get("hostModels", [])

    # Aggregate per (model_name, creator_name)
    best: dict[tuple, dict] = {}
    # Models the catalog cannot carry. Counting them is the point: "148 tracked"
    # silently meant "162 upstream minus 14" until this existed, and a silent
    # drop is how this project has been bitten before.
    skipped: dict[str, set] = {"no_score": set(), "no_price": set()}

    for hm in host_models:
        model_obj = hm.get("model") or {}

        # Quality
        quality = model_obj.get("intelligence_index")
        if quality is None or quality <= 0:
            skipped["no_score"].add(model_obj.get("name") or "")
            continue

        # Model name
        model_name = model_obj.get("name") or hm.get("model_label") or ""
        if not model_name:
            continue

        # Provider = AI lab that created the model
        creators = model_obj.get("model_creators") or {}
        provider = (
            creators.get("name") if isinstance(creators, dict) else ""
        ) or ""

        # Context window — the MODEL's, not the host's.
        #
        # `hm` is one host's offering of the model, and a host may serve it with
        # a smaller window than the model supports. Preferring the host record
        # meant the emitted figure was whichever host AA happened to list first
        # (verified: it matched the first host 79/79 times and the cheapest host
        # only 62/79), so Nemotron 3.5 Lightning published 29k against a real
        # 1,000,000 — 34.5x understated — and this catalogue disagreed with
        # data/local_scraper.py, which already reads the canonical value, on 40
        # of 91 shared models.
        #
        # Unlike price/speed/latency there is no cheapest-host convention to
        # honour here: the dedup below never updates ctx, so a host-derived
        # window was not "the cheapest host's window", just an arbitrary one.
        ctx_tokens = (
            model_obj.get("context_window_tokens")
            or hm.get("context_window_tokens")
            or 0
        )
        # Format from the value actually used. Taking the host's preformatted
        # string would reintroduce the mismatch in the label while the number
        # underneath said something else. Millions keep the "1m" form the UI
        # has always shown (static_helpers.ctx_to_k parses both).
        if ctx_tokens >= 1_000_000:
            m = ctx_tokens / 1_000_000
            ctx_str = f"{m:.0f}m" if abs(m - round(m)) < 0.05 else f"{m:.1f}m"
        elif ctx_tokens >= 1000:
            ctx_str = f"{ctx_tokens // 1000}k"
        else:
            ctx_str = str(ctx_tokens)

        # Price — OUR blend, computed here: 3 parts output to 1 part input, per
        # 1M tokens. Input and output are kept alongside it, because the blend
        # is the right basis for cost estimates but is not the number anyone
        # quotes, so the UI shows all three.
        #
        # This used to read `price_1m_blended_3_to_1` first and treat the
        # arithmetic as a fallback. That key is absent from all 443 live
        # records, so every row took the fallback anyway — the read was dead
        # code that made this look like an upstream figure. It is not:
        # Artificial Analysis publishes `price_1m_blended_0_3_1`, which is
        # (3*in + out)/4 — the OPPOSITE weighting, a median 1.91x lower.
        #
        # We keep the output-weighted basis deliberately. Output-only would
        # overstate cost for input-heavy agentic/RAG workloads, re-rank Value
        # toward models with cheap output and expensive input, and break
        # continuity with 90 blended-only history snapshots. But nothing may
        # credit AA for it — see the labels in app.py, docs/app.js and README.md.
        p_in  = hm.get("price_1m_input_tokens")
        p_out = hm.get("price_1m_output_tokens")
        if p_in is None or p_out is None or p_in < 0 or p_out < 0:
            skipped["no_price"].add(model_name)
            continue
        price = (3 * p_out + 1 * p_in) / 4
        if price <= 0:
            skipped["no_price"].add(model_name)
            continue

        # Speed and latency
        ts = hm.get("timescaleData") or {}
        speed   = ts.get("median_output_speed") or 0
        latency = ts.get("median_time_to_first_chunk") or 0

        key = (model_name, provider)
        if key not in best:
            best[key] = {
                "ctx": ctx_str, "quality": quality,
                "price": price, "speed": speed, "latency": latency,
                "price_in": p_in, "price_out": p_out,
            }
        else:
            entry = best[key]
            # Keep the cheapest host's full record so price, speed, and
            # latency all describe the same real API offering. The previous
            # "elif speed > entry['speed']" branch produced synthetic records
            # where price came from host A and speed from host B — a SKU
            # that doesn't exist. We drop that branch entirely.
            if price < entry["price"]:
                entry["price"]     = price
                entry["speed"]     = speed
                entry["latency"]   = latency
                entry["price_in"]  = p_in
                entry["price_out"] = p_out

    rows = []
    for (model_name, provider), v in best.items():
        rows.append([
            model_name,
            v["ctx"],
            provider,
            str(round(v["quality"], 2)),
            f"${v['price']}",
            str(v["speed"]),
            str(v["latency"]),
            "" if v.get("price_in") is None else str(v["price_in"]),
            "" if v.get("price_out") is None else str(v["price_out"]),
        ])

    kept = {name for name, _ in best}
    _last_coverage.clear()
    _last_coverage.update({
        "upstream_records": len(host_models),
        "kept": len(kept),
        "skipped_no_score": sorted(n for n in skipped["no_score"] if n and n not in kept),
        "skipped_no_price": sorted(n for n in skipped["no_price"] if n and n not in kept),
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
        resp = requests.get(_API_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        if "hostModels" not in data:
            print("[scraper] Unexpected API response structure")
            return False

        rows = _parse_api_response(data)
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
