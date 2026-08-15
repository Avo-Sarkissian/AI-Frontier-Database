"""
Scrapes live image generation ELO data from the Artificial Analysis site.

Source: https://artificialanalysis.ai/text-to-image

This used to call /api/text-to-image/arena/preferences directly. That endpoint
is now key-gated — it answers every request with
``400 {"error":"User key is required"}`` — and because the refresh workflow
tolerated scraper failures, the job stayed green and quietly republished the
cache for 29 days (2026-07-11 → 2026-08-09) while the Image Gen tab served
frozen numbers.

The comparison page renders the same data server-side, so we read it from the
React Server Components payload the page ships instead. Each model carries:
  - global ELO (the ``tag: null`` entry in ``elos[]``)
  - ~34 per-category ELOs (``tag.displayName``; this was ``tag.label`` under the
    old API, another reason the old parser could not have survived)

Saves to data/raw/aa_image_models.csv. Returns False on any failure.

Run standalone:  python -m data.image_scraper
"""

import json
import re
import sys
import threading
import time
from pathlib import Path

import requests
import pandas as pd

from data import scrape_status
from static_helpers import csv_safe

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://artificialanalysis.ai/",
}

_PAGE_URL = "https://artificialanalysis.ai/text-to-image"
_TIMEOUT  = 45          # the page is ~2 MB of HTML
_CACHE    = Path(__file__).parent / "raw" / "aa_image_models.csv"

# Next.js streams its RSC payload as a series of self.__next_f.push([1,"…"])
# calls, each carrying a JS string literal. Concatenating those literals yields
# the flight payload that holds the model records.
_RSC_CHUNK_RE = re.compile(r'self\.__next_f\.push\(\[1,\s*("(?:[^"\\]|\\.)*")\]\)')
_MODELS_KEY = '"textToImage":'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(label: str) -> str:
    """'Text & Typography' → 'elo_text_typography'"""
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"elo_{s}"


def _slice_json_array(text: str, start: int) -> str:
    """Return the complete JSON array beginning at ``start``.

    The payload is one long flight string, so we cannot json.loads the tail —
    bracket-match to find where the array ends, skipping brackets inside
    strings.
    """
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unterminated JSON array in RSC payload")


def _extract_models(html: str) -> list[dict]:
    """Pull the textToImage model records out of the page's RSC payload."""
    chunks = _RSC_CHUNK_RE.findall(html)
    if not chunks:
        raise ValueError("no RSC payload found — page structure changed")
    payload = "".join(json.loads(c) for c in chunks)

    # The key also appears as a UI label ("textToImage":"Text to Image"), so
    # take the first occurrence whose value is actually an array.
    at = payload.find(_MODELS_KEY)
    while at != -1:
        cursor = at + len(_MODELS_KEY)
        while cursor < len(payload) and payload[cursor].isspace():
            cursor += 1
        if cursor < len(payload) and payload[cursor] == "[":
            return json.loads(_slice_json_array(payload, cursor))
        at = payload.find(_MODELS_KEY, at + 1)
    raise ValueError(f"no array found for {_MODELS_KEY} in RSC payload")


# ── Parser ────────────────────────────────────────────────────────────────────

def _tag_label(tag: dict) -> str | None:
    """Category name. displayName is the current field; label was the old API's."""
    return tag.get("displayName") or tag.get("label") or tag.get("slug")


def _parse(models: list[dict]) -> pd.DataFrame | None:
    if not models:
        return None

    # Collect all category labels across all models
    all_labels: list[str] = []
    seen: set[str] = set()
    for m in models:
        for elo_obj in m.get("elos", []):
            tag = elo_obj.get("tag")
            if isinstance(tag, dict):
                lbl = _tag_label(tag)
                if lbl and lbl not in seen:
                    seen.add(lbl)
                    all_labels.append(lbl)

    rows = []
    for m in models:
        name = (m.get("name") or "").strip()
        if not name:
            continue
        creator  = m.get("creator") or {}
        provider = (creator.get("name") or "").strip()
        price    = m.get("pricePer1kImages")   # float or None
        ow       = bool(m.get("openWeightsUrl"))

        global_elo: float | None = None
        cat_elos: dict[str, float] = {}
        for elo_obj in m.get("elos", []):
            tag = elo_obj.get("tag")
            val = elo_obj.get("elo")
            if val is None:
                continue
            if tag is None:
                global_elo = float(val)
            elif isinstance(tag, dict):
                lbl = _tag_label(tag)
                if lbl:
                    cat_elos[lbl] = float(val)

        if global_elo is None:
            continue

        row: dict = {
            "model":        name,
            "provider":     provider,
            "elo":          global_elo,
            "price_per_1k": price,
            "open_weights": ow,
        }
        for lbl in all_labels:
            row[_col(lbl)] = cat_elos.get(lbl)
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("elo", ascending=False).reset_index(drop=True)
    return df



MAX_SHRINK_PCT = 20.0

# Share of rows each critical column must actually carry. A rename degrades to a
# constant rather than raising, so without this a schema change publishes a full
# row count with an all-zero column — see data/scraper.py for the hosted case.
_COLUMN_HEALTH = {"model": 0.95, "provider": 0.90, "elo": 0.90}


def _column_violations(df) -> list[str]:
    from data.scraper import column_health_violations
    return column_health_violations(df, _COLUMN_HEALTH)


def _shrink_violations(df) -> list[str]:
    """Refuse a scrape that loses an implausible share of the existing cache.

    data_guard.py enforces this in CI against committed files; this is the same
    rule where the write actually happens, so the Dash-boot path cannot quietly
    replace the cache with a fraction of it.
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


# ── Public entry points ───────────────────────────────────────────────────────

def _scrape_and_save() -> bool:
    """Fetch live data and write to cache CSV. Returns True on success."""
    try:
        resp = requests.get(_PAGE_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        models = _extract_models(resp.text)
    except Exception as exc:
        print(f"[image_scraper] Fetch error: {exc}")
        return False

    df = _parse(models)
    if df is None or df.empty:
        print("[image_scraper] No valid rows parsed")
        return False

    violations = _shrink_violations(df) + _column_violations(df)
    if violations:
        for v in violations:
            print(f"[image_scraper] {v}")
        print("[image_scraper] Refusing to publish — cache left unchanged")
        return False

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    # Sanitised like the hosted catalogue — these files are committed
    # hourly and opened by hand. See static_helpers.csv_safe.
    csv_safe(df).to_csv(_CACHE, index=False)
    print(f"[image_scraper] Saved {len(df)} image models")
    return True


def scrape_and_save() -> bool:
    """Records the outcome so the freshness badge reports real fetch times.

    See data/scrape_status.py: the badge used to show the BUILD time, so one
    succeeding scraper reset the clock for all three datasets.
    """
    ok = _scrape_and_save()
    rows = None
    if ok:
        try:
            cached = load_cached()
            rows = None if cached is None else len(cached)
        except Exception:
            rows = None
    scrape_status.record("image", ok, rows)
    return ok


def load_cached() -> pd.DataFrame | None:
    if _CACHE.exists():
        return pd.read_csv(_CACHE)
    return None


def _loop(interval_s: int = 3600):
    scrape_and_save()
    while True:
        time.sleep(interval_s)
        scrape_and_save()


def start_background_image_scraper(interval_s: int = 3600):
    """Start image scraper as a daemon thread."""
    t = threading.Thread(target=_loop, args=(interval_s,), daemon=True)
    t.start()
    return t


# ── Standalone ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok = scrape_and_save()
    if ok:
        df = load_cached()
        print(df[["model", "provider", "elo", "price_per_1k"]].head(20).to_string())
    else:
        print("[image_scraper] Failed")
    # Exit non-zero on failure so .github/workflows/refresh.yml can see it: the
    # workflow records failures with `python -m data.image_scraper || failed=...`,
    # which is dead code unless this process actually reports the failure.
    sys.exit(0 if ok else 1)
