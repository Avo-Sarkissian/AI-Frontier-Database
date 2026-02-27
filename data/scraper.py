"""
Scrapes the Artificial Analysis leaderboard and updates the local CSV cache.

Strategy:
  1. Fetch https://artificialanalysis.ai/models with a browser-like User-Agent.
  2. Extract __NEXT_DATA__ JSON embedded in the HTML (Next.js SSR).
  3. Walk the page props to find the model rows.
  4. Fall back to the existing cache silently on any failure.

Run standalone:  python -m data.scraper
Integrated:      from data.scraper import scrape_and_save; scrape_and_save()
"""

import json
import re
import threading
import time
from pathlib import Path

import requests

from data.ingest import load_from_raw, save_cache, load_cached

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_URL     = "https://artificialanalysis.ai/models"
_TIMEOUT = 25   # seconds


# ── Parsers ───────────────────────────────────────────────────────────────────

def _extract_next_data(html: str) -> dict | None:
    """Pull the __NEXT_DATA__ JSON blob from a Next.js page."""
    m = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        html, re.S
    )
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _rows_from_next_data(data: dict) -> list[list] | None:
    """
    Navigate Next.js page props to find a list of model records.
    AA stores the leaderboard table under different keys depending on deploy.
    We try a few known paths.
    """
    def _walk(obj, depth=0):
        """Recursively look for a list that looks like model rows."""
        if depth > 8:
            return None
        if isinstance(obj, list) and len(obj) >= 5:
            # Check if it looks like model table rows: list of lists with 7+ items
            if all(isinstance(r, (list, tuple)) and len(r) >= 7 for r in obj[:3]):
                return [list(r) for r in obj]
            # Or list of dicts with model/quality/price keys
            if all(isinstance(r, dict) and "model" in r for r in obj[:3]):
                return _dicts_to_rows(obj)
        if isinstance(obj, dict):
            for v in obj.values():
                result = _walk(v, depth + 1)
                if result:
                    return result
        if isinstance(obj, list):
            for item in obj:
                result = _walk(item, depth + 1)
                if result:
                    return result
        return None

    return _walk(data)


def _dicts_to_rows(records: list[dict]) -> list[list]:
    """Convert list-of-dicts (AA API format) to the raw_rows format expected by load_from_raw."""
    rows = []
    for r in records:
        rows.append([
            r.get("model", r.get("name", "")),
            str(r.get("context", r.get("context_window", ""))),
            r.get("provider", r.get("organization", "")),
            str(r.get("quality", r.get("intelligence_index", r.get("mmlu", "")))),
            f"${r.get('price', r.get('output_price', r.get('blended_price', '')))}",
            str(r.get("speed", r.get("tokens_per_second", r.get("throughput", "")))),
            str(r.get("latency", r.get("time_to_first_token", ""))),
        ])
    return rows


# ── Public entry point ────────────────────────────────────────────────────────

def scrape_and_save() -> bool:
    """
    Fetch fresh data from AA, update cache.
    Returns True on success, False on failure (cache unchanged).
    """
    try:
        resp = requests.get(_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()

        data = _extract_next_data(resp.text)
        if data is None:
            print("[scraper] __NEXT_DATA__ not found in page HTML")
            return False

        rows = _rows_from_next_data(data)
        if not rows:
            print("[scraper] Could not locate model rows in __NEXT_DATA__")
            return False

        df = load_from_raw(rows)
        if df.empty:
            print("[scraper] Parsed DataFrame is empty — skipping cache update")
            return False

        save_cache(df)
        print(f"[scraper] Updated cache with {len(df)} models")
        return True

    except requests.RequestException as exc:
        print(f"[scraper] Network error: {exc}")
        return False
    except Exception as exc:
        print(f"[scraper] Unexpected error: {exc}")
        return False


def _scraper_loop(interval_s: int = 1800):
    """Background thread: scrape every `interval_s` seconds."""
    while True:
        time.sleep(interval_s)
        scrape_and_save()


def start_background_scraper(interval_s: int = 1800):
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
