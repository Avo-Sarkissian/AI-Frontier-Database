"""
Scrapes live image generation ELO data from the Artificial Analysis Image Arena API.
Endpoint: https://artificialanalysis.ai/api/text-to-image/arena/preferences?supports_image_input=false

Each model has:
  - global ELO (tag=null entry in elos[])
  - 15 per-category ELOs (tag.label entries)

Saves to data/raw/aa_image_models.csv. Falls back silently on any failure.

Run standalone:  python -m data.image_scraper
"""

import re
import threading
import time
from pathlib import Path

import requests
import pandas as pd

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://artificialanalysis.ai/text-to-image",
}

_API_URL  = "https://artificialanalysis.ai/api/text-to-image/arena/preferences?supports_image_input=false"
_TIMEOUT  = 30
_CACHE    = Path(__file__).parent / "raw" / "aa_image_models.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(label: str) -> str:
    """'Text & Typography' → 'elo_text_typography'"""
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"elo_{s}"


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse(data: dict) -> pd.DataFrame | None:
    models = data.get("models", [])
    if not models:
        return None

    # Collect all category labels across all models
    all_labels: list[str] = []
    seen: set[str] = set()
    for m in models:
        for elo_obj in m.get("elos", []):
            tag = elo_obj.get("tag")
            if tag and isinstance(tag, dict):
                lbl = tag["label"]
                if lbl not in seen:
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
                cat_elos[tag["label"]] = float(val)

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


# ── Public entry points ───────────────────────────────────────────────────────

def scrape_and_save() -> bool:
    """Fetch live data and write to cache CSV. Returns True on success."""
    try:
        resp = requests.get(_API_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[image_scraper] Fetch error: {exc}")
        return False

    df = _parse(data)
    if df is None or df.empty:
        print("[image_scraper] No valid rows parsed")
        return False

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_CACHE, index=False)
    print(f"[image_scraper] Saved {len(df)} image models")
    return True


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
