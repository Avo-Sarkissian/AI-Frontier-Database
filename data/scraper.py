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

import threading
import time

import requests

from data.ingest import load_from_raw, save_cache, load_cached

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

    for hm in host_models:
        model_obj = hm.get("model") or {}

        # Quality
        quality = model_obj.get("intelligence_index")
        if quality is None or quality <= 0:
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

        # Context window
        ctx_tokens = (
            hm.get("context_window_tokens")
            or model_obj.get("context_window_tokens")
            or 0
        )
        ctx_str = hm.get("context_window_formatted") or (
            f"{ctx_tokens // 1000}k" if ctx_tokens >= 1000 else str(ctx_tokens)
        )

        # Price — blended 3:1 output:input ratio, per 1M tokens
        price = hm.get("price_1m_blended_3_to_1")
        if price is None or price <= 0:
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
            }
        else:
            entry = best[key]
            # Keep cheapest price; use that host's speed/latency
            if price < entry["price"]:
                entry["price"]   = price
                entry["speed"]   = speed
                entry["latency"] = latency
            # If same price or more expensive, take faster speed
            elif speed > entry["speed"]:
                entry["speed"]   = speed
                entry["latency"] = latency

    rows = []
    for (model_name, provider), v in best.items():
        rows.append([
            model_name,
            v["ctx"],
            provider,
            str(v["quality"]),
            f"${v['price']}",
            str(v["speed"]),
            str(v["latency"]),
        ])

    return rows


# ── Public entry point ────────────────────────────────────────────────────────

def scrape_and_save() -> bool:
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

        save_cache(df)
        print(f"[scraper] Updated cache with {len(df)} models")
        return True

    except requests.RequestException as exc:
        print(f"[scraper] Network error: {exc}")
        return False
    except Exception as exc:
        print(f"[scraper] Unexpected error: {exc}")
        return False


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
