"""
Scrapes open-weight model specs from the AA API and saves to aa_local_models.csv.
This powers the Run Local tab's model catalog automatically, replacing the old
hand-maintained _MODELS_RAW list in local_models.py.

Fields pulled per model:
  name, family, params_b (total), active_b (active/forward-pass), context_k,
  quality (AA Intelligence Index), license, tags (csv string), moe (bool)

Run standalone:  python -m data.local_scraper
"""

import sys
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
    "Referer": "https://artificialanalysis.ai/models",
}
_API_URL = (
    "https://artificialanalysis.ai/api/data/website/host-models/performance"
    "?prompt_length=1000"
)
_TIMEOUT = 45
_CACHE = Path(__file__).parent / "raw" / "aa_local_models.csv"


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse(data: dict) -> pd.DataFrame | None:
    seen: set[tuple] = set()
    rows = []

    for hm in data.get("hostModels", []):
        model_obj = hm.get("model") or {}

        if not model_obj.get("is_open_weights"):
            continue

        quality = model_obj.get("intelligence_index")
        if not quality or quality <= 0:
            continue

        name = (model_obj.get("name") or "").strip()
        if not name:
            continue

        creators = model_obj.get("model_creators") or {}
        family = (creators.get("name") if isinstance(creators, dict) else "") or "Other"

        key = (name, family)
        if key in seen:
            continue
        seen.add(key)

        params_b = model_obj.get("parameters")        # total, integer billions
        active_b = model_obj.get("inference_parameters_active_billions")
        if params_b is None or params_b <= 0:
            continue
        # Dense models don't expose active_b separately — fall back to total
        if active_b is None or active_b <= 0:
            active_b = params_b

        ctx_tokens = model_obj.get("context_window_tokens") or 0
        context_k  = max(1, round(ctx_tokens / 1000)) if ctx_tokens else 128

        # Tags derived from modality + model type flags
        tags = []
        if model_obj.get("input_modality_image"):
            tags.append("vision")
        if model_obj.get("input_modality_speech"):
            tags.append("audio")
        if model_obj.get("reasoning_model"):
            tags.append("reasoning")
        name_lower = name.lower()
        if any(w in name_lower for w in ["coder", "code", "codex", "coding"]):
            tags.append("code")

        # MoE when total params are materially larger than active params
        moe = (float(params_b) / float(active_b)) > 1.5

        license_name = (model_obj.get("license_name") or "").strip() or "Unknown"

        rows.append({
            "name":      name,
            "family":    family,
            "params_b":  float(params_b),
            "active_b":  float(active_b),
            "context_k": context_k,
            "quality":   round(float(quality), 2),
            "license":   license_name,
            "tags":      ",".join(tags),
            "moe":       moe,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("quality", ascending=False).reset_index(drop=True)
    return df


# ── Public entry points ───────────────────────────────────────────────────────

def scrape_and_save() -> bool:
    """Fetch open-weight model specs and write to cache CSV. Returns True on success."""
    try:
        resp = requests.get(_API_URL, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[local_scraper] Fetch error: {exc}")
        return False

    df = _parse(data)
    if df is None or df.empty:
        print("[local_scraper] No valid open-weight model rows parsed")
        return False

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_CACHE, index=False)
    print(f"[local_scraper] Saved {len(df)} open-weight models")
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


def start_background_local_scraper(interval_s: int = 3600):
    """Start the local-model scraper as a daemon thread."""
    t = threading.Thread(target=_loop, args=(interval_s,), daemon=True)
    t.start()
    return t


# ── Standalone ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok = scrape_and_save()
    if ok:
        df = load_cached()
        print(df[["name", "family", "params_b", "active_b", "quality"]].head(30).to_string())
    else:
        print("[local_scraper] Failed — no cache updated")
    # Exit non-zero on failure so .github/workflows/refresh.yml can see it: the
    # workflow records failures with `python -m data.local_scraper || failed=...`,
    # which is dead code unless this process actually reports the failure.
    sys.exit(0 if ok else 1)
