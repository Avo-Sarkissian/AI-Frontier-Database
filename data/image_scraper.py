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
React Server Components payload the page ships instead.

SCHEMA HISTORY — two shapes, both parsed
----------------------------------------
Until 2026-09-07 each ``textToImage`` record nested its scores in ``elos[]``:
  - global ELO (the ``tag: null`` entry)
  - ~34 per-category ELOs (``tag.displayName``; ``tag.label`` under the old API)

Between 00:00Z and 04:52Z on 2026-09-07 AA rebuilt the page and flattened it:

  ``elos[]``            -> ``elo`` + ``lower95ci`` / ``upper95ci``
  ``pricePer1kImages``  -> ``price``            (same unit, per 1k images)
  ``openWeightsUrl``    -> gone, no replacement
  ~34 per-category ELOs -> gone from every public page

The parser looked for ``elos[]``, found none, dropped all 158 models and
returned no rows — so the run went red rather than republishing frozen numbers,
which is exactly what the 2026-08-09 rework of this file was for. Both shapes
are read now, because AA has reverted a redesign before.

The two columns AA stopped publishing are still real: 51 of 158 models are
open-weights, and the per-category ELOs drive the Image Gen tab's three facets.
They survive only behind the key-gated ``/api/text-to-image/arena/preferences``
endpoint (still ``400 {"error":"User key is required"}``), and no public page
carries them — ``/image/arena`` renders the voting UI, not the leaderboard.
``_merge_cached_columns`` therefore carries them forward from the committed CSV
instead of overwriting 34 populated columns with blanks. Open-weights status is
a durable property of a model, so carrying it is retention, not staleness; the
category ELOs are a retired metric AA no longer scores, and a model AA has
never scored keeps an empty cell rather than an invented one.

Saves to data/raw/aa_image_models.csv. Returns False on any failure.

Run standalone:  python -m data.image_scraper
"""

import re
import sys
import threading
import time
from pathlib import Path

import requests
import pandas as pd

from data import scrape_status
from data.rsc import find_array, payload_from_html
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
# calls; data/rsc.py concatenates those literals and bracket-matches the array
# out of the result. Shared with data/video_scraper.py, whose page is built the
# same way.
_MODELS_KEY = "textToImage"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(label: str) -> str:
    """'Text & Typography' → 'elo_text_typography'"""
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"elo_{s}"


def _extract_models(html: str) -> list[dict]:
    """Pull the textToImage model records out of the page's RSC payload."""
    models = find_array(payload_from_html(html), _MODELS_KEY)
    if models is None:
        raise ValueError(f"no array found for {_MODELS_KEY} in RSC payload")
    return models


# ── Parser ────────────────────────────────────────────────────────────────────

def _tag_label(tag: dict) -> str | None:
    """Category name. displayName is the current field; label was the old API's."""
    return tag.get("displayName") or tag.get("label") or tag.get("slug")


def _model_scores(m: dict) -> tuple[float | None, dict[str, float]]:
    """(global ELO, {category label: ELO}) from either payload shape.

    The flat shape carries one score and no categories; the nested shape puts
    the global score under ``tag: null`` alongside the tagged ones.
    """
    elos = m.get("elos")
    if isinstance(elos, list) and elos:
        global_elo: float | None = None
        cats: dict[str, float] = {}
        for elo_obj in elos:
            if not isinstance(elo_obj, dict):
                continue
            tag = elo_obj.get("tag")
            val = elo_obj.get("elo")
            if val is None:
                continue
            if tag is None:
                global_elo = float(val)
            elif isinstance(tag, dict):
                lbl = _tag_label(tag)
                if lbl:
                    cats[lbl] = float(val)
        return global_elo, cats

    val = m.get("elo")
    return (float(val) if val is not None else None), {}


def _model_price(m: dict):
    """Price per 1k images. Renamed pricePer1kImages -> price on 2026-09-07.

    Read by presence of the key, not by truthiness: a genuine 0 and an absent
    field are different facts, and `or` collapses them.
    """
    for key in ("pricePer1kImages", "price"):
        if key in m:
            return m[key]
    return None


def _model_open_weights(m: dict):
    """True/False when AA says, None when AA no longer publishes it.

    Defaulting to False would assert that FLUX, Qwen and Stable Diffusion are
    closed — a false fact about 51 of 158 models, and one that silently empties
    the Open Weights filter. None means "unknown", and _merge_cached_columns
    fills it from what we last knew.
    """
    if "openWeightsUrl" in m:
        return bool(m.get("openWeightsUrl"))
    return None


def _parse(models: list[dict]) -> pd.DataFrame | None:
    if not models:
        return None

    # Collect all category labels across all models, in first-seen order.
    all_labels: list[str] = []
    seen: set[str] = set()
    for m in models:
        for lbl in _model_scores(m)[1]:
            if lbl not in seen:
                seen.add(lbl)
                all_labels.append(lbl)

    rows = []
    for m in models:
        name = (m.get("name") or "").strip()
        if not name:
            continue
        creator  = m.get("creator") or {}
        provider = _canonical_image_provider((creator.get("name") or "").strip())

        global_elo, cat_elos = _model_scores(m)
        if global_elo is None:
            continue

        row: dict = {
            "model":        name,
            "provider":     provider,
            "elo":          global_elo,
            "price_per_1k": _model_price(m),
            "open_weights": _model_open_weights(m),
        }
        for lbl in all_labels:
            row[_col(lbl)] = cat_elos.get(lbl)
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Model name breaks ELO ties. Without it the order of two models on the same
    # ELO follows AA's payload order, so a tie could swap on any refresh and
    # rewrite a line in a CSV that is committed hourly — the change guard in
    # .github/workflows/refresh.yml would read that as "data moved" and publish.
    df = df.sort_values(["elo", "model"], ascending=[False, True],
                        kind="mergesort").reset_index(drop=True)
    return df


# Columns the live payload is authoritative for. A cached value must never win
# over one AA just published, or the scrape would pin itself to its own history.
_LIVE_COLUMNS = frozenset({"model", "provider", "elo", "price_per_1k"})


def _merge_cached_columns(df: pd.DataFrame,
                          cached: pd.DataFrame | None) -> pd.DataFrame:
    """Fill columns the live payload no longer carries from the committed CSV.

    Joins on model name and touches only holes: a column AA still publishes is
    left exactly as scraped, and a model absent from the cache keeps empty
    cells rather than being handed another model's numbers.
    """
    if cached is None or cached.empty or "model" not in cached.columns:
        return df
    if "model" not in df.columns or df.empty:
        return df

    lookup = cached.drop_duplicates(subset="model").set_index("model")
    carried: list[str] = []

    for col in lookup.columns:
        if col in _LIVE_COLUMNS:
            continue
        prior = pd.Series(lookup[col].reindex(df["model"]).to_numpy(),
                          index=df.index)
        if col in df.columns:
            live = df[col]
            # Nothing live to protect only when the column is entirely empty.
            if not live.notna().any() and prior.notna().any():
                carried.append(col)
            df[col] = live.where(live.notna(), prior)
        else:
            df[col] = prior
            if prior.notna().any():
                carried.append(col)

    if carried:
        filled = int(df[carried].notna().any(axis=1).sum())
        print(f"[image_scraper] {len(carried)} column(s) carried forward from "
              f"cache for {filled}/{len(df)} models — upstream stopped "
              f"publishing them (see module docstring): "
              f"{', '.join(carried[:4])}"
              f"{f' +{len(carried) - 4} more' if len(carried) > 4 else ''}")

    # Cache order first, new columns appended: this CSV is committed hourly and
    # a reshuffled header would rewrite all 158 lines in every diff.
    ordered = [c for c in cached.columns if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    return df[ordered]


MAX_SHRINK_PCT = 20.0

# Share of rows each critical column must actually carry. A rename degrades to a
# constant rather than raising, so without this a schema change publishes a full
# row count with an all-zero column — see data/scraper.py for the hosted case.
# Artificial Analysis's image arena spells some labs two ways in the same feed
# ("Bytedance" 6 models, "ByteDance Seed" 3), so they counted as two providers,
# fragmented the leaderboard, and made get_image_providers() return 39 uniques
# where the real count is 38. The palette already mapped both spellings to one
# colour with an "(alt spelling)" comment — for colour only, never identity.
# Collapsing at parse time makes it one provider everywhere.
_PROVIDER_ALIASES = {
    "bytedance seed":  "Bytedance",
    "stability ai":    "Stability.ai",
    "playground":      "Playground AI",
    "leonardo ai":     "Leonardo.Ai",
    "microsoft azure": "Microsoft AI",
    "xai":             "SpaceXAI",
}


def _canonical_image_provider(name: str) -> str:
    return _PROVIDER_ALIASES.get(name.strip().lower(), name.strip())


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

    try:
        cached = load_cached()
    except Exception:
        cached = None
    df = _merge_cached_columns(df, cached)

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
