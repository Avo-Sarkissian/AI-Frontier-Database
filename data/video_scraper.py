"""Scrapes the live Artificial Analysis Video Arena.

Source: https://artificialanalysis.ai/video/models  (+ /video/providers)

WHY THIS EXISTS
---------------
The Video Gen tab was the last hand-maintained catalogue in the dashboard.
data/video_models.py was a literal list of 20 rows whose content had not changed
since 2026-03-13 — Veo 2/3, Sora HD, Gen-4, Kling 1.6, SVD 1.1 — sitting under a
header badge that reports an hourly refresh, while the LLM tabs carried GPT-5.6
and Claude Opus 5. Its `quality` numbers were hand-typed "approximate" human
preference scores with no published source, and they fed a Pareto frontier.

Artificial Analysis does publish this data. There is no JSON endpoint — every
``/api/data/website/video-*`` path answers 404 — but ``/video/models`` renders
all six leaderboards server-side into its RSC payload, exactly the way
``/image/models`` does. data/rsc.py holds the parsing the two share.

WHAT THE PAGE CARRIES
---------------------
Six independent Elo pools, because AA never compares across them: text-to-video,
image-to-video, each of those again with audio, and video-to-video with and
without audio. A model's price differs per pool — Veo 3.1 is $12/min silent and
$24/min with audio; Vidu Q3 Pro is $6/min from text and $9.60/min from an image —
so there is no single "the price of this model" to publish, and a scraper that
picked one arena and called it "video" would repeat the narrowing this project
already paid for on the Run Local tab.

We carry the two silent generation arenas as peers: 89 text-to-video models, 84
image-to-video, 109 distinct. Each keeps its own Elo, its own price and its own
29 per-category Elos. The audio arenas are read only to answer a capability
question — does this model generate synced audio, and what does that cost —
because with 33 models they are too thin to rank as a third mode. video-to-video
(8 models) is left alone entirely: it is below data_guard's 50-row floor and is
an editing benchmark, not a generation one.

WHAT IS NOT HERE, AND WHY
-------------------------
The old dataset's `max_res`, `max_duration_s` and its 0-100 `quality` have no
upstream equivalent and are gone rather than carried forward as invented
constants. Elo replaces quality: AA computes it by Bradley-Terry MLE over
pairwise human votes, rescaled to an Elo-like range, recomputed hourly.

`gen_time_s` survives, but only where it was actually measured. AA speed-tests a
handful of endpoints (currently 6 models of the 109) and publishes a median over
14 trailing days on /video/providers. We read the endpoint AA itself marks
``isRepresentative`` so the time, the host and the reading describe one real
offering — the "synthetic SKU" mistake data/scraper.py documents. Every other
row's gen_time_s is null, and the charts render it as absent rather than as
zero. That second fetch is strictly optional: if it fails, the catalogue still
publishes.

Saves to data/raw/aa_video_models.csv. Returns False on any failure.

Run standalone:  python -m data.video_scraper
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

_PAGE_URL      = "https://artificialanalysis.ai/video/models"
_PROVIDERS_URL = "https://artificialanalysis.ai/video/providers"
_TIMEOUT       = 45          # the page is ~2.3 MB of HTML
_CACHE         = Path(__file__).parent / "raw" / "aa_video_models.csv"

# The two arenas the tab presents as peer modes, and the column prefix each
# writes. Keep these keys in sync with data/video_models.VIDEO_MODES.
_MODES = (("t2v", "textToVideo"), ("i2v", "imageToVideo"))

# Read for capability only — see the module docstring.
_AUDIO_KEYS = ("textToVideoAudio", "imageToVideoAudio")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(mode: str, slug: str) -> str:
    """('t2v', 'moving-camera') → 'elo_t2v_moving_camera'."""
    s = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    return f"elo_{mode}_{s}"


def _tag_slug(tag: dict) -> str | None:
    """Stable key for a category.

    Prefer `slug`: AA has already renamed a display label once (`tag.label` →
    `tag.displayName` on the image arena, which is why data/image_scraper.py
    carries a three-way fallback). The slug is the identifier that survived that
    rename, so deriving column names from it keeps a re-labelling from churning
    the whole schema.
    """
    return tag.get("slug") or tag.get("label") or tag.get("displayName")


def _global_elo(record: dict) -> float | None:
    """The `tag: null` entry in elos[] — the model's overall Elo in that arena.

    The record's own `overallElo` field cannot be used: it is an RSC back
    reference ("$f:props:children:2:props:textToVideo:0:elos:0"), not a number.
    """
    for entry in record.get("elos") or []:
        if entry.get("tag") is None and entry.get("elo") is not None:
            return float(entry["elo"])
    return None


def _category_elos(record: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for entry in record.get("elos") or []:
        tag, val = entry.get("tag"), entry.get("elo")
        if isinstance(tag, dict) and val is not None:
            slug = _tag_slug(tag)
            if slug:
                out[slug] = float(val)
    return out


def _price(record: dict) -> float | None:
    """Published USD to generate one minute at the arena's default settings.

    Nullable upstream (7 of 89 text-to-video models have no published rate), and
    kept null rather than zero-filled: data/image_models.py fills a missing price
    with 0.0 and components/charts/image_scatter.py:43-52 documents the result —
    "no published price" and "genuinely free" become indistinguishable, so the
    UI printed "free" for models nobody can price.
    """
    val = record.get("pricePerMinute")
    if val is None:
        return None
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _extract_arenas(html: str) -> dict[str, list[dict]]:
    """The six leaderboards, keyed by their RSC key. Missing ones come back []."""
    payload = payload_from_html(html)
    keys = [k for _, k in _MODES] + list(_AUDIO_KEYS)
    return {k: (find_array(payload, k) or []) for k in keys}


def _generation_times(html: str) -> dict[str, dict]:
    """model slug → {gen_time_s, gen_time_host} from AA's representative endpoint.

    One model is served by several hosts at wildly different speeds (MiniMax H3
    is 16s on MachGen and 127s on Runware), so there is no single true number.
    AA flags one endpoint per model as `isRepresentative`; we take that one and
    record which host it was, rather than averaging hosts into a figure that
    describes no purchasable offering.
    """
    hosts = find_array(payload_from_html(html), "hostModels") or []
    out: dict[str, dict] = {}
    for hm in hosts:
        if not hm.get("isRepresentative"):
            continue
        slug = ((hm.get("model") or {}).get("slug") or "").strip()
        secs = (hm.get("performance") or {}).get("medianGenerationTimeSeconds")
        if not slug or secs is None:
            continue
        try:
            secs = float(secs)
        except (TypeError, ValueError):
            continue
        if secs <= 0:
            continue
        out[slug] = {
            "gen_time_s":    secs,
            "gen_time_host": ((hm.get("host") or {}).get("name") or "").strip(),
        }
    return out


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse(arenas: dict[str, list[dict]],
           gen_times: dict[str, dict] | None = None) -> pd.DataFrame | None:
    """One row per distinct model across the two silent generation arenas."""
    gen_times = gen_times or {}

    # Identity comes from the slug, not the display name: the same model appears
    # in several arenas and the slug is what ties those records together.
    rows: dict[str, dict] = {}
    order: list[str] = []

    for mode, key in _MODES:
        for rec in arenas.get(key) or []:
            slug = (rec.get("slug") or "").strip()
            name = (rec.get("name") or "").strip()
            if not slug or not name:
                continue
            elo = _global_elo(rec)
            if elo is None:
                continue        # no score in this arena — nothing to publish
            if slug not in rows:
                creator = rec.get("creator") or {}
                family  = rec.get("family") or {}
                rows[slug] = {
                    "model":        name,
                    "slug":         slug,
                    "provider":     _canonical_video_provider(
                                        (creator.get("name") or "").strip()),
                    "family":       (family.get("name") or "").strip(),
                    "release_date": (rec.get("releaseDate") or "").strip(),
                    # AA retires preview/dated variants but keeps them ranked.
                    # Carried so the UI can default to shipping models without
                    # deleting the history a comparison needs.
                    "is_current":   bool(rec.get("isCurrent")),
                    "open_weights": bool(rec.get("openWeightsUrl")),
                }
                order.append(slug)
            row = rows[slug]
            row[f"elo_{mode}"] = elo
            row[f"price_per_min_{mode}"] = _price(rec)
            for cat_slug, val in _category_elos(rec).items():
                row[_col(mode, cat_slug)] = val

    if not rows:
        return None

    # Audio is a capability, not a third mode: presence in either audio arena
    # means the model emits synced sound, and its audio arena price is what that
    # costs. Models that exist ONLY in an audio arena are not added here — they
    # have no silent Elo to rank against.
    for key in _AUDIO_KEYS:
        for rec in arenas.get(key) or []:
            row = rows.get((rec.get("slug") or "").strip())
            if row is None:
                continue
            row["audio"] = True
            price = _price(rec)
            if price is not None:
                row["price_per_min_audio"] = min(
                    price, row.get("price_per_min_audio", price))

    for slug, row in rows.items():
        row.setdefault("audio", False)
        row.update(gen_times.get(slug, {}))

    df = pd.DataFrame([rows[s] for s in order])
    for col in ("elo_t2v", "elo_i2v", "price_per_min_t2v", "price_per_min_i2v",
                "price_per_min_audio", "gen_time_s", "gen_time_host"):
        if col not in df.columns:
            df[col] = pd.NA
    df["audio"] = df["audio"].fillna(False).astype(bool)

    # Rank by the primary arena so the committed CSV is readable by hand, with
    # image-to-video-only models after it rather than dropped.
    return (df.sort_values("elo_t2v", ascending=False, na_position="last")
              .reset_index(drop=True))


# Upstream spelling drift silently greys a provider out — components/charts/
# constants.py:120-128 exists because an AA rename dropped Microsoft into the
# default bucket. Only genuine spellings of the SAME entity belong here: AA's
# "Alibaba-ATH" and "Alibaba" are listed as separate creators upstream and are
# left separate, because merging them would be an identity claim the source does
# not make.
_PROVIDER_ALIASES = {
    "xai":             "SpaceXAI",
    "microsoft azure": "Microsoft",
    "kuaishou":        "KlingAI",
}


def _canonical_video_provider(name: str) -> str:
    return _PROVIDER_ALIASES.get(name.strip().lower(), name.strip())


# ── Write-time guards ─────────────────────────────────────────────────────────

MAX_SHRINK_PCT = 20.0

# Share of rows each column must actually carry. A renamed upstream key degrades
# to a constant rather than raising, so without this a schema change publishes a
# full row count with an all-null column — see data/scraper.py for the hosted
# case that made the site look 90% healthy while two panels were empty.
#
# The Elo and price floors are deliberately well below today's coverage (t2v Elo
# 82%, i2v Elo 77%, t2v price 74%): a model is only in one arena if AA ran it
# there, so genuine gaps are normal and the guard is looking for collapse, not
# for gaps. Booleans are not listed — column_health_violations counts a numeric
# zero as unpopulated, so `open_weights` (14 True of 109) would read as 87%
# missing rather than as mostly-False.
_COLUMN_HEALTH = {
    "model":             0.99,
    "provider":          0.95,
    "slug":              0.99,
    "release_date":      0.90,
    "elo_t2v":           0.60,
    "elo_i2v":           0.55,
    "price_per_min_t2v": 0.50,
}


def _column_violations(df) -> list[str]:
    from data.scraper import column_health_violations
    return column_health_violations(df, _COLUMN_HEALTH)


def _shrink_violations(df) -> list[str]:
    """Refuse a scrape that loses an implausible share of the existing cache.

    data_guard.py enforces this in CI against committed files; this is the same
    rule where the write actually happens, so no path can quietly replace the
    cache with a fraction of it.
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

def _fetch(url: str) -> str:
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _scrape_and_save() -> bool:
    """Fetch live data and write to cache CSV. Returns True on success."""
    try:
        arenas = _extract_arenas(_fetch(_PAGE_URL))
    except Exception as exc:
        print(f"[video_scraper] Fetch error: {exc}")
        return False

    # Enrichment only. A speed page that moves or fails must not cost us the
    # catalogue — 103 of 109 models have no measured time either way.
    gen_times: dict[str, dict] = {}
    try:
        gen_times = _generation_times(_fetch(_PROVIDERS_URL))
    except Exception as exc:
        print(f"[video_scraper] Generation times unavailable ({exc}) — "
              f"publishing catalogue without them")

    df = _parse(arenas, gen_times)
    if df is None or df.empty:
        print("[video_scraper] No valid rows parsed")
        return False

    violations = _shrink_violations(df) + _column_violations(df)
    if violations:
        for v in violations:
            print(f"[video_scraper] {v}")
        print("[video_scraper] Refusing to publish — cache left unchanged")
        return False

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    # Sanitised like every other committed catalogue — these files are written
    # hourly and opened by hand. See static_helpers.csv_safe.
    csv_safe(df).to_csv(_CACHE, index=False)
    print(f"[video_scraper] Saved {len(df)} video models "
          f"({int(df['elo_t2v'].notna().sum())} text-to-video, "
          f"{int(df['elo_i2v'].notna().sum())} image-to-video, "
          f"{int(df['gen_time_s'].notna().sum())} speed-tested)")
    return True


def scrape_and_save() -> bool:
    """Records the outcome so the freshness badge reports real fetch times.

    See data/scrape_status.py: the badge used to show the BUILD time, so one
    succeeding scraper reset the clock for every dataset.
    """
    ok = _scrape_and_save()
    rows = None
    if ok:
        try:
            cached = load_cached()
            rows = None if cached is None else len(cached)
        except Exception:
            rows = None
    scrape_status.record("video", ok, rows)
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


def start_background_video_scraper(interval_s: int = 3600):
    """Start the video scraper as a daemon thread."""
    t = threading.Thread(target=_loop, args=(interval_s,), daemon=True)
    t.start()
    return t


# ── Standalone ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok = scrape_and_save()
    if ok:
        df = load_cached()
        print(df[["model", "provider", "elo_t2v", "elo_i2v",
                  "price_per_min_t2v"]].head(20).to_string())
    else:
        print("[video_scraper] Failed")
    # Exit non-zero on failure so .github/workflows/refresh.yml can see it: the
    # workflow records failures with `python -m data.video_scraper || failed=...`,
    # which is dead code unless this process actually reports the failure.
    sys.exit(0 if ok else 1)
