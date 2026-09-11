"""
Scrapes open-weight model specs from the AA leaderboard and saves to
aa_local_models.csv. This powers the Run Local tab's model catalog.

Source: https://artificialanalysis.ai/models/<slug>  (entry point: /leaderboards/models)

WHY NOT THE API ENDPOINT
------------------------
This used to read ``/api/data/website/host-models/performance`` — the same URL
data/scraper.py uses for the hosted LLM catalogue. That endpoint returns
host x model ROWS, so a model appears only if some commercial provider sells API
access to it. For a tab captioned "open-weight models you can run on your own
hardware" that is exactly the wrong filter: a model nobody hosts is often
precisely the one you were going to run yourself.

The gap was not marginal. On 2026-08-17 the endpoint carried 169 models and this
catalogue published 100, while the leaderboard carried 177 open-weight scored
models — 77 missing, 65 of them because no host sells them at all. The top miss
was Qwen3.8 27B (Intelligence Index 52.0, Apache 2.0, 27B dense, 256k context,
released 2026-08-14), which would rank 5th here and is aimed squarely at the
24 GB consumer cards this tab exists to serve. data/pending_models.py had been
carrying it as an unbenchmarked curated entry for days after AA scored it,
because the self-expiry check compares against a scrape that structurally could
not see it.

The page carries every model AA tracks with its metrics attached, including the
unhosted ones, and everything this catalogue needs: total and active
parameters, context window, Intelligence Index, licence, creator and the
modality flags the tags are built from. Nothing was lost in the move — the new
catalogue is a strict superset of the old one.

WHY A MODEL PAGE AND NOT THE LEADERBOARD
---------------------------------------
AA restructured /leaderboards/models on 2026-09-10 (see data/scraper.py for the
full account) and the slimmed records it now serves dropped totalParameters,
activeParameters and every inputModality flag. What is left of model size is
`paramClass` — "small" / "medium" / "large" — which cannot size a model against
a card's VRAM or drive the roofline in data/run_local.py. This tab exists to
answer "what can I run on my hardware"; a coarse bucket does not answer it.

Those fields were not deleted, only moved. Every page under /models/<slug>
ships the full 88-field catalogue for ALL models, not just the one named in the
URL, with totalParameters renamed to `parameters`, activeParameters to
`inferenceParametersActiveBillions` and modelCreatorName to a nested `creator`
object. So this module makes two hops: the leaderboard names a live model, and
that model's page carries the catalogue. The slug is read at run time rather
than pinned, because a pinned slug dies the day AA deprecates that model — the
failure mode that has already cost this project three endpoints.

A second benefit, restored: data/scraper.py and this module used to hit the
byte-identical URL, so the hosted and local datasets always failed together and
the freshness badge could never distinguish them (see data/scrape_status.py).
The catalogue each one parses now comes from a different page again.

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

from data import scrape_status
from data.rsc import find_array, payload_from_html
from data.scraper import _real as _aa_real
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
_PAGE_URL = "https://artificialanalysis.ai/leaderboards/models"
# Any model's own page ships the FULL catalogue, not just the model in the URL,
# so this is an entry point rather than a lookup — the slug is discovered at
# run time (see _entry_slug) because pinning one means the scrape dies the day
# that model is deprecated.
_CATALOGUE_URL = "https://artificialanalysis.ai/models/{slug}"
_TIMEOUT = 45          # the leaderboard is ~2.8 MB, a model page ~3.6 MB
_CACHE = Path(__file__).parent / "raw" / "aa_local_models.csv"



# ── Parser ────────────────────────────────────────────────────────────────────

def _entry_slug(payload: str) -> str:
    """Any live model's slug, taken from the leaderboard's picker index.

    Every /models/<slug> page serves the same 646-model catalogue, so this only
    has to name a page that exists. Deprecated models are skipped because their
    pages are the ones AA eventually removes.
    """
    index = find_array(
        payload, "models",
        where=lambda a: bool(a) and isinstance(a[0], dict) and "slug" in a[0],
    )
    if not index:
        raise ValueError("no model index found for 'models' in RSC payload")
    for entry in index:
        if isinstance(entry, dict) and entry.get("slug") and not entry.get("deprecated"):
            return entry["slug"]
    raise ValueError("model index carries no live slug")


def _extract_models(html: str) -> list[dict]:
    """The full catalogue records from a /models/<slug> page.

    ``models`` appears more than once in this payload. The one wanted is the
    88-field catalogue: it is the only array carrying ``parameters``, which is
    also the reason this module reads a model page at all. The slimmed
    leaderboard array would satisfy an ``isOpenWeights`` predicate and publish a
    full row count with no parameter counts behind it — the silent-degradation
    shape this project keeps paying for.
    """
    models = find_array(
        payload_from_html(html), "models",
        where=lambda a: bool(a) and isinstance(a[0], dict) and "parameters" in a[0],
    )
    if models is None:
        raise ValueError("no catalogue array found for 'models' in RSC payload")
    return [m for m in models if isinstance(m, dict)]


def _creator_name(m: dict) -> str:
    """The lab, from whichever shape the page ships.

    /models/<slug> nests it as a ``creator`` object; the leaderboard's slim
    array still uses the flat ``modelCreatorName``. Reading only one of them is
    how a rename published blank providers before.
    """
    creator = m.get("creator")
    if isinstance(creator, dict):
        name = (creator.get("name") or "").strip()
        if name:
            return name
    elif isinstance(creator, str) and creator.strip() and creator != "$undefined":
        return creator.strip()
    return (m.get("modelCreatorName") or "").strip() or "Other"


def _num(v) -> float | None:
    """A real number from an RSC field, or None.

    Guards the "$undefined" string and JSON null alike; bools are rejected
    because `True` would otherwise arrive as a 1-billion-parameter model.
    """
    if not _aa_real(v) or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse(models: list[dict]) -> pd.DataFrame | None:
    seen: set[tuple] = set()
    rows = []

    for m in models:
        if not m.get("isOpenWeights"):
            continue
        # Deprecated models stay on the leaderboard for history. The tab answers
        # "what can I run today", so they are dropped here rather than ranked.
        if m.get("deprecated"):
            continue

        # _real() first: RSC encodes JS undefined as the STRING "$undefined",
        # so an unpublished score arrives truthy and `<= 0` raises TypeError
        # against a str, taking the whole catalogue down with it.
        quality = m.get("intelligenceIndex")
        if not _aa_real(quality) or not isinstance(quality, (int, float)) \
                or isinstance(quality, bool) or quality <= 0:
            continue

        name = (m.get("name") or "").strip()
        if not name:
            continue

        family = _creator_name(m)

        key = (name, family)
        if key in seen:
            continue
        seen.add(key)

        # Renamed by AA's 2026-09-10 restructure: totalParameters ->
        # parameters, activeParameters -> inferenceParametersActiveBillions.
        # Both are still billions, so nothing downstream rescales.
        params_b = _num(m.get("parameters"))
        active_b = _num(m.get("inferenceParametersActiveBillions"))
        if params_b is None or params_b <= 0:
            continue
        # Dense models report no active count at all — 98 of 193 open-weight
        # rows — so a null falls back to total rather than dropping the row.
        if active_b is None or active_b <= 0:
            active_b = params_b

        ctx_tokens = _num(m.get("contextWindowTokens")) or 0
        context_k  = max(1, round(ctx_tokens / 1000)) if ctx_tokens else 128

        # Tags derived from modality + model type flags
        tags = []
        if m.get("inputModalityImage"):
            tags.append("vision")
        if m.get("inputModalitySpeech"):
            tags.append("audio")
        if m.get("isReasoning"):
            tags.append("reasoning")
        name_lower = name.lower()
        if any(w in name_lower for w in ["coder", "code", "codex", "coding"]):
            tags.append("code")

        # MoE when total params are materially larger than active params
        moe = (float(params_b) / float(active_b)) > 1.5

        license_name = (m.get("licenseName") or "").strip() or "Unknown"

        # AA's 2026-09 intelligence overhaul — same three sibling indices and
        # the same estimated flag as the hosted catalogue (data/scraper.py).
        # The share is HIGHER here: 147 of 190 open-weight models carry an
        # estimated index against 105 of 198 hosted, because AA re-runs the
        # full eval suite on frontier API models first. NaN, never 0: 0 is a
        # real score on all three scales, and AA-Omniscience is negative for
        # most of the catalogue by construction.
        def _idx(v):
            # Rounded like `quality` above: this CSV is committed hourly, and
            # full float precision (74.757248...) churns lines on noise.
            if not _aa_real(v) or isinstance(v, bool):
                return float("nan")
            try:
                return round(float(v), 2)
            except (TypeError, ValueError):
                return float("nan")

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
            "quality_estimated": m.get("intelligenceIndexIsEstimated") is True,
            "coding":      _idx(m.get("codingIndex")),
            "agentic":     _idx(m.get("agenticIndex")),
            "omniscience": _idx(m.get("omniscience")),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("quality", ascending=False).reset_index(drop=True)
    return df


MAX_SHRINK_PCT = 20.0

# Share of rows each critical column must actually carry. A rename degrades to a
# constant rather than raising, so without this a schema change publishes a full
# row count with an all-zero column — see data/scraper.py for the hosted case.
# The overhaul columns are absent on purpose — see the note in data/scraper.py:
# quality_estimated is boolean and coding/agentic are sparse upstream.
_COLUMN_HEALTH = {"name": 0.95, "family": 0.90, "params_b": 0.90, "quality": 0.80}


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
    """Fetch open-weight model specs and write to cache CSV. Returns True on success."""
    try:
        # Two hops. The leaderboard names a live model; that model's own page
        # carries the catalogue with the parameter counts this tab is built on.
        index = requests.get(_PAGE_URL, headers=_HEADERS, timeout=_TIMEOUT)
        index.raise_for_status()
        slug = _entry_slug(payload_from_html(index.text))

        page = requests.get(_CATALOGUE_URL.format(slug=slug),
                            headers=_HEADERS, timeout=_TIMEOUT)
        page.raise_for_status()
        models = _extract_models(page.text)
    except Exception as exc:
        print(f"[local_scraper] Fetch error: {exc}")
        return False

    df = _parse(models)
    if df is None or df.empty:
        print("[local_scraper] No valid open-weight model rows parsed")
        return False

    # AA dropped codingIndex and agenticIndex on 2026-09-10 — see the note in
    # data/scraper.py. Same treatment here, joined on this CSV's name column.
    from data.scraper import _carry_dropped_columns
    df = _carry_dropped_columns(df, load_cached(), key="name", tag="local_scraper")

    violations = _shrink_violations(df) + _column_violations(df)
    if violations:
        for v in violations:
            print(f"[local_scraper] {v}")
        print("[local_scraper] Refusing to publish — cache left unchanged")
        return False

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    # Sanitised like the hosted catalogue — these files are committed
    # hourly and opened by hand. See static_helpers.csv_safe.
    csv_safe(df).to_csv(_CACHE, index=False)
    print(f"[local_scraper] Saved {len(df)} open-weight models")
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
    scrape_status.record("local", ok, rows)
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
