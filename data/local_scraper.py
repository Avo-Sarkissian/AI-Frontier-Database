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

SPECS BY SLUG (2026-10-10)
-------------------------
The full-catalogue pages went away in turn: from ~2026-10-05 a /models/<slug>
page ships its own record plus ~25 headline models, and the entry-page scrape
failed on most hourly runs while the tab served a stale cache. So the two
halves are now read separately. Scores and flags come from the leaderboard,
which still lists every model and matched the old catalogue field-for-field.
The four static facts it lacks — parameters, active parameters, modality flags,
licence — live in data/carried/local_specs.json, keyed by slug. Each run reads
the page of every open-weight model the cache has not seen (at most
_MAX_NEW_PAGES) plus a rotating _ROTATE_PER_RUN known ones, and every record
those pages carry refreshes the cache. A newly released model appears within
one run; a page that fails costs that model one run, never the catalogue.

A second benefit, restored: data/scraper.py and this module used to hit the
byte-identical URL, so the hosted and local datasets always failed together and
the freshness badge could never distinguish them (see data/scrape_status.py).
The catalogue each one parses now comes from a different page again.

Fields pulled per model:
  name, family, params_b (total), active_b (active/forward-pass), context_k,
  quality (AA Intelligence Index), license, tags (csv string), moe (bool)

Run standalone:  python -m data.local_scraper
"""

import json
import sys
import threading
import time
from pathlib import Path

import requests
import pandas as pd

from data import carried, scrape_status
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
# A model's own page: its full record, plus ~25 headline models' records.
_CATALOGUE_URL = "https://artificialanalysis.ai/models/{slug}"
# Spec cache under data/carried/ — see SPECS BY SLUG in the module docstring.
_SPECS_FILE = "local_specs.json"
_SPEC_FIELDS = ("parameters", "inferenceParametersActiveBillions",
                "inputModalityImage", "inputModalitySpeech", "licenseName")
_MAX_NEW_PAGES = 25      # model pages fetched per run for models with no specs yet
_ROTATE_PER_RUN = 2      # known models re-read per run, so corrections land (~8 days/cycle)
_PAGE_DELAY_S = 0.3
_TIMEOUT = 45          # the leaderboard is ~2.8 MB, a model page ~3.6 MB
_CACHE = Path(__file__).parent / "raw" / "aa_local_models.csv"
# Under data/carried/ beside the hosted scraper's deprecation_state.json, and
# separate from it — see _scrape_and_save.
_DEPRECATION_STATE_FILE = "local_deprecation_state.json"



# ── Parser ────────────────────────────────────────────────────────────────────

def _leaderboard(payload: str) -> list[dict]:
    """The leaderboard's metrics records — every model AA tracks, scored.

    Matched field-for-field against the old full catalogue on 2026-10-10: name,
    Intelligence Index, estimated flag, reasoning flag, context window,
    deprecated, omniscience and isOpenWeights agreed on all 372 open-weight
    models. What it lacks is _SPEC_FIELDS, which come from the spec cache.
    """
    models = find_array(
        payload, "models",
        where=lambda a: bool(a) and isinstance(a[0], dict) and "isOpenWeights" in a[0],
    )
    if not models:
        raise ValueError("no leaderboard metrics array found for 'models' in RSC payload")
    return [m for m in models if isinstance(m, dict) and m.get("slug")]


def _spec_of(m: dict) -> dict:
    """The static facts this tab needs from one catalogue record, normalised."""
    lic = m.get("licenseName")
    return {
        "parameters": _num(m.get("parameters")),
        "inferenceParametersActiveBillions": _num(m.get("inferenceParametersActiveBillions")),
        "inputModalityImage": m.get("inputModalityImage") is True,
        "inputModalitySpeech": m.get("inputModalitySpeech") is True,
        "licenseName": lic.strip() if isinstance(lic, str) and lic != "$undefined" else None,
    }


def _harvest_specs(payload: str) -> dict[str, dict]:
    """Specs for every open-weight record a model page carries, by slug.

    A page whose catalogue array survives yields all of them at once. Since
    ~2026-10-05 most pages carry only their own record plus ~25 headline
    models, scattered through the payload rather than in one array, so each
    record carrying `parameters` is decoded where it stands.
    """
    out: dict[str, dict] = {}

    def keep(rec):
        if isinstance(rec, dict) and rec.get("slug") and rec.get("isOpenWeights") is True \
                and "parameters" in rec:
            out[rec["slug"]] = _spec_of(rec)

    catalogue = find_array(
        payload, "models",
        where=lambda a: bool(a) and isinstance(a[0], dict) and "parameters" in a[0],
    )
    for rec in catalogue or []:
        keep(rec)

    dec = json.JSONDecoder()
    pos = 0
    while (i := payload.find('"parameters":', pos)) >= 0:
        pos = i + 1
        j = i
        for _ in range(200):                      # nearest enclosing object
            j = payload.rfind("{", 0, j)
            if j < 0:
                break
            try:
                rec, end = dec.raw_decode(payload, j)
            except ValueError:
                continue
            if end > i and isinstance(rec, dict) and "parameters" in rec:
                keep(rec)
                break
    return out


def _specs_path() -> Path:
    return carried.CARRIED_DIR / _SPECS_FILE


def load_specs() -> dict[str, dict]:
    try:
        return json.loads(_specs_path().read_text())
    except (OSError, ValueError):
        return {}


def save_specs(specs: dict[str, dict]) -> None:
    """Sorted and timestamp-free, so an unchanged cache is byte-identical and
    the hourly bot's change guard makes no commit for it."""
    path = _specs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(specs, sort_keys=True, indent=1) + "\n"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def _pages_to_fetch(board: list[dict], specs: dict, hour: int) -> list[str]:
    """Model pages this run reads: new models first, then a rotating few known ones."""
    live = sorted(m["slug"] for m in board
                  if m.get("isOpenWeights") and not m.get("deprecated")
                  and _aa_real(m.get("intelligenceIndex")))
    missing = [s for s in live if s not in specs][:_MAX_NEW_PAGES]
    known = [s for s in live if s in specs]
    if not known:
        return missing
    start = (hour * _ROTATE_PER_RUN) % len(known)
    rotate = (known + known)[start:start + min(_ROTATE_PER_RUN, len(known))]
    return missing + rotate


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


def _parse(models: list[dict], hold_deprecated: set[str] | None = None) -> pd.DataFrame | None:
    seen: set[tuple] = set()
    rows = []
    hold_deprecated = hold_deprecated or set()

    for m in models:
        if not m.get("isOpenWeights"):
            continue
        # Deprecated models stay on the leaderboard for history. The tab answers
        # "what can I run today", so they are dropped here rather than ranked —
        # unless this catalogue already publishes them and their deprecated
        # streak is still too short to trust (data/carried.py): AA's flag
        # flapped hourly on 2026-09-22, and each flap dropped then re-added rows.
        if m.get("deprecated") and (m.get("name") or "").strip() not in hold_deprecated:
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
        board = _leaderboard(payload_from_html(index.text))
    except Exception as exc:
        print(f"[local_scraper] Fetch error: {exc}")
        return False

    # Specs for models the cache has not seen, plus a rotating few it has. One
    # page failing costs that model this run, not the run.
    specs = load_specs()
    misses = []
    for n, slug in enumerate(_pages_to_fetch(board, specs, int(time.time() // 3600))):
        if n:
            time.sleep(_PAGE_DELAY_S)
        try:
            page = requests.get(_CATALOGUE_URL.format(slug=slug),
                                headers=_HEADERS, timeout=_TIMEOUT)
            page.raise_for_status()
            specs.update(_harvest_specs(payload_from_html(page.text)))
        except Exception as exc:
            misses.append(f"{slug}: {exc}")
    if misses:
        print(f"[local_scraper] {len(misses)} model page(s) unread this run: "
              + "; ".join(misses[:5]))
    models = [{**m, **specs.get(m["slug"], {})} for m in board]

    # Deprecation hysteresis, as in data/scraper.py, with its OWN state file:
    # deprecation_hysteresis keeps only names in `published`, so sharing the
    # hosted scraper's file would let each run erase the other's streaks.
    # Persisted only once this scrape publishes, so a refused scrape does not
    # advance anyone's streak.
    try:
        prior = load_cached()
        published = set(prior["name"].astype(str)) if prior is not None \
            and "name" in prior.columns else set()
    except Exception:
        published = set()
    flagged = {(m.get("name") or "").strip() for m in models
               if m.get("isOpenWeights") and m.get("deprecated")} - {""}
    state_path = carried.CARRIED_DIR / _DEPRECATION_STATE_FILE
    hold, dep_state = carried.deprecation_hysteresis(
        carried.load_deprecation_state(state_path), flagged, published)

    df = _parse(models, hold_deprecated=hold)
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
    carried.save_deprecation_state(dep_state, state_path)
    save_specs(specs)
    # Fold this scrape's frozen coding/agentic scores into the sidecar that
    # _carry_dropped_columns reads back — without this the local sidecar only
    # ever held what backfill_from_git found, and a new score was one bad
    # hour from being lost.
    carried.remember(df, "local_scraper")
    if hold:
        print(f"[local_scraper] {len(hold)} deprecated model(s) held until their "
              f"streak reaches {carried.DEPRECATED_DROP_AFTER_SCRAPES} scrapes "
              f"/ {carried.DEPRECATED_DROP_AFTER_HOURS:g}h: {', '.join(sorted(hold))}")
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
