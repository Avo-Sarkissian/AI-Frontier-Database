"""Per-dataset scrape status, so the freshness badge cannot lie.

The badge's only source was `manifest.generated_iso` — the moment the site was
BUILT. Three datasets refresh independently, and the hourly workflow rebuilds
whenever *any* file under data/raw/ moves, so one succeeding scraper reset the
clock for all three.

This is not hypothetical. On CI run 31558618824 (2026-08-12T02:59Z) the hosted
and local scrapers both failed and only the image scrape succeeded; the site
published, the badge read "just now", and the catalogue behind it was 5h12m
stale while the local one was 19h18m stale. The "report failures" step is
deliberately last, so the run went red *after* the lie was pushed.

That was made worse by a coupling since removed: data/scraper.py and
data/local_scraper.py used to hit the byte-identical URL, so those two always
failed together while the image scraper (a different URL, a churning ELO arena)
changed almost every hour — the one dataset that reliably reset the clock was
the one least coupled to the others. All four scrapers now read different
sources and fail independently, but the badge still reports per dataset, because
independence makes a shared build timestamp less honest, not more.

Each scraper now records whether it actually fetched, when, and how many rows.
build_static folds the file into the manifest and the badge reports the OLDEST
successful fetch, with a warning when any dataset is failing or stale.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Overridable so a test can drive a scraper through its FAILURE path without
# writing ok=false into the committed file the live badge reads. Without this,
# `pytest` marked all three datasets as failing and the deployed site showed a
# staleness warning because someone had run the tests — the same class of bug as
# the suite republishing docs/.
STATUS_PATH = Path(
    os.environ.get("AI_FRONTIER_STATUS_PATH")
    or Path(__file__).parent / "raw" / "scrape_status.json"
)

# Datasets the dashboard presents. A dataset missing from the file is treated as
# unknown rather than fresh — absence of evidence is not evidence of freshness.
DATASETS = ("hosted", "local", "image", "video")

# How old the newest successful fetch may get before the badge warns.
#
# This was 3h, chosen against a cron that claims to run hourly. GitHub does not
# honour that claim: across 59 successful runs (2026-08-24..31) the median gap
# was 1.3h, but p90 was 6.9h and the largest was 19.7h. So the badge spent a
# good part of every week warning about GitHub's scheduler while all four
# scrapers were healthy — and a warning that common is one nobody reads, which
# is the same failure as the image endpoint's 29 silent days, just louder.
#
# 12h sits above the p90 gap and well under a day, so it flags a pipeline that
# has genuinely stopped. It costs nothing in sensitivity to the failure that
# actually matters: a scraper that breaks reports ok=false and flags instantly,
# with no reference to the clock.
STALE_AFTER_HOURS = 12.0

# How old the PUBLISHED freshness stamp may get before a run republishes it even
# though no number moved. See heartbeat_due().
HEARTBEAT_AFTER_HOURS = 4.0


def _read() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        data = json.loads(STATUS_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def record(dataset: str, ok: bool, rows: int | None = None) -> None:
    """Note the outcome of one scrape.

    A failure keeps the previous `fetched_at` — the data really is that old —
    and flips `ok`, so the badge can say "stale, and we know it" rather than
    silently reporting the build time.
    """
    status = _read()
    entry = dict(status.get(dataset) or {})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry["ok"] = bool(ok)
    entry["checked_at"] = now
    if ok:
        entry["fetched_at"] = now
        if rows is not None:
            entry["rows"] = int(rows)
    status[dataset] = entry
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    except OSError:
        pass          # a status file we cannot write must never fail a scrape


def load() -> dict:
    """Status for every known dataset, with unknowns made explicit."""
    status = _read()
    return {name: status.get(name) or {"ok": None} for name in DATASETS}


def oldest_successful_fetch(status: dict | None = None) -> str | None:
    """The ISO timestamp the badge should show: the oldest dataset's fetch.

    A dashboard is only as fresh as its stalest panel, so the honest headline is
    the minimum, not the maximum and certainly not the build time.
    """
    status = load() if status is None else status
    stamps = [e.get("fetched_at") for e in status.values()
              if isinstance(e, dict) and e.get("fetched_at")]
    return min(stamps) if len(stamps) == len(DATASETS) else None


def heartbeat_due(published_iso: str | None,
                  now: datetime | None = None) -> bool:
    """Should this run publish its freshness even though no data moved?

    THE BUG THIS EXISTS FOR. record() stamps fetched_at on every successful
    scrape, so the runner always knows the data was just verified. But
    refresh.yml only commits when a NUMBER moves — scrape_status.json is
    excluded from the change guard precisely so it cannot trigger a commit by
    itself — so on a quiet upstream that fresh stamp was written on the runner
    and thrown away. The site kept serving the stamp from the last run where
    some price happened to change, the browser aged it past STALE_AFTER_HOURS,
    and refreshing could not help: the published manifest really did hold that
    stamp. On 2026-08-30 the data last moved at 06:46, runs at 12:58 and 17:34
    both scraped all four datasets successfully and committed nothing, and the
    badge read "Updated 11 hours ago" under a warning triangle.

    A failing scrape needs publishing for the same reason and more urgently: if
    every scraper broke on a quiet upstream, nothing would move, nothing would
    be committed, and the site would go on showing the last happy status. The
    heartbeat is what carries ok=false to the badge.

    Answering "is the published stamp old" rather than "did anything change"
    keeps the original guard intact — the bot still skips the commit on the
    runs that have nothing to say.
    """
    if not published_iso:
        return True
    now = now or datetime.now(timezone.utc)
    try:
        when = datetime.fromisoformat(str(published_iso))
    except ValueError:
        return True
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (now - when).total_seconds() > HEARTBEAT_AFTER_HOURS * 3600


def stale_datasets(status: dict | None = None,
                   max_age_hours: float = STALE_AFTER_HOURS) -> list[str]:
    """Datasets that failed their last scrape or have not refreshed in a while."""
    status = load() if status is None else status
    now = datetime.now(timezone.utc)
    out = []
    for name, entry in status.items():
        if not isinstance(entry, dict) or entry.get("ok") is None:
            out.append(name)
            continue
        if entry.get("ok") is False:
            out.append(name)
            continue
        fetched = entry.get("fetched_at")
        if not fetched:
            out.append(name)
            continue
        try:
            when = datetime.fromisoformat(fetched)
        except ValueError:
            out.append(name)
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if (now - when).total_seconds() > max_age_hours * 3600:
            out.append(name)
    return sorted(out)


if __name__ == "__main__":
    # `--heartbeat-due <manifest.json>`: prints true/false for refresh.yml.
    # The decision lives here, with a test, rather than as a shell expression in
    # the workflow — the alarm is the thing that has to work, not the YAML that
    # describes it.
    import sys

    if "--heartbeat-due" in sys.argv:
        i = sys.argv.index("--heartbeat-due")
        published = None
        if len(sys.argv) > i + 1:
            manifest = Path(sys.argv[i + 1])
            if manifest.exists():
                try:
                    published = json.loads(manifest.read_text()).get("data_fetched_iso")
                except (ValueError, OSError):
                    published = None
        print("true" if heartbeat_due(published) else "false")
