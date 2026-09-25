"""State the hosted and open-weight scrapes need to keep between runs.

Two things live here, both committed under data/carried/ rather than
data/raw/, because data/raw/ is what the scrape OVERWRITES every hour and both
of these exist precisely to outlive one bad hour.

1. FROZEN SCORES (coding, agentic).
AA stopped publishing codingIndex and agenticIndex on 2026-09-10. From then on
the committed CSV was the only place those numbers existed, and
_carry_dropped_columns filled each scrape from it. That is a one-hop memory: a
model missing from ONE published scrape (AA flagged it deprecated for an hour)
left the CSV without its scores, and when it came back there was nothing left
to carry. Measured in git history, hosted coding went 143 -> 120 -> 102 -> 100
non-null (b2ecf2c -> f64facc -> 9427066 -> c51aeda) and agentic 106 -> 72;
17 Claude Opus 5 / GPT-5.6 rows returned with their scores gone for good.

The sidecar is keyed by model name and only ever GAINS or UPDATES rows — a
model absent from a scrape keeps its line. backfill_from_git() rebuilt it from
every historical version of the CSVs, which is where the lost values still were.

2. DEPRECATION HYSTERESIS.
AA's `deprecated` flag toggled hourly on 2026-09-22 for the same 17 models
(5 Claude Opus 5 efforts + 12 GPT-5.6 Luna/Sol): 196 -> 180 -> 196 -> 180 rows.
Each flip was 8.7%, under the 20% shrink guard, so every one was published and
shared ?models= links silently lost the rows. A model this catalogue already
publishes is now dropped as deprecated only after DEPRECATED_DROP_AFTER_SCRAPES
consecutive deprecated scrapes spanning at least DEPRECATED_DROP_AFTER_HOURS.
One clean scrape resets the streak.

The streak is counted in SCRAPES, as asked, with an hours floor on top because
the refresh can fire several times an hour (four cron entries plus an external
dispatch) — three runs fifteen minutes apart would otherwise ride out none of
the flaps observed, which lasted 40-90 minutes each.

Run standalone:  python -m data.carried --backfill
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Overridable for the same reason as AI_FRONTIER_STATUS_PATH: a test that drives
# a scrape end to end must not write into the files the bot commits.
CARRIED_DIR = Path(
    os.environ.get("AI_FRONTIER_CARRIED_DIR")
    or Path(__file__).parent / "carried"
)

# Columns AA no longer publishes. Kept in step with data.scraper._CARRIED_COLUMNS
# (which imports this) — one list, not two.
FROZEN_COLUMNS = ("coding", "agentic")

# tag (as passed to _carry_dropped_columns) -> (sidecar file, key column,
# the committed CSV whose history backfills it).
SIDECARS: dict[str, tuple[str, str, str]] = {
    "scraper":       ("hosted_scores.csv", "model", "data/raw/aa_models.csv"),
    "local_scraper": ("local_scores.csv",  "name",  "data/raw/aa_local_models.csv"),
}

DEPRECATION_STATE_FILE = "deprecation_state.json"
DEPRECATED_DROP_AFTER_SCRAPES = 3
DEPRECATED_DROP_AFTER_HOURS = 3.0


# ── Frozen-score sidecar ──────────────────────────────────────────────────────

def sidecar_path(tag: str) -> Path | None:
    spec = SIDECARS.get(tag)
    return CARRIED_DIR / spec[0] if spec else None


def sidecar_key(tag: str) -> str | None:
    spec = SIDECARS.get(tag)
    return spec[1] if spec else None


def load_sidecar(path: Path | None, key: str) -> pd.DataFrame:
    """The sidecar as a frame with `key` + FROZEN_COLUMNS; empty when absent."""
    cols = [key, *FROZEN_COLUMNS]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except (ValueError, OSError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=cols)
    if key not in df.columns:
        return pd.DataFrame(columns=cols)
    for col in FROZEN_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns \
            else float("nan")
    return df[cols].drop_duplicates(subset=key, keep="last")


def merge_scores(sidecar: pd.DataFrame, df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Upsert every non-null frozen score in `df` into `sidecar`.

    Gains and updates only. A model missing from `df` keeps its row, and a NaN
    in `df` never overwrites a stored number — a hole in one scrape is exactly
    the case this file exists to survive.
    """
    cols = [key, *FROZEN_COLUMNS]
    out = sidecar.copy() if not sidecar.empty else pd.DataFrame(columns=cols)
    if df is None or df.empty or key not in df.columns:
        return out
    out = out.set_index(key)
    for col in FROZEN_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    present = [c for c in FROZEN_COLUMNS if c in df.columns]
    if not present:
        return out.reset_index()[cols]
    fresh = df[[key, *present]].copy()
    for col in present:
        # float, always: an all-integer snapshot column would otherwise make
        # combine_first try to cast the stored NaNs to int.
        fresh[col] = pd.to_numeric(fresh[col], errors="coerce").astype(float)
    fresh = fresh[fresh[key].astype(str).str.strip() != ""]
    fresh = fresh[fresh[present].notna().any(axis=1)]
    fresh = fresh.drop_duplicates(subset=key, keep="last").set_index(key)
    # combine_first: a non-null fresh value wins, a NaN falls through to the
    # stored one, and the index is the union — so nothing is ever removed.
    out = fresh.combine_first(out[list(FROZEN_COLUMNS)])
    out.index.name = key
    return out.reset_index()[cols]


def _canonical(df: pd.DataFrame, key: str) -> str:
    """Stable text form: sorted by name, 2dp like the scrapers write, so a
    rewrite with no new information produces a byte-identical file and the
    refresh workflow's change guard stays quiet."""
    out = df.dropna(subset=list(FROZEN_COLUMNS), how="all").copy()
    out = out.sort_values(key, kind="mergesort").reset_index(drop=True)
    for col in FROZEN_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    return out[[key, *FROZEN_COLUMNS]].to_csv(index=False)


def save_sidecar(df: pd.DataFrame, path: Path, key: str) -> bool:
    """Write only on a real change. Returns True when the file moved."""
    text = _canonical(df, key)
    path = Path(path)
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def remember(df: pd.DataFrame, tag: str, path: Path | None = None) -> bool:
    """Fold a published scrape's frozen scores into its sidecar."""
    key = sidecar_key(tag)
    path = path or sidecar_path(tag)
    if key is None or path is None:
        return False
    merged = merge_scores(load_sidecar(path, key), df, key)
    return save_sidecar(merged, path, key)


# ── Backfill from git history ─────────────────────────────────────────────────

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def _history(csv_path: str, refs: list[str]) -> list[str]:
    """Every commit that touched `csv_path` on any of `refs`, oldest first."""
    seen: dict[str, int] = {}
    for ref in refs:
        try:
            out = _git("log", "--format=%H %ct", ref, "--", csv_path)
        except subprocess.CalledProcessError:
            continue
        for line in out.splitlines():
            sha, _, ts = line.partition(" ")
            if sha and ts.isdigit():
                seen[sha] = int(ts)
    return sorted(seen, key=lambda s: (seen[s], s))


def backfill_from_git(tag: str, refs: list[str] | None = None,
                      path: Path | None = None,
                      include_worktree: bool = True) -> dict:
    """Rebuild a sidecar from every historical version of its CSV.

    Walks oldest to newest, so each model ends on the LAST score it was ever
    published with — the live value up to 2026-09-10, the frozen one after.
    Read-only on git and on data/raw/; the only write is the sidecar.
    """
    name, key, csv_path = SIDECARS[tag]
    path = path or CARRIED_DIR / name
    refs = refs or ["HEAD", "origin/main"]
    sidecar = load_sidecar(path, key)
    before = int(sidecar[list(FROZEN_COLUMNS)].notna().any(axis=1).sum()) \
        if not sidecar.empty else 0

    versions = 0
    for sha in _history(csv_path, refs):
        try:
            text = _git("show", f"{sha}:{csv_path}")
        except subprocess.CalledProcessError:
            continue
        try:
            snap = pd.read_csv(io.StringIO(text))
        except (ValueError, pd.errors.ParserError):
            continue
        if key not in snap.columns or not any(c in snap.columns for c in FROZEN_COLUMNS):
            continue
        versions += 1
        sidecar = merge_scores(sidecar, snap, key)

    worktree = ROOT / csv_path
    if include_worktree and worktree.exists():
        try:
            sidecar = merge_scores(sidecar, pd.read_csv(worktree), key)
        except (ValueError, pd.errors.ParserError):
            pass

    save_sidecar(sidecar, path, key)
    after = int(sidecar[list(FROZEN_COLUMNS)].notna().any(axis=1).sum())
    return {"tag": tag, "path": str(path), "versions_read": versions,
            "models_before": before, "models_after": after,
            **{f"{c}_scored": int(sidecar[c].notna().sum()) for c in FROZEN_COLUMNS}}


# ── Deprecation hysteresis ────────────────────────────────────────────────────

def _state_path(path: Path | None) -> Path:
    return Path(path) if path is not None else CARRIED_DIR / DEPRECATION_STATE_FILE


def load_deprecation_state(path: Path | None = None) -> dict:
    """{model: {"count": n, "first_seen": iso}} for models on a deprecated
    streak; {} when the file is absent or unreadable."""
    p = _state_path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (ValueError, OSError):
        return {}
    models = data.get("models", {}) if isinstance(data, dict) else {}
    return {str(k): v for k, v in models.items()
            if isinstance(v, dict) and isinstance(v.get("count"), int)}


def save_deprecation_state(state: dict, path: Path | None = None) -> bool:
    p = _state_path(path)
    text = json.dumps({"models": dict(sorted(state.items()))}, indent=2) + "\n"
    if p.exists() and p.read_text() == text:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return True


def _parse_iso(s) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def deprecation_hysteresis(state: dict, deprecated_now: set[str],
                           published: set[str], now: datetime | None = None,
                           after_scrapes: int | None = None,
                           after_hours: float | None = None,
                           ) -> tuple[set[str], dict]:
    """Which deprecated models to keep this run, and the state to persist.

    Only models the catalogue ALREADY publishes are tracked. The ~95 models AA
    has long marked superseded were never published here; tracking them would
    hold every one for a few hours on first deploy — a flap of its own — and
    rewrite the state file on every run forever.

    A model is held while its streak is short on either axis (fewer than
    `after_scrapes` consecutive deprecated scrapes, or under `after_hours`
    since the first). Once both are met it is dropped, leaves the published
    set, and so leaves the state on the next run. A scrape that sees it
    undeprecated clears its streak.
    """
    now = now or datetime.now(timezone.utc)
    # Read at call time, not bound as defaults, so the thresholds stay one
    # module constant each that callers and tests can see and adjust.
    after_scrapes = DEPRECATED_DROP_AFTER_SCRAPES if after_scrapes is None else after_scrapes
    after_hours = DEPRECATED_DROP_AFTER_HOURS if after_hours is None else after_hours
    new_state: dict = {}
    hold: set[str] = set()
    for name in sorted(deprecated_now & published):
        prev = state.get(name) or {}
        first = _parse_iso(prev.get("first_seen")) or now
        count = int(prev.get("count", 0)) + 1
        hours = (now - first).total_seconds() / 3600
        if count < after_scrapes or hours < after_hours:
            hold.add(name)
            new_state[name] = {"count": count, "first_seen": first.isoformat(timespec="seconds")}
    return hold, new_state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backfill", action="store_true",
                    help="rebuild the frozen-score sidecars from git history")
    ap.add_argument("--ref", action="append", dest="refs",
                    help="git ref(s) whose history to read (default HEAD + origin/main)")
    args = ap.parse_args()
    if not args.backfill:
        ap.print_help()
        return 0
    for tag in SIDECARS:
        print(json.dumps(backfill_from_git(tag, refs=args.refs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
