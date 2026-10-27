"""Seed data/carried/open_weight_names.csv from git history. Run once; the
hourly scrape grows it from then on (data/carried.learn_open_weights).

Why history: the gap view classifies every archived hosted snapshot back to
2026-02-25, and those rows carry no AA open-weights flag (the scrape kept it
only from 2026-10-27). Today's open-weights catalogue alone misses models that
have since left it — Kimi K2.6, AA's top open model on 2026-04-30, is gone —
so the early gap would be overstated. Every committed revision of
data/raw/aa_local_models.csv (AA's open-weights catalogue, scraped since
2026-05-06) is read and the union of its names recorded, each with the date of
the first commit that listed it. Then the current hosted catalogue's AA flags
are folded in, which is also what every hourly scrape does.

Read-only on git and data/raw/; the only write is the registry.

Usage:
    .venv/bin/python scripts/seed_open_weight_registry.py [--ref REF ...]
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import carried  # noqa: E402
from data.open_weights import REGISTRY_COLUMNS, SOURCE_CATALOGUE, load_registry  # noqa: E402

CATALOGUE = "data/raw/aa_local_models.csv"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def catalogue_history(refs: list[str]) -> dict[str, str]:
    """name -> ISO date of the first commit whose catalogue listed it."""
    commits: dict[str, str] = {}
    for ref in refs:
        try:
            out = _git("log", "--format=%H %cs", ref, "--", CATALOGUE)
        except subprocess.CalledProcessError:
            continue
        for line in out.splitlines():
            sha, _, day = line.partition(" ")
            if sha and day:
                commits[sha] = day
    first: dict[str, str] = {}
    for sha, day in sorted(commits.items(), key=lambda kv: (kv[1], kv[0])):
        try:
            snap = pd.read_csv(io.StringIO(_git("show", f"{sha}:{CATALOGUE}")),
                               dtype=str, keep_default_na=False)
        except (subprocess.CalledProcessError, ValueError, pd.errors.ParserError):
            continue
        col = "name" if "name" in snap.columns else ("model" if "model" in snap.columns else None)
        if col is None:
            continue
        for name in snap[col]:
            name = name.strip()
            if name and (name not in first or day < first[name]):
                first[name] = day
    return first


def seed(refs: list[str], path: Path | None = None) -> dict:
    path = Path(path) if path is not None else carried.CARRIED_DIR / "open_weight_names.csv"
    reg = load_registry(path)
    have = set(reg["name"])
    hist = catalogue_history(refs)
    worktree = ROOT / CATALOGUE
    if worktree.exists():
        for name in pd.read_csv(worktree, dtype=str, keep_default_na=False)["name"]:
            hist.setdefault(name.strip(), carried._today_iso())
    added = [{"name": n, "open_weights": "1", "source": SOURCE_CATALOGUE, "first_seen": d}
             for n, d in sorted(hist.items()) if n and n not in have]
    reg = pd.concat([reg, pd.DataFrame(added, columns=list(REGISTRY_COLUMNS))],
                    ignore_index=True)
    # The hosted catalogue's AA flags, exactly as the hourly scrape folds them.
    from data.ingest import load_cached
    reg = carried.merge_open_weights(reg, load_cached())
    carried.save_open_weight_registry(reg, path)
    return {"path": str(path), "catalogue_names": len(hist), "rows": len(reg),
            "aa_rows": int((reg["source"] == "aa_isOpenWeights").sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", action="append", dest="refs",
                    help="git ref(s) whose history to read (default HEAD + origin/main)")
    args = ap.parse_args()
    print(seed(args.refs or ["HEAD", "origin/main"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
