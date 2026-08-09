"""Refuse to publish a scrape that lost an implausible share of its rows.

The refresh workflow's only sanity check used to be an absolute floor:

    assert len(df) > 50, f'{f} only {len(df)} rows — refusing to publish'

On 2026-07-24 Artificial Analysis pruned ~181 legacy models in one go and
data/raw/aa_models.csv went 329 -> 155 rows. A 53% contraction is well clear of
a floor of 50, so the pipeline published it unremarked and the dashboard's
headline model count halved without anyone noticing for weeks.

An absolute floor cannot catch that; a *relative* one can. This compares each
raw CSV against the copy committed at HEAD and fails when the drop exceeds a
threshold. A genuine upstream pruning is then a deliberate act: re-run the
workflow with allow_shrink, or commit the smaller file by hand.

Run standalone:  python data_guard.py [--max-drop 20] [--allow-shrink]
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_CSVS = [
    "data/raw/aa_models.csv",
    "data/raw/aa_local_models.csv",
    "data/raw/aa_image_models.csv",
]

DEFAULT_MAX_DROP_PCT = 20.0
MIN_ROWS = 50            # the original absolute floor, kept as a backstop


def _row_count(text: str) -> int:
    """Data rows in a CSV, excluding the header."""
    import pandas as pd
    return len(pd.read_csv(io.StringIO(text)))


def current_rows(path: str) -> int | None:
    p = ROOT / path
    if not p.exists():
        return None
    return _row_count(p.read_text())


def committed_rows(path: str, rev: str = "HEAD") -> int | None:
    """Row count of the copy at ``rev``; None when the file is new there."""
    try:
        out = subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return _row_count(out)


def check(paths: list[str] | None = None,
          max_drop_pct: float = DEFAULT_MAX_DROP_PCT,
          rev: str = "HEAD") -> list[str]:
    """Return a list of human-readable violations; empty means safe to publish."""
    problems: list[str] = []
    for path in paths if paths is not None else DATA_CSVS:
        now = current_rows(path)
        if now is None:
            problems.append(f"{path}: missing")
            continue
        if now < MIN_ROWS:
            problems.append(f"{path}: only {now} rows (floor {MIN_ROWS})")
            continue

        before = committed_rows(path, rev)
        if before is None or before == 0:
            print(f"  {path}: {now} rows (no committed baseline — skipping delta)")
            continue

        drop_pct = (before - now) / before * 100
        if drop_pct > max_drop_pct:
            problems.append(
                f"{path}: {before} -> {now} rows, a {drop_pct:.0f}% drop "
                f"(limit {max_drop_pct:.0f}%)"
            )
        else:
            delta = now - before
            print(f"  {path}: {now} rows ({delta:+d})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-drop", type=float, default=DEFAULT_MAX_DROP_PCT,
                    help="percent of rows a dataset may lose in one refresh")
    ap.add_argument("--allow-shrink", action="store_true",
                    help="report the deltas but never fail (deliberate pruning)")
    ap.add_argument("--rev", default="HEAD", help="baseline revision")
    args = ap.parse_args()

    print(f"Comparing data/raw against {args.rev} (max drop {args.max_drop:.0f}%)")
    problems = check(max_drop_pct=args.max_drop, rev=args.rev)

    if not problems:
        print("data guard OK")
        return 0

    for p in problems:
        print(f"::error::{p}")
    if args.allow_shrink:
        print("allow-shrink set — publishing anyway")
        return 0
    print("::error::Refusing to publish. If upstream really did prune this "
          "much, re-run the workflow with allow_shrink=true.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
