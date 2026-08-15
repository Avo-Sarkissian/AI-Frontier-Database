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

# How far back the baseline reaches. The bot commits every run, so a HEAD-only
# baseline is *the previous hour* — and five consecutive 19% drops each pass on
# their own while compounding to a 66% loss. Comparing against a commit roughly
# a day old makes the cumulative slide visible; the per-run check still runs
# against HEAD, so both a cliff and a slope fail.
CUMULATIVE_BASELINE_HOURS = 24
CUMULATIVE_MAX_DROP_PCT = 35.0

# Value-level drift. Row counts are dimensionally blind: across the 2026-05-06
# price-basis flip every price doubled while the row count moved +11% — in the
# safe direction — so the guard passed. A median that moves this much between
# refreshes is a units change or a parsing fault, not the market.
_NUMERIC_MEDIAN_MAX_SHIFT_PCT = 40.0
_WATCHED_COLUMNS = {
    "data/raw/aa_models.csv":       ["price", "quality", "speed", "latency"],
    "data/raw/aa_local_models.csv": ["params_b", "quality"],
    "data/raw/aa_image_models.csv": ["elo", "price_per_1k"],
}


def _read(text: str):
    import pandas as pd
    return pd.read_csv(io.StringIO(text))


def _row_count(text: str) -> int:
    """Data rows in a CSV, excluding the header."""
    return len(_read(text))


def _medians(text: str, columns: list[str]) -> dict[str, float]:
    """Median of each watched column, ignoring nulls and zeros.

    Zeros are excluded because a collapsed column reads as a legitimate median
    of 0 otherwise; the *presence* check below is what catches that case.
    """
    import pandas as pd

    df = _read(text)
    out: dict[str, float] = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        s = s[s.notna() & (s != 0)]
        if len(s):
            out[col] = float(s.median())
    return out


def _baseline_rev(hours: int) -> str | None:
    """The newest commit at least `hours` old, or None in a shallow checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-list", "-1", f"--before={hours}.hours.ago", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _show(path: str, rev: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


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

        problems.extend(_cumulative_violations(path, now))
        problems.extend(_value_violations(path, rev))
    return problems


def _cumulative_violations(path: str, now: int) -> list[str]:
    """Catch the slow drain a per-run check cannot see."""
    rev = _baseline_rev(CUMULATIVE_BASELINE_HOURS)
    if rev is None:
        return []
    text = _show(path, rev)
    if not text:
        return []
    then = _row_count(text)
    if then == 0:
        return []
    drop = (then - now) / then * 100
    if drop > CUMULATIVE_MAX_DROP_PCT:
        return [
            f"{path}: {then} -> {now} rows over the last "
            f"{CUMULATIVE_BASELINE_HOURS}h, a {drop:.0f}% cumulative drop "
            f"(limit {CUMULATIVE_MAX_DROP_PCT:.0f}%) — each hourly step may have "
            f"looked small"
        ]
    return []


def _value_violations(path: str, rev: str) -> list[str]:
    """Catch a units change or parsing fault that leaves the row count intact."""
    columns = _WATCHED_COLUMNS.get(path)
    if not columns:
        return []
    before_text = _show(path, rev)
    if not before_text:
        return []
    p = ROOT / path
    if not p.exists():
        return []

    before = _medians(before_text, columns)
    after = _medians(p.read_text(), columns)

    out = []
    for col, was in before.items():
        if col not in after:
            out.append(f"{path}: column '{col}' lost every usable value")
            continue
        now = after[col]
        if was == 0:
            continue
        shift = abs(now - was) / abs(was) * 100
        if shift > _NUMERIC_MEDIAN_MAX_SHIFT_PCT:
            out.append(
                f"{path}: median '{col}' moved {was:.4g} -> {now:.4g} "
                f"({shift:.0f}%, limit {_NUMERIC_MEDIAN_MAX_SHIFT_PCT:.0f}%) — "
                f"a units change or parsing fault, not the market"
            )
    return out


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
