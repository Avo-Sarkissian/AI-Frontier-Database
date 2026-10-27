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
                                     [--require-baseline]
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
    "data/raw/aa_video_models.csv",
]

# Which scraper's shrink state speaks for each CSV. A scraper that watched a
# shrink hold for SHRINK_CONFIRM_HOURS accepts it as an upstream prune
# (data/carried.py, "Shrink confirmation"); both checks below then measure
# against the accepted size, or the workflow would refuse what the scraper
# just — deliberately — published. Honoured for two baseline windows, after
# which the 24h baseline is post-prune anyway.
_SHRINK_DATASET = {
    "data/raw/aa_models.csv":       "hosted",
    "data/raw/aa_local_models.csv": "local",
    "data/raw/aa_image_models.csv": "image",
    "data/raw/aa_video_models.csv": "video",
}

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
    # The video catalogue changed units once already — the hand-written dataset
    # it replaced priced per SECOND, this one per MINUTE, a 60x shift. Watching
    # both arenas' Elo and the primary price is what would catch that happening
    # again, or an arena's records being read into the wrong column.
    "data/raw/aa_video_models.csv": ["elo_t2v", "elo_i2v", "price_per_min_t2v"],
}


# Economics archives and mirrors written by data/econ_archive.py on their own
# cadence. They are NOT in DATA_CSVS: they are not hourly scrapes, the MIN_ROWS
# floor of 50 does not fit them, and an append-only archive must never lose a
# single row, which the 20% rule would allow. econ_archive runs these same
# rules on every candidate file BEFORE writing it, so a fetch cannot produce a
# file that trips the publish guard; only a hand-edited archive can.
#   min_rows     absolute floor
#   append_only  True: any row loss is a violation; False (a mirror of an
#                upstream that revises in place): DEFAULT_MAX_DROP_PCT applies
#   watch        columns under the _NUMERIC_MEDIAN_MAX_SHIFT_PCT units check
ARCHIVE_CSVS: dict[str, dict] = {
    "data/carried/econ/gpu_prices_weekly.csv":
        {"min_rows": 8000, "append_only": True, "watch": ["median_price"]},
    "data/carried/econ/sofr.csv":
        {"min_rows": 1000, "append_only": True, "watch": ["percent_rate"]},
    "data/carried/econ/inferencex_rates.csv":
        {"min_rows": 8, "append_only": True,
         "watch": ["cost_own_usd_per_chip_hr", "cost_rent_usd_per_chip_hr"]},
    "data/carried/econ/inferencex_points.csv":
        {"min_rows": 200, "append_only": True, "watch": ["tput_tok_s_chip"]},
    "data/raw/epoch/ml_hardware.csv":
        {"min_rows": 150, "append_only": False, "watch": ["TDP (W)"]},
    "data/raw/epoch/ai_companies_revenue_reports.csv":
        {"min_rows": 40, "append_only": False, "watch": []},
    "data/raw/epoch/ai_companies_compute_spend.csv":
        {"min_rows": 8, "append_only": False, "watch": []},
    "data/raw/epoch/ai_companies_funding_rounds.csv":
        {"min_rows": 30, "append_only": False, "watch": []},
    "data/raw/epoch/ai_companies_usage_reports.csv":
        {"min_rows": 40, "append_only": False, "watch": []},
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


def _is_shallow() -> bool:
    """True in a shallow clone — actions/checkout's default of depth 1."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return out == "true"


def baseline_problem(hours: int = CUMULATIVE_BASELINE_HOURS) -> str | None:
    """Why the cumulative check cannot run here, or None when it can.

    The refresh workflow checked out at depth 1 until 2026-09-24, so the only
    commit present was HEAD — always under 24h old on an hourly bot —
    _baseline_rev returned None, and _cumulative_violations returned [] without
    a word. The slow-drain case this guard was written for was never checked
    in the one place it runs. Saying so is the minimum; --require-baseline
    turns it into a failure.
    """
    if _baseline_rev(hours) is not None:
        return None
    if _is_shallow():
        return (f"shallow checkout: no commit {hours}h old is present, so the "
                f"{hours}h cumulative row-loss check cannot run — check out with "
                f"fetch-depth: 0")
    return (f"no commit at least {hours}h old in this history — the "
            f"cumulative row-loss check has no baseline")


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

        before = _accepted_floor(path, committed_rows(path, rev))
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


def _accepted_floor(path: str, rows: int | None) -> int | None:
    """`rows`, lowered to a shrink the scraper accepted as an upstream prune."""
    dataset = _SHRINK_DATASET.get(path)
    if rows is None or dataset is None:
        return rows
    from data import carried
    accepted = carried.accepted_shrink(dataset, 2 * CUMULATIVE_BASELINE_HOURS)
    return rows if accepted is None else min(rows, accepted)


def _cumulative_violations(path: str, now: int) -> list[str]:
    """Catch the slow drain a per-run check cannot see."""
    rev = _baseline_rev(CUMULATIVE_BASELINE_HOURS)
    if rev is None:
        return []
    text = _show(path, rev)
    if not text:
        return []
    then = _accepted_floor(path, _row_count(text))
    if not then:
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


def archive_violations(path: str, before_text: str | None, after_text: str | None,
                       max_drop_pct: float = DEFAULT_MAX_DROP_PCT,
                       allow_shrink: bool = False) -> list[str]:
    """Violations for one economics archive or mirror; pure (texts in, list out).

    `before_text` is the committed copy (None when the file is new), `after_text`
    the candidate. Paths outside ARCHIVE_CSVS have no rules.

    allow_shrink (econ_archive --allow-shrink, i.e. the workflow's allow_shrink
    input) waives the two RELATIVE rules, the mirror row-drop limit and the
    median shift, because an upstream that really pruned trips both on every
    retry. The absolute floor and the append-only no-row-loss rule still hold:
    a merge can never lose a row, so no upstream change excuses one.
    """
    rule = ARCHIVE_CSVS.get(path)
    if rule is None:
        return []
    if not after_text or not after_text.strip():
        return [f"{path}: empty"]
    try:
        now = _row_count(after_text)
    except Exception as e:                        # a file pandas cannot read
        return [f"{path}: unreadable ({type(e).__name__}: {e})"]
    out = []
    if now < rule["min_rows"]:
        out.append(f"{path}: only {now} rows (floor {rule['min_rows']})")
    if before_text and before_text.strip():
        before = _row_count(before_text)
        if rule["append_only"]:
            if now < before:
                out.append(f"{path}: {before} -> {now} rows; an append-only archive "
                           f"must never lose a row")
        elif (not allow_shrink and before
              and (before - now) / before * 100 > max_drop_pct):
            out.append(f"{path}: {before} -> {now} rows, a "
                       f"{(before - now) / before * 100:.0f}% drop (limit {max_drop_pct:.0f}%)")
        if rule["watch"]:
            was_m = _medians(before_text, rule["watch"])
            now_m = _medians(after_text, rule["watch"])
            for col, was in was_m.items():
                if col not in now_m:
                    out.append(f"{path}: column '{col}' lost every usable value")
                    continue
                if was == 0 or allow_shrink:
                    continue
                shift = abs(now_m[col] - was) / abs(was) * 100
                if shift > _NUMERIC_MEDIAN_MAX_SHIFT_PCT:
                    out.append(
                        f"{path}: median '{col}' moved {was:.4g} -> {now_m[col]:.4g} "
                        f"({shift:.0f}%, limit {_NUMERIC_MEDIAN_MAX_SHIFT_PCT:.0f}%) — "
                        f"a units change or parsing fault")
    return out


def check_archives(rev: str = "HEAD", paths: list[str] | None = None) -> list[str]:
    """archive_violations for every ARCHIVE_CSVS file against `rev`.

    A file absent both here and at `rev` has simply not been archived yet; one
    that was committed and is now gone is a violation.
    """
    problems: list[str] = []
    for path in paths if paths is not None else ARCHIVE_CSVS:
        before = _show(path, rev)
        p = ROOT / path
        if not p.exists():
            if before:
                problems.append(f"{path}: missing")
            continue
        after = p.read_text(encoding="utf-8")
        found = archive_violations(path, before, after)
        if not found:
            print(f"  {path}: {_row_count(after)} rows")
        problems.extend(found)
    return problems

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-drop", type=float, default=DEFAULT_MAX_DROP_PCT,
                    help="percent of rows a dataset may lose in one refresh")
    ap.add_argument("--allow-shrink", action="store_true",
                    help="report the deltas but never fail (deliberate pruning)")
    ap.add_argument("--rev", default="HEAD", help="baseline revision")
    ap.add_argument("--require-baseline", action="store_true",
                    help="fail when the 24h cumulative check has no baseline "
                         "(a shallow CI checkout) instead of warning")
    args = ap.parse_args()

    print(f"Comparing data/raw against {args.rev} (max drop {args.max_drop:.0f}%)")
    problems = check(max_drop_pct=args.max_drop, rev=args.rev)
    print(f"Comparing the economics archives against {args.rev}")
    problems += check_archives(rev=args.rev)

    # Not folded into check(): the suite calls check() from shallow CI
    # checkouts too, and a missing baseline is a property of the checkout, not
    # of the data. It is never waived by --allow-shrink, which is about
    # accepting a real drop, not about running blind.
    missing = baseline_problem()
    if missing:
        if args.require_baseline:
            print(f"::error::{missing}")
            return 1
        print(f"::warning::{missing}")

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
