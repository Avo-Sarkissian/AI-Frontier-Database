"""Archive external economics data the bot would otherwise never see again.

Bot-only. Nothing in the browser imports this module (it is deliberately NOT on
build_static's pycode include list); the Economics views derive small files from
what it writes at build time.

Why archive at all: every source below publishes only "now", or a rolling
window, and nobody publishes the history.

- GetDeploying's weekly GPU-price CSV keeps a rolling ~52 weeks. Between the
  2026-10-23 and 2026-10-27 downloads, the 145 rows of week 2025-10-20 left the
  window for good. The archive keeps every row it has ever seen.
- SemiAnalysis InferenceX publishes per-GPU throughput against interactivity
  and its own $/chip-hr marks (h = owning, r = renting). Its views show recent
  runs, and the marks are edited in place, so the archive is append-only for
  points and a change log for the marks. Two upstream conventions are
  normalised on the way in (see parse_ix_view): gpu_count is the physical GPU
  count, and every throughput is per GPU of the whole deployment.
- The NY Fed publishes SOFR daily (the "SOFR + spread" financing input).
- Epoch AI's hardware and AI Companies CSVs are revised in place; they are
  mirrored, and git history is the archive.

Each source has its own cadence (SOURCES), so most hourly bot runs fetch
nothing. Every source is fetched, validated and written on its own: one failing
never stops the others, and the workflow step that runs this module can never
block the AA publish (continue-on-error + a step timeout in refresh.yml). A
source that keeps failing for ESCALATE_AFTER_H turns the run red, so a dead
endpoint cannot stay green for weeks, which is how the image scrape once failed
silently for 29 days.

Writes are atomic (temp file + os.replace) and skipped when the bytes are
unchanged, so an unchanged upstream produces no diff and no commit. Every
candidate file is checked with data_guard.archive_violations BEFORE it is
written, with the same rules the publish guard applies, so a fetch can never
produce a file that would then block the publish.

fetch_state.json is the only status record for these sources. They are not
datasets the freshness badge reports on, so scrape_status is not touched.

    python -m data.econ_archive                 # whatever is due
    python -m data.econ_archive --force         # everything, now
    python -m data.econ_archive --only sofr,inferencex
    python -m data.econ_archive --allow-shrink  # upstream really pruned (see run())
    python -m data.econ_archive --seed-getdeploying PATH   # one-time, archive wins

Terms (checked 2026-10-27): GetDeploying and Epoch AI data are CC BY 4.0. The
NY Fed reference-rate API is public. InferenceX code is Apache-2.0 and its data
is published as freely available, with no formal data licence; the name is a
SemiAnalysis trademark, so credit it and link the runs. Vast.ai, RunPod, Ornn,
Silicon Data and OpenRouter's rankings are excluded on purpose: their terms
forbid what a public dashboard would do with them (a test enforces this).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent

# Same override as data/carried.py, so a test that drives a run end to end
# never writes into the files the bot commits.
ECON_DIR = Path(
    os.environ.get("AI_FRONTIER_CARRIED_DIR") or ROOT / "data" / "carried"
) / "econ"
EPOCH_DIR = Path(os.environ.get("AI_FRONTIER_EPOCH_DIR") or ROOT / "data" / "raw" / "epoch")
STATE_FILE = "fetch_state.json"

ESCALATE_AFTER_H = 24.0
DEFAULT_BUDGET_S = 240.0
_TIMEOUT = 60
_HEADERS = {"User-Agent": "AI-Frontier-Dashboard/1.0 (+github.com/Avo-Sarkissian)"}

# Canonical repo paths. data_guard.ARCHIVE_CSVS is keyed on these.
GPU_PRICES = "data/carried/econ/gpu_prices_weekly.csv"
SOFR_CSV = "data/carried/econ/sofr.csv"
IX_RATES = "data/carried/econ/inferencex_rates.csv"
IX_POINTS = "data/carried/econ/inferencex_points.csv"
EPOCH_FILES = {
    "ml_hardware.csv": "data/raw/epoch/ml_hardware.csv",
    "ai_companies_revenue_reports.csv": "data/raw/epoch/ai_companies_revenue_reports.csv",
    "ai_companies_compute_spend.csv": "data/raw/epoch/ai_companies_compute_spend.csv",
    "ai_companies_funding_rounds.csv": "data/raw/epoch/ai_companies_funding_rounds.csv",
    "ai_companies_usage_reports.csv": "data/raw/epoch/ai_companies_usage_reports.csv",
}

GD_URL = "https://getdeploying.com/dataset/gpu-prices/weekly.csv"
SOFR_LAST_URL = "https://markets.newyorkfed.org/api/rates/secured/sofr/last/10.json"
SOFR_SEARCH_URL = "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
SOFR_FIRST_DATE = "2018-04-02"     # first published SOFR (NY Fed)
IX_BASE = "https://inferencex.semianalysis.com/api/v1"
IX_RUN_URL = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/{run_id}"
EPOCH_BASE = "https://epoch.ai/data/"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    terms_url: str
    licence: str
    credit: str
    cadence_h: float          # after a success
    retry_h: float            # after a failure
    outputs: tuple[str, ...]  # canonical repo paths
    verified: str = "2026-10-27"


# Cadences are a little under the upstream period (6 days, not 7) so that
# GitHub's cron jitter can never make the bot skip a week.
SOURCES: dict[str, Source] = {
    "sofr": Source(
        "sofr", SOFR_LAST_URL,
        "https://www.newyorkfed.org/markets/reference-rates/sofr",
        "public reference-rate API (Federal Reserve Bank of New York)",
        "Federal Reserve Bank of New York, SOFR",
        20, 3, (SOFR_CSV,)),
    "getdeploying": Source(
        "getdeploying", GD_URL,
        "https://getdeploying.com/dataset/gpu-prices",
        "CC BY 4.0", "GetDeploying (CC BY 4.0)",
        144, 6, (GPU_PRICES,)),
    "epoch_hardware": Source(
        "epoch_hardware", EPOCH_BASE + "ml_hardware.csv",
        "https://epoch.ai/data/machine-learning-hardware-documentation",
        "CC BY 4.0", "Epoch AI, 'Data on Machine Learning Hardware' (CC BY 4.0)",
        144, 6, (EPOCH_FILES["ml_hardware.csv"],)),
    "epoch_companies": Source(
        "epoch_companies", EPOCH_BASE + "ai_companies_revenue_reports.csv",
        "https://epoch.ai/data/ai-companies",
        "CC BY 4.0",
        "Epoch AI, 'Data on AI companies' (Josh You, John Croxton, Venkat Somala, "
        "Yafah Edelman; CC BY 4.0)",
        144, 6, tuple(v for k, v in EPOCH_FILES.items() if k != "ml_hardware.csv")),
    "inferencex": Source(
        "inferencex", IX_BASE + "/views/inference",
        "https://github.com/SemiAnalysisAI/InferenceX",
        "code Apache-2.0; data published as freely available (no formal data licence)",
        "SemiAnalysis InferenceX (per-run links to github.com/SemiAnalysisAI/InferenceX/actions/runs)",
        144, 6, (IX_RATES, IX_POINTS)),
}

# Hosts whose terms forbid what a public dashboard does with their data.
BANNED_HOSTS = ("vast.ai", "runpod", "ornn", "openrouter.ai", "silicondata")


# ── small utilities ───────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_date(s: str) -> date | None:
    try:
        return date.fromisoformat(str(s).strip()[:10])
    except ValueError:
        return None


def _num(s) -> float | None:
    try:
        v = float(str(s).strip())
    except (TypeError, ValueError):
        return None
    return v if v == v else None   # NaN -> None


def _g(v) -> str:
    """A float as '.6g': stable across re-fetches that differ in the 15th digit."""
    return "" if v is None else format(float(v), ".6g")


def resolve(canonical: str) -> Path:
    """Where a canonical repo path lives on disk (honours the test overrides)."""
    p = Path(canonical)
    if canonical.startswith("data/carried/econ/"):
        return ECON_DIR / p.name
    if canonical.startswith("data/raw/epoch/"):
        return EPOCH_DIR / p.name
    return ROOT / canonical


def _write_if_changed(path: Path, text: str) -> bool:
    """Atomically write `text`; False (file untouched) when the bytes match."""
    data = text.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _parse_csv(text: str | None) -> tuple[list[str], list[list[str]]]:
    if not text:
        return [], []
    rows = list(csv.reader(io.StringIO(text, newline="")))
    rows = [r for r in rows if r]
    return (rows[0], rows[1:]) if rows else ([], [])


def _csv_text(header: list[str], rows: Iterable[list[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


# Same leading characters as static_helpers._FORMULA_LEAD (a test pins that).
_FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")


def sanitise_cell(v: str) -> str:
    """static_helpers.csv_safe for one cell, except numbers are left alone.

    Embedded newlines and tabs are collapsed (171 Epoch cells carry them, and a
    newline inside a quoted field still splits the record in some readers),
    then a formula-lead cell is prefixed with ' so a spreadsheet cannot run it.
    A cell that parses as a number is skipped, so '-0.4' stays a number.
    """
    if not isinstance(v, str) or v == "":
        return v
    if _num(v) is not None:
        return v
    v = v.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return "'" + v if v[:1] in _FORMULA_LEAD else v


def sanitise_rows(rows: Iterable[list[str]]) -> list[list[str]]:
    return [[sanitise_cell(c) for c in r] for r in rows]


def _guard(canonical: str, before: str | None, after: str,
           allow_shrink: bool = False) -> list[str]:
    """The publish guard's verdict on a candidate file, before it is written.
    allow_shrink waives the relative rules only (see data_guard)."""
    import data_guard
    return data_guard.archive_violations(canonical, before, after,
                                         allow_shrink=allow_shrink)


def _conform(header: list[str], rows: list[list[str]], target: list[str],
             aliases: dict[str, str] | None = None) -> list[list[str]]:
    """Archive rows written under an older header, re-laid out by column name
    under `target`. A column the old file lacks is blank; `aliases` maps an
    old name to its new one. Columns are only ever appended or renamed, so an
    archive committed before a schema change still merges."""
    if header == target or not rows:
        return rows
    names = [(aliases or {}).get(h, h) for h in header]
    idx = [names.index(c) if c in names else None for c in target]
    return [[(r[i] if i is not None and i < len(r) else "") for i in idx] for r in rows]


# ── HTTP ──────────────────────────────────────────────────────────────────────

class _Requests:
    """Default session. Looks requests.get up at call time, so a test that
    replaces requests.get (tests/test_refresh_workflow._FAIL_DRIVER) is obeyed,
    and importing this module makes no request."""

    def get(self, url, **kw):
        import requests
        return requests.get(url, **kw)


def _get(session, url: str, params: dict | None = None):
    r = session.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r


def _utf8(resp) -> str:
    """A CSV body decoded as UTF-8, never through `resp.text`.

    epoch.ai serves `Content-Type: text/csv` with no charset, and for text/*
    without one requests falls back to ISO-8859-1, so `.text` turned every
    curly quote and en dash into mojibake (the funding-round Id "OpenAI, en
    dash, Nvidia" was stored as "OpenAI" + U+00E2 U+0080 U+0093 + "Nvidia";
    verified 2026-10-27). Both CSV hosts publish
    UTF-8. A body that is not UTF-8 is refused rather than guessed at."""
    try:
        return resp.content.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise SourceError(f"response is not UTF-8 ({e})") from None


class SourceError(Exception):
    """A fetch or validation failure; the message is recorded in fetch_state."""


@dataclass
class Outcome:
    ok: bool
    rows: int | None = None
    upstream_rows: int | None = None
    error: str = ""
    wrote: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # A success with something wrong in it (main() prints it as ::warning).
    warn: str = ""
    # Per-view bookkeeping for multi-call sources (InferenceX): the views that
    # raised, and the planned views not attempted (budget). None = not tracked.
    failed_views: list[str] | None = None
    pending_views: list[str] = field(default_factory=list)


# ── GetDeploying ──────────────────────────────────────────────────────────────

GD_HEADER = ["gpu_slug", "date", "billing_type", "reservation_months", "min_price",
             "max_price", "median_price", "provider_median_price", "provider_count",
             "offering_count"]
GD_MIN_UPSTREAM_ROWS = 8000
GD_MAX_AGE_DAYS = 21


def _gd_key(r: list[str]) -> tuple:
    return (r[0], r[1], r[2], r[3])


def _gd_sort(r: list[str]) -> tuple:
    rm = r[3].strip()
    return (r[0], r[2], (0, 0) if rm == "" else (1, int(float(rm))), r[1])


def merge_getdeploying(archive_rows: list[list[str]], upstream_rows: list[list[str]],
                       upstream_wins: bool = True) -> list[list[str]]:
    """Union on (gpu_slug, date, billing_type, reservation_months).

    Cells stay the upstream strings (never float-parsed), so the file is
    byte-stable. upstream_wins=True for live fetches: GetDeploying revises the
    in-progress week (59 rows of 2026-10-19 changed between the 10-23 and
    10-27 downloads). Rows that left the upstream window are always kept.
    """
    merged: dict[tuple, list[str]] = {}
    first, second = (archive_rows, upstream_rows) if upstream_wins else (upstream_rows, archive_rows)
    for r in first:
        merged[_gd_key(r)] = r
    for r in second:
        merged[_gd_key(r)] = r
    return sorted(merged.values(), key=_gd_sort)


def validate_getdeploying(header: list[str], rows: list[list[str]], now: datetime,
                          last_upstream_rows: int | None, live: bool = True) -> list[str]:
    problems: list[str] = []
    if header != GD_HEADER:
        return [f"header changed: {header}"]
    if any(len(r) != len(GD_HEADER) for r in rows):
        problems.append("ragged rows")
        return problems
    if live:
        if len(rows) < GD_MIN_UPSTREAM_ROWS:
            problems.append(f"only {len(rows)} upstream rows (floor {GD_MIN_UPSTREAM_ROWS})")
        if last_upstream_rows and len(rows) < 0.8 * last_upstream_rows:
            problems.append(f"upstream fell {last_upstream_rows} -> {len(rows)} rows (>20%)")
    keys = [_gd_key(r) for r in rows]
    if len(set(keys)) != len(keys):
        problems.append("duplicate (gpu_slug, date, billing_type, reservation_months) keys")
    dates = [_parse_date(r[1]) for r in rows]
    if any(d is None for d in dates):
        problems.append("unparseable date")
        return problems
    if live and dates and (now.date() - max(dates)).days > GD_MAX_AGE_DAYS:
        problems.append(f"newest week {max(dates)} is over {GD_MAX_AGE_DAYS} days old")
    bad = [r for r in rows if not (0 < (_num(r[6]) or 0) <= 200)]
    if bad:
        problems.append(f"{len(bad)} median_price values outside (0, 200] $/GPU-hr, e.g. {bad[0]}")
    h100 = [r for r in rows if r[0] == "nvidia-h100" and r[2] == "ON_DEMAND" and r[3] == ""]
    if live:
        if not h100:
            problems.append("no nvidia-h100 ON_DEMAND rows")
        else:
            latest = max(h100, key=lambda r: r[1])
            v = _num(latest[6]) or 0
            if not 0.5 <= v <= 20:
                problems.append(f"nvidia-h100 ON_DEMAND median {v} outside [0.5, 20] (units?)")
    return problems


def run_getdeploying(session, now: datetime, prev: dict, allow_shrink: bool = False,
                     **_) -> Outcome:
    text = _utf8(_get(session, GD_URL))
    header, rows = _parse_csv(text)
    # allow_shrink: upstream really did trim its window. Without it, the last
    # successful count is the baseline forever and a real prune fails forever.
    last = None if allow_shrink else prev.get("upstream_rows")
    problems = validate_getdeploying(header, rows, now, last)
    if problems:
        raise SourceError("; ".join(problems))
    return _merge_gd_into_archive(rows, upstream_wins=True, upstream_rows=len(rows),
                                  allow_shrink=allow_shrink)


def _merge_gd_into_archive(rows: list[list[str]], upstream_wins: bool,
                           upstream_rows: int, allow_shrink: bool = False) -> Outcome:
    path = resolve(GPU_PRICES)
    before = _read_text(path)
    _, archive = _parse_csv(before)
    merged = merge_getdeploying(archive, sanitise_rows(rows), upstream_wins=upstream_wins)
    if len(merged) < len(archive):          # cannot happen; belt and braces
        raise SourceError(f"merge shrank the archive {len(archive)} -> {len(merged)}")
    text = _csv_text(GD_HEADER, merged)
    problems = _guard(GPU_PRICES, before, text, allow_shrink)
    if problems:
        raise SourceError("; ".join(problems))
    wrote = [GPU_PRICES] if _write_if_changed(path, text) else []
    return Outcome(True, rows=len(merged), upstream_rows=upstream_rows, wrote=wrote,
                   notes=[f"+{len(merged) - len(archive)} rows"])


def seed_getdeploying(snapshot_path: str | Path) -> Outcome:
    """One-time merge of an older download, archive-wins (recovers weeks that
    have already left the rolling window; never overrides newer rows)."""
    header, rows = _parse_csv(Path(snapshot_path).read_text(encoding="utf-8"))
    problems = validate_getdeploying(header, rows, _utcnow(), None, live=False)
    if problems:
        raise SourceError("; ".join(problems))
    return _merge_gd_into_archive(rows, upstream_wins=False, upstream_rows=len(rows))


# ── SOFR ──────────────────────────────────────────────────────────────────────

SOFR_HEADER = ["effective_date", "percent_rate", "volume_bn", "revision_indicator"]
SOFR_MAX_AGE_DAYS = 7


def parse_sofr(payload: dict) -> list[list[str]]:
    """NY Fed refRates -> rows. percent_rate is % p.a. (ACT/360), '%.2f'."""
    rows, problems = [], []
    for rec in (payload or {}).get("refRates") or []:
        if rec.get("type") != "SOFR":
            problems.append(f"type {rec.get('type')!r} is not SOFR")
            continue
        d = _parse_date(rec.get("effectiveDate", ""))
        rate = _num(rec.get("percentRate"))
        if d is None:
            problems.append(f"bad effectiveDate {rec.get('effectiveDate')!r}")
            continue
        if rate is None or not 0 <= rate <= 25:
            problems.append(f"{d}: percentRate {rec.get('percentRate')!r} outside [0, 25] "
                            f"(a basis-point/percent flip?)")
            continue
        vol = _num(rec.get("volumeInBillions"))
        rows.append([d.isoformat(), f"{rate:.2f}", "" if vol is None else str(int(round(vol))),
                     str(rec.get("revisionIndicator") or "").strip()])
    if problems:
        raise SourceError("; ".join(problems[:3]))
    if not rows:
        raise SourceError("no SOFR rows in the response")
    return rows


def merge_sofr(archive_rows: list[list[str]], new_rows: list[list[str]]) -> list[list[str]]:
    """Union on effective_date; a revised rate replaces the old one."""
    merged = {r[0]: r for r in archive_rows}
    merged.update({r[0]: r for r in new_rows})
    return [merged[k] for k in sorted(merged)]


def sofr_request(archive_rows: list[list[str]], today: date) -> tuple[str, dict | None]:
    """Backfill from SOFR's first day when empty; last/10 when current; a
    search from a week before the newest row when the bot has been away."""
    if not archive_rows:
        return SOFR_SEARCH_URL, {"startDate": SOFR_FIRST_DATE, "endDate": today.isoformat()}
    newest = max(_parse_date(r[0]) for r in archive_rows)
    if (today - newest).days <= 10:
        return SOFR_LAST_URL, None
    return SOFR_SEARCH_URL, {"startDate": (newest - timedelta(days=7)).isoformat(),
                             "endDate": today.isoformat()}


def run_sofr(session, now: datetime, prev: dict, allow_shrink: bool = False,
             **_) -> Outcome:
    path = resolve(SOFR_CSV)
    before = _read_text(path)
    _, archive = _parse_csv(before)
    url, params = sofr_request(archive, now.date())
    new = parse_sofr(_get(session, url, params).json())
    newest = max(_parse_date(r[0]) for r in new)
    if (now.date() - newest).days > SOFR_MAX_AGE_DAYS:
        raise SourceError(f"newest SOFR {newest} is over {SOFR_MAX_AGE_DAYS} days old")
    merged = merge_sofr(archive, new)
    text = _csv_text(SOFR_HEADER, merged)
    problems = _guard(SOFR_CSV, before, text, allow_shrink)
    if problems:
        raise SourceError("; ".join(problems))
    wrote = [SOFR_CSV] if _write_if_changed(path, text) else []
    return Outcome(True, rows=len(merged), upstream_rows=len(new), wrote=wrote,
                   notes=[f"latest {merged[-1][0]} {merged[-1][1]}%"])


# ── InferenceX ────────────────────────────────────────────────────────────────

# observed_on is the UTC date the bot first saw a mark, NOT an effective date:
# SemiAnalysis publishes neither when a mark was set nor when it changed. Use
# the latest row per hw_key as "current SemiAnalysis mark, observed <date>",
# never as the mark that applied on a point's run_date.
IX_RATES_HEADER = ["observed_on", "hw_key", "label", "vendor", "arch", "tdp_w",
                   "cost_own_usd_per_chip_hr", "cost_rent_usd_per_chip_hr"]
IX_RATES_ALIASES = {"valid_from": "observed_on"}      # the name before 2026-10-27
IX_POINTS_HEADER = ["point_id", "model", "sequence", "x_field", "hw_key", "gpu", "framework",
                    "spec_method", "deployment", "kv_offload", "precision", "concurrency",
                    "gpu_count", "tensor_parallel", "expert_parallel",
                    "interactivity_tok_s_user", "tput_tok_s_chip", "output_tput_tok_s_chip",
                    "input_tput_tok_s_chip", "run_date", "run_id", "first_seen",
                    # appended 2026-10-27 (older rows conform with blanks)
                    "gpu_count_upstream", "prefill_gpus", "decode_gpus"]
IX_MIN_HARDWARE = 8
# Tolerance for recognising which basis a point's output/input throughput uses.
_IX_BASIS_TOL = 0.02


def run_url(run_id) -> str:
    """The GitHub Actions run behind a point (the html_url /workflow-info returns)."""
    s = str(run_id).strip()
    return IX_RUN_URL.format(run_id=s) if s.isdigit() else ""


def parse_ix_options(payload: dict, fetched_on: str) -> list[list[str]]:
    """options.hardware -> rate rows. h = owning, r = renting, USD per chip-hour
    (SemiAnalysis's large-hyperscaler-volume marks, not market prices).
    `fetched_on` lands in observed_on: the source publishes no effective date."""
    rows, bad = [], []
    for hw in (payload or {}).get("hardware") or []:
        cost = hw.get("costPerHour") or {}
        h, r = _num(cost.get("h")), _num(cost.get("r"))
        key = str(hw.get("key") or "").strip()
        if not key or h is None or r is None or not (0 < h < 50 and 0 < r < 50):
            bad.append(key or "?")
            continue
        rows.append([fetched_on, key, str(hw.get("label") or key), str(hw.get("vendor") or ""),
                     str(hw.get("arch") or ""), _g(_num(hw.get("tdpW"))), _g(h), _g(r)])
    if len(rows) < IX_MIN_HARDWARE:
        raise SourceError(f"options lists {len(rows)} priced hardware (floor {IX_MIN_HARDWARE}); "
                          f"unusable: {bad}")
    return sanitise_rows(rows)


def merge_ix_rates(archive_rows: list[list[str]], new_rows: list[list[str]]) -> list[list[str]]:
    """A change log: a row is appended only when a hardware's (label, vendor,
    arch, tdp_w, h, r) differs from its latest row. Vanished hardware keeps its
    rows; the latest row per hw_key is the current mark."""
    latest: dict[str, list[str]] = {}
    for r in sorted(archive_rows, key=lambda r: (r[1], r[0])):
        latest[r[1]] = r
    out = list(archive_rows)
    for r in new_rows:
        cur = latest.get(r[1])
        if cur is None or cur[2:] != r[2:]:
            out.append(r)
            latest[r[1]] = r
    return sorted(out, key=lambda r: (r[1], r[0]))


def plan_ix_requests(options: dict, availability: list[dict]) -> list[tuple[str, str]]:
    """(model display name, sequence key) pairs that have data.

    dbKeys map to the display name the view endpoint takes; (isl, osl) maps to
    a sequence key, null isl/osl to agentic-traces. Deprecated sequences
    (1k/1k, 1k/8k) are skipped: nothing on the site reads them.
    """
    names = {k: m["name"] for m in options.get("models") or [] for k in m.get("dbKeys") or []}
    seqs = {}
    for s in options.get("sequences") or []:
        if s.get("deprecated"):
            continue
        seqs[(s.get("isl"), s.get("osl"))] = s["key"]
    pairs = set()
    for row in availability or []:
        name = names.get(row.get("model"))
        seq = seqs.get((row.get("isl"), row.get("osl")))
        if name and seq:
            pairs.add((name, seq))
    return sorted(pairs)


def _view_key(model: str, sequence: str) -> str:
    return f"{model} | {sequence}"


def ix_view_params(options: dict, model: str, sequence: str) -> dict:
    """One call per (model, sequence) returns every point.

    Three traps, each verified live on 2026-10-27:
    - without explicit gpus the view silently drops hardware (Qwen3.5 loses
      H100/H200/MI300X/MI325X), so every hardware key is passed;
    - best=true keeps one framework per chip, which drops every spec_method
      'none' and most FP8 series, so best=false;
    - optimal=true returns only each series' frontier for the requested metric
      (and the total/output/input frontiers differ), so optimal=false returns
      all points and each carries all three throughputs.
    JSON, not the brief's format=csv: the CSV form has no precision and no
    point id, and both are needed for a stable append-only key.
    """
    return {
        "model": model, "sequence": sequence, "metric": "tpPerGpu",
        "xmode": "interactivity", "optimal": "false", "best": "false",
        "precisions": ",".join(options.get("precisions") or []),
        "gpus": ",".join(h["key"] for h in options.get("hardware") or []),
        "format": "json",
    }


def _topology(topology_key: str) -> dict[str, str]:
    out = {}
    for tok in str(topology_key or "").split("|"):
        k, sep, v = tok.partition("=")
        if sep and k not in out:
            out[k] = "" if v.strip() == "?" else v.strip()
    return out


def _int_str(v) -> str:
    n = _num(v)
    return "" if n is None else str(int(n))


def physical_gpus(single_node: bool, upstream_gpus, tp, ep) -> int | None:
    """The GPUs a configuration really runs on.

    InferenceX's point.tp (= the topologyKey's GPU=) is TP x EP on single-node
    expert-parallel configurations, not a GPU count: 'Single|GPU=64|TP=8|EP=8'
    runs on one 8-GPU node. Its tpPerGpu is normalised by the 8 physical GPUs
    (DeepSeek-V4-Pro 8k/1k, B200 vLLM, conc 64, 2026-09-03: 1977 tok/s/GPU
    at 'GPU=64' against 1984 at 'Single|GPU=8|TP=8|EP=1'; verified live
    2026-10-27 across 443 such points). There the expert-parallel group reuses
    the tensor-parallel ranks, so the node holds max(TP, EP) GPUs. Multi-node
    and prefill/decode labels already count GPUs (P-GPU + D-GPU = GPU on every
    disaggregated point) and are kept.
    """
    g, t, e = _num(upstream_gpus), _num(tp), _num(ep)
    if g is None:
        return None
    if single_node and t and e and e > 1 and g == t * e:
        return int(max(t, e))
    return int(g)


def ix_rebase_throughput(tp: float, out_t: float | None, in_t: float | None,
                         gpus: int | None, prefill: int | None,
                         decode: int | None) -> tuple[float | None, float | None]:
    """Output and input tok/s per GPU of the WHOLE deployment, whatever basis
    upstream used.

    Upstream's outputTputPerGpu / inputTputPerGpu are per total GPU almost
    everywhere (output + input = tpPerGpu), but on fixed-sequence (8k/1k)
    disaggregated points they are per DECODE GPU and per PREFILL GPU: point
    438832 (GB300, P-GPU=60, D-GPU=12): 13716.5 x 72 / 9 / 9143.6 = 12.0
    decode GPUs. Pricing that column at a per-chip rate would bill only the
    decode pool (0.07 instead of 0.46 $/M output on point 424199). The basis
    is recognised from the numbers, not from the sequence name: a point whose
    basis matches neither is returned blank rather than guessed.
    """
    if out_t is None or in_t is None or not tp:
        return out_t, in_t
    if abs(out_t + in_t - tp) <= _IX_BASIS_TOL * tp:
        return out_t, in_t
    if gpus and prefill and decode and \
            abs(out_t * decode + in_t * prefill - tp * gpus) <= _IX_BASIS_TOL * tp * gpus:
        return out_t * decode / gpus, in_t * prefill / gpus
    return None, None


def parse_ix_view(payload: dict, model: str, sequence: str, fetched_on: str,
                  labels: dict[str, str] | None = None) -> tuple[list[list[str]], int]:
    """A view -> point rows, and the number of points dropped as invalid.

    hw_key is the hardware key (joins inferencex_rates.hw_key); gpu is its
    display label. x_field records whether interactivity is the median
    (fixed-sequence views) or p90 (agentic views): never mix the two.

    Normalised so that gpu_count x tok/s/GPU is the deployment's throughput
    and a per-chip-hour price applies to every throughput column alike:
    - gpu_count is physical_gpus(); upstream's label is kept in
      gpu_count_upstream;
    - tput / output / input are per GPU of the whole deployment
      (ix_rebase_throughput); upstream's per-pool values are
      output x gpu_count / decode_gpus and input x gpu_count / prefill_gpus;
    - prefill_gpus / decode_gpus come from the topologyKey (P-GPU, D-GPU) on
      disaggregated points and are blank elsewhere.
    """
    labels = labels or {}
    x_field = str(((payload or {}).get("xAxis") or {}).get("field") or "")
    rows, dropped = [], 0
    for s in (payload or {}).get("series") or []:
        hw = str(s.get("gpu") or "")
        for p in s.get("points") or []:
            m = p.get("metrics") or {}
            x, tp = _num(p.get("x")), _num(m.get("tpPerGpu"))
            pid = _int_str(p.get("id"))
            if not pid or x is None or x <= 0 or tp is None or tp <= 0:
                dropped += 1
                continue
            key = str(p.get("topologyKey") or "")
            topo = _topology(key)
            upstream_gpus = p.get("tp") if _num(p.get("tp")) is not None else topo.get("GPU")
            gpus = physical_gpus(key.split("|", 1)[0] == "Single", upstream_gpus,
                                 topo.get("TP"), topo.get("EP"))
            pre, dec = _num(topo.get("P-GPU")), _num(topo.get("D-GPU"))
            out_t, in_t = ix_rebase_throughput(
                tp, _num(m.get("outputTputPerGpu")), _num(m.get("inputTputPerGpu")),
                gpus, pre, dec)
            rows.append([
                pid, model, sequence, x_field, hw, labels.get(hw, hw),
                str(s.get("framework") or ""), str(s.get("specMethod") or ""),
                str(s.get("deployment") or ""), str(bool(s.get("kvOffload"))).lower(),
                str(p.get("precision") or ""), _int_str(p.get("concurrency")),
                "" if gpus is None else str(gpus), topo.get("TP", ""), topo.get("EP", ""),
                _g(x), _g(tp), _g(out_t), _g(in_t),
                str(p.get("date") or ""), _int_str(p.get("runId")), fetched_on,
                _int_str(upstream_gpus), _int_str(pre), _int_str(dec),
            ])
    return sanitise_rows(rows), dropped


_IX_COL = {c: i for i, c in enumerate(IX_POINTS_HEADER)}
_IX_FIRST_SEEN = _IX_COL["first_seen"]


def conform_ix_points(header: list[str], rows: list[list[str]]) -> list[list[str]]:
    """Archive rows from before the 2026-10-27 schema change, laid out under
    IX_POINTS_HEADER. Their gpu_count was upstream's label, so it moves to
    gpu_count_upstream and gpu_count is recomputed (every single-node point is
    a 'Single' topology). Their throughputs keep upstream's basis and their
    prefill/decode split stays blank: a point still in the view is rewritten
    with both on the next fetch (the merge is upstream-wins)."""
    if header == IX_POINTS_HEADER:
        return rows
    out = _conform(header, rows, IX_POINTS_HEADER)
    if "gpu_count_upstream" not in header:
        c = _IX_COL
        for r in out:
            r[c["gpu_count_upstream"]] = r[c["gpu_count"]]
            g = physical_gpus(r[c["deployment"]] == "single-node", r[c["gpu_count"]],
                              r[c["tensor_parallel"]], r[c["expert_parallel"]])
            r[c["gpu_count"]] = "" if g is None else str(g)
    return out


def _ix_key(r: list[str]) -> tuple[str, str]:
    return (r[0], r[3])        # point_id, x_field


def _ix_sort(r: list[str]) -> tuple:
    def i(v):
        n = _num(v)
        return -1 if n is None else int(n)
    return (r[1], r[2], r[4], r[6], r[7], r[10], i(r[12]), i(r[11]), i(r[0]), r[3])


def merge_ix_points(archive_rows: list[list[str]], new_rows: list[list[str]]) -> list[list[str]]:
    """Union keyed by (point_id, x_field). Never deletes: a point that leaves
    the view's window is exactly what the archive is for. first_seen is the
    archive's when the point was seen before; the other cells are upstream's."""
    merged = {_ix_key(r): r for r in archive_rows}
    for r in new_rows:
        k = _ix_key(r)
        if k in merged:
            r = list(r)
            r[_IX_FIRST_SEEN] = merged[k][_IX_FIRST_SEEN]
        merged[k] = r
    return sorted(merged.values(), key=_ix_sort)


def run_inferencex(session, now: datetime, prev: dict, deadline: float | None = None,
                   sleep_s: float = 1.0, allow_shrink: bool = False, **_) -> Outcome:
    today = now.date().isoformat()
    options = _get(session, IX_BASE + "/views/options", {"format": "json"}).json()
    rate_rows = parse_ix_options(options, today)
    availability = _get(session, IX_BASE + "/availability").json()
    if isinstance(availability, dict):
        availability = availability.get("rows") or availability.get("data") or []
    plan = plan_ix_requests(options, availability)
    if not plan:
        raise SourceError("availability planned no (model, sequence) views")
    labels = {h["key"]: h.get("label") or h["key"] for h in options.get("hardware") or []}

    new_points: list[list[str]] = []
    failed, failed_keys, pending = [], [], []
    done, dropped, budget_hit = 0, 0, False
    for i, (model, seq) in enumerate(plan):
        if budget_hit or (deadline is not None and time.monotonic() >= deadline):
            budget_hit = True
            pending.append(_view_key(model, seq))
            continue
        if i and sleep_s:
            time.sleep(sleep_s)
        try:
            payload = _get(session, IX_BASE + "/views/inference",
                           ix_view_params(options, model, seq)).json()
            rows, d = parse_ix_view(payload, model, seq, today, labels)
            new_points.extend(rows)
            dropped += d
            done += 1
        except Exception as e:                        # one view never sinks the rest
            failed.append(f"{model} {seq}: {type(e).__name__}: {str(e)[:120]}")
            failed_keys.append(_view_key(model, seq))

    # Rates first: they come from one call that already succeeded.
    wrote = []
    rp = resolve(IX_RATES)
    before_r = _read_text(rp)
    head_r, arch_r = _parse_csv(before_r)
    arch_r = _conform(head_r, arch_r, IX_RATES_HEADER, IX_RATES_ALIASES)
    text_r = _csv_text(IX_RATES_HEADER, merge_ix_rates(arch_r, rate_rows))
    problems = _guard(IX_RATES, before_r, text_r, allow_shrink)
    if problems:
        raise SourceError("; ".join(problems))
    if _write_if_changed(rp, text_r):
        wrote.append(IX_RATES)

    pp = resolve(IX_POINTS)
    before_p = _read_text(pp)
    head_p, arch_p = _parse_csv(before_p)
    arch_p = conform_ix_points(head_p, arch_p)
    merged = merge_ix_points(arch_p, new_points)
    if new_points:              # partial results still merge: the archive is append-only
        text_p = _csv_text(IX_POINTS_HEADER, merged)
        problems = _guard(IX_POINTS, before_p, text_p, allow_shrink)
        if problems:
            raise SourceError("; ".join(problems))
        if _write_if_changed(pp, text_p):
            wrote.append(IX_POINTS)

    notes = [f"{done}/{len(plan)} views, {len(new_points)} points, "
             f"+{len(merged) - len(arch_p)} new, {dropped} invalid dropped"]
    notes.append(_ix_summary(new_points))
    if failed:
        notes.append("failed views: " + " | ".join(failed[:5]))
    error = warn = ""
    if budget_hit:
        error = f"budget: stopped after {done + len(failed)}/{len(plan)} views"
    elif done * 2 <= len(plan):
        error = f"only {done}/{len(plan)} views succeeded"
    elif failed:
        # A minority failing is still a success (the rest merged), but never a
        # silent one: it is a ::warning now, the source retries on its short
        # interval, and a view failing for ESCALATE_AFTER_H turns the run red.
        warn = f"{len(failed)}/{len(plan)} views failed: " + " | ".join(failed[:3])
    return Outcome(not error, rows=len(merged), upstream_rows=len(new_points),
                   error=error, wrote=wrote, notes=notes, warn=warn,
                   failed_views=failed_keys, pending_views=pending)


def _ix_summary(rows: list[list[str]]) -> str:
    """Distinct deployment / spec_method / precision values, the physical
    gpu_count range and how many points had an unrecognised throughput basis,
    logged so a consumer can see what like-for-like slices exist."""
    if not rows:
        return "no points"
    col = _IX_COL
    def vals(c):
        return sorted({r[col[c]] for r in rows})
    counts = [int(r[col["gpu_count"]]) for r in rows if r[col["gpu_count"]]]
    unknown = sum(1 for r in rows if r[col["output_tput_tok_s_chip"]] == "")
    rng = f"{min(counts)}..{max(counts)}" if counts else "?"
    return (f"deployment={vals('deployment')} spec_method={vals('spec_method')} "
            f"precision={vals('precision')} gpu_count={rng} "
            f"unrecognised_throughput_basis={unknown}")


# ── Epoch AI mirrors ──────────────────────────────────────────────────────────

# Required columns (matched case-insensitively: Epoch uses both 'Source type'
# and 'Source Type'), and per-file unit ranges so bad units never land.
EPOCH_REQUIRED = {
    "ml_hardware.csv": ["Hardware name", "Release price (USD)", "TDP (W)",
                        "Memory bandwidth (byte/s)", "FP8 performance (FLOP/s)"],
    "ai_companies_revenue_reports.csv": ["Company", "Date", "Annualized revenue (USD)",
                                         "Confidence", "Source type"],
    "ai_companies_compute_spend.csv": ["Company", "Inference compute spend",
                                       "Total compute spend", "Confidence"],
    "ai_companies_funding_rounds.csv": ["Company", "Close date", "Valuation (post-money)",
                                        "Confidence"],
    "ai_companies_usage_reports.csv": ["Company", "Date", "Daily tokens"],
}
EPOCH_RANGES = {
    "ai_companies_revenue_reports.csv": {"Annualized revenue (USD)": (1e6, 1e12)},
    "ai_companies_usage_reports.csv": {"Daily tokens": (1e8, 1e16)},
}
# Event dates in the company files. Hardware release dates go back to 2008, so
# ml_hardware is not date-bounded.
EPOCH_DATES = {
    "ai_companies_revenue_reports.csv": ["Date"],
    "ai_companies_compute_spend.csv": ["Date"],
    "ai_companies_funding_rounds.csv": ["Close date"],
    "ai_companies_usage_reports.csv": ["Date"],
}
EPOCH_MIN_DATE = date(2015, 1, 1)


def _col_index(header: list[str], name: str) -> int | None:
    want = name.strip().lower()
    for i, h in enumerate(header):
        if h.strip().lower() == want:
            return i
    return None


def validate_epoch(basename: str, header: list[str], rows: list[list[str]],
                   today: date) -> list[str]:
    problems = []
    missing = [c for c in EPOCH_REQUIRED.get(basename, []) if _col_index(header, c) is None]
    if missing:
        return [f"{basename}: missing columns {missing}"]
    if any(len(r) != len(header) for r in rows):
        problems.append(f"{basename}: ragged rows")
    for col, (lo, hi) in EPOCH_RANGES.get(basename, {}).items():
        i = _col_index(header, col)
        for r in rows:
            cell = r[i].strip() if i < len(r) else ""
            if not cell:
                continue
            v = _num(cell.replace(",", ""))
            if v is None or not lo <= v <= hi:
                problems.append(f"{basename}: {col} {cell!r} outside [{lo:g}, {hi:g}]")
                break
    latest = today + timedelta(days=31)
    for col in EPOCH_DATES.get(basename, []):
        i = _col_index(header, col)
        for r in rows:
            cell = r[i].strip() if i < len(r) else ""
            if not cell:
                continue
            d = _parse_date(cell)
            if d is None or not EPOCH_MIN_DATE <= d <= latest:
                problems.append(f"{basename}: {col} {cell!r} unparseable or outside "
                                f"[{EPOCH_MIN_DATE}, {latest}]")
                break
    return problems


def mirror_epoch_file(session, basename: str, today: date,
                      allow_shrink: bool = False) -> tuple[bool, int, list[str]]:
    """Fetch, validate, sanitise and write one Epoch CSV. Returns
    (written, rows, problems); nothing is written when problems is non-empty.
    Decoded with _utf8: epoch.ai sends no charset (see there)."""
    canonical = EPOCH_FILES[basename]
    header, rows = _parse_csv(_utf8(_get(session, EPOCH_BASE + basename)))
    problems = validate_epoch(basename, header, rows, today)
    if problems:
        return False, len(rows), problems
    path = resolve(canonical)
    before = _read_text(path)
    text = _csv_text(sanitise_rows([header])[0], sanitise_rows(rows))
    problems = _guard(canonical, before, text, allow_shrink)
    if problems:
        return False, len(rows), problems
    return _write_if_changed(path, text), len(rows), []


def _run_epoch(session, now: datetime, basenames: list[str],
               allow_shrink: bool = False) -> Outcome:
    wrote, problems, total = [], [], 0
    for b in basenames:
        try:
            changed, n, p = mirror_epoch_file(session, b, now.date(), allow_shrink)
        except Exception as e:
            changed, n, p = False, 0, [f"{b}: {type(e).__name__}: {str(e)[:160]}"]
        total += n
        problems.extend(p)
        if changed:
            wrote.append(EPOCH_FILES[b])
    return Outcome(not problems, rows=total, upstream_rows=total,
                   error="; ".join(problems), wrote=wrote)


def run_epoch_hardware(session, now: datetime, prev: dict, allow_shrink: bool = False,
                       **_) -> Outcome:
    return _run_epoch(session, now, ["ml_hardware.csv"], allow_shrink)


def run_epoch_companies(session, now: datetime, prev: dict, allow_shrink: bool = False,
                        **_) -> Outcome:
    return _run_epoch(session, now, [b for b in EPOCH_FILES if b != "ml_hardware.csv"],
                      allow_shrink)


RUNNERS: dict[str, Callable[..., Outcome]] = {
    "sofr": run_sofr,
    "getdeploying": run_getdeploying,
    "epoch_hardware": run_epoch_hardware,
    "epoch_companies": run_epoch_companies,
    "inferencex": run_inferencex,           # last: the long one, under the budget
}


# ── cadence, state, escalation ───────────────────────────────────────────────

def load_state(path: Path | None = None) -> dict:
    path = path or ECON_DIR / STATE_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict, path: Path | None = None) -> bool:
    path = path or ECON_DIR / STATE_FILE
    return _write_if_changed(path, json.dumps(state, sort_keys=True, indent=2) + "\n")


def due(state: dict, name: str, now: datetime, force: bool = False) -> bool:
    """Never checked -> due. Otherwise the source's cadence after a clean
    success, and its shorter retry interval after a failure or after a success
    in which some views failed (so one dead view is retried, not left alone
    for six days)."""
    if force:
        return True
    entry = state.get(name) or {}
    checked = _parse_ts(entry.get("checked_at"))
    if checked is None:
        return True
    src = SOURCES[name]
    clean = entry.get("ok") and not entry.get("view_failing_since")
    wait_h = src.cadence_h if clean else src.retry_h
    return (now - checked) >= timedelta(hours=wait_h)


def escalate(state: dict, now: datetime) -> list[str]:
    """Sources, and single views of a multi-call source, failing for
    ESCALATE_AFTER_H or longer: these fail the run."""
    out = []
    limit = timedelta(hours=ESCALATE_AFTER_H)
    for name in sorted(state):
        entry = state.get(name) or {}
        since = _parse_ts(entry.get("failing_since"))
        if since is not None and now - since >= limit:
            hours = (now - since).total_seconds() / 3600
            out.append(f"{name}: failing for {hours:.0f}h: {entry.get('error', '')}")
        for view, ts in sorted((entry.get("view_failing_since") or {}).items()):
            vsince = _parse_ts(ts)
            if vsince is not None and now - vsince >= limit:
                hours = (now - vsince).total_seconds() / 3600
                out.append(f"{name}: view '{view}' failing for {hours:.0f}h")
    return out


def _record(state: dict, name: str, now: datetime, out: Outcome) -> None:
    prev = state.get(name) or {}
    entry = {
        "checked_at": _iso(now),
        "fetched_at": _iso(now) if out.ok else prev.get("fetched_at"),
        "ok": bool(out.ok),
        "rows": out.rows if out.rows is not None else prev.get("rows"),
        "upstream_rows": (out.upstream_rows if out.ok and out.upstream_rows is not None
                          else prev.get("upstream_rows")),
        "error": out.error[:500],
        "failing_since": None if out.ok else (prev.get("failing_since") or _iso(now)),
    }
    # Per-view failure clocks. A view that failed keeps the time it first
    # failed; one that succeeded (or left the plan) is dropped; one the budget
    # never reached keeps its clock. A run that never got as far as the views
    # (out.failed_views is None) leaves the clocks as they were.
    prev_views = prev.get("view_failing_since") or {}
    if out.failed_views is None:
        views = dict(prev_views)
    else:
        views = {v: prev_views.get(v) or _iso(now) for v in out.failed_views}
        views.update({v: prev_views[v] for v in out.pending_views if v in prev_views})
    if views:
        entry["view_failing_since"] = dict(sorted(views.items()))
    state[name] = entry


def run(only: Iterable[str] | None = None, force: bool = False, now: datetime | None = None,
        session=None, budget_s: float = DEFAULT_BUDGET_S, sleep_s: float = 1.0,
        allow_shrink: bool = False) -> dict:
    """Fetch every due source; each in its own try/except. Returns
    {source: {"status": skipped|ok|failed, ...}}. fetch_state.json is written
    only when something was attempted.

    allow_shrink is the remedy for an upstream that really pruned: it waives
    the relative checks (GetDeploying's upstream-shrink, the mirror row-drop
    limit, the median shift) for this run, which resets their baselines, and
    makes every source whose last attempt failed due at once. Absolute floors,
    validation ranges and the append-only rule still apply."""
    now = now or _utcnow()
    session = session or _Requests()
    state = load_state()
    names = [n for n in RUNNERS if only is None or n in set(only)]
    deadline = time.monotonic() + budget_s
    report: dict[str, dict] = {}
    attempted = False
    for name in names:
        entry = state.get(name) or {}
        retry_now = allow_shrink and entry.get("checked_at") and not entry.get("ok")
        if not (retry_now or due(state, name, now, force)):
            report[name] = {"status": "skipped"}
            continue
        if time.monotonic() >= deadline:
            report[name] = {"status": "skipped", "reason": "budget"}
            continue
        attempted = True
        prev = state.get(name) or {}
        try:
            out = RUNNERS[name](session, now, prev, deadline=deadline, sleep_s=sleep_s,
                                allow_shrink=allow_shrink)
        except Exception as e:
            out = Outcome(False, error=f"{type(e).__name__}: {str(e)[:400]}")
        _record(state, name, now, out)
        report[name] = {"status": "ok" if out.ok else "failed", "error": out.error,
                        "rows": out.rows, "wrote": out.wrote, "notes": out.notes,
                        "warn": out.warn}
    if attempted:
        save_state(state)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Archive external economics sources.")
    ap.add_argument("--force", action="store_true", help="fetch every source now")
    ap.add_argument("--only", default="", help="comma-separated source names")
    ap.add_argument("--allow-shrink", action="store_true",
                    help="upstream really pruned: waive the relative row-drop and "
                         "median checks for this run (the workflow's allow_shrink input)")
    ap.add_argument("--seed-getdeploying", metavar="PATH",
                    help="one-time: merge an older GetDeploying weekly.csv (archive wins)")
    args = ap.parse_args(argv)

    if args.seed_getdeploying:
        out = seed_getdeploying(args.seed_getdeploying)
        print(f"[econ] seeded getdeploying: {out.rows} rows ({', '.join(out.notes)}); "
              f"wrote {out.wrote or 'nothing'}")
        return 0

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    unknown = [s for s in only or [] if s not in RUNNERS]
    if unknown:
        print(f"[econ] unknown source(s): {unknown}; known: {list(RUNNERS)}")
        return 2
    now = _utcnow()
    report = run(only=only, force=args.force, now=now, allow_shrink=args.allow_shrink)
    for name, r in report.items():
        line = f"[econ] {name}: {r['status']}"
        if r.get("reason"):
            line += f" ({r['reason']})"
        if r["status"] != "skipped":
            line += f", {r.get('rows')} rows, wrote {r.get('wrote') or 'nothing'}"
            for n in r.get("notes") or []:
                line += f"\n[econ]   {n}"
        if r.get("error"):
            line += f"\n[econ]   error: {r['error']}"
        print(line)
        if r["status"] == "failed":
            # A failure inside the grace period is a warning; escalate() below
            # turns a persistent one into a failed run.
            print(f"::warning title=Economics archive::{name} failed: {r.get('error', '')[:200]}")
        elif r.get("warn"):
            print(f"[econ]   warning: {r['warn']}")
            print(f"::warning title=Economics archive::{name}: {r['warn'][:200]}")
    problems = escalate(load_state(), now)
    for p in problems:
        print(f"::error title=Economics archive stale::{p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
