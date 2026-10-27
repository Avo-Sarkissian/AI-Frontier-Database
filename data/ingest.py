"""
Data ingestion: scrapes live model data from Artificial Analysis leaderboard.
Falls back to cached CSV if scraping fails.
Each successful scrape is also saved as a timestamped snapshot for trend tracking.
"""
import re
import json
from datetime import date
import pandas as pd
from pathlib import Path

RAW_DIR    = Path(__file__).parent / "raw"
CACHE_PATH = RAW_DIR / "aa_models.csv"
HIST_DIR   = RAW_DIR / "history"


def _parse_price(val: str) -> float:
    """'$4.50' → 4.50, '$0.00' → 0.0"""
    try:
        return float(val.replace("$", "").replace(",", ""))
    except (ValueError, AttributeError):
        return float("nan")


def _parse_numeric(val: str) -> float:
    """'1,196' → 1196.0, '34E' → 34.0, '--' → nan"""
    if not val or val in ("--", ""):
        return float("nan")
    # Remove thousands-separator commas before anything else, so "1,196" → "1196"
    val = val.replace(",", "")
    # Strip trailing 'E' unit suffix (e.g. speed values like "34E")
    val = re.sub(r"[Ee].*$", "", val)
    try:
        return float(val)
    except ValueError:
        return float("nan")


def _parse_float(val) -> float:
    """A plain float, or NaN — safe for the e-notation `str(float)` can write.

    The late AA fields (cost per task, cache prices) are written by
    `str(float(v))`, which renders small values as "5e-05". `_parse_numeric`
    strips everything from the first E onward (it was written for a "34E" speed
    suffix), so "5e-05" would come back as 5.0 — a 100,000x error with nothing
    to flag it. `_parse_price` would survive, but it also strips "$" and ",",
    which a cost-per-task field never carries. This parses exactly one number.
    """
    if val is None or val is pd.NA:
        return float("nan")
    try:
        out = float(str(val).strip())
    except (TypeError, ValueError):
        return float("nan")
    return out if out == out and out not in (float("inf"), float("-inf")) else float("nan")


def _tri_bool(val):
    """True / False / pd.NA from whatever a CSV round-trip hands back.

    AA's isOpenWeights and isReasoning are written as "1" / "0" / "" — and an
    empty cell means "AA did not say", which is NOT False. Collapsing it to
    False would file every model scraped before the field was captured (every
    history snapshot) as closed-weights and non-reasoning, which is a claim
    about the model rather than about our records.
    read_csv returns these as int64 (no blanks), float64 (blanks) or object
    (mixed with older "True"/"False" text), so all three shapes are accepted.
    """
    if val is None or val is pd.NA:
        return pd.NA
    if isinstance(val, bool):
        return val
    try:
        import numpy as _np
        if isinstance(val, _np.bool_):
            return bool(val)
        if isinstance(val, (int, float, _np.integer, _np.floating)):
            if val != val:
                return pd.NA
            return True if val == 1 else False if val == 0 else pd.NA
    except ImportError:          # pragma: no cover - numpy ships with pandas
        pass
    text = str(val).strip().lower()
    if text in ("1", "1.0", "true"):
        return True
    if text in ("0", "0.0", "false"):
        return False
    return pd.NA


def _tri_bool_str(val) -> str:
    """The CSV spelling of a nullable boolean: "1", "0", or "" for unknown."""
    v = _tri_bool(val)
    return "" if v is pd.NA else ("1" if v else "0")


def load_from_raw(raw_rows: list) -> pd.DataFrame:
    """Parse raw scraped row data into a clean DataFrame."""
    records = []
    for row in raw_rows:
        if len(row) < 7:
            continue
        model     = row[0].strip()
        context   = row[1].strip()
        provider  = row[2].strip()
        quality   = _parse_numeric(row[3])
        price     = _parse_price(row[4])
        speed     = _parse_numeric(row[5])
        latency   = _parse_numeric(row[6])
        # Input/output rates are optional: `price` is the 3:1 blend and stays the
        # cost basis, but the blend is not the number anyone quotes, so the raw
        # sides are carried through for display. Older rows have only 7 fields.
        price_in  = _parse_price(row[7]) if len(row) > 7 else float("nan")
        price_out = _parse_price(row[8]) if len(row) > 8 else float("nan")
        # AA's 2026-09 intelligence overhaul, appended at 9..12 so the 7- and
        # 9-field rows already on disk keep parsing. quality_estimated defaults
        # to False for those: a cache written before AA published the flag says
        # nothing about estimation, and False is what every consumer already
        # assumed. The three sibling indices default to NaN, never 0 — 0 is a
        # real score on all three scales.
        quality_estimated = str(row[9]).strip() in ("1", "True", "true") \
            if len(row) > 9 else False
        coding      = _parse_numeric(row[10]) if len(row) > 10 else float("nan")
        agentic     = _parse_numeric(row[11]) if len(row) > 11 else float("nan")
        omniscience = _parse_numeric(row[12]) if len(row) > 12 else float("nan")
        # Fields AA's leaderboard records carried all along and the scrape used
        # to drop, appended at 13..19 (2026-10-27). Parsed with _parse_float,
        # never _parse_numeric: these are written with e-notation when small.
        def _at(i):
            return row[i] if len(row) > i else None
        cost_per_task     = _parse_float(_at(13))
        # AA publishes 0 for five models whose index runs were never costed;
        # a $0 task is not a measurement, so it reads as "not published".
        if not cost_per_task > 0:
            cost_per_task = float("nan")
        cache_hit_price   = _parse_float(_at(14))
        cache_write_price = _parse_float(_at(15))
        if cache_hit_price < 0:
            cache_hit_price = float("nan")
        if cache_write_price < 0:
            cache_write_price = float("nan")
        is_open_weights = _tri_bool(_at(16))
        is_reasoning    = _tri_bool(_at(17))
        release_date    = _iso_date(_at(18))
        slug_text       = "" if _at(19) is None or _at(19) is pd.NA else str(_at(19)).strip()
        aa_slug         = slug_text if slug_text and slug_text.lower() != "nan" else float("nan")

        # Skip rows with no useful data
        if pd.isna(quality) or model == "":
            continue

        records.append({
            "model":     model,
            "provider":  provider,
            "context":   context,
            "quality":   quality,
            "price":     price,
            "speed":     speed,
            "latency":   latency,
            "price_in":  price_in,
            "price_out": price_out,
            "quality_estimated": quality_estimated,
            "coding":      coding,
            "agentic":     agentic,
            "omniscience": omniscience,
            "cost_per_task":     cost_per_task,
            "cache_hit_price":   cache_hit_price,
            "cache_write_price": cache_write_price,
            "is_open_weights":   is_open_weights,
            "is_reasoning":      is_reasoning,
            "release_date":      release_date,
            "aa_slug":           aa_slug,
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["model", "provider"])
    df = df[df["price"] > 0].copy()          # only models with public pricing
    df = df[df["quality"] > 0].copy()        # drop models with no quality score
    df = df.sort_values("quality", ascending=False).reset_index(drop=True)
    return _coerce_late_dtypes(df)


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _iso_date(val):
    """'2026-09-22' -> '2026-09-22'; anything that is not a real ISO date -> NaN.

    Kept as text, not a datetime: it round-trips through the CSV and the
    history snapshots unchanged, and the consumers that need arithmetic parse
    it themselves.
    """
    if val is None or val is pd.NA:
        return float("nan")
    text = str(val).strip()[:10]
    if not _ISO_DATE_RE.match(text):
        return float("nan")
    try:
        date.fromisoformat(text)
    except ValueError:
        return float("nan")
    return text


def load_cached() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return _with_price_sides(pd.read_csv(CACHE_PATH))
    raise FileNotFoundError("No cached data found. Run scrape first.")


# Columns added after the first cache was written, with the value that means
# "we do not know" for each. Every archived history snapshot under
# data/raw/history/ predates all of them.
_LATE_COLUMNS: dict[str, object] = {
    "price_in":   float("nan"),
    "price_out":  float("nan"),
    # AA's 2026-09 overhaul. False, not NaN: "not flagged as estimated" is what
    # every pre-overhaul row meant and what every consumer already assumed, and
    # a NaN here would make `if row["quality_estimated"]` truthy.
    "quality_estimated": False,
    # NaN, not 0 — 0 is a real score on all three of these scales.
    "coding":      float("nan"),
    "agentic":     float("nan"),
    "omniscience": float("nan"),
    # Fields AA's leaderboard records carried all along and the scrape dropped
    # until 2026-10-27 (data/scraper.py, "THE LATE AA FIELDS"). Every one is
    # "unknown" when absent: NaN for the numbers and text, pd.NA for the two
    # booleans — never False, because "AA did not say" is not "closed weights".
    # cost_per_task is USD per AA Intelligence Index task; the cache prices are
    # USD per 1M cached-input tokens (0 is a real, published price).
    "cost_per_task":     float("nan"),
    "cache_hit_price":   float("nan"),
    "cache_write_price": float("nan"),
    "is_open_weights":   pd.NA,
    "is_reasoning":      pd.NA,
    "release_date":      float("nan"),
    "aa_slug":           float("nan"),
}

# The nullable booleans among them: "1"/"0"/"" on disk, pandas "boolean" in
# memory, so `.eq(True)` / `.eq(False)` read them and NA stays NA.
TRI_BOOL_COLUMNS = ("is_open_weights", "is_reasoning")
_LATE_FLOAT_COLUMNS = ("cost_per_task", "cache_hit_price", "cache_write_price")
# The columns appended on 2026-10-27, in their on-disk order (indices 13..19).
AA_LATE_FIELDS = ("cost_per_task", "cache_hit_price", "cache_write_price",
                  "is_open_weights", "is_reasoning", "release_date", "aa_slug")


def _coerce_late_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """One dtype per late column, whatever read_csv or a raw parse produced.

    Without this the same column is int64 in one snapshot, float64 in the next
    and object in a third, and a concat of history turns the booleans into
    floats where NaN is truthy.
    """
    for col in TRI_BOOL_COLUMNS:
        if col in df.columns and str(df[col].dtype) != "boolean":
            df[col] = pd.array([_tri_bool(v) for v in df[col]], dtype="boolean")
    for col in _LATE_FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def _with_price_sides(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the late-added columns exist.

    A cache written before they existed — and every archived history snapshot —
    has only the older shape. Charts and the detail panel read these columns
    unconditionally, so materialise them rather than making every caller test
    for them; a missing value renders as "—", never as a wrong number.
    """
    for col, missing in _LATE_COLUMNS.items():
        if col not in df.columns:
            df[col] = missing
    return _coerce_late_dtypes(df)


def _upgrade_snapshot(snap_path: Path, safe: pd.DataFrame) -> bool:
    """Add columns today's snapshot lacks, without touching what it holds.

    One snapshot is written per calendar day, by the first scrape of the day.
    On the day a new column ships, that first scrape ran the OLD code, so the
    day's snapshot would miss the column until tomorrow — losing one day of a
    series (cost per task) whose whole value is that it accrues from now on.

    Every existing cell is kept byte-for-byte (read as text, NA handling off):
    the snapshot still records the first scrape of the day. Only the missing
    columns are filled, joined on model + provider from the current scrape,
    and a model the current scrape no longer lists gets blanks, not a guess.
    Returns True when the file was rewritten.
    """
    import io

    have = pd.read_csv(snap_path, dtype=str, keep_default_na=False)
    missing = [c for c in safe.columns if c not in have.columns]
    if not missing or not {"model", "provider"} <= set(have.columns):
        return False
    buf = io.StringIO()
    safe.to_csv(buf, index=False)
    buf.seek(0)
    fresh = pd.read_csv(buf, dtype=str, keep_default_na=False)
    fresh = fresh.drop_duplicates(subset=["model", "provider"])
    out = have.merge(fresh[["model", "provider"] + missing],
                     on=["model", "provider"], how="left").fillna("")
    order = [c for c in safe.columns if c in out.columns]
    order += [c for c in have.columns if c not in order and c != "scraped_at"]
    if "scraped_at" in out.columns:
        order.append("scraped_at")
    out[order].to_csv(snap_path, index=False)
    return True


def save_cache(df: pd.DataFrame):
    from static_helpers import csv_safe

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Sanitised on the way IN as well as on export: these files are committed
    # hourly and opened by hand. See static_helpers.csv_safe.
    safe = csv_safe(df)
    # Nullable booleans as "1" / "0" / "" — pandas would write "True"/"False"
    # and an empty cell, and the explicit spelling is what _tri_bool documents.
    for col in TRI_BOOL_COLUMNS:
        if col in safe.columns:
            safe[col] = safe[col].map(_tri_bool_str)
    safe.to_csv(CACHE_PATH, index=False)
    # Also save a timestamped snapshot for trend tracking
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    snap_path = HIST_DIR / f"aa_models_{date.today().isoformat()}.csv"
    if not snap_path.exists():           # one snapshot per calendar day
        snap = safe.copy()
        snap["scraped_at"] = date.today().isoformat()
        snap.to_csv(snap_path, index=False)
    else:
        # The day a column ships, the day's snapshot predates it (see
        # _upgrade_snapshot). A no-op on every other scrape of every other day.
        _upgrade_snapshot(snap_path, safe)


def load_history() -> pd.DataFrame:
    """Load all timestamped snapshots into one long DataFrame.

    Each snapshot is normalised on its own before the concat: snapshots span
    four column shapes, and a column present in some files but not others
    would otherwise reach the concat as float NaN and turn the nullable
    booleans into floats where NaN is truthy.
    """
    if not HIST_DIR.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(HIST_DIR.glob("aa_models_*.csv")):
        snap = pd.read_csv(path)
        if "scraped_at" not in snap.columns:
            snap["scraped_at"] = path.stem.replace("aa_models_", "")
        frames.append(_with_price_sides(snap))
    if not frames:
        return pd.DataFrame()
    return _with_price_sides(pd.concat(frames, ignore_index=True))


def get_models(raw_rows: list | None = None) -> pd.DataFrame:
    """
    Primary entry point. Pass raw_rows from Playwright scrape,
    or leave None to load from cache.
    """
    if raw_rows:
        df = load_from_raw(raw_rows)
        save_cache(df)
        return df
    return load_cached()
