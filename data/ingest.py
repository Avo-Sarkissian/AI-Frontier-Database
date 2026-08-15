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
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["model", "provider"])
    df = df[df["price"] > 0].copy()          # only models with public pricing
    df = df[df["quality"] > 0].copy()        # drop models with no quality score
    df = df.sort_values("quality", ascending=False).reset_index(drop=True)
    return df


def load_cached() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return _with_price_sides(pd.read_csv(CACHE_PATH))
    raise FileNotFoundError("No cached data found. Run scrape first.")


def _with_price_sides(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee price_in / price_out exist.

    A cache written before those columns existed — and every one of the archived
    history snapshots — has only the blended `price`. Charts read these columns
    unconditionally, so materialise them as NaN rather than making every caller
    test for them; a missing rate renders as "—", never as a wrong number.
    """
    for col in ("price_in", "price_out"):
        if col not in df.columns:
            df[col] = float("nan")
    return df


def save_cache(df: pd.DataFrame):
    from static_helpers import csv_safe

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Sanitised on the way IN as well as on export: these files are committed
    # hourly and opened by hand. See static_helpers.csv_safe.
    safe = csv_safe(df)
    safe.to_csv(CACHE_PATH, index=False)
    # Also save a timestamped snapshot for trend tracking
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    snap_path = HIST_DIR / f"aa_models_{date.today().isoformat()}.csv"
    if not snap_path.exists():           # one snapshot per calendar day
        snap = safe.copy()
        snap["scraped_at"] = date.today().isoformat()
        snap.to_csv(snap_path, index=False)


def load_history() -> pd.DataFrame:
    """Load all timestamped snapshots into one long DataFrame."""
    if not HIST_DIR.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(HIST_DIR.glob("aa_models_*.csv")):
        snap = pd.read_csv(path)
        if "scraped_at" not in snap.columns:
            snap["scraped_at"] = path.stem.replace("aa_models_", "")
        frames.append(snap)
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
