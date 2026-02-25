"""
Data ingestion: scrapes live model data from Artificial Analysis leaderboard.
Falls back to cached CSV if scraping fails.
"""
import re
import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
CACHE_PATH = RAW_DIR / "aa_models.csv"


def _parse_price(val: str) -> float:
    """'$4.50' → 4.50, '$0.00' → 0.0"""
    try:
        return float(val.replace("$", "").replace(",", ""))
    except (ValueError, AttributeError):
        return float("nan")


def _parse_numeric(val: str) -> float:
    """'1,196' → 1196, '34E' → 34, '--' → nan"""
    if not val or val in ("--", ""):
        return float("nan")
    val = re.sub(r"[E,].*", "", val)   # strip 'E' suffix and commas
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

        # Skip rows with no useful data
        if pd.isna(quality) or model == "":
            continue

        records.append({
            "model":    model,
            "provider": provider,
            "context":  context,
            "quality":  quality,
            "price":    price,
            "speed":    speed,
            "latency":  latency,
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["model", "provider"])
    df = df[df["price"] > 0].copy()          # only models with public pricing
    df = df.sort_values("quality", ascending=False).reset_index(drop=True)
    return df


def load_cached() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH)
    raise FileNotFoundError("No cached data found. Run scrape first.")


def save_cache(df: pd.DataFrame):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)


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
