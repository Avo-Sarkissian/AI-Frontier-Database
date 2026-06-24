"""Pure, dash-free helpers shared by app.py, static_api.py, build_static.py.
Imports only stdlib + pandas so this module loads under Pyodide."""
import re
import pandas as pd


def apply_filters(df, providers, min_quality, search: str = "") -> pd.DataFrame:
    filtered = df.copy()
    if providers:
        filtered = filtered[filtered["provider"].isin(providers)]
    if min_quality and float(min_quality) > 0:
        filtered = filtered[filtered["quality"] >= float(min_quality)]
    if search and search.strip():
        # Escape user input so characters like (, [, * are treated as
        # plain text rather than regex metacharacters.
        pat = re.escape(search.strip())
        mask = (
            filtered["model"].str.contains(pat, case=False, na=False) |
            filtered["provider"].str.contains(pat, case=False, na=False)
        )
        filtered = filtered[mask]
    return filtered


def compute_diverse5(dataframe: pd.DataFrame) -> list[str]:
    """Pick 5 diverse models spanning quality, value, speed, and budget tiers."""
    valid = dataframe[(dataframe["quality"] > 0) & (dataframe["price"] > 0)].copy()
    if valid.empty:
        return []
    picks: list = []

    def _add(rows):
        for _, row in rows.iterrows():
            if row["model"] not in picks:
                picks.append(row["model"])
                return

    # 1. Best intelligence
    _add(valid.sort_values("quality", ascending=False))
    # 2. Best value (quality/price) with a quality floor so weak-but-cheap
    #    models don't monopolise the "value" pick in Compare defaults.
    v = valid[valid["quality"] >= 35].copy()
    v["_val"] = v["quality"] / v["price"]
    _add(v.sort_values("_val", ascending=False) if not v.empty else valid.assign(_val=valid["quality"]/valid["price"]).sort_values("_val", ascending=False))
    # 3. Fastest model with quality >= 40
    fast = valid[(valid["speed"] > 0) & (valid["quality"] >= 40)]
    _add(fast.sort_values("speed", ascending=False) if not fast.empty else valid.sort_values("speed", ascending=False))
    # 4. Cheapest with quality >= 50
    cheap = valid[valid["quality"] >= 50]
    _add(cheap.sort_values("price") if not cheap.empty else valid.sort_values("price"))
    # 5. Mid-tier (40-70 quality range)
    mid = valid[(valid["quality"] >= 40) & (valid["quality"] <= 70)]
    _add(mid.sort_values("quality", ascending=False) if not mid.empty else valid.sort_values("quality", ascending=False))

    return picks[:5]


def ctx_to_k(c) -> float | None:
    """Convert context string ('400k', '1m', '128k') to numeric thousands for sorting."""
    if not c:
        return None
    s = str(c).strip().lower()
    try:
        if s.endswith('m'):
            return float(s[:-1]) * 1000
        if s.endswith('k'):
            return float(s[:-1])
        return float(s) / 1000
    except ValueError:
        return None


def quality_label(pct: float) -> str:
    """pct is quality normalised to 0–100 relative to the dataset max."""
    if pct >= 90: return "Exceptional"
    if pct >= 75: return "Strong"
    if pct >= 55: return "Capable"
    if pct >= 35: return "Average"
    return "Limited"


def provider_options(dataframe: pd.DataFrame) -> list:
    return [{"label": p, "value": p} for p in sorted(dataframe["provider"].unique())]


def model_options(dataframe: pd.DataFrame) -> list:
    top = dataframe[dataframe["quality"] > 0].sort_values("quality", ascending=False)
    return [{"label": f"{r['model']} ({r['provider']})", "value": r["model"]}
            for _, r in top.iterrows()]
