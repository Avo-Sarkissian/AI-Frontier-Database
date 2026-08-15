"""Pure, dash-free helpers shared by app.py, static_api.py, build_static.py.
Imports only stdlib + pandas so this module loads under Pyodide."""
import math
import re
import pandas as pd

from components.charts.constants import canonical_provider


def coerce_number(value, default: float, minimum: float | None = None) -> float:
    """Read a number the user typed, distinguishing 0 from "left blank".

    Every numeric input used `float(value or DEFAULT)`, and 0 is falsy — so
    typing 0 silently produced the default. A 0-token budget priced 1M tokens,
    0 GB of VRAM selected a whole GPU's worth of "models that fit", and the
    Agent Stack recommended a 21 GB model for a stated 0 GB.

    Only None, "" or an unparseable value means "not given". `minimum` clamps
    hostile input: the Budget box's HTML `min` attribute is a validation hint,
    not a guard, so a negative volume reached the cost model and its ascending
    sort put the most expensive model under a callout reading CHEAPEST.

    NaN and infinity count as unparseable. They must take the `default` branch,
    not the clamp: `max(0.0, nan)` is `nan` — NaN compares False against
    everything — so a NaN bandwidth used to become 0 GB/s and every model on the
    Local tab read 0 tok/s, defeating the very default it was paired with. And
    `max(1, inf)` is `inf`, which raises OverflowError on the int() that follows,
    or serialises as the literal `Infinity`, which JSON.parse rejects.

    The blank test sits inside the guard because it is not total: `pd.NA == ""`
    returns NA, and bool(NA) raises.
    """
    try:
        out = float(default) if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    return out if minimum is None else max(float(minimum), out)


def apply_filters(df, providers, min_quality, search: str = "") -> pd.DataFrame:
    """Global PROVIDER / MIN SCORE / SEARCH filter, shared by both renderings.

    `providers` is falsy-means-all here, and that is deliberate — this backs a
    multi-select whose placeholder literally reads "All providers", so an empty
    box is the unfiltered state. The Agent Stack's `select_stack` uses the
    opposite rule (`[]` means none) because it backs *checkboxes*, where every
    box unticked is an explicit choice. Two controls, two idioms; the rule is
    pinned by tests on both sides so the asymmetry stays deliberate.

    Names are canonicalised on both sides of the comparison so a retired
    spelling still selects: an `?p=xAI` link shared before Artificial Analysis
    renamed the provider matched zero rows and silently rendered the entire
    catalogue.
    """
    filtered = df.copy()
    if providers:
        wanted = {canonical_provider(str(p)) for p in providers}
        filtered = filtered[
            filtered["provider"].astype(str).map(canonical_provider).isin(wanted)
        ]
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
