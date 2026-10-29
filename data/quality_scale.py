"""Quality floors that survive an Intelligence Index rescale.

Every "good enough" threshold in the dashboard used to be a literal on the AA
Intelligence Index: Agent Stack's Fast/Balanced floors (28 / 30), Run Local's
(10 / 12), Value Leaders' "quality >= 20", the Compare defaults' 35/40/50, the
Budget score colours (45 / 30), the MIN SCORE ladder (0 ... 50) and the Budget
slider's max of 65. All were picked on index v4. AA's v5.0 (2026-10-28) moved
scores NON-uniformly — x1.08 at the top, x0.36 at the bottom — so the literals
now meant something else: on a 32 GB card the three local tiers all showed
Qwen3.8 27B, because v5 put every mid-size open model under the old 10 / 12.

The rule now: a floor is a QUANTILE of the current catalogue's family scores,
never a number. A quantile is invariant under any monotone rescale of the
index — uniform or not — so the next rescale moves every floor with the scores
it is compared against, with no code change. Family scores (each model
family's best variant) keep a lab that publishes five effort tiers from
weighting the distribution five times.

Calibration (documented, 2026-10-29, the v5 switch): each v4 literal was
carried onto v5 through the models AA scored on BOTH versions — 75 of them,
fitted with a monotone (isotonic) map from v4 score to v5 score — and the
floor became the share of v5 model families scoring below the mapped value,
rounded. So every floor admits, on v5, the models it admitted on v4:

    floor            v4 literal  ->  v5 score  ->  quantile
    stack_fast            28           16.2          0.18
    stack_balanced        30           18.2          0.20
    value_leaders         20            9.5          0.05
    budget_strong         45           34.0          0.65
    budget_mid            30           18.2          0.20
    compare_value         35           24.3          0.30
    compare_fast          40           30.1          0.50
    compare_strong        50           50.6          0.80
    compare_mid_lo/hi  40 / 70     30.1 / 58.4   0.50 / 0.95
    local_fast            10            4.6          0.05  (open-weight families)
    local_balanced        12            5.6          0.05

The v4 literals read as v5 scores were far too strict: v5 compressed the
middle of the scale (v4 30 -> v5 18), so on a 32 GB card one model cleared the
local floors and all three tiers recommended Qwen3.8 27B, and the Agent
Stack's Fast tier was empty on its default providers.

Pure pandas: imported by chart builders, app.py, static_api (Pyodide) and
build_static.
"""
from __future__ import annotations

import math

import pandas as pd

# key -> quantile of family scores. Hosted keys read the hosted catalogue,
# local_* keys the open-weight catalogue.
FLOOR_QUANTILES: dict[str, float] = {
    "stack_fast":       0.18,   # Agent Stack › Fast (API)
    "stack_balanced":   0.20,   # Agent Stack › Balanced (API)
    "value_leaders":    0.05,   # Value Leaders: cheap-but-weak models excluded
    "compare_value":    0.30,   # Compare defaults: best value among these
    "compare_fast":     0.50,   # fastest among these
    "compare_strong":   0.80,   # cheapest among these
    "compare_mid_lo":   0.50,   # mid-tier band
    "compare_mid_hi":   0.95,
    "budget_strong":    0.65,   # Budget score colour: strong
    "budget_mid":       0.20,   # Budget score colour: capable
    "local_fast":       0.05,   # Agent Stack / Run Local › Fast (local)
    "local_balanced":   0.05,   # Agent Stack › Balanced (local)
}


def family_scores(df: pd.DataFrame | None, name_col: str = "model") -> pd.Series:
    """Positive scores, one per model family (its best variant)."""
    if df is None or df.empty or "quality" not in df.columns or name_col not in df.columns:
        return pd.Series(dtype=float)
    from components.charts.constants import base_model_name

    q = pd.to_numeric(df["quality"], errors="coerce")
    ok = q.notna() & (q > 0)
    if not ok.any():
        return pd.Series(dtype=float)
    fam = df.loc[ok, name_col].astype(str).map(base_model_name)
    return q[ok].groupby(fam.values).max().sort_values().reset_index(drop=True)


def floor(df: pd.DataFrame | None, key: str, name_col: str = "model",
          default: float = 0.0) -> float:
    """The score at FLOOR_QUANTILES[key] of `df`'s family scores (linear
    interpolation, unrounded: callers compare with >=)."""
    scores = family_scores(df, name_col)
    if scores.empty:
        return float(default)
    return float(scores.quantile(FLOOR_QUANTILES[key]))


def peak(df: pd.DataFrame | None) -> float:
    """The catalogue's best score (0 when nothing is scored)."""
    if df is None or df.empty or "quality" not in df.columns:
        return 0.0
    q = pd.to_numeric(df["quality"], errors="coerce")
    q = q[q > 0]
    return float(q.max()) if len(q) else 0.0


def ladder(peak_score: float | None) -> list[float]:
    """MIN SCORE steps: 0 then every 5 points below the current peak (every 10
    on a scale whose peak passes 100). The old fixed 0..50 ladder sat on v4's
    range; a ladder that stops short of the peak cannot reach the top tier,
    and one that runs past it offers settings that match nothing."""
    p = float(peak_score or 0)
    if not math.isfinite(p) or p <= 0:
        return [0.0]
    step = 5 if p <= 100 else 10
    top = int(math.ceil(p / step) - 1) * step       # the last step BELOW the peak
    return [0.0] + [float(v) for v in range(step, top + 1, step)]


def slider_max(peak_score: float | None) -> int:
    """The Budget MIN INTELLIGENCE slider's max: the peak, rounded up. Past it
    the slider only reaches "no model qualifies"."""
    p = float(peak_score or 0)
    return int(math.ceil(p)) if math.isfinite(p) and p > 0 else 100
