"""Cost per Intelligence Index task, and the "token burn" it implies.

`cost_per_task` is Artificial Analysis's own number: the USD it spent, per task,
running a model through the Intelligence Index, built from the tokens each
provider reported (input, cache read and write, reasoning and answer) at that
provider's prices. It is scraped as `intelligenceIndexCostPerTask` from
https://artificialanalysis.ai/leaderboards/models (column 13 of aa_models.csv
since 2026-10-27); the method is described at
https://artificialanalysis.ai/methodology/intelligence-benchmarking (read
2026-10-27). AA publishes it for about two thirds of the catalogue and nothing
here fills the rest: a model without one is counted and named as missing,
never estimated.

Token burn = cost per task ÷ the model's list price per token, i.e. how many
tokens at this dashboard's default blend one task costs. It is always taken
against the LIST blend (`price_list_blend` when a MIX re-priced the frame,
otherwise `price`), so the multiple describes the model, not the visitor's
MIX setting, and its label names that blend.

Pure functions only: the Dash app, the static site (Pyodide) and the tests all
call these, so the two renderings cannot drift.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from data.workload_mix import resolve_mix

TASK_COST_SOURCE_URL = "https://artificialanalysis.ai/leaderboards/models"
TASK_COST_METHOD_URL = "https://artificialanalysis.ai/methodology/intelligence-benchmarking"
TASK_COST_AS_OF = "2026-10-27"

# What the burn is denominated in. "blend-eq." because the tokens are
# equivalents at the list blend, not a count of tokens AA observed.
BURN_UNIT = "blend-eq. tok/task"


def list_blend_short() -> str:
    """The name of the blend token burn divides by: the DEFAULT mix's label.
    Never a hard-coded ratio string (B's grep test bans the old ones)."""
    return resolve_mix(None).short


def task_costs(df: pd.DataFrame) -> pd.Series:
    """AA's cost per task where it published a positive one, else NaN.

    A frame without the column (history before 2026-10-27, older fixtures)
    yields all-NaN rather than raising."""
    if "cost_per_task" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    c = pd.to_numeric(df["cost_per_task"], errors="coerce")
    return c.where(c > 0)


def has_task_cost(df: pd.DataFrame) -> pd.Series:
    return task_costs(df).notna()


def list_blend(df: pd.DataFrame) -> pd.Series:
    """The stored default-blend $/1M price, whatever MIX priced `df` at."""
    col = "price_list_blend" if "price_list_blend" in df.columns else "price"
    p = pd.to_numeric(df[col], errors="coerce")
    return p.where(p > 0)


def token_burn(df: pd.DataFrame) -> pd.Series:
    """Blend-equivalent tokens per task: cost/task ÷ ($/1M at the list blend)
    × 1e6. NaN where either side is missing."""
    return task_costs(df) / list_blend(df) * 1e6


def median_token_burn(df: pd.DataFrame) -> float | None:
    """Median burn over every row that has one. Callers pass the UNMIXED
    catalogue (static_api._DF / app.df), so the reference does not move with
    MIX or with a filter."""
    b = token_burn(df).dropna()
    return float(b.median()) if len(b) else None


def _scalar(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def row_task_cost(row) -> float | None:
    return _scalar(row.get("cost_per_task"))


def row_token_burn(row) -> float | None:
    cost = row_task_cost(row)
    price = _scalar(row.get("price_list_blend", row.get("price")))
    if cost is None or price is None:
        return None
    return cost / price * 1e6


def fmt_tokens(v) -> str:
    """374K, 1.98M, 11.3K: three significant digits, never scientific."""
    v = _scalar(v)
    if v is None:
        return "—"
    if v >= 1e6:
        return f"{v / 1e6:.3g}M"
    if v >= 1e3:
        return f"{v / 1e3:.3g}K"
    return f"{v:.3g}"


def fmt_task_cost(v) -> str:
    """Same rule as workload_mix.fmt_price: AA's floor is under half a cent."""
    from data.workload_mix import fmt_price
    v = _scalar(v)
    return fmt_price(v) if v is not None else "—"


def burn_text(v) -> str:
    """'374K blend-eq. tok/task', or '—'."""
    return f"{fmt_tokens(v)} {BURN_UNIT}" if _scalar(v) is not None else "—"


def burn_multiple_text(v, median) -> str:
    """'2.3× median' against the catalogue median, or '' when either is unknown."""
    v, median = _scalar(v), _scalar(median)
    if v is None or median is None:
        return ""
    m = v / median
    return f"{m:.2g}× median" if m < 10 else f"{m:.0f}× median"


def detail_rows(row, median) -> list[tuple[str, str]]:
    """The detail panel's two task-cost lines, shared by Dash and static_api:
    [("Cost / task", ...), ("Token burn", ...)]."""
    cost = row_task_cost(row)
    if cost is None:
        return [("Cost / task", "—  ·  AA publishes none"), ("Token burn", "—")]
    burn = row_token_burn(row)
    mult = burn_multiple_text(burn, median)
    return [
        ("Cost / task", f"{fmt_task_cost(cost)}  ·  AA-measured"),
        ("Token burn", burn_text(burn) + (f"  ·  {mult}" if mult else "")),
    ]


def detail_note(row) -> str:
    """The small line under the panel's token-burn row, naming the blend;
    empty when the row has no burn to explain."""
    if row_token_burn(row) is None:
        return ""
    return (f"Token burn = AA cost per Intelligence Index task ÷ list price at "
            f"{list_blend_short()}")


def rank_flip_share(df: pd.DataFrame) -> tuple[float, int] | None:
    """Share of model pairs where the cheaper-per-token model (list blend)
    costs MORE per task, over rows that have both. Pairs tied on either axis
    are left out. Returns (share, pairs counted) or None below two rows."""
    cost, price = task_costs(df), list_blend(df)
    ok = cost.notna() & price.notna()
    c, p = cost[ok].to_numpy(float), price[ok].to_numpy(float)
    if len(c) < 2:
        return None
    iu = np.triu_indices(len(c), k=1)
    dp = np.sign(p[:, None] - p[None, :])[iu]
    dc = np.sign(c[:, None] - c[None, :])[iu]
    counted = (dp != 0) & (dc != 0)
    n = int(counted.sum())
    if n == 0:
        return None
    return float((dp[counted] != dc[counted]).sum()) / n, n
