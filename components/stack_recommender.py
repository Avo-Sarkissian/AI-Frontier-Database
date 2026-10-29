"""
Stack Recommender — opinionated 3-tier model recommendations.

Modes:
  api    — all tiers use cloud API models
  hybrid — Fast tier = local (priced at InferenceX's renting rate on the seven
           datacenter cards it prices; other cards get no price chip),
           Balanced + Reasoning = API
  local  — all tiers use open-weight models filtered to user's hardware

Tiers:
  fast      — maximize quality/price × speed (API) or raw speed (local)
  balanced  — best quality/price value (API) or quality/VRAM efficiency (local)
  reasoning — raw quality, no price constraint (API or local)
"""
import re
import pandas as pd
# Aliased: the name `html` below is Dash's html module, not the stdlib one.
from html import escape as _escape
try:
    from dash import html
except ImportError:  # dash-free env (Pyodide): only the HTML-string renderer is used
    html = None

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR, canonical_provider, base_model_name,
)
from data.quality_scale import floor as _q_floor
from data.local_models import FAMILY_COLORS as _FAMILY_COLORS, DEFAULT_FAMILY_COLOR
from data.workload_mix import DEFAULT_MIX, fmt_price

# Agent Stack does not read the MIX setting: the global filter bar (MIX with it)
# is hidden on this tab, and its tier caps below ($3 / $8) and the value sort
# are cut on the dashboard's default blend. Its price chips used to print a
# bare "$x/M", which beside a header floor stat at an agentic mix read as an
# agentic price. They now name the basis, and their tooltip says why.
_PRICE_BASIS_TAG = "1:3 in:out"
_PRICE_BASIS_TITLE = (f"Price at {DEFAULT_MIX.long}. Agent Stack does not follow the MIX "
                      "setting: its tiers (Fast ≤ $3/M, Balanced ≤ $8/M) are cut on this blend.")


# "(Max, Default Fallback)": the qualifier is on every Claude 5.5 tier and
# carries nothing, while it pushed the effort itself past the 36-char cut.
_DEFAULT_FALLBACK_RE = re.compile(r",\s*Default\s+Fallback\s*\)", re.I)


def _price_chip_text(price) -> str:
    return f"{fmt_price(price)}/M · {_PRICE_BASIS_TAG}"


# Local inference on a datacenter card InferenceX prices is not free: the row
# carries usd_per_m_out (static_helpers.with_local_rent_cost), SemiAnalysis
# InferenceX's renting $/GPU-hr for the preset spread over the tab's MAX
# CONCURRENT throughput (`sessions` streams at once) at full use. The tok/s chip
# beside it is ONE stream, so the price chip names its own basis: per output
# token, the session count it assumes, full use. A bare "$x/M rent" beside
# "2,050 tok/s" read as 22-37x cheaper than that speed implies. Cards
# InferenceX does not price get no chip.
_RENT_TITLE = ("GPU cost per 1M OUTPUT tokens: SemiAnalysis InferenceX's renting "
               "$/GPU-hr (large-hyperscaler volume) for this card x the GPU count, "
               "over this tab's max-throughput figure, the total across the session "
               "count shown, at full use. The tok/s chip is one stream, so the "
               "price is far lower than one stream would cost. Decode only (no "
               "prefill), uncalibrated; Economics › Serve has the calibrated "
               "datacenter figure.")


def _rent_chip_text(row) -> str | None:
    v = row.get("usd_per_m_out") if hasattr(row, "get") else None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not (v == v and v > 0):
        return None
    money = f"${v:.2f}" if v >= 0.01 else f"${v:.3g}"
    try:
        n = int(float(row.get("sessions")))
    except (TypeError, ValueError):
        n = 0
    if n > 1:
        basis = f"{n:,} sessions, full use"
    elif n == 1:
        basis = "1 session, full use"
    else:
        basis = "full use"
    return f"{money}/M out · rent, {basis}"

_FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

# ── API tier config ────────────────────────────────────────────────────────────
# Time-to-first-token beyond this reads as "not fast" whatever else is true.
# Used both as a hard tier filter and as the cap on the latency score term.
FAST_MAX_LATENCY_S = 10.0
_LATENCY_SCORE_CAP_S = 5.0
# Balanced's cap, and the TTFT above which ANY recommended API row carries a
# warning chip (Reasoning has no cap: an orchestrator can wait, but the reader
# should see that it will).
BALANCED_MAX_LATENCY_S = 60.0
SLOW_TTFT_WARN_S = 30.0


_API_TIERS = [
    {
        "key":         "fast",
        "label":       "Fast",
        "tagline":     "Sub-agent workhorse — cheap, high-throughput, parallel calls",
        "color":       "#00d4ff",
        "max_price":    3.0,
        # Quality floors are QUANTILES of the catalogue's family scores
        # (data/quality_scale), not index points: v4's 28 / 30 meant something
        # else after AA's non-uniform v5 rescale (the calibration table is in
        # data/quality_scale).
        "floor_key":    "stack_fast",
        "min_speed":    30,
        # A tier whose whole promise is "parallel sub-agents" cannot recommend
        # a model that takes a minute to say its first word.
        "max_latency":  FAST_MAX_LATENCY_S,
        "sort":        "composite_fast",
        "n":            5,
    },
    {
        "key":         "balanced",
        "label":       "Balanced",
        "tagline":     "Daily driver — coding, writing, tool use, file editing",
        "color":       "#34d399",
        "max_price":    8.0,
        "floor_key":    "stack_balanced",
        "min_speed":    0,
        # A "daily driver" that takes minutes to say its first word is not
        # one: Claude Haiku 5.5 (Max) topped this tier at 340 s TTFT. Unmeasured
        # rows stay in (see _tier_pool).
        "max_latency":  BALANCED_MAX_LATENCY_S,
        "sort":        "value",
        "n":            5,
    },
    {
        "key":         "reasoning",
        "label":       "Reasoning",
        "tagline":     "Orchestrator — plans, delegates, and reviews sub-agent work",
        "color":       "#c084fc",
        "max_price":    None,
        "floor_key":    None,
        "min_speed":    0,
        "sort":        "quality",
        "n":            5,
    },
]

# Tier marks. These were "⚡", "⚙" and "🧠": one text glyph, one that some
# fonts draw as text and some as emoji, and one colour emoji — three sizes and
# three styles side by side. A plain square in the tier's colour is the same
# mark everywhere.
_TIER_MARK_STYLE = {"display": "inline-block", "width": "8px", "height": "8px",
                    "borderRadius": "2px", "marginRight": "8px", "flexShrink": "0"}

_TIER_ADVICE = {
    "fast": {
        "best_for":  "Parallel sub-tasks, file search, grep, classification, boilerplate generation",
        "tradeoff":  "Lower reasoning depth — not suitable for complex logic or multi-step planning",
        "avoid_if":  "Task requires synthesizing many sources or multi-hop reasoning",
    },
    "balanced": {
        "best_for":  "Feature implementation, code review, refactoring, writing, debugging",
        "tradeoff":  "More expensive than Fast — avoid for high-volume simple operations",
        "avoid_if":  "You need the absolute best quality or are running > 10M tokens/month at cost",
    },
    "reasoning": {
        "best_for":  "Orchestration, system design, hard bugs, planning multi-agent workflows",
        "tradeoff":  "Slowest and most expensive — reserve for the top-level orchestrator only",
        "avoid_if":  "The task is clear enough for the Balanced tier; cost is very sensitive",
    },
}

_USE_CASES = {
    "fast": [
        "Run in parallel across many files",
        "Grep, search, and classify at scale",
        "Route and filter agent outputs",
        "Generate boilerplate & tests",
    ],
    "balanced": [
        "Implement features end-to-end",
        "Edit, refactor, and review code",
        "Answer questions about the codebase",
        "Write docs, PRs, and commit messages",
    ],
    "reasoning": [
        "Plan multi-step agent workflows",
        "Delegate to Fast + Balanced sub-agents",
        "Architect systems and APIs",
        "Debug hard cross-file problems",
    ],
}

# ── API picking ────────────────────────────────────────────────────────────────

def _score_fast_api(row, q_max, v_max, s_max):
    """Score for the tier captioned "cheap, high-throughput, parallel calls".

    Latency used to appear nowhere in this function — the whole Fast tier was
    scored on quality, value and throughput — while the card rendered a TTFT
    chip right beside the pick. On the default page state, with no interaction,
    it recommended GPT-5.6 Luna (max) at 57.45s time-to-first-token, when the
    same pool held GPT-5.6 Luna (low) at 1.63s: same provider, same $0.950/M,
    slightly higher throughput, 35x faster to first token, ranked 6th because
    quality carried 0.45 of the weight.

    Latency now takes 0.20, out of throughput's share. Throughput matters once a
    response is streaming; for a sub-agent fired in parallel, the wait before it
    starts is the cost the user actually feels.
    """
    q = row["quality"] / q_max if q_max else 0
    v = (row["quality"] / row["price"]) / v_max if (v_max and row["price"] > 0) else 0
    s = row["speed"] / s_max if s_max else 0
    lat = row.get("latency", None)
    if lat is None or not pd.notna(lat) or lat <= 0:
        l = 0.5          # unmeasured: neither rewarded nor punished
    else:
        l = 1.0 - min(float(lat), _LATENCY_SCORE_CAP_S) / _LATENCY_SCORE_CAP_S
    # Quality-first, but a "fast" model that keeps you waiting is not fast.
    return q * 0.45 + v * 0.25 + s * 0.10 + l * 0.20


def tier_floor(tier: dict, ref_df: pd.DataFrame | None, name_col: str = "model") -> float:
    """The tier's quality floor on `ref_df` (data/quality_scale), 0 for none."""
    key = tier.get("floor_key")
    return _q_floor(ref_df, key, name_col) if key else 0.0


def _tier_pool(df: pd.DataFrame, tier: dict, ref_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows eligible for a tier, before ranking. The quality floor is read off
    `ref_df` (default `df`): the unfiltered catalogue, so a provider filter
    cannot move it."""
    pool = df[(df["quality"] > 0) & (df["price"] > 0)].copy()
    if tier["max_price"] is not None:
        pool = pool[pool["price"] <= tier["max_price"]]
    q_min = tier_floor(tier, df if ref_df is None else ref_df)
    if q_min > 0:
        pool = pool[pool["quality"] >= q_min]
    if tier["min_speed"] > 0:
        pool = pool[pool["speed"] >= tier["min_speed"]]
    if tier.get("max_latency"):
        # Keep unmeasured rows — dropping them would silently narrow the pool to
        # whichever models happen to have a TTFT figure this hour.
        lat = pd.to_numeric(pool["latency"], errors="coerce")
        pool = pool[lat.isna() | (lat <= tier["max_latency"])]
    return pool


def _pick_api_tier(df: pd.DataFrame, tier: dict, full_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank a tier's candidates.

    ``full_df`` is the catalogue before the provider filter. The composite Fast
    score divides quality, value and speed by three *separate* maxima; taking
    them from the provider-filtered pool changed the relative weighting, so a
    model's score — and the rendered order — moved when the user ticked an
    unrelated provider box. Ordering was not even preserved: pairs demonstrably
    swapped rank between a single-provider and an all-provider selection.
    """
    pool = _tier_pool(df, tier, ref_df=full_df)
    if pool.empty:
        return pool
    ref = pool if full_df is None else _tier_pool(full_df, tier)
    if ref.empty:
        ref = pool

    # Every ranking below is a stable sort: `head(n)` cuts through ties, and a
    # quicksort let the CI build and the in-browser render pick different cards.
    if tier["sort"] == "composite_fast":
        q_max = ref["quality"].max() or 1
        v_max = (ref["quality"] / ref["price"].replace(0, float("nan"))).max() or 1
        s_max = ref["speed"].replace(0, float("nan")).max() or 1
        pool["_score"] = pool.apply(
            lambda r: _score_fast_api(r, q_max, v_max, s_max), axis=1
        )
        pool = pool.sort_values("_score", ascending=False, kind="mergesort")
    elif tier["sort"] == "value":
        pool["_score"] = pool["quality"] / pool["price"]
        pool = pool.sort_values("_score", ascending=False, kind="mergesort")
    else:
        pool = pool.sort_values("quality", ascending=False, kind="mergesort")

    # One row per model family: its best-ranked tier for THIS tier's sort.
    # Under "All variants" the five alternates were Claude Haiku 5.5 (Max) and
    # Claude Haiku 5.5 (Xhigh) — one model at one price, twice.
    pool = pool[~pool["model"].astype(str).map(base_model_name).duplicated()]
    return pool.head(tier["n"])


# ── Local picking ──────────────────────────────────────────────────────────────

def _pick_local_tier(local_df: pd.DataFrame, tier_key: str, n: int = 5,
                     full_local_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank the local candidates for a tier.

    ``full_local_df`` is the catalogue before the TAG filter, used only for the
    normalising maxima. The composite Fast score divides quality and speed by
    two separate pool-dependent maxima, so taking them from the filtered pool
    changes the relative weighting and does not preserve order — the same defect
    _pick_api_tier had, where seven provider selections visibly reordered the
    rendered top five.

    The hardware filter deliberately stays in the reference: "fastest model that
    fits your GPU" is a claim about your GPU, so the pool is the point.
    """
    runnable = local_df[local_df["fits"].isin(["yes", "tight"])].copy()
    if runnable.empty:
        return runnable
    ref_all = runnable if full_local_df is None else \
        full_local_df[full_local_df["fits"].isin(["yes", "tight"])].copy()
    if ref_all.empty:
        ref_all = runnable

    # Quality floors as quantiles of the whole open-weight catalogue's family
    # scores (data/quality_scale) — not of what fits, and not v4's 10 / 12,
    # which on v5 left one model a 32 GB card can run above both floors, so
    # all three tiers recommended Qwen3.8 27B.
    _catalogue = local_df if full_local_df is None else full_local_df
    if tier_key == "fast":
        # Fast but intelligent — enforce a real quality floor, then balance
        # speed and quality so tiny dumb models don't win on speed alone
        f_fast = _q_floor(_catalogue, "local_fast", "name")
        runnable = runnable[runnable["quality"] >= f_fast].copy()
        ref = ref_all[ref_all["quality"] >= f_fast]
        if ref.empty:
            ref = runnable
        s_max = ref["speed_tps"].replace(0, float("nan")).max() or 1
        q_max = ref["quality"].max() or 1
        runnable["_score"] = (
            runnable["quality"] / q_max * 0.50 +
            runnable["speed_tps"] / s_max * 0.50
        )
        runnable = runnable.sort_values("_score", ascending=False, kind="mergesort")
    elif tier_key == "balanced":
        # Quality-weighted efficiency — quality matters more than VRAM footprint
        f_bal = _q_floor(_catalogue, "local_balanced", "name")
        runnable = runnable[runnable["quality"] >= f_bal].copy()
        _ref_b = ref_all[ref_all["quality"] >= f_bal]
        q_max = (_ref_b if not _ref_b.empty else runnable)["quality"].max() or 1
        # The VRAM term used to be a RAW RECIPROCAL in GB against a min-max
        # normalised quality term: measured spreads were 0.474 for quality and
        # 0.095 for VRAM, so the argmax was simply the argmax of quality — which
        # is the Reasoning tier's own sort key. Balanced matched Reasoning on 90
        # of 137 GPU presets, and all three tiers were identical on 27.
        _v = runnable["vram_req_gb"].astype(float)
        _v_lo, _v_hi = _v.min(), _v.max()
        runnable["_vram_fit"] = (
            (_v_hi - _v) / (_v_hi - _v_lo) if _v_hi > _v_lo else 1.0
        )
        runnable["_score"] = (
            (runnable["quality"] / q_max) * 0.70 +
            runnable["_vram_fit"] * 0.30
        )
        runnable = runnable.sort_values("_score", ascending=False, kind="mergesort")
    else:  # reasoning
        # Scored rows only. Fast and Balanced get this from their `>=` floors
        # (NaN fails a comparison); Reasoning had no floor, so a NaN quality
        # merely sorted LAST — and head(n) still returned it whenever fewer
        # than n scored models fit. After AA's v5.0 rescale left three scored
        # models fitting 32 GB, the card recommended both unbenchmarked
        # Ornith entries (data/pending_models.py) on "intelligence".
        runnable = runnable[runnable["quality"] > 0]
        runnable = runnable.sort_values("quality", ascending=False, kind="mergesort")

    # One row per family (GLM-5.3 (Max) and GLM-5.3 (Low) are one download
    # choice, not two alternates): its best-ranked row for this tier.
    if not runnable.empty:
        runnable = runnable[~runnable["name"].astype(str).map(base_model_name).duplicated()]
    return runnable.head(n)


# ── Shared chip helper ─────────────────────────────────────────────────────────

def _slow(lat) -> bool:
    try:
        return pd.notna(lat) and float(lat) >= SLOW_TTFT_WARN_S
    except (TypeError, ValueError):
        return False


def _ttft_text(lat) -> str | None:
    """'1.36s TTFT', or '⚠ 341s to first token' past SLOW_TTFT_WARN_S: a pick
    that keeps you waiting minutes must say so on the card, not only in a
    number styled like every other chip."""
    if not (pd.notna(lat) and lat > 0):
        return None
    if _slow(lat):
        return f"⚠ {lat:.0f}s to first token"
    return f"{lat:.2f}s TTFT"


def _ttft_style(lat) -> tuple[str, str | None]:
    """(chip background, tooltip) for the TTFT chip."""
    if _slow(lat):
        return ("#3a2410", f"Time to first token at or above {SLOW_TTFT_WARN_S:.0f} s "
                           "(Artificial Analysis's median). Reasoning at high effort "
                           "thinks before it answers.")
    return ("#2a1a1a", None)


def _chip(text: str, bg: str = "#1e2a1e", title: str | None = None) -> "html.Span":
    return html.Span(text, title=title, style={
        "display": "inline-block",
        "padding": "2px 7px",
        "borderRadius": "3px",
        "fontSize": "10px",
        "fontFamily": _FONT,
        "fontWeight": "500",
        "color": "#ccc",
        "background": bg,
        "marginRight": "4px",
        "letterSpacing": "0.02em",
        "whiteSpace": "nowrap",
    })


# ── Row renderers ──────────────────────────────────────────────────────────────

def _api_row(row: pd.Series, is_top: bool = False) -> "html.Div":
    # Canonical at the render site too: the palette is keyed by the current
    # spelling, so a row that reached here by any other path still gets its
    # colour rather than the grey fallback.
    provider  = canonical_provider(str(row.get("provider", "")))
    pcolor    = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
    price     = row["price"]
    speed     = row["speed"]
    quality   = row["quality"]

    price_str = _price_chip_text(price)
    speed_str = f"{int(speed):,} tok/s" if speed > 0 else "—"
    lat       = row.get("latency", float("nan"))
    lat_str   = _ttft_text(lat)

    name = _DEFAULT_FALLBACK_RE.sub(")", str(row["model"]))
    if len(name) > 36:
        name = name[:35] + "…"

    chips = html.Div([
        _chip(price_str, "#1a2a1a", title=_PRICE_BASIS_TITLE),
        _chip(speed_str, "#1a1a2a"),
        *([_chip(lat_str, *_ttft_style(lat))] if lat_str else []),
    ], style={"marginTop": "5px", "display": "flex", "flexWrap": "wrap", "gap": "2px"})

    return _row_shell(name, quality, provider, pcolor, chips, is_top)


def _local_row(row: pd.Series, is_top: bool = False) -> "html.Div":
    family   = str(row.get("family", ""))
    fcolor   = _FAMILY_COLORS.get(family, DEFAULT_FAMILY_COLOR)
    quality  = row["quality"]
    speed    = row.get("speed_tps", 0)
    vram     = row.get("vram_req_gb", 0)
    tight    = row.get("fits") == "tight"

    speed_str = f"{int(speed):,} tok/s" if speed > 0 else "—"
    vram_str  = f"{vram:.1f} GB VRAM" + (" ⚠" if tight else "")

    name = str(row["name"])
    if len(name) > 36:
        name = name[:35] + "…"

    rent = _rent_chip_text(row)
    chips = html.Div([
        _chip(vram_str, "#1a1a2e"),
        _chip(speed_str, "#1a2a1a"),
        *([_chip(rent, "#2a1a1a", title=_RENT_TITLE)] if rent else []),
    ], style={"marginTop": "5px", "display": "flex", "flexWrap": "wrap", "gap": "2px"})

    return _row_shell(name, quality, family, fcolor, chips, is_top)


def _row_shell(name, quality, sub_label, sub_color, chips, is_top):
    if is_top:
        return html.Div([
            html.Div([
                html.Span(name, style={
                    "fontSize": "13px", "fontWeight": "600",
                    "color": "#f2f2f2", "fontFamily": _FONT,
                }),
                html.Span(f"{quality:.1f}", style={
                    "fontSize": "13px", "fontWeight": "700",
                    "color": "#f2f2f2", "fontFamily": _FONT, "marginLeft": "auto",
                }),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "baseline"}),
            html.Div(sub_label, style={
                "fontSize": "10px", "color": sub_color, "fontFamily": _FONT,
                "marginTop": "2px", "fontWeight": "500",
            }),
            chips,
        ], style={
            "padding": "12px 14px", "borderRadius": "6px",
            "background": "rgba(255,255,255,0.05)",
            "border": "1px solid rgba(255,255,255,0.1)",
            "marginBottom": "8px",
        })
    else:
        return html.Div([
            html.Div([
                html.Span(name, style={
                    "fontSize": "11px", "color": "#ccc",
                    "fontFamily": _FONT, "fontWeight": "500",
                }),
                html.Span(f"{quality:.1f}", style={
                    "fontSize": "11px", "color": "#888",
                    "fontFamily": _FONT, "marginLeft": "auto",
                }),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "baseline"}),
            html.Div(sub_label, style={
                "fontSize": "10px", "color": sub_color,
                "fontFamily": _FONT, "marginTop": "1px",
            }),
            chips,
        ], style={
            "padding": "9px 12px", "borderRadius": "4px",
            "background": "rgba(255,255,255,0.02)",
            "border": "1px solid rgba(255,255,255,0.05)",
            "marginBottom": "6px",
        })


# ── Card builder ───────────────────────────────────────────────────────────────

def _tier_card(tier: dict, picks: pd.DataFrame, source: str) -> "html.Div":
    """
    source: "API" | "LOCAL"
    """
    color = tier["color"]

    # Source badge
    badge_bg  = "rgba(0,212,255,0.12)" if source == "API" else "rgba(134,239,172,0.12)"
    badge_col = "#00d4ff"              if source == "API" else "#86efac"
    badge = html.Span(source, style={
        "fontSize": "9px", "letterSpacing": "0.08em", "fontWeight": "700",
        "color": badge_col, "background": badge_bg,
        "padding": "2px 6px", "borderRadius": "3px",
        "fontFamily": _FONT, "marginLeft": "8px",
        "verticalAlign": "middle",
    })

    if picks.empty:
        body = html.Div(
            "No models match — try adjusting VRAM or quant level." if source == "LOCAL"
            else "No models match these criteria.",
            style={"color": "#555", "fontSize": "12px", "padding": "16px 0"},
        )
    else:
        row_fn = _local_row if source == "LOCAL" else _api_row
        body   = html.Div([row_fn(row, is_top=(i == 0))
                           for i, (_, row) in enumerate(picks.iterrows())])

    advice    = tier["advice"] if "advice" in tier else _TIER_ADVICE[tier["key"]]
    use_cases = html.Div([
        html.Div("USE CASES", style={
            "fontSize": "9px", "letterSpacing": "0.1em", "color": "#555",
            "fontFamily": _FONT, "marginBottom": "6px", "marginTop": "14px",
        }),
        html.Div([
            html.Div(f"· {uc}", style={
                "fontSize": "11px", "color": "#666", "fontFamily": _FONT,
                "marginBottom": "3px",
            })
            for uc in _USE_CASES[tier["key"]]
        ]),
    ])

    def _advice_row(label: str, text: str, text_color: str = "#555") -> html.Div:
        return html.Div([
            html.Span(label, style={
                "fontSize": "9px", "letterSpacing": "0.08em", "color": "#444",
                "fontFamily": _FONT, "fontWeight": "600", "marginRight": "6px",
                "flexShrink": "0",
            }),
            html.Span(text, style={
                "fontSize": "10px", "color": text_color, "fontFamily": _FONT,
                "lineHeight": "1.4",
            }),
        ], style={"display": "flex", "marginBottom": "5px"})

    advisor = html.Div([
        html.Div(style={"height": "1px", "background": "rgba(255,255,255,0.05)", "margin": "14px 0 12px"}),
        *([
            _advice_row("BEST FOR",  advice["best_for"],  "#5a8a5a"),
            _advice_row("TRADEOFF",  advice["tradeoff"],  "#666"),
            _advice_row("AVOID IF",  advice["avoid_if"],  "#7a4a4a"),
        ] if advice else [
            _advice_row(
                "NOTE",
                f"Same model as {tier.get('duplicate_of') or 'another tier'} — the "
                f"selected catalogue does not offer a distinct option at this tier.",
                "#7a6a3a",
            ),
        ]),
    ])

    return html.Div([
        html.Div([
            html.Span(style={**_TIER_MARK_STYLE, "background": color}),
            html.Span(tier["label"], style={
                "fontSize": "14px", "fontWeight": "700", "color": color,
                "fontFamily": _FONT, "letterSpacing": "0.02em",
            }),
            badge,
        ], style={
            "borderBottom": f"2px solid {color}",
            "paddingBottom": "10px", "marginBottom": "14px",
            "display": "flex", "alignItems": "center",
        }),
        html.Div(tier["tagline"], style={
            "fontSize": "11px", "color": "#777", "fontFamily": _FONT,
            "marginBottom": "14px", "lineHeight": "1.5",
        }),
        body,
        use_cases,
        advisor,
    ], style={
        "flex": "1",
        "minWidth": "260px",
        "background": "#111",
        "border": "1px solid rgba(255,255,255,0.07)",
        "borderRadius": "8px",
        "padding": "20px",
    })


# ── Pure selection logic (no Dash dependency) ──────────────────────────────────

def select_stack(
    df:        pd.DataFrame,
    providers: list[str] | None = None,
    mode:      str = "api",
    local_df:  pd.DataFrame | None = None,
    full_local_df: pd.DataFrame | None = None,
) -> dict:
    """
    Pure model-selection logic — returns plain data with no Dash objects.

    Returns:
        {
            "tiers": [
                {
                    "name":      str,          # e.g. "Fast"
                    "key":       str,          # e.g. "fast"
                    "color":     str,          # hex accent color
                    "tagline":   str,
                    "picks":     pd.DataFrame, # selected model rows
                    "source":    str,          # "API" | "LOCAL"
                    "use_cases": list[str],
                    "advice":    dict,         # best_for / tradeoff / avoid_if
                },
                ...  # 3 tiers total
            ]
        }

    mode: "api" | "hybrid" | "hybrid2" | "local"
    providers: filter cloud models to these providers (None = all)
    local_df:  output of get_local_df(); required for hybrid/local modes
    hybrid2: Fast=local, Balanced=local, Reasoning=API
    """
    api_pool = df.copy()
    # Canonicalise once, here, for every mode — not just inside the filter
    # branch. Filtering on the canonical name while the card still printed the
    # raw one meant ticking "SpaceXAI" returned the right rows and then
    # captioned them "xAI" in fallback grey, because PROVIDER_COLORS is keyed by
    # the canonical spelling only. One spelling in, one spelling out.
    api_pool["provider"] = api_pool["provider"].astype(str).map(canonical_provider)
    # `is not None`, not truthiness: `[]` is a deliberate "I unchecked every
    # provider" and must return nothing, while `None` means "no filter". Both are
    # falsy, so `if providers:` sent an empty selection down the no-filter branch
    # and recommended the very providers the user had just excluded.
    if providers is not None:
        # Resolve retired upstream spellings (xAI -> SpaceXAI) so a control
        # labelled with the old name still matches. A raw .isin() meant the UI's
        # own "xAI" checkbox selected zero rows.
        wanted = {canonical_provider(str(p)) for p in providers}
        api_pool = api_pool[api_pool["provider"].isin(wanted)]

    tiers_out = []
    cands = []                    # (tier, candidate picks, source), claim order
    _taken: dict[str, str] = {}   # model FAMILY -> the tier key that claimed it

    # Claim order, not display order. Tiers are shown Fast -> Balanced ->
    # Reasoning, but if Fast and Balanced pick first they can take the very model
    # Reasoning exists to name — leaving the max-quality card rendering the
    # SECOND-best model in the catalogue. Reasoning claims first; the cards are
    # re-sorted back into display order afterwards.
    _CLAIM_ORDER = {"reasoning": 0, "balanced": 1, "fast": 2}
    for tier in sorted(_API_TIERS, key=lambda t: _CLAIM_ORDER.get(t["key"], 9)):
        key = tier["key"]
        duplicate_of = None

        if mode == "api":
            picks  = _pick_api_tier(api_pool, tier, full_df=df)
            source = "API"
        elif mode == "local":
            picks  = _pick_local_tier(local_df, key, full_local_df=full_local_df) \
                         if local_df is not None else pd.DataFrame()
            source = "LOCAL"
        elif mode == "hybrid2":  # Fast + Balanced = local, Reasoning = API
            if key in ("fast", "balanced"):
                picks  = _pick_local_tier(local_df, key, full_local_df=full_local_df) \
                         if local_df is not None else pd.DataFrame()
                source = "LOCAL"
            else:
                picks  = _pick_api_tier(api_pool, tier, full_df=df)
                source = "API"
        else:  # hybrid — Fast = local, Balanced + Reasoning = API
            if key == "fast":
                picks  = _pick_local_tier(local_df, key, full_local_df=full_local_df) \
                         if local_df is not None else pd.DataFrame()
                source = "LOCAL"
            else:
                picks  = _pick_api_tier(api_pool, tier, full_df=df)
                source = "API"

        cands.append((tier, picks, source))

    # Don't hand the same model to two tiers. The point of the card is that
    # these are three DIFFERENT recommendations; when they collapsed (11 of 32
    # single-provider selections in API mode, 27 of 137 GPU presets in local
    # mode) the page asserted, of one model at one price, that it was
    # simultaneously "not suitable for complex logic", "more expensive than
    # Fast" and "slowest and most expensive". By FAMILY: Opus 5.5 at Max effort
    # for Reasoning and Opus 5.5 at Low for Balanced is one model at one price.
    def _fams(frame):
        col = "model" if "model" in frame.columns else "name"
        return frame[col].astype(str).map(base_model_name) if col in frame.columns \
            else pd.Series(dtype=str)

    assigned: dict[str, pd.DataFrame] = {}
    dup_of: dict[str, str | None] = {}
    for tier, picks, _src in cands:
        key = tier["key"]
        dup_of[key] = None
        if picks.empty or not len(_fams(picks)):
            assigned[key] = picks
            continue
        fams = _fams(picks)
        fresh = picks[~fams.isin(_taken)]
        if fresh.empty:
            assigned[key] = picks
            dup_of[key] = _taken.get(fams.iloc[0])
        else:
            assigned[key] = fresh
            _taken.setdefault(_fams(fresh).iloc[0], key)

    # Repair: a tier left repeating another can take the family it wants when
    # the tier holding it (never Reasoning, whose point is the top model) has
    # a free alternate. With Anthropic + Google + OpenAI, Balanced claimed
    # Claude Haiku 5.5 — Fast's only candidate — while holding four others.
    for tier, picks, _src in cands:
        key = tier["key"]
        if dup_of[key] is None or picks.empty:
            continue
        for fam in _fams(picks):
            owner = _taken.get(fam)
            if owner in (None, "reasoning", key):
                continue
            mine = assigned[owner]
            alt = mine[~_fams(mine).isin(set(_taken) | {fam})]
            if alt.empty:
                continue
            assigned[owner] = alt
            _taken[_fams(alt).iloc[0]] = owner
            _taken[fam] = key
            assigned[key] = picks[_fams(picks).eq(fam)
                                  | ~_fams(picks).isin(_taken)]
            # The wanted family leads.
            assigned[key] = pd.concat([assigned[key][_fams(assigned[key]).eq(fam)],
                                       assigned[key][~_fams(assigned[key]).eq(fam)]])
            dup_of[key] = None
            break

    label_of = {t["key"]: t["label"] for t in _API_TIERS}
    for tier, _picks, source in cands:
        key = tier["key"]
        picks = assigned[key]
        duplicate_of = label_of.get(dup_of[key], dup_of[key])
        tiers_out.append({
            "name":      tier["label"],
            "key":       key,
            "color":     tier["color"],
            "tagline":   tier["tagline"],
            "picks":     picks,
            "source":    source,
            "use_cases": _USE_CASES[key],
            # A repeat makes the tier-specific tradeoff/avoid_if lines false, so
            # they are suppressed rather than printed about the wrong model.
            "advice":    None if duplicate_of else _TIER_ADVICE[key],
            "duplicate_of": duplicate_of,
        })

    display_order = [t["key"] for t in _API_TIERS]
    tiers_out.sort(key=lambda t: display_order.index(t["key"]))
    return {"tiers": tiers_out}


# ── HTML-string renderer (Pyodide-safe, no Dash) ───────────────────────────────

def _h(tag: str, inner: str = "", style: str | None = None, cls: str | None = None,
       title: str | None = None) -> str:
    """Minimal HTML tag builder mirroring dash.html structure."""
    s = f' style="{style}"' if style else ""
    c = f' class="{cls}"' if cls else ""
    t = f' title="{_escape(title, quote=True)}"' if title else ""
    return f"<{tag}{c}{s}{t}>{inner}</{tag}>"


def _dict_to_style(d: dict) -> str:
    """Convert a CSS dict (camelCase keys) to an inline style string."""
    def _camel_to_kebab(k: str) -> str:
        return re.sub(r'([A-Z])', r'-\1', k).lower()

    parts = []
    for k, v in d.items():
        parts.append(f"{_camel_to_kebab(k)}:{v}")
    return ";".join(parts)


def _chip_html(text: str, bg: str = "#1e2a1e", title: str | None = None) -> str:
    style = _dict_to_style({
        "display": "inline-block",
        "padding": "2px 7px",
        "borderRadius": "3px",
        "fontSize": "10px",
        "fontFamily": _FONT,
        "fontWeight": "500",
        "color": "#ccc",
        "background": bg,
        "marginRight": "4px",
        "letterSpacing": "0.02em",
        "whiteSpace": "nowrap",
    })
    return _h("span", text, style=style, title=title)


def _row_shell_html(name: str, quality: float, sub_label: str, sub_color: str,
                    chips_html: str, is_top: bool) -> str:
    # `name` and `sub_label` are scraped third-party text (model/provider from the
    # AA feed, name/family from the local-model catalog) and this string is assigned
    # to innerHTML at docs/app.js:494 — escape before interpolating. Both callers
    # truncate first, so escaping here cannot split an entity. `sub_color` and
    # `chips_html` are generated internally and are not escaped.
    name = _escape(str(name), quote=True)
    sub_label = _escape(str(sub_label), quote=True)
    if is_top:
        header = _h("div",
            _h("span", name, style=_dict_to_style({
                "fontSize": "13px", "fontWeight": "600",
                "color": "#f2f2f2", "fontFamily": _FONT,
            })) +
            _h("span", f"{quality:.1f}", style=_dict_to_style({
                "fontSize": "13px", "fontWeight": "700",
                "color": "#f2f2f2", "fontFamily": _FONT, "marginLeft": "auto",
            })),
            style=_dict_to_style({"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "baseline"}),
        )
        sublabel_div = _h("div", sub_label, style=_dict_to_style({
            "fontSize": "10px", "color": sub_color, "fontFamily": _FONT,
            "marginTop": "2px", "fontWeight": "500",
        }))
        chips_div = _h("div", chips_html, style=_dict_to_style({
            "marginTop": "5px", "display": "flex", "flexWrap": "wrap", "gap": "2px",
        }))
        return _h("div", header + sublabel_div + chips_div, style=_dict_to_style({
            "padding": "12px 14px", "borderRadius": "6px",
            "background": "rgba(255,255,255,0.05)",
            "border": "1px solid rgba(255,255,255,0.1)",
            "marginBottom": "8px",
        }))
    else:
        header = _h("div",
            _h("span", name, style=_dict_to_style({
                "fontSize": "11px", "color": "#ccc",
                "fontFamily": _FONT, "fontWeight": "500",
            })) +
            _h("span", f"{quality:.1f}", style=_dict_to_style({
                "fontSize": "11px", "color": "#888",
                "fontFamily": _FONT, "marginLeft": "auto",
            })),
            style=_dict_to_style({"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "baseline"}),
        )
        sublabel_div = _h("div", sub_label, style=_dict_to_style({
            "fontSize": "10px", "color": sub_color,
            "fontFamily": _FONT, "marginTop": "1px",
        }))
        chips_div = _h("div", chips_html, style=_dict_to_style({
            "marginTop": "5px", "display": "flex", "flexWrap": "wrap", "gap": "2px",
        }))
        return _h("div", header + sublabel_div + chips_div, style=_dict_to_style({
            "padding": "9px 12px", "borderRadius": "4px",
            "background": "rgba(255,255,255,0.02)",
            "border": "1px solid rgba(255,255,255,0.05)",
            "marginBottom": "6px",
        }))


def _api_row_html(row: pd.Series, is_top: bool = False) -> str:
    # See _api_row — same rule, other rendering.
    provider  = canonical_provider(str(row.get("provider", "")))
    pcolor    = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
    price     = row["price"]
    speed     = row["speed"]
    quality   = row["quality"]

    price_str = _price_chip_text(price)
    speed_str = f"{int(speed):,} tok/s" if speed > 0 else "—"
    lat       = row.get("latency", float("nan"))
    lat_str   = _ttft_text(lat)

    name = _DEFAULT_FALLBACK_RE.sub(")", str(row["model"]))
    if len(name) > 36:
        name = name[:35] + "…"

    chips_html = (
        _chip_html(price_str, "#1a2a1a", title=_PRICE_BASIS_TITLE) +
        _chip_html(speed_str, "#1a1a2a") +
        (_chip_html(lat_str, *_ttft_style(lat)) if lat_str else "")
    )
    return _row_shell_html(name, quality, provider, pcolor, chips_html, is_top)


def _local_row_html(row: pd.Series, is_top: bool = False) -> str:
    family  = str(row.get("family", ""))
    fcolor  = _FAMILY_COLORS.get(family, DEFAULT_FAMILY_COLOR)
    quality = row["quality"]
    speed   = row.get("speed_tps", 0)
    vram    = row.get("vram_req_gb", 0)
    tight   = row.get("fits") == "tight"

    speed_str = f"{int(speed):,} tok/s" if speed > 0 else "—"
    vram_str  = f"{vram:.1f} GB VRAM" + (" ⚠" if tight else "")

    name = str(row["name"])
    if len(name) > 36:
        name = name[:35] + "…"

    rent = _rent_chip_text(row)
    chips_html = (_chip_html(vram_str, "#1a1a2e") + _chip_html(speed_str, "#1a2a1a")
                  + (_chip_html(rent, "#2a1a1a", title=_RENT_TITLE) if rent else ""))
    return _row_shell_html(name, quality, family, fcolor, chips_html, is_top)


def _tier_card_html(tier_data: dict) -> str:
    """Render a single tier card as an HTML string."""
    name      = tier_data["name"]
    key       = tier_data["key"]
    color     = tier_data["color"]
    tagline   = tier_data["tagline"]
    picks     = tier_data["picks"]
    source    = tier_data["source"]
    use_cases = tier_data["use_cases"]
    advice    = tier_data["advice"]
    mark      = _h("span", "", style=_dict_to_style({**_TIER_MARK_STYLE, "background": color}))

    # Source badge
    badge_bg  = "rgba(0,212,255,0.12)" if source == "API" else "rgba(134,239,172,0.12)"
    badge_col = "#00d4ff"              if source == "API" else "#86efac"
    badge = _h("span", source, style=_dict_to_style({
        "fontSize": "9px", "letterSpacing": "0.08em", "fontWeight": "700",
        "color": badge_col, "background": badge_bg,
        "padding": "2px 6px", "borderRadius": "3px",
        "fontFamily": _FONT, "marginLeft": "8px",
        "verticalAlign": "middle",
    }))

    # Header row
    header = _h("div",
        mark + _h("span", name, style=_dict_to_style({
            "fontSize": "14px", "fontWeight": "700", "color": color,
            "fontFamily": _FONT, "letterSpacing": "0.02em",
        })) + badge,
        style=_dict_to_style({
            "borderBottom": f"2px solid {color}",
            "paddingBottom": "10px", "marginBottom": "14px",
            "display": "flex", "alignItems": "center",
        }),
    )

    # Tagline
    tagline_div = _h("div", tagline, style=_dict_to_style({
        "fontSize": "11px", "color": "#777", "fontFamily": _FONT,
        "marginBottom": "14px", "lineHeight": "1.5",
    }))

    # Model rows / empty state
    if picks.empty:
        body = _h("div",
            "No models match — try adjusting VRAM or quant level." if source == "LOCAL"
            else "No models match these criteria.",
            style=_dict_to_style({"color": "#555", "fontSize": "12px", "padding": "16px 0"}),
        )
    else:
        row_fn = _local_row_html if source == "LOCAL" else _api_row_html
        body = _h("div", "".join(
            row_fn(row, is_top=(i == 0))
            for i, (_, row) in enumerate(picks.iterrows())
        ))

    # Use cases section
    uc_label = _h("div", "USE CASES", style=_dict_to_style({
        "fontSize": "9px", "letterSpacing": "0.1em", "color": "#555",
        "fontFamily": _FONT, "marginBottom": "6px", "marginTop": "14px",
    }))
    uc_items = "".join(
        _h("div", f"· {uc}", style=_dict_to_style({
            "fontSize": "11px", "color": "#666", "fontFamily": _FONT, "marginBottom": "3px",
        }))
        for uc in use_cases
    )
    use_cases_div = _h("div", uc_label + uc_items)

    # Advisor section
    def _advice_row_html(label: str, text: str, text_color: str = "#555") -> str:
        lbl = _h("span", label, style=_dict_to_style({
            "fontSize": "9px", "letterSpacing": "0.08em", "color": "#444",
            "fontFamily": _FONT, "fontWeight": "600", "marginRight": "6px",
            "flexShrink": "0",
        }))
        val = _h("span", text, style=_dict_to_style({
            "fontSize": "10px", "color": text_color, "fontFamily": _FONT, "lineHeight": "1.4",
        }))
        return _h("div", lbl + val, style=_dict_to_style({"display": "flex", "marginBottom": "5px"}))

    divider = _h("div", "", style=_dict_to_style({
        "height": "1px", "background": "rgba(255,255,255,0.05)", "margin": "14px 0 12px",
    }))
    if advice:
        advisor = _h("div",
            divider +
            _advice_row_html("BEST FOR",  advice["best_for"],  "#5a8a5a") +
            _advice_row_html("TRADEOFF",  advice["tradeoff"],  "#666") +
            _advice_row_html("AVOID IF",  advice["avoid_if"],  "#7a4a4a"),
        )
    else:
        # The catalogue on screen cannot differentiate this tier, so the fixed
        # advice would describe the wrong model — it once told the reader that
        # one model at one price was both "not suitable for complex logic" and
        # "slowest and most expensive", on the same page.
        dup = _escape(str(tier_data.get("duplicate_of") or "another tier"))
        advisor = _h("div",
            divider +
            _advice_row_html(
                "NOTE",
                f"Same model as {dup} — the selected catalogue does not offer a "
                f"distinct option at this tier. Narrow or widen the provider "
                f"filter to see alternatives.",
                "#7a6a3a",
            ),
        )

    return _h("div",
        header + tagline_div + body + use_cases_div + advisor,
        style=_dict_to_style({
            "flex": "1",
            "minWidth": "260px",
            "background": "#111",
            "border": "1px solid rgba(255,255,255,0.07)",
            "borderRadius": "8px",
            "padding": "20px",
        }),
    )


def build_stack_cards_html(
    df:        pd.DataFrame,
    providers: list[str] | None = None,
    mode:      str = "api",
    local_df:  pd.DataFrame | None = None,
    full_local_df=None,
) -> str:
    """
    HTML-string renderer — Pyodide-safe, no Dash dependency.
    Returns the same tier cards as build_stack_cards() but as a raw HTML string.

    mode: "api" | "hybrid" | "hybrid2" | "local"
    providers: filter cloud models to these providers (None = all)
    local_df:  output of get_local_df(); required for hybrid/local modes
    """
    data = select_stack(df, providers=providers, mode=mode, local_df=local_df,
                        full_local_df=full_local_df)
    cards_html = "".join(_tier_card_html(t) for t in data["tiers"])
    return _h("div", cards_html, style=_dict_to_style({
        "display": "flex",
        "gap": "16px",
        "flexWrap": "wrap",
        "alignItems": "flex-start",
    }))


# ── Public API ─────────────────────────────────────────────────────────────────

def build_stack_cards(
    df:           pd.DataFrame,
    providers:    list[str] | None = None,
    mode:         str = "api",
    local_df:     pd.DataFrame | None = None,
    full_local_df=None,
) -> "html.Div":
    """
    mode: "api" | "hybrid" | "hybrid2" | "local"
    providers: filter cloud models to these providers (None = all)
    local_df:  output of get_local_df(); required for hybrid/local modes
    hybrid2: Fast=local, Balanced=local, Reasoning=API
    """
    if html is None:
        raise RuntimeError("dash is required for build_stack_cards; "
                           "use build_stack_cards_html in dash-free environments")
    data = select_stack(df, providers=providers, mode=mode, local_df=local_df,
                        full_local_df=full_local_df)
    cards = [_tier_card(tier_cfg, t["picks"], t["source"])
             for tier_cfg, t in zip(_API_TIERS, data["tiers"])]
    return html.Div(cards, style={
        "display": "flex",
        "gap": "16px",
        "flexWrap": "wrap",
        "alignItems": "flex-start",
    })
