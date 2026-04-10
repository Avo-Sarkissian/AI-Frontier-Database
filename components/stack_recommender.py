"""
Stack Recommender — opinionated 3-tier model recommendations.

Modes:
  api    — all tiers use cloud API models
  hybrid — Fast tier = local (free), Balanced + Reasoning = API
  local  — all tiers use open-weight models filtered to user's hardware

Tiers:
  fast      — maximize quality/price × speed (API) or raw speed (local)
  balanced  — best quality/price value (API) or quality/VRAM efficiency (local)
  reasoning — raw quality, no price constraint (API or local)
"""
import pandas as pd
from dash import html

from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR
from data.local_models import FAMILY_COLORS as _FAMILY_COLORS, DEFAULT_FAMILY_COLOR

_FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

# ── API tier config ────────────────────────────────────────────────────────────
_API_TIERS = [
    {
        "key":         "fast",
        "label":       "Fast",
        "tagline":     "Sub-agent workhorse — cheap, high-throughput, parallel calls",
        "color":       "#00d4ff",
        "max_price":    3.0,
        "min_quality":  28.0,
        "min_speed":    30,
        "sort":        "composite_fast",
        "n":            5,
    },
    {
        "key":         "balanced",
        "label":       "Balanced",
        "tagline":     "Daily driver — coding, writing, tool use, file editing",
        "color":       "#34d399",
        "max_price":    8.0,
        "min_quality":  30.0,
        "min_speed":    0,
        "sort":        "value",
        "n":            5,
    },
    {
        "key":         "reasoning",
        "label":       "Reasoning",
        "tagline":     "Orchestrator — plans, delegates, and reviews sub-agent work",
        "color":       "#c084fc",
        "max_price":    None,
        "min_quality":  0.0,
        "min_speed":    0,
        "sort":        "quality",
        "n":            5,
    },
]

_TIER_ICONS = {"fast": "⚡", "balanced": "⚙", "reasoning": "🧠"}

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
    q = row["quality"] / q_max if q_max else 0
    v = (row["quality"] / row["price"]) / v_max if (v_max and row["price"] > 0) else 0
    s = row["speed"] / s_max if s_max else 0
    # Quality-first: even fast models need real intelligence
    return q * 0.45 + v * 0.30 + s * 0.25


def _pick_api_tier(df: pd.DataFrame, tier: dict) -> pd.DataFrame:
    pool = df[(df["quality"] > 0) & (df["price"] > 0)].copy()
    if tier["max_price"] is not None:
        pool = pool[pool["price"] <= tier["max_price"]]
    if tier.get("min_quality", 0) > 0:
        pool = pool[pool["quality"] >= tier["min_quality"]]
    if tier["min_speed"] > 0:
        pool = pool[pool["speed"] >= tier["min_speed"]]
    if pool.empty:
        return pool

    if tier["sort"] == "composite_fast":
        q_max = pool["quality"].max() or 1
        v_max = (pool["quality"] / pool["price"].replace(0, float("nan"))).max() or 1
        s_max = pool["speed"].replace(0, float("nan")).max() or 1
        pool["_score"] = pool.apply(
            lambda r: _score_fast_api(r, q_max, v_max, s_max), axis=1
        )
        pool = pool.sort_values("_score", ascending=False)
    elif tier["sort"] == "value":
        pool["_score"] = pool["quality"] / pool["price"]
        pool = pool.sort_values("_score", ascending=False)
    else:
        pool = pool.sort_values("quality", ascending=False)

    return pool.head(tier["n"])


# ── Local picking ──────────────────────────────────────────────────────────────

def _pick_local_tier(local_df: pd.DataFrame, tier_key: str, n: int = 5) -> pd.DataFrame:
    runnable = local_df[local_df["fits"].isin(["yes", "tight"])].copy()
    if runnable.empty:
        return runnable

    if tier_key == "fast":
        # Fast but intelligent — enforce a real quality floor, then balance
        # speed and quality so tiny dumb models don't win on speed alone
        runnable = runnable[runnable["quality"] >= 10]
        s_max = runnable["speed_tps"].replace(0, float("nan")).max() or 1
        q_max = runnable["quality"].max() or 1
        runnable["_score"] = (
            runnable["quality"] / q_max * 0.50 +
            runnable["speed_tps"] / s_max * 0.50
        )
        runnable = runnable.sort_values("_score", ascending=False)
    elif tier_key == "balanced":
        # Quality-weighted efficiency — quality matters more than VRAM footprint
        runnable = runnable[runnable["quality"] >= 12]
        q_max = runnable["quality"].max() or 1
        runnable["_score"] = (
            (runnable["quality"] / q_max) * 0.70 +
            (1 / runnable["vram_req_gb"].replace(0, float("nan")).fillna(99)) * 0.30
        )
        runnable = runnable.sort_values("_score", ascending=False)
    else:  # reasoning
        runnable = runnable.sort_values("quality", ascending=False)

    return runnable.head(n)


# ── Shared chip helper ─────────────────────────────────────────────────────────

def _chip(text: str, bg: str = "#1e2a1e") -> html.Span:
    return html.Span(text, style={
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

def _api_row(row: pd.Series, is_top: bool = False) -> html.Div:
    provider  = str(row.get("provider", ""))
    pcolor    = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
    price     = row["price"]
    speed     = row["speed"]
    quality   = row["quality"]

    price_str = f"${price:.4f}/M" if price < 0.01 else f"${price:.3f}/M" if price < 1 else f"${price:.2f}/M"
    speed_str = f"{int(speed):,} tok/s" if speed > 0 else "—"
    lat       = row.get("latency", float("nan"))
    lat_str   = f"{lat:.2f}s TTFT" if pd.notna(lat) and lat > 0 else None

    name = str(row["model"])
    if len(name) > 36:
        name = name[:35] + "…"

    chips = html.Div([
        _chip(price_str, "#1a2a1a"),
        _chip(speed_str, "#1a1a2a"),
        *([_chip(lat_str, "#2a1a1a")] if lat_str else []),
    ], style={"marginTop": "5px", "display": "flex", "flexWrap": "wrap", "gap": "2px"})

    return _row_shell(name, quality, provider, pcolor, chips, is_top)


def _local_row(row: pd.Series, is_top: bool = False) -> html.Div:
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

    chips = html.Div([
        _chip(vram_str, "#1a1a2e"),
        _chip(speed_str, "#1a2a1a"),
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

def _tier_card(tier: dict, picks: pd.DataFrame, source: str) -> html.Div:
    """
    source: "API" | "LOCAL"
    """
    color = tier["color"]
    icon  = _TIER_ICONS[tier["key"]]

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

    advice    = _TIER_ADVICE[tier["key"]]
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
        _advice_row("BEST FOR",  advice["best_for"],  "#5a8a5a"),
        _advice_row("TRADEOFF",  advice["tradeoff"],  "#666"),
        _advice_row("AVOID IF",  advice["avoid_if"],  "#7a4a4a"),
    ])

    return html.Div([
        html.Div([
            html.Span(icon + " " + tier["label"], style={
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


# ── Public API ─────────────────────────────────────────────────────────────────

def build_stack_cards(
    df:           pd.DataFrame,
    providers:    list[str] | None = None,
    mode:         str = "api",
    local_df:     pd.DataFrame | None = None,
) -> html.Div:
    """
    mode: "api" | "hybrid" | "hybrid2" | "local"
    providers: filter cloud models to these providers (None = all)
    local_df:  output of get_local_df(); required for hybrid/local modes
    hybrid2: Fast=local, Balanced=local, Reasoning=API
    """
    # Filtered cloud pool
    api_pool = df.copy()
    if providers:
        api_pool = api_pool[api_pool["provider"].isin(providers)]

    cards = []

    for tier in _API_TIERS:
        key = tier["key"]

        if mode == "api":
            picks  = _pick_api_tier(api_pool, tier)
            source = "API"
        elif mode == "local":
            picks  = _pick_local_tier(local_df, key) if local_df is not None else pd.DataFrame()
            source = "LOCAL"
        elif mode == "hybrid2":  # Fast + Balanced = local, Reasoning = API
            if key in ("fast", "balanced"):
                picks  = _pick_local_tier(local_df, key) if local_df is not None else pd.DataFrame()
                source = "LOCAL"
            else:
                picks  = _pick_api_tier(api_pool, tier)
                source = "API"
        else:  # hybrid — Fast = local, Balanced + Reasoning = API
            if key == "fast":
                picks  = _pick_local_tier(local_df, key) if local_df is not None else pd.DataFrame()
                source = "LOCAL"
            else:
                picks  = _pick_api_tier(api_pool, tier)
                source = "API"

        cards.append(_tier_card(tier, picks, source))

    return html.Div(cards, style={
        "display": "flex",
        "gap": "16px",
        "flexWrap": "wrap",
        "alignItems": "flex-start",
    })
