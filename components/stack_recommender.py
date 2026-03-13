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
        "tagline":     "High-volume tasks, automation, quick responses",
        "color":       "#00d4ff",
        "max_price":    2.0,
        "min_quality":  10.0,
        "min_speed":    50,
        "sort":        "composite_fast",
        "n":            3,
    },
    {
        "key":         "balanced",
        "label":       "Balanced",
        "tagline":     "Coding, writing, day-to-day tasks",
        "color":       "#34d399",
        "max_price":    8.0,
        "min_quality":  30.0,
        "min_speed":    0,
        "sort":        "value",
        "n":            3,
    },
    {
        "key":         "reasoning",
        "label":       "Reasoning",
        "tagline":     "Complex planning, orchestration, delegation",
        "color":       "#c084fc",
        "max_price":    None,
        "min_quality":  0.0,
        "min_speed":    0,
        "sort":        "quality",
        "n":            3,
    },
]

_TIER_ICONS = {"fast": "⚡", "balanced": "⚙", "reasoning": "🧠"}

_USE_CASES = {
    "fast": [
        "Tagging & classification",
        "Summarisation at scale",
        "Routing & intent detection",
        "Code autocomplete",
    ],
    "balanced": [
        "Feature implementation",
        "Writing & editing",
        "Data analysis & charts",
        "API integration work",
    ],
    "reasoning": [
        "Architecture & planning",
        "Multi-step research",
        "Orchestrating other models",
        "Hard maths / logic",
    ],
}

# ── API picking ────────────────────────────────────────────────────────────────

def _score_fast_api(row, q_max, v_max, s_max):
    q = row["quality"] / q_max if q_max else 0
    v = (row["quality"] / row["price"]) / v_max if (v_max and row["price"] > 0) else 0
    s = row["speed"] / s_max if s_max else 0
    return q * 0.25 + v * 0.45 + s * 0.30


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

def _pick_local_tier(local_df: pd.DataFrame, tier_key: str, n: int = 3) -> pd.DataFrame:
    runnable = local_df[local_df["fits"].isin(["yes", "tight"])].copy()
    if runnable.empty:
        return runnable

    if tier_key == "fast":
        # Fastest models — sorted by speed, decent quality minimum
        runnable = runnable[runnable["quality"] >= 3]
        runnable = runnable.sort_values("speed_tps", ascending=False)
    elif tier_key == "balanced":
        # Best quality-per-GB-of-VRAM — good quality without eating all your RAM
        runnable["_score"] = runnable["quality"] / runnable["vram_req_gb"].replace(0, float("nan"))
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
    mode: "api" | "hybrid" | "local"
    providers: filter cloud models to these providers (None = all)
    local_df:  output of get_local_df(); required for hybrid/local modes
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
