"""
Stack Recommender — build opinionated 3-tier model recommendations.

Tiers:
  fast      — maximize quality/price × speed (price < $2/M, speed > 50 tok/s)
  balanced  — maximize quality within price < $8/M
  reasoning — raw quality, no price filter
"""
import pandas as pd
from dash import html

from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR

_FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

TIERS = [
    {
        "key":       "fast",
        "label":     "Fast",
        "tagline":   "High-volume tasks, automation, quick responses",
        "color":     "#00d4ff",
        "max_price":  2.0,
        "min_quality": 10.0,
        "min_speed":  50,
        "sort":      "composite_fast",
        "n":         3,
    },
    {
        "key":       "balanced",
        "label":     "Balanced",
        "tagline":   "Coding, writing, day-to-day tasks",
        "color":     "#34d399",
        "max_price": 8.0,
        "min_quality": 30.0,
        "min_speed": 0,
        "sort":      "value",   # quality / price — best bang for buck
        "n":         3,
    },
    {
        "key":       "reasoning",
        "label":     "Reasoning",
        "tagline":   "Complex planning, orchestration, delegation",
        "color":     "#c084fc",
        "max_price": None,
        "min_quality": 0.0,
        "min_speed": 0,
        "sort":      "quality",  # raw quality, no price constraint
        "n":         3,
    },
]

_TIER_ICONS = {
    "fast":      "⚡",
    "balanced":  "⚙",
    "reasoning": "🧠",
}

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


def _score_fast(row, q_max: float, v_max: float, s_max: float) -> float:
    q = row["quality"] / q_max if q_max else 0
    v = (row["quality"] / row["price"]) / v_max if (v_max and row["price"] > 0) else 0
    s = row["speed"] / s_max if s_max else 0
    return q * 0.25 + v * 0.45 + s * 0.30


def _pick_tier(df: pd.DataFrame, tier: dict) -> pd.DataFrame:
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
            lambda r: _score_fast(r, q_max, v_max, s_max), axis=1
        )
        pool = pool.sort_values("_score", ascending=False)
    elif tier["sort"] == "value":
        pool["_score"] = pool["quality"] / pool["price"]
        pool = pool.sort_values("_score", ascending=False)
    else:
        pool = pool.sort_values("quality", ascending=False)

    return pool.head(tier["n"])


def _chip(text: str, color: str = "#444") -> html.Span:
    return html.Span(text, style={
        "display": "inline-block",
        "padding": "2px 7px",
        "borderRadius": "3px",
        "fontSize": "10px",
        "fontFamily": _FONT,
        "fontWeight": "500",
        "color": "#ccc",
        "background": color,
        "marginRight": "4px",
        "letterSpacing": "0.02em",
        "whiteSpace": "nowrap",
    })


def _model_row(row: pd.Series, is_top: bool = False) -> html.Div:
    provider = str(row.get("provider", ""))
    pcolor   = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
    price    = row["price"]
    speed    = row["speed"]
    quality  = row["quality"]

    price_str = f"${price:.4f}/M" if price < 0.01 else f"${price:.3f}/M" if price < 1 else f"${price:.2f}/M"
    speed_str = f"{int(speed):,} tok/s" if speed > 0 else "—"
    lat       = row.get("latency", float("nan"))
    lat_str   = f"{lat:.2f}s" if pd.notna(lat) and lat > 0 else None

    name = str(row["model"])
    if len(name) > 36:
        name = name[:35] + "…"

    chips = html.Div([
        _chip(price_str, "#1a2a1a"),
        _chip(speed_str, "#1a1a2a"),
        *([_chip(lat_str, "#2a1a1a")] if lat_str else []),
    ], style={"marginTop": "5px", "display": "flex", "flexWrap": "wrap", "gap": "2px"})

    if is_top:
        return html.Div([
            html.Div([
                html.Span(name, style={
                    "fontSize": "13px", "fontWeight": "600",
                    "color": "#f2f2f2", "fontFamily": _FONT,
                }),
                html.Span(f"{quality:.1f}", style={
                    "fontSize": "13px", "fontWeight": "700",
                    "color": "#f2f2f2", "fontFamily": _FONT,
                    "marginLeft": "auto",
                }),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "baseline"}),
            html.Div(provider, style={
                "fontSize": "10px", "color": pcolor, "fontFamily": _FONT,
                "marginTop": "2px", "fontWeight": "500",
            }),
            chips,
        ], style={
            "padding": "12px 14px",
            "borderRadius": "6px",
            "background": "rgba(255,255,255,0.05)",
            "border": "1px solid rgba(255,255,255,0.1)",
            "marginBottom": "8px",
        })
    else:
        return html.Div([
            html.Div([
                html.Span(name, style={
                    "fontSize": "11px", "color": "#ccc", "fontFamily": _FONT,
                    "fontWeight": "500",
                }),
                html.Span(f"{quality:.1f}", style={
                    "fontSize": "11px", "color": "#888", "fontFamily": _FONT,
                    "marginLeft": "auto",
                }),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "baseline"}),
            html.Div(provider, style={
                "fontSize": "10px", "color": pcolor, "fontFamily": _FONT,
                "marginTop": "1px",
            }),
            chips,
        ], style={
            "padding": "9px 12px",
            "borderRadius": "4px",
            "background": "rgba(255,255,255,0.02)",
            "border": "1px solid rgba(255,255,255,0.05)",
            "marginBottom": "6px",
        })


def _tier_card(tier: dict, picks: pd.DataFrame) -> html.Div:
    color = tier["color"]
    icon  = _TIER_ICONS[tier["key"]]

    if picks.empty:
        body = html.Div("No models match these criteria.",
                        style={"color": "#555", "fontSize": "12px", "padding": "16px 0"})
    else:
        rows = []
        for i, (_, row) in enumerate(picks.iterrows()):
            rows.append(_model_row(row, is_top=(i == 0)))
        body = html.Div(rows)

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
        # Header
        html.Div([
            html.Span(icon + " " + tier["label"], style={
                "fontSize": "14px", "fontWeight": "700", "color": color,
                "fontFamily": _FONT, "letterSpacing": "0.02em",
            }),
        ], style={
            "borderBottom": f"2px solid {color}",
            "paddingBottom": "10px", "marginBottom": "14px",
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


def build_stack_cards(df: pd.DataFrame, providers: list[str] | None = None) -> html.Div:
    """
    Build 3-column tier recommendation cards.
    providers: list of provider names to restrict to; None = all providers.
    """
    pool = df.copy()
    if providers:
        pool = pool[pool["provider"].isin(providers)]

    if pool.empty:
        return html.Div("No models match the selected providers.",
                        style={"color": "#555", "padding": "32px", "fontFamily": _FONT})

    cards = []
    for tier in TIERS:
        picks = _pick_tier(pool, tier)
        cards.append(_tier_card(tier, picks))

    return html.Div(cards, style={
        "display": "flex",
        "gap": "16px",
        "flexWrap": "wrap",
        "alignItems": "flex-start",
    })
