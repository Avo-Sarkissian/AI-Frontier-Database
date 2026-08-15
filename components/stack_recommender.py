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
import re
import pandas as pd
# Aliased: the name `html` below is Dash's html module, not the stdlib one.
from html import escape as _escape
try:
    from dash import html
except ImportError:  # dash-free env (Pyodide): only the HTML-string renderer is used
    html = None

from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR, canonical_provider
from data.local_models import FAMILY_COLORS as _FAMILY_COLORS, DEFAULT_FAMILY_COLOR

_FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

# ── API tier config ────────────────────────────────────────────────────────────
# Time-to-first-token beyond this reads as "not fast" whatever else is true.
# Used both as a hard tier filter and as the cap on the latency score term.
FAST_MAX_LATENCY_S = 10.0
_LATENCY_SCORE_CAP_S = 5.0


_API_TIERS = [
    {
        "key":         "fast",
        "label":       "Fast",
        "tagline":     "Sub-agent workhorse — cheap, high-throughput, parallel calls",
        "color":       "#00d4ff",
        "max_price":    3.0,
        "min_quality":  28.0,
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


def _tier_pool(df: pd.DataFrame, tier: dict) -> pd.DataFrame:
    """Rows eligible for a tier, before ranking."""
    pool = df[(df["quality"] > 0) & (df["price"] > 0)].copy()
    if tier["max_price"] is not None:
        pool = pool[pool["price"] <= tier["max_price"]]
    if tier.get("min_quality", 0) > 0:
        pool = pool[pool["quality"] >= tier["min_quality"]]
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
    pool = _tier_pool(df, tier)
    if pool.empty:
        return pool
    ref = pool if full_df is None else _tier_pool(full_df, tier)
    if ref.empty:
        ref = pool

    if tier["sort"] == "composite_fast":
        q_max = ref["quality"].max() or 1
        v_max = (ref["quality"] / ref["price"].replace(0, float("nan"))).max() or 1
        s_max = ref["speed"].replace(0, float("nan")).max() or 1
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

    if tier_key == "fast":
        # Fast but intelligent — enforce a real quality floor, then balance
        # speed and quality so tiny dumb models don't win on speed alone
        runnable = runnable[runnable["quality"] >= 10].copy()
        ref = ref_all[ref_all["quality"] >= 10]
        if ref.empty:
            ref = runnable
        s_max = ref["speed_tps"].replace(0, float("nan")).max() or 1
        q_max = ref["quality"].max() or 1
        runnable["_score"] = (
            runnable["quality"] / q_max * 0.50 +
            runnable["speed_tps"] / s_max * 0.50
        )
        runnable = runnable.sort_values("_score", ascending=False)
    elif tier_key == "balanced":
        # Quality-weighted efficiency — quality matters more than VRAM footprint
        runnable = runnable[runnable["quality"] >= 12].copy()
        _ref_b = ref_all[ref_all["quality"] >= 12]
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
        runnable = runnable.sort_values("_score", ascending=False)
    else:  # reasoning
        runnable = runnable.sort_values("quality", ascending=False)

    return runnable.head(n)


# ── Shared chip helper ─────────────────────────────────────────────────────────

def _chip(text: str, bg: str = "#1e2a1e") -> "html.Span":
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

def _api_row(row: pd.Series, is_top: bool = False) -> "html.Div":
    # Canonical at the render site too: the palette is keyed by the current
    # spelling, so a row that reached here by any other path still gets its
    # colour rather than the grey fallback.
    provider  = canonical_provider(str(row.get("provider", "")))
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

def _tier_card(tier: dict, picks: pd.DataFrame, source: str) -> "html.Div":
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

    advice    = tier.get("advice", _TIER_ADVICE[tier["key"]])
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
    _taken: dict[str, str] = {}   # model name -> the tier that claimed it

    for tier in _API_TIERS:
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

        # Don't hand the same model to two tiers. The point of the card is that
        # these are three DIFFERENT recommendations; when they collapsed (11 of
        # 32 single-provider selections in API mode, 27 of 137 GPU presets in
        # local mode) the page asserted, of one model at one price, that it was
        # simultaneously "not suitable for complex logic", "more expensive than
        # Fast" and "slowest and most expensive".
        name_col = "model" if "model" in picks.columns else "name"
        if not picks.empty and name_col in picks.columns:
            fresh = picks[~picks[name_col].isin(_taken)]
            if not fresh.empty:
                picks = fresh
            else:
                # Every candidate is already spoken for. Show the repeat rather
                # than an empty card, but say so and drop the advice, which is
                # written on the assumption the tiers differ.
                duplicate_of = _taken.get(str(picks.iloc[0][name_col]))
        if not picks.empty and name_col in picks.columns:
            top = str(picks.iloc[0][name_col])
            duplicate_of = duplicate_of or (_taken.get(top) if top in _taken else None)
            _taken.setdefault(top, tier["label"])

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

    return {"tiers": tiers_out}


# ── HTML-string renderer (Pyodide-safe, no Dash) ───────────────────────────────

def _h(tag: str, inner: str = "", style: str | None = None, cls: str | None = None) -> str:
    """Minimal HTML tag builder mirroring dash.html structure."""
    s = f' style="{style}"' if style else ""
    c = f' class="{cls}"' if cls else ""
    return f"<{tag}{c}{s}>{inner}</{tag}>"


def _dict_to_style(d: dict) -> str:
    """Convert a CSS dict (camelCase keys) to an inline style string."""
    def _camel_to_kebab(k: str) -> str:
        return re.sub(r'([A-Z])', r'-\1', k).lower()

    parts = []
    for k, v in d.items():
        parts.append(f"{_camel_to_kebab(k)}:{v}")
    return ";".join(parts)


def _chip_html(text: str, bg: str = "#1e2a1e") -> str:
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
    return _h("span", text, style=style)


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

    price_str = f"${price:.4f}/M" if price < 0.01 else f"${price:.3f}/M" if price < 1 else f"${price:.2f}/M"
    speed_str = f"{int(speed):,} tok/s" if speed > 0 else "—"
    lat       = row.get("latency", float("nan"))
    lat_str   = f"{lat:.2f}s TTFT" if pd.notna(lat) and lat > 0 else None

    name = str(row["model"])
    if len(name) > 36:
        name = name[:35] + "…"

    chips_html = (
        _chip_html(price_str, "#1a2a1a") +
        _chip_html(speed_str, "#1a1a2a") +
        (_chip_html(lat_str, "#2a1a1a") if lat_str else "")
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

    chips_html = _chip_html(vram_str, "#1a1a2e") + _chip_html(speed_str, "#1a2a1a")
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
    icon      = _TIER_ICONS[key]

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
        _h("span", icon + " " + name, style=_dict_to_style({
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
