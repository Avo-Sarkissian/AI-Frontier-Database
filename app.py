"""
AI Frontier — Main Dash Application
Interactive dashboard comparing 100+ LLMs on cost, speed, and quality.

Architecture: all dcc.Graph components live inside dcc.Tab.children so their
IDs are always in the DOM regardless of active tab. This prevents Dash 4's
"nonexistent object" callback errors.
"""
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs

import dash
from dash import ctx, dcc, html, dash_table, Input, Output, State, callback, clientside_callback, no_update
import pandas as pd

from data.ingest import get_models, load_history
from data.scraper import start_background_scraper
from data.image_scraper import start_background_image_scraper
from data.local_models import get_local_df, get_gpu_options, GPU_BY_NAME, QUANT_LEVELS
from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR
from components.charts.pareto               import build_pareto_scatter
from components.charts.quadrant             import build_quadrant
from components.charts.treemap              import build_treemap
from components.charts.rankings             import build_rankings
from components.charts.radar                import build_radar
from components.charts.cost_calc            import build_cost_calc
from components.charts.provider_leaderboard import build_provider_leaderboard
from components.charts.local_scatter        import build_local_scatter
from components.charts.local_compat         import build_local_compat
from components.charts.image_scatter        import build_image_faceted, build_image_rankings
from components.charts.video_chart          import build_video_rankings, build_video_scatter
from components.charts.embedding_chart      import build_embedding_scatter, build_embedding_rankings
from components.stack_recommender           import build_stack_cards
from data.image_models                      import get_image_df, get_image_providers, PROVIDER_COLORS as IMG_PROVIDER_COLORS
from data.video_models                      import get_video_df, get_video_providers
from data.embedding_models                  import get_embedding_df, get_embedding_providers
from components.charts.bump_chart          import build_bump_chart

# ── Data ─────────────────────────────────────────────────────────────────────
df         = get_models()
history_df = load_history()

# Kick off background scrapers — run immediately, then every hour.
# Guard: in Werkzeug debug-reload mode two Python processes exist — the
# watchdog (WERKZEUG_RUN_MAIN not set) and the real worker child
# (WERKZEUG_RUN_MAIN=true). Without this check both processes start a
# scraper thread and race to write the cache on every file-save reload.
_debug_mode       = os.getenv("DEBUG", "false").lower() == "true"
_is_worker_child  = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
if not _debug_mode or _is_worker_child:
    start_background_scraper(interval_s=3600)
    start_background_image_scraper(interval_s=3600)

_CACHE_PATH  = Path(__file__).parent / "data" / "raw" / "aa_models.csv"
_data_lock   = threading.Lock()
_cache_mtime = _CACHE_PATH.stat().st_mtime if _CACHE_PATH.exists() else 0.0


def _cache_ts() -> str:
    try:
        return datetime.fromtimestamp(_CACHE_PATH.stat().st_mtime).strftime("%b %d  %H:%M")
    except Exception:
        return "—"


def _reload_if_stale():
    """Re-read df/history_df if the cache file has changed on disk."""
    global df, history_df, _cache_mtime
    try:
        mtime = _CACHE_PATH.stat().st_mtime
        if mtime != _cache_mtime:
            with _data_lock:
                # Double-check inside the lock to avoid redundant reloads
                # if two callbacks both saw the stale mtime simultaneously.
                if mtime != _cache_mtime:
                    df           = get_models()
                    history_df   = load_history()
                    _cache_mtime = mtime
    except Exception:
        pass

def _compute_diverse5(dataframe: pd.DataFrame) -> list[str]:
    """Pick 5 diverse models spanning quality, value, speed, and budget tiers."""
    valid = dataframe[(dataframe["quality"] > 0) & (dataframe["price"] > 0)].copy()
    if valid.empty:
        return []
    picks: list[str] = []

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


_DIVERSE5    = _compute_diverse5(df)


def _compute_insights(dataframe: pd.DataFrame, hist: pd.DataFrame) -> dict:
    """Pre-compute narrative statistics shown in the Insights tab."""
    out: dict = {}
    valid = dataframe[(dataframe["quality"] > 0) & (dataframe["price"] > 0)]
    if valid.empty:
        return out

    # Best value model — apply a quality floor so cheap-but-weak models
    # don't dominate the "best value" headline.
    valid = valid.copy()
    valid["_val"] = valid["quality"] / valid["price"]
    quality_floor = max(valid["quality"].quantile(0.20), 25.0)
    bv_pool = valid[valid["quality"] >= quality_floor]
    bv = bv_pool.loc[bv_pool["_val"].idxmax()] if not bv_pool.empty else valid.loc[valid["_val"].idxmax()]
    out["best_value_model"]    = bv["model"]
    out["best_value_provider"] = bv["provider"]
    out["best_value_score"]    = f"{bv['quality']:.0f}"
    out["best_value_price"]    = f"${bv['price']:.4f}"
    out["best_value_ratio"]    = f"{bv['_val']:.1f}"

    # Cheapest frontier model with meaningful quality (≥ 30).
    # The raw cheapest Pareto entry is often a sub-30 model which isn't
    # actionable for most use-cases, so we skip those.
    sorted_p = valid.sort_values("price")
    max_q = 0.0
    frontier = []
    for _, row in sorted_p.iterrows():
        if row["quality"] > max_q:
            frontier.append(row)
            max_q = float(row["quality"])
    useful = [f for f in frontier if f["quality"] >= 30]
    if useful:
        cheapest_f = useful[0]
        out["frontier_cheapest_model"]    = cheapest_f["model"]
        out["frontier_cheapest_price"]    = f"${cheapest_f['price']:.4f}/M"
        out["frontier_cheapest_quality"]  = f"{cheapest_f['quality']:.0f}"
    elif frontier:
        cheapest_f = frontier[0]
        out["frontier_cheapest_model"]    = cheapest_f["model"]
        out["frontier_cheapest_price"]    = f"${cheapest_f['price']:.4f}/M"
        out["frontier_cheapest_quality"]  = f"{cheapest_f['quality']:.0f}"

    # Price compression: first vs last snapshot
    if not hist.empty:
        h = hist.copy()
        h["scraped_at"] = pd.to_datetime(h["scraped_at"]).dt.date
        h = h[(h["quality"] > 0) & (h["price"] > 0)]
        dates = sorted(h["scraped_at"].unique())
        if len(dates) >= 2:
            first_avg = h[h["scraped_at"] == dates[0]]["price"].median()
            last_avg  = h[h["scraped_at"] == dates[-1]]["price"].median()
            if first_avg > 0:
                pct = (last_avg - first_avg) / first_avg * 100
                out["price_change_pct"]  = f"{pct:+.0f}%"
                out["price_change_dir"]  = "down" if pct < 0 else "up"
                out["snapshot_window"]   = f"{dates[0].strftime('%b %d')} – {dates[-1].strftime('%b %d')}"
                out["n_snapshots"]       = len(dates)

    # Speed vs quality correlation
    sq = valid[(valid["speed"] > 0)]
    if len(sq) >= 5:
        import numpy as _np
        r = _np.corrcoef(sq["speed"], sq["quality"])[0, 1]
        out["speed_quality_r"] = f"{r:.2f}"

    return out


_INSIGHTS = _compute_insights(df, history_df)
_N_SNAPSHOTS = history_df["scraped_at"].nunique() if not history_df.empty else 0
# Percentile thresholds for preset filters (recomputed on each restart)
_P75 = round(df["quality"].quantile(0.75), 1)   # top 25%
_P90 = round(df["quality"].quantile(0.90), 1)   # top 10%

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="AI Frontier",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server


# ── Helpers ───────────────────────────────────────────────────────────────────
def _stat(value: str, label: str, accent: bool = False) -> html.Div:
    return html.Div([
        html.Div(value, className="stat-value",
                 style={"color": "#00d4ff" if accent else "#f2f2f2"}),
        html.Div(label, className="stat-label"),
    ], className="stat")


def _provider_options(dataframe: pd.DataFrame) -> list[dict]:
    return [{"label": p, "value": p} for p in sorted(dataframe["provider"].unique())]


def _model_options(dataframe: pd.DataFrame) -> list[dict]:
    top = dataframe[dataframe["quality"] > 0].sort_values("quality", ascending=False)
    return [{"label": f"{r['model']} ({r['provider']})", "value": r["model"]}
            for _, r in top.iterrows()]


def _apply_filters(providers, min_quality, search: str = "") -> pd.DataFrame:
    filtered = df.copy()
    if providers:
        filtered = filtered[filtered["provider"].isin(providers)]
    if min_quality and min_quality > 0:
        filtered = filtered[filtered["quality"] >= min_quality]
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


def _quality_label(pct: float) -> str:
    """pct is quality normalised to 0–100 relative to the dataset max."""
    if pct >= 90: return "Exceptional"
    if pct >= 75: return "Strong"
    if pct >= 55: return "Capable"
    if pct >= 35: return "Average"
    return "Limited"


def _desc(text: str) -> html.Div:
    return html.Div(text, className="chart-caption")


def _build_raw_table(dataframe: pd.DataFrame, selected_models: list[str]) -> html.Div:
    """Raw values table rendered below the radar chart."""
    _FONT_S = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    rows = dataframe[dataframe["model"].isin(selected_models)].copy()
    if rows.empty:
        return html.Div()

    header_style = {
        "padding": "8px 14px", "fontSize": "9px", "letterSpacing": "0.08em",
        "color": "#555", "fontFamily": _FONT_S, "fontWeight": "600",
        "textTransform": "uppercase", "textAlign": "right", "whiteSpace": "nowrap",
    }
    cell_style_base = {
        "padding": "7px 14px", "fontSize": "11px", "fontFamily": _FONT_S,
        "borderBottom": "1px solid rgba(255,255,255,0.04)", "textAlign": "right",
        "whiteSpace": "nowrap",
    }

    def _cell(text, color="#888"):
        return html.Td(text, style={**cell_style_base, "color": color})

    header = html.Tr([
        html.Th("Model",           style={**header_style, "textAlign": "left"}),
        html.Th("Provider",        style={**header_style, "textAlign": "left"}),
        html.Th("Intelligence",    style=header_style),
        html.Th("Price ($/M tok)", style=header_style),
        html.Th("Speed (tok/s)",   style=header_style),
        html.Th("Latency (TTFT)",  style=header_style),
        html.Th("Context",         style=header_style),
    ])

    table_rows = []
    for _, r in rows.sort_values("quality", ascending=False).iterrows():
        pcolor    = PROVIDER_COLORS.get(r["provider"], DEFAULT_COLOR)
        price_str = f"${r['price']:.4f}" if pd.notna(r["price"]) and r["price"] > 0 else "—"
        speed_str = f"{int(r['speed']):,}" if pd.notna(r["speed"]) and r["speed"] > 0 else "—"
        lat_str   = f"{r['latency']:.2f}s" if pd.notna(r["latency"]) and r["latency"] > 0 else "—"
        ctx_str   = str(r["context"]) if pd.notna(r.get("context")) else "—"
        table_rows.append(html.Tr([
            html.Td(r["model"], style={**cell_style_base, "textAlign": "left",
                                       "color": "#ccc", "maxWidth": "260px",
                                       "overflow": "hidden", "textOverflow": "ellipsis"}),
            html.Td(r["provider"], style={**cell_style_base, "textAlign": "left", "color": pcolor}),
            _cell(f"{r['quality']:.1f}", "#f2f2f2"),
            _cell(price_str),
            _cell(speed_str),
            _cell(lat_str),
            _cell(ctx_str),
        ]))

    return html.Div([
        html.Div("RAW VALUES", style={
            "fontSize": "9px", "letterSpacing": "0.1em", "color": "#444",
            "fontFamily": _FONT_S, "padding": "14px 14px 6px", "fontWeight": "600",
        }),
        html.Table(
            [html.Thead(header), html.Tbody(table_rows)],
            style={"width": "100%", "borderCollapse": "collapse", "overflowX": "auto"},
        ),
    ])


_GRAPH_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"],
}
_LOADING = dict(type="dot", color="#00d4ff")


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Invisible infrastructure
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="url-sync"),
    dcc.Store(id="local-hw-meta",     data={"bandwidth_gbps": 1792, "hw_type": "nvidia"}),
    dcc.Store(id="detail-model-name", data=None),
    dcc.Store(id="refresh-sink"),
    dcc.Store(id="share-sink"),
    dcc.Store(id="data-version",      data=0),
    dcc.Download(id="download-csv"),
    dcc.Interval(id="data-refresh-interval", interval=10 * 60 * 1000, n_intervals=0),

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("AI FRONTIER"),
            html.Span("LLM comparison dashboard", className="subtitle"),
        ], className="header-left"),
        html.Div([
            html.Button("⟳", id="btn-refresh", className="header-icon-btn",
                        title="Reload page to fetch latest data"),
            html.Button("↗", id="btn-share", className="header-icon-btn",
                        title="Copy URL to clipboard"),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], className="header"),

    # ── Stat bar — all values are callback-driven so they stay live ───────────
    html.Div([
        html.Div([
            html.Div(id="stat-model-count",
                     children=str(len(df)), className="stat-value"),
            html.Div("Models tracked", className="stat-label"),
        ], className="stat"),
        html.Div([
            html.Div(id="stat-provider-count",
                     children=str(df["provider"].nunique()), className="stat-value"),
            html.Div("Providers", className="stat-label"),
        ], className="stat"),
        html.Div([
            html.Div(id="stat-floor-price",
                     children=f"${df['price'].min():.3f}",
                     className="stat-value", style={"color": "#00d4ff"}),
            html.Div("Floor price / 1M", className="stat-label"),
        ], className="stat"),
        html.Div([
            html.Div(id="stat-peak-quality",
                     children=f"{df['quality'].max():.1f}", className="stat-value"),
            html.Div("Peak intelligence", className="stat-label"),
        ], className="stat"),
    ], className="stat-bar"),

    # ── Global filters ────────────────────────────────────────────────────────
    html.Div([
        html.Span("PROVIDER", className="filter-label"),
        dcc.Dropdown(
            id="filter-provider",
            options=_provider_options(df),
            multi=True,
            placeholder="All providers",
            style={"minWidth": "220px"},
        ),
        html.Div(className="filter-sep"),
        html.Span("MIN SCORE", className="filter-label"),
        dcc.Dropdown(
            id="filter-quality",
            options=[{"label": f"≥ {v}", "value": v}
                     for v in [0, 10, 15, 20, 25, 30, 35, 40, 45, 50]],
            value=0,
            clearable=False,
            style={"width": "96px"},
        ),
        html.Div(className="filter-sep"),
        html.Span("SEARCH", className="filter-label"),
        dcc.Input(
            id="model-search",
            type="text",
            placeholder="model or provider…",
            debounce=True,
            className="search-input",
        ),
        html.Div(className="filter-sep"),
        html.Button("↓ CSV", id="btn-export", className="export-btn",
                    title="Download filtered data as CSV"),
        html.Div(style={"flex": "1"}),   # spacer
        html.Button("All",      id="preset-all",    className="preset-btn"),
        html.Button("Top 25%",  id="preset-strong", className="preset-btn"),
        html.Button("Top 10%",  id="preset-elite",  className="preset-btn"),
    ], className="filters"),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dcc.Tabs(id="tabs", value="insights", className="tabs", children=[

        # Insights ─────────────────────────────────────────────────────────────
        dcc.Tab(label="Insights", value="insights",
                className="tab", selected_className="tab--selected", children=[
            html.Div([
                html.Div("What the data says", className="insight-heading"),
                html.Div(
                    "Key findings from the current snapshot — patterns that stand out "
                    "when you look across 100+ models at once.",
                    className="insight-subheading",
                ),
            ], className="insight-hero"),

            # ── Callout cards row ────────────────────────────────────────────
            html.Div([

                # Card 1: Price compression
                html.Div([
                    html.Div("PRICE COMPRESSION", className="insight-card-label"),
                    html.Div(
                        _INSIGHTS.get("price_change_pct", "—"),
                        className="insight-card-value",
                        style={"color": "#22c55e" if _INSIGHTS.get("price_change_dir") == "down" else "#f87171"},
                    ),
                    html.Div(
                        f"Median API cost change over "
                        f"{_INSIGHTS.get('snapshot_window', 'the tracked window')} "
                        f"({_INSIGHTS.get('n_snapshots', '—')} daily snapshots).",
                        className="insight-card-body",
                    ),
                    html.Div(
                        "Frontier models are getting cheaper without quality loss — "
                        "the animated chart on the Overview tab shows this shift.",
                        className="insight-card-footnote",
                    ),
                ], className="insight-card"),

                # Card 2: Best value model
                html.Div([
                    html.Div("BEST VALUE RIGHT NOW", className="insight-card-label"),
                    html.Div(
                        _INSIGHTS.get("best_value_model", "—"),
                        className="insight-card-value",
                        style={"fontSize": "16px"},
                    ),
                    html.Div(
                        f"{_INSIGHTS.get('best_value_provider', '')} · "
                        f"Score {_INSIGHTS.get('best_value_score', '—')} · "
                        f"{_INSIGHTS.get('best_value_price', '—')}/M tokens · "
                        f"{_INSIGHTS.get('best_value_ratio', '—')} pts per dollar",
                        className="insight-card-body",
                    ),
                    html.Div(
                        "Value = Intelligence ÷ Price. "
                        "This model delivers the most benchmark performance per dollar.",
                        className="insight-card-footnote",
                    ),
                ], className="insight-card"),

                # Card 3: Speed vs quality correlation
                html.Div([
                    html.Div("SPEED vs. QUALITY", className="insight-card-label"),
                    html.Div(
                        f"r = {_INSIGHTS.get('speed_quality_r', '—')}",
                        className="insight-card-value",
                    ),
                    html.Div(
                        "Pearson correlation between throughput (tok/s) and "
                        "AA Intelligence Index across all models with valid speed data.",
                        className="insight-card-body",
                    ),
                    html.Div(
                        "A value near 0 means fast models are no dumber than slow ones — "
                        "speed is an infrastructure choice, not a quality trade-off.",
                        className="insight-card-footnote",
                    ),
                ], className="insight-card"),

                # Card 4: Cheapest frontier model
                html.Div([
                    html.Div("CHEAPEST FRONTIER ENTRY", className="insight-card-label"),
                    html.Div(
                        _INSIGHTS.get("frontier_cheapest_model", "—"),
                        className="insight-card-value",
                        style={"fontSize": "16px"},
                    ),
                    html.Div(
                        f"{_INSIGHTS.get('frontier_cheapest_price', '—')} · "
                        f"Score {_INSIGHTS.get('frontier_cheapest_quality', '—')}",
                        className="insight-card-body",
                    ),
                    html.Div(
                        "This model sits on the Pareto frontier — "
                        "no cheaper model matches its quality.",
                        className="insight-card-footnote",
                    ),
                ], className="insight-card"),

            ], className="insight-cards"),

            # ── Bump chart ───────────────────────────────────────────────────
            _desc(
                "Rank evolution: how the top 12 models' intelligence rankings have shifted "
                "across daily snapshots. Rank 1 = highest AA Intelligence Index. "
                "A rising line means improving rank. Hover for details."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(
                    id="bump-chart",
                    figure=build_bump_chart(history_df),
                    config=_GRAPH_CONFIG,
                    style={"height": "580px"},
                ),
            ])], className="chart-card"),
        ]),

        # Overview ─────────────────────────────────────────────────────────────
        dcc.Tab(label="Overview", value="overview",
                className="tab", selected_className="tab--selected", children=[
            html.Div([
                html.Span("X AXIS", className="filter-label"),
                dcc.RadioItems(
                    id="overview-xaxis",
                    options=[
                        {"label": "Price",  "value": "price"},
                        {"label": "Speed",  "value": "speed"},
                    ],
                    value="price",
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "20px", "cursor": "pointer",
                                "fontSize": "12px", "color": "#aaa"},
                ),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div(id="overview-desc", children=[_desc(
                "Each bubble is one model. X = price per 1M tokens (log scale), "
                "Y = AA Intelligence Index. Bubble size = throughput (tok/s). "
                "Dotted line = Pareto frontier. Click any bubble for full details."
            )]),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="pareto-chart", figure=build_pareto_scatter(df),
                          config=_GRAPH_CONFIG, style={"height": "620px"}),
            ])], className="chart-card"),
        ]),

        # Agent Stack ──────────────────────────────────────────────────────────
        dcc.Tab(label="Agent Stack", value="recommend",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Build your Claude Code model stack. "
                "Pick a workflow — API Only (cloud, any machine), "
                "Hybrid (local Fast tier, API for heavy tasks), "
                "or Local Only (fully offline, filtered to your hardware). "
                "Fast = haiku-class, high-volume sub-tasks. "
                "Balanced = coding and writing. "
                "Reasoning = planning and delegation."
            ),
            # Row 1: workflow mode
            html.Div([
                html.Span("WORKFLOW", className="filter-label"),
                dcc.RadioItems(
                    id="recommend-mode",
                    options=[
                        {"label": "API Only",   "value": "api"},
                        {"label": "Hybrid",     "value": "hybrid"},
                        {"label": "Local Only", "value": "local"},
                    ],
                    value="api",
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "20px", "cursor": "pointer",
                                "fontSize": "12px", "color": "#aaa"},
                ),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            # Row 2: provider filter (hidden in local mode)
            html.Div([
                html.Span("PROVIDERS", className="filter-label"),
                dcc.Checklist(
                    id="recommend-providers",
                    options=[
                        {"label": "Anthropic",  "value": "Anthropic"},
                        {"label": "Google",     "value": "Google"},
                        {"label": "OpenAI",     "value": "OpenAI"},
                        {"label": "xAI",        "value": "xAI"},
                        {"label": "DeepSeek",   "value": "DeepSeek"},
                        {"label": "Mistral",    "value": "Mistral"},
                        {"label": "All",        "value": "__all__"},
                    ],
                    value=["Anthropic", "Google", "OpenAI"],
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "20px", "cursor": "pointer",
                                "fontSize": "12px", "color": "#aaa"},
                ),
            ], id="recommend-providers-row", className="filters",
               style={"paddingTop": "0"}),
            # Row 3: hardware controls (shown in hybrid/local modes)
            html.Div([
                html.Span("GPU", className="filter-label"),
                dcc.Dropdown(
                    id="recommend-gpu-preset",
                    options=get_gpu_options(),
                    value="NVIDIA RTX 5090",
                    placeholder="Select GPU…",
                    clearable=False,
                    style={"minWidth": "260px"},
                ),
                html.Div(className="filter-sep"),
                html.Span("VRAM (GB)", className="filter-label"),
                dcc.Input(
                    id="recommend-vram",
                    type="number", value=32, min=1, step=1, debounce=True,
                    style={
                        "background": "var(--bg-card)", "border": "1px solid var(--border)",
                        "borderRadius": "4px", "color": "#f2f2f2",
                        "fontFamily": "Inter, sans-serif", "fontSize": "13px",
                        "padding": "6px 10px", "width": "72px", "outline": "none",
                    },
                ),
                html.Div(className="filter-sep"),
                html.Span("GPUs", className="filter-label"),
                dcc.Dropdown(
                    id="recommend-num-gpus",
                    options=[{"label": f"×{n}", "value": n} for n in [1, 2, 4, 8]],
                    value=1, clearable=False,
                    style={"width": "72px"},
                ),
                html.Div(className="filter-sep"),
                html.Span("QUANT", className="filter-label"),
                dcc.Dropdown(
                    id="recommend-quant",
                    options=[{"label": q, "value": q} for q in QUANT_LEVELS],
                    value="Q4", clearable=False,
                    style={"width": "88px"},
                ),
            ], id="recommend-hw-row", className="filters",
               style={"paddingTop": "0", "display": "none"}),
            html.Div(id="recommend-cards",
                     children=build_stack_cards(df, ["Anthropic", "Google", "OpenAI"]),
                     className="chart-card"),
        ]),

        # Landscape ────────────────────────────────────────────────────────────
        dcc.Tab(label="Landscape", value="landscape",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "AI ecosystem by provider. Tile area = number of models in the dataset. "
                "Color intensity = average intelligence score."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="treemap-chart", figure=build_treemap(df),
                          config=_GRAPH_CONFIG, style={"height": "600px"}),
            ])], className="chart-card"),
            _desc(
                "Provider leaderboard: bar length = best model's intelligence. "
                "Tick mark = provider average. Right labels show model count and top model."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="provider-leaderboard-chart",
                          figure=build_provider_leaderboard(df),
                          config=_GRAPH_CONFIG, style={"minHeight": "400px"}),
            ])], className="chart-card"),
        ]),

        # Rankings ─────────────────────────────────────────────────────────────
        dcc.Tab(label="Rankings", value="rankings",
                className="tab", selected_className="tab--selected", children=[
            html.Div([
                html.Span("SORT BY", className="filter-label"),
                dcc.RadioItems(
                    id="rankings-sort",
                    options=[
                        {"label": "Intelligence", "value": "intelligence"},
                        {"label": "Value (score/$)", "value": "value"},
                        {"label": "Speed",        "value": "speed"},
                    ],
                    value="intelligence",
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "20px", "cursor": "pointer",
                                "fontSize": "12px", "color": "#aaa"},
                ),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            _desc(
                "Intelligence = AA Intelligence Index (composite benchmark). "
                "Value = Intelligence ÷ Price (higher = more score per dollar). "
                "Speed = throughput in tokens/second. "
                "Models within ±2 points of each other are effectively tied — small deltas are within measurement variance."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="rankings-chart", figure=build_rankings(df, top_n=25),
                          config=_GRAPH_CONFIG, style={"height": "750px"}),
            ])], className="chart-card"),
        ]),

        # Compare ──────────────────────────────────────────────────────────────
        dcc.Tab(label="Compare", value="compare",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Radar comparing up to 5 models across 5 dimensions — all normalized to 0–100% "
                "relative to the best model in the full dataset. "
                "Affordability = inverted price (100% = cheapest). Latency = inverted TTFT (100% = fastest). "
                "Raw values for each model are shown in the table below the chart."
            ),
            html.Div([
                html.Span("SELECT MODELS", className="filter-label"),
                dcc.Dropdown(
                    id="radar-model-select",
                    options=_model_options(df),
                    value=_DIVERSE5,
                    multi=True,
                    placeholder="Select up to 5 models…",
                    style={"minWidth": "500px"},
                ),
                html.Span("max 5", className="filter-label",
                          style={"color": "var(--text-3)", "paddingLeft": "8px"}),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="radar-chart", figure=build_radar(df, _DIVERSE5),
                          config=_GRAPH_CONFIG, style={"height": "560px"}),
            ])], className="chart-card"),
            html.Div(id="compare-raw-table", className="chart-card",
                     style={"padding": "0"},
                     children=_build_raw_table(df, _DIVERSE5)),
        ]),

        # Budget ───────────────────────────────────────────────────────────────
        dcc.Tab(label="Budget", value="budget",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Estimate monthly API cost. Price uses Artificial Analysis's blended rate "
                "(assumes 3:1 output/input token ratio). Enter volume in millions — "
                "1M tokens ≈ 750,000 words or ~1,500 pages. Chart sorts cheapest-first."
            ),
            html.Div([
                html.Span("MONTHLY TOKENS", className="filter-label"),
                dcc.Input(
                    id="budget-tokens",
                    type="number", value=1.0, min=0.001, step=0.5,
                    debounce=True,
                    placeholder="e.g. 10",
                    style={
                        "background": "var(--bg-card)", "border": "1px solid var(--border)",
                        "borderRadius": "4px", "color": "#f2f2f2",
                        "fontFamily": "Inter, sans-serif", "fontSize": "13px",
                        "padding": "6px 10px", "width": "100px", "outline": "none",
                    },
                ),
                html.Span("million tokens / month", className="budget-unit",
                          style={"fontSize": "11px", "paddingLeft": "6px"}),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="cost-calc-chart",
                          figure=build_cost_calc(df, monthly_tokens_m=1.0),
                          config=_GRAPH_CONFIG, style={"height": "850px"}),
            ])], className="chart-card"),
        ]),

        # Table ────────────────────────────────────────────────────────────────
        dcc.Tab(label="Table", value="table",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Full sortable model table. Score = AA Intelligence Index (composite benchmark, higher = better). "
                "Value = Score ÷ Price (quality per dollar). Price = blended $/M tokens (3:1 output/input). "
                "Latency = time-to-first-token (TTFT) in seconds. Click any column header to sort."
            ),
            html.Div([
                dash_table.DataTable(
                    id="model-table",
                    columns=[
                        {"name": "Model",            "id": "model",    "type": "text"},
                        {"name": "Provider",          "id": "provider", "type": "text"},
                        {"name": "Score",             "id": "quality",  "type": "numeric",
                         "format": {"specifier": ".1f"}},
                        {"name": "Value (score/$)",   "id": "value",    "type": "numeric",
                         "format": {"specifier": ".2f"}},
                        {"name": "Price ($/M tok)",   "id": "price",    "type": "numeric",
                         "format": {"specifier": ".4f"}},
                        {"name": "Speed (tok/s)",     "id": "speed",    "type": "numeric",
                         "format": {"specifier": ".0f"}},
                        {"name": "Latency (s)",       "id": "latency",  "type": "numeric",
                         "format": {"specifier": ".2f"}},
                        {"name": "Context",           "id": "context",  "type": "text"},
                    ],
                    data=(lambda _df: _df.assign(
                        value=_df.apply(
                            lambda r: r["quality"] / r["price"] if r["price"] > 0 else None, axis=1
                        )
                    )[["model", "provider", "quality", "value", "price", "speed", "latency", "context"]].to_dict("records"))(df),
                    sort_action="native",
                    sort_mode="single",
                    filter_action="none",
                    page_action="native",
                    page_size=50,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#111111",
                        "color": "#aaaaaa",
                        "fontFamily": "Inter, sans-serif",
                        "fontSize": "11px",
                        "fontWeight": "600",
                        "letterSpacing": "0.06em",
                        "textTransform": "uppercase",
                        "borderBottom": "1px solid rgba(255,255,255,0.08)",
                        "borderTop": "none",
                        "paddingTop": "10px",
                        "paddingBottom": "10px",
                    },
                    style_cell={
                        "backgroundColor": "#0e0e0e",
                        "color": "#888888",
                        "fontFamily": "Inter, sans-serif",
                        "fontSize": "12px",
                        "padding": "8px 14px",
                        "border": "none",
                        "borderBottom": "1px solid rgba(255,255,255,0.04)",
                        "textAlign": "left",
                        "whiteSpace": "normal",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "maxWidth": "260px",
                    },
                    style_cell_conditional=[
                        {"if": {"column_id": "quality"},  "color": "#f2f2f2", "textAlign": "right"},
                        {"if": {"column_id": "value"},    "color": "#34d399",  "textAlign": "right"},
                        {"if": {"column_id": "price"},    "textAlign": "right"},
                        {"if": {"column_id": "speed"},    "textAlign": "right"},
                        {"if": {"column_id": "latency"},  "textAlign": "right"},
                    ],
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#0a0a0a"},
                        {"if": {"state": "active"},
                         "backgroundColor": "rgba(0,212,255,0.06)",
                         "border": "1px solid rgba(0,212,255,0.2)"},
                        *[
                            {"if": {"filter_query": f'{{provider}} = "{p}"',
                                    "column_id": "provider"},
                             "color": PROVIDER_COLORS.get(p, DEFAULT_COLOR)}
                            for p in sorted(df["provider"].unique())
                        ],
                    ],
                    style_as_list_view=True,
                ),
            ], className="chart-card", style={"padding": "0"}),
        ]),

        # Run Local ────────────────────────────────────────────────────────────
        dcc.Tab(label="Run Local", value="local",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Find open-weight models you can run on your own hardware. "
                "Select your GPU (or enter VRAM manually), choose a quantization level, "
                "and see which models fit — with estimated inference speed."
            ),
            html.Div([
                html.Span("GPU", className="filter-label"),
                dcc.Dropdown(
                    id="local-gpu-preset",
                    options=get_gpu_options(),
                    value="NVIDIA RTX 5090",
                    placeholder="Select GPU…",
                    clearable=False,
                    style={"minWidth": "280px"},
                ),
                html.Div(className="filter-sep"),
                html.Span("VRAM (GB)", className="filter-label"),
                dcc.Input(
                    id="local-vram",
                    type="number", value=32, min=1, step=1, debounce=True,
                    style={
                        "background": "var(--bg-card)", "border": "1px solid var(--border)",
                        "borderRadius": "4px", "color": "#f2f2f2",
                        "fontFamily": "Inter, sans-serif", "fontSize": "13px",
                        "padding": "6px 10px", "width": "72px", "outline": "none",
                    },
                ),
                html.Div(className="filter-sep"),
                html.Span("GPUs", className="filter-label"),
                dcc.Dropdown(
                    id="local-num-gpus",
                    options=[{"label": f"×{n}", "value": n} for n in [1, 2, 4, 8]],
                    value=1, clearable=False,
                    style={"width": "72px"},
                ),
                html.Div(className="filter-sep"),
                html.Span("QUANT", className="filter-label"),
                dcc.Dropdown(
                    id="local-quant",
                    options=[{"label": q, "value": q} for q in QUANT_LEVELS],
                    value="Q4", clearable=False,
                    style={"width": "88px"},
                ),
                html.Div(className="filter-sep"),
                html.Span("TAGS", className="filter-label"),
                dcc.Dropdown(
                    id="local-tags",
                    options=[
                        {"label": "Code",         "value": "code"},
                        {"label": "Reasoning",    "value": "reasoning"},
                        {"label": "Vision",       "value": "vision"},
                        {"label": "Multilingual", "value": "multilingual"},
                    ],
                    multi=True,
                    placeholder="All capabilities",
                    style={"minWidth": "180px"},
                ),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(
                    id="local-scatter",
                    figure=build_local_scatter(
                        get_local_df(quant="Q4", vram_gb=32,
                                     bandwidth_gbps=1792, hw_type="nvidia"),
                        vram_gb=32, quant="Q4",
                    ),
                    config=_GRAPH_CONFIG, style={"height": "640px"},
                ),
            ])], className="chart-card"),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(
                    id="local-compat-chart",
                    figure=build_local_compat(
                        get_local_df(quant="Q4", vram_gb=32,
                                     bandwidth_gbps=1792, hw_type="nvidia"),
                        quant="Q4",
                    ),
                    config=_GRAPH_CONFIG, style={"minHeight": "400px"},
                ),
            ])], className="chart-card"),
        ]),

        # Image Gen ────────────────────────────────────────────────────────────
        dcc.Tab(label="Image Gen", value="image",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Compare image generation models by quality and style. "
                "ELO scores from Artificial Analysis Image Arena — blind human comparisons. "
                "Each column shows the best models for that style. Annotations show generation time."
            ),
            html.Div([
                html.Span("PROVIDERS", className="filter-label"),
                dcc.Dropdown(
                    id="image-provider-filter",
                    options=[{"label": p, "value": p} for p in get_image_providers()],
                    multi=True,
                    placeholder="All providers",
                    style={"minWidth": "320px"},
                ),
                html.Div(className="filter-sep"),
                html.Span("TAGS", className="filter-label"),
                dcc.Dropdown(
                    id="image-tag-filter",
                    options=[
                        {"label": "Photorealistic", "value": "photorealistic"},
                        {"label": "Artistic",       "value": "artistic"},
                        {"label": "Text & Type",    "value": "text"},
                        {"label": "Fast",           "value": "fast"},
                        {"label": "Open Weights",   "value": "open_weights"},
                    ],
                    multi=True,
                    placeholder="All types",
                    style={"minWidth": "240px"},
                ),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="image-faceted-chart",
                          figure=build_image_faceted(get_image_df()),
                          config=_GRAPH_CONFIG, style={"minHeight": "380px"}),
            ])], className="chart-card"),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="image-rankings-chart",
                          figure=build_image_rankings(get_image_df()),
                          config=_GRAPH_CONFIG, style={"minHeight": "400px"}),
            ])], className="chart-card"),
        ]),

        # Video Gen ────────────────────────────────────────────────────────────
        dcc.Tab(label="Video Gen", value="video",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Compare video generation models on quality, speed, and cost. "
                "Quality scores are human preference ratings (0–100). "
                "Price is USD per second of generated video. "
                "Open-weights models can be self-hosted for free."
            ),
            html.Div([
                html.Span("PROVIDERS", className="filter-label"),
                dcc.Dropdown(
                    id="video-provider-filter",
                    options=[{"label": p, "value": p} for p in get_video_providers()],
                    multi=True,
                    placeholder="All providers",
                    style={"minWidth": "280px"},
                ),
                html.Div(className="filter-sep"),
                html.Span("TAGS", className="filter-label"),
                dcc.Dropdown(
                    id="video-tag-filter",
                    options=[
                        {"label": "Cinematic",    "value": "cinematic"},
                        {"label": "Realistic",    "value": "realistic"},
                        {"label": "Artistic",     "value": "artistic"},
                        {"label": "Fast",         "value": "fast"},
                        {"label": "Open Weights", "value": "open-weights"},
                    ],
                    multi=True,
                    placeholder="All types",
                    style={"minWidth": "220px"},
                ),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="video-rankings-chart",
                          figure=build_video_rankings(get_video_df()),
                          config=_GRAPH_CONFIG, style={"minHeight": "400px"}),
            ])], className="chart-card"),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="video-scatter-chart",
                          figure=build_video_scatter(
                              get_video_df()[get_video_df()["price_per_sec"] > 0]
                          ),
                          config=_GRAPH_CONFIG, style={"height": "520px"}),
            ])], className="chart-card"),
        ]),

        # Embeddings ───────────────────────────────────────────────────────────
        dcc.Tab(label="Embeddings", value="embeddings",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Compare text embedding models for RAG, semantic search, and vector databases. "
                "MTEB (Massive Text Embedding Benchmark) measures retrieval quality across 56 tasks. "
                "Open-weights models are free to self-host — shown as faded bars and open circles."
            ),
            html.Div([
                html.Span("PROVIDERS", className="filter-label"),
                dcc.Dropdown(
                    id="embedding-provider-filter",
                    options=[{"label": p, "value": p} for p in get_embedding_providers()],
                    multi=True,
                    placeholder="All providers",
                    style={"minWidth": "280px"},
                ),
                html.Div(className="filter-sep"),
                html.Span("TAGS", className="filter-label"),
                dcc.Dropdown(
                    id="embedding-tag-filter",
                    options=[
                        {"label": "English",      "value": "english"},
                        {"label": "Multilingual", "value": "multilingual"},
                        {"label": "Code",         "value": "code"},
                        {"label": "Open Weights", "value": "open-weights"},
                        {"label": "Fast / Tiny",  "value": "fast"},
                    ],
                    multi=True,
                    placeholder="All types",
                    style={"minWidth": "220px"},
                ),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="embedding-scatter-chart",
                          figure=build_embedding_scatter(get_embedding_df()),
                          config=_GRAPH_CONFIG, style={"height": "520px"}),
            ])], className="chart-card"),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="embedding-rankings-chart",
                          figure=build_embedding_rankings(get_embedding_df()),
                          config=_GRAPH_CONFIG, style={"minHeight": "400px"}),
            ])], className="chart-card"),
        ]),

    ]),

    # ── Model detail panel (slide-in from right) ───────────────────────────────
    html.Div(id="detail-panel", className="detail-panel", children=[
        html.Button("✕", id="detail-close", className="detail-panel-close"),
        html.Div(id="detail-panel-body"),
        html.Button("→ Compare", id="detail-add-compare",
                    className="detail-compare-btn"),
    ]),

    # ── Footer ────────────────────────────────────────────────────────────────
    html.Div([
        html.Span("Source: Artificial Analysis · Updated live"),
        html.Span("AI Frontier"),
    ], className="footer"),

], style={"minHeight": "100vh", "background": "#0a0a0a"})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# ── URL: restore state on load ────────────────────────────────────────────────
@callback(
    Output("tabs",            "value"),
    Output("filter-provider", "value"),
    Output("filter-quality",  "value"),
    Input("url", "search"),
    prevent_initial_call=False,
)
def init_from_url(search: str):
    if not search:
        return "insights", [], 0
    params    = parse_qs(search.lstrip("?"))
    tab       = params.get("tab", ["insights"])[0]
    raw_p     = params.get("p",   [""])[0]
    providers = [x for x in raw_p.split(",") if x] if raw_p else []
    try:
        quality = int(params.get("q", [0])[0])
    except (ValueError, TypeError):
        quality = 0
    return tab, providers, quality


# ── URL: write state on change ────────────────────────────────────────────────
clientside_callback(
    """
    function(tab, providers, quality) {
        var params = new URLSearchParams();
        if (tab)                           params.set('tab', tab);
        if (providers && providers.length) params.set('p', providers.join(','));
        if (quality > 0)                   params.set('q', quality);
        var qs = params.toString();
        history.replaceState(null, '', qs ? '?' + qs : window.location.pathname);
        return window.location.href;
    }
    """,
    Output("url-sync", "data"),
    Input("tabs",            "value"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    prevent_initial_call=True,
)

# ── Refresh: reload page ───────────────────────────────────────────────────────
clientside_callback(
    "function(n) { if (n) { window.location.reload(); } return null; }",
    Output("refresh-sink", "data"),
    Input("btn-refresh", "n_clicks"),
    prevent_initial_call=True,
)

# ── Share: copy URL to clipboard ──────────────────────────────────────────────
clientside_callback(
    """
    function(n) {
        if (!n) return null;
        var url = window.location.href;
        if (navigator.clipboard) {
            navigator.clipboard.writeText(url).catch(function() {});
        } else {
            var el = document.createElement('input');
            el.value = url;
            document.body.appendChild(el);
            el.select();
            document.execCommand('copy');
            document.body.removeChild(el);
        }
        return url;
    }
    """,
    Output("share-sink", "data"),
    Input("btn-share", "n_clicks"),
    prevent_initial_call=True,
)

# ── Preset quick-filters ───────────────────────────────────────────────────────
@callback(
    Output("filter-quality",  "value",  allow_duplicate=True),
    Output("filter-provider", "value",  allow_duplicate=True),
    Output("model-search",    "value"),
    Input("preset-all",    "n_clicks"),
    Input("preset-strong", "n_clicks"),
    Input("preset-elite",  "n_clicks"),
    prevent_initial_call=True,
)
def apply_preset(*_):
    trigger = ctx.triggered_id
    if trigger == "preset-all":
        return 0, [], ""
    if trigger == "preset-strong":
        return _P75, [], no_update
    if trigger == "preset-elite":
        return _P90, [], no_update
    return no_update, no_update, no_update


# ── Recommend tab ─────────────────────────────────────────────────────────────
@callback(
    Output("recommend-cards",        "children"),
    Output("recommend-providers-row", "style"),
    Output("recommend-hw-row",        "style"),
    Input("recommend-providers",  "value"),
    Input("recommend-mode",       "value"),
    Input("recommend-gpu-preset", "value"),
    Input("recommend-vram",       "value"),
    Input("recommend-num-gpus",   "value"),
    Input("recommend-quant",      "value"),
    prevent_initial_call=True,
)
def update_recommend(selected, mode, gpu_preset, vram_per_gpu, num_gpus, quant):
    _reload_if_stale()
    mode = mode or "api"

    # Show/hide provider row and hardware row based on mode
    prov_style = {"paddingTop": "0", "display": "none"} if mode == "local" \
                 else {"paddingTop": "0"}
    hw_style   = {"paddingTop": "0"} if mode in ("hybrid", "local") \
                 else {"paddingTop": "0", "display": "none"}

    # Resolve providers for API tiers
    if mode == "local":
        providers = None
    elif not selected:
        providers = []
    elif "__all__" in selected:
        providers = None
    else:
        providers = selected

    # Build local_df when needed
    local_df = None
    if mode in ("hybrid", "local"):
        gpu_meta       = GPU_BY_NAME.get(gpu_preset or "", {})
        vram_gb        = float(vram_per_gpu or 32) * int(num_gpus or 1)
        bandwidth_gbps = gpu_meta.get("bandwidth_gbps", 1792)
        hw_type        = gpu_meta.get("hw_type", "nvidia")
        gpu_count      = int(num_gpus or 1)
        eff_bw         = bandwidth_gbps * (1 + (gpu_count - 1) * 0.85) if gpu_count > 1 else bandwidth_gbps
        local_df = get_local_df(
            quant=quant or "Q4",
            vram_gb=vram_gb,
            bandwidth_gbps=eff_bw,
            hw_type=hw_type,
        )

    cards = build_stack_cards(df, providers, mode=mode, local_df=local_df)
    return cards, prov_style, hw_style


# ── Chart update callbacks (all respond to global filters + search) ───────────
@callback(
    Output("pareto-chart",  "figure"),
    Output("overview-desc", "children"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    Input("overview-xaxis",  "value"),
    Input("data-version",    "data"),
    prevent_initial_call=True,
)
def update_overview(providers, min_quality, search, xaxis, _v):
    filtered = _apply_filters(providers, min_quality, search or "")
    if xaxis == "speed":
        desc = _desc(
            "Speed (tok/s) vs. AA Intelligence Index. Top-right = fast and smart. "
            "Bubble size = affordability (larger = cheaper). "
            "Click any bubble for full details."
        )
        return build_quadrant(filtered), desc
    desc = _desc(
        "Each bubble is one model. X = price per 1M tokens (log scale), "
        "Y = AA Intelligence Index. Bubble size = throughput (tok/s). "
        "Dotted line = Pareto frontier. Click any bubble for full details."
    )
    return build_pareto_scatter(filtered), desc


@callback(
    Output("treemap-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    Input("data-version",    "data"),
    prevent_initial_call=True,
)
def update_treemap(providers, min_quality, search, _v):
    return build_treemap(_apply_filters(providers, min_quality, search or ""))


@callback(
    Output("rankings-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    Input("rankings-sort",   "value"),
    Input("data-version",    "data"),
    prevent_initial_call=True,
)
def update_rankings(providers, min_quality, search, sort_by, _v):
    filtered = _apply_filters(providers, min_quality, search or "")
    return build_rankings(filtered, top_n=min(25, len(filtered)), metric=sort_by or "intelligence")



@callback(
    Output("provider-leaderboard-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    prevent_initial_call=True,
)
def update_provider_leaderboard(providers, min_quality, search):
    return build_provider_leaderboard(_apply_filters(providers, min_quality, search or ""))


@callback(
    Output("radar-chart",        "figure"),
    Output("radar-model-select", "options"),
    Output("radar-model-select", "value"),
    Output("compare-raw-table",  "children"),
    Input("filter-provider",     "value"),
    Input("filter-quality",      "value"),
    Input("model-search",        "value"),
    Input("radar-model-select",  "value"),
    prevent_initial_call=True,
)
def update_compare(providers, min_quality, search, selected_models):
    filtered  = _apply_filters(providers, min_quality, search or "")
    options   = _model_options(filtered)
    triggered = ctx.triggered_id

    if triggered in ("filter-provider", "filter-quality", "model-search"):
        capped = _compute_diverse5(filtered)
    else:
        capped = (selected_models or [])[:5]

    raw_table = _build_raw_table(filtered, capped)
    return build_radar(filtered, capped), options, capped, raw_table


@callback(
    Output("cost-calc-chart", "figure"),
    Input("budget-tokens",   "value"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    prevent_initial_call=True,
)
def update_cost_calc(monthly_tokens_m, providers, min_quality, search):
    filtered = _apply_filters(providers, min_quality, search or "")
    tokens   = float(monthly_tokens_m) if monthly_tokens_m else 1.0
    return build_cost_calc(filtered, monthly_tokens_m=tokens)


# ── Agent Stack tab — sync VRAM from GPU preset ───────────────────────────────
@callback(
    Output("recommend-vram", "value"),
    Input("recommend-gpu-preset", "value"),
    prevent_initial_call=True,
)
def update_recommend_hw(gpu_name: str):
    gpu = GPU_BY_NAME.get(gpu_name)
    if not gpu:
        return no_update
    return gpu["vram_gb"]


# ── Local tab ─────────────────────────────────────────────────────────────────
@callback(
    Output("local-vram",    "value"),
    Output("local-hw-meta", "data"),
    Input("local-gpu-preset", "value"),
    prevent_initial_call=True,
)
def update_local_hw(gpu_name: str):
    gpu = GPU_BY_NAME.get(gpu_name)
    if not gpu:
        return no_update, no_update
    return gpu["vram_gb"], {"bandwidth_gbps": gpu["bandwidth_gbps"], "hw_type": gpu["hw_type"]}


@callback(
    Output("local-scatter",      "figure"),
    Output("local-compat-chart", "figure"),
    Input("local-vram",     "value"),
    Input("local-num-gpus", "value"),
    Input("local-quant",    "value"),
    Input("local-hw-meta",  "data"),
    Input("local-tags",     "value"),
    prevent_initial_call=True,
)
def update_local_charts(vram_per_gpu, num_gpus, quant, hw_meta, tags):
    vram_gb        = float(vram_per_gpu or 8) * int(num_gpus or 1)
    bandwidth_gbps = (hw_meta or {}).get("bandwidth_gbps", 1792)
    hw_type        = (hw_meta or {}).get("hw_type", "nvidia")
    gpu_count      = int(num_gpus or 1)
    eff_bw         = bandwidth_gbps * (1 + (gpu_count - 1) * 0.85) if gpu_count > 1 else bandwidth_gbps

    local_df = get_local_df(
        quant=quant or "Q4",
        vram_gb=vram_gb,
        bandwidth_gbps=eff_bw,
        hw_type=hw_type,
        tags=tags or None,
    )
    return (
        build_local_scatter(local_df, vram_gb=vram_gb, quant=quant or "Q4"),
        build_local_compat(local_df, quant=quant or "Q4"),
    )


# ── Model detail panel ────────────────────────────────────────────────────────
@callback(
    Output("detail-panel",      "className"),
    Output("detail-panel-body", "children"),
    Output("detail-model-name", "data"),
    Input("pareto-chart",   "clickData"),
    Input("quadrant-chart", "clickData"),
    Input("detail-close",   "n_clicks"),
    prevent_initial_call=True,
)
def toggle_detail_panel(pareto_click, quadrant_click, _close):
    trigger = ctx.triggered_id

    if trigger == "detail-close":
        return "detail-panel", [], None

    click = pareto_click if trigger == "pareto-chart" else quadrant_click
    if not click or not click.get("points"):
        return no_update, no_update, no_update

    pt         = click["points"][0]
    customdata = pt.get("customdata", [])
    if not customdata or len(customdata) < 2:
        return no_update, no_update, no_update

    model_name = customdata[0]
    provider   = customdata[1]

    rows = df[df["model"] == model_name]
    if rows.empty:
        return no_update, no_update, no_update
    row = rows.iloc[0]

    color      = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
    quality    = float(row["quality"])
    q_max      = df["quality"].max() or 1
    quality_pct = quality / q_max * 100
    qlabel     = _quality_label(quality_pct)

    speed_val   = row["speed"]
    speed_str   = f"{int(speed_val):,} tok/s" if pd.notna(speed_val) and speed_val > 0 else "N/A"
    latency_val = row.get("latency", float("nan"))
    latency_str = f"{latency_val:.2f}s" if pd.notna(latency_val) and latency_val > 0 else "N/A"

    # Percentile rank among all models
    n_total = len(df[df["quality"] > 0])
    n_below = int((df["quality"] < quality).sum())
    pct     = round(n_below / n_total * 100) if n_total else 0

    def _metric(lbl, val, accent=False):
        return html.Div([
            html.Span(lbl, className="detail-metric-label"),
            html.Span(val, className=f"detail-metric-value{' accent' if accent else ''}"),
        ], className="detail-metric")

    body = [
        html.Div(provider,   className="detail-panel-provider", style={"color": color}),
        html.Div(model_name, className="detail-panel-model"),

        # Visual quality bar
        html.Div([
            html.Div([
                html.Span("INTELLIGENCE", className="detail-metric-label"),
                html.Span(f"{quality:.0f}  ·  {qlabel}",
                          className="detail-metric-value accent"),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "baseline", "marginBottom": "6px"}),
            html.Div(
                html.Div(style={
                    "width":        f"{quality_pct:.1f}%",
                    "height":       "3px",
                    "background":   "linear-gradient(90deg,#00d4ff,#4c9eff)",
                    "borderRadius": "2px",
                    "transition":   "width 0.4s ease",
                }),
                style={
                    "width": "100%", "height": "3px",
                    "background": "rgba(255,255,255,0.06)",
                    "borderRadius": "2px",
                    "marginBottom": "4px",
                },
            ),
            html.Div(f"Top {100 - pct}% of all models",
                     style={"fontSize": "10px", "color": "var(--text-3)",
                            "marginBottom": "16px"}),
        ]),

        html.Div(className="detail-panel-divider"),
        _metric("Price",   f"${row['price']:.4f} / 1M tokens"),
        _metric("Speed",   speed_str),
        _metric("Latency", latency_str),
        _metric("Context", str(row.get("context", "N/A")) or "N/A"),
    ]

    return "detail-panel open", body, model_name


# ── Add to Compare (from detail panel) ────────────────────────────────────────
@callback(
    Output("radar-model-select", "value", allow_duplicate=True),
    Output("tabs",               "value", allow_duplicate=True),
    Input("detail-add-compare",  "n_clicks"),
    State("detail-model-name",   "data"),
    State("radar-model-select",  "value"),
    prevent_initial_call=True,
)
def add_to_compare(n_clicks, model_name, current_selection):
    if not n_clicks or not model_name:
        return no_update, no_update
    current = list(current_selection or [])
    if model_name not in current:
        current = (current + [model_name])[:5]
    return current, "compare"


# ── Table view ────────────────────────────────────────────────────────────────
@callback(
    Output("model-table", "data"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    prevent_initial_call=True,
)
def update_table(providers, min_quality, search):
    filtered = _apply_filters(providers, min_quality, search or "").copy()
    filtered["value"] = filtered.apply(
        lambda r: r["quality"] / r["price"] if r["price"] > 0 else None, axis=1
    )
    cols = ["model", "provider", "quality", "value", "price", "speed", "latency", "context"]
    return filtered[cols].to_dict("records")


# ── Export CSV ────────────────────────────────────────────────────────────────
@callback(
    Output("download-csv",   "data"),
    Input("btn-export",      "n_clicks"),
    State("filter-provider", "value"),
    State("filter-quality",  "value"),
    State("model-search",    "value"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, providers, min_quality, search):
    if not n_clicks:
        return no_update
    filtered = _apply_filters(providers, min_quality, search or "")
    return dcc.send_data_frame(filtered.to_csv, "ai_frontier_export.csv", index=False)


# ── Auto data refresh — drives ALL stat bar values and data-version ───────────
@callback(
    Output("stat-model-count",    "children"),
    Output("stat-provider-count", "children"),
    Output("stat-floor-price",    "children"),
    Output("stat-peak-quality",   "children"),
    Output("data-version",        "data"),
    Input("data-refresh-interval", "n_intervals"),
    State("data-version",          "data"),
    prevent_initial_call=True,
)
def auto_refresh_data(_, current_version):
    prev_mtime = _cache_mtime
    _reload_if_stale()
    data_changed = (_cache_mtime != prev_mtime)
    new_version  = (current_version or 0) + (1 if data_changed else 0)
    return (
        str(len(df)),
        str(df["provider"].nunique()),
        f"${df['price'].min():.3f}",
        f"{df['quality'].max():.1f}",
        new_version,
    )


# ── Image Gen tab ─────────────────────────────────────────────────────────────
@callback(
    Output("image-faceted-chart",  "figure"),
    Output("image-rankings-chart", "figure"),
    Input("image-provider-filter", "value"),
    Input("image-tag-filter",      "value"),
    prevent_initial_call=True,
)
def update_image_charts(providers, tags):
    img_df = get_image_df()
    if providers:
        img_df = img_df[img_df["provider"].isin(providers)]
    if tags:
        for tag in tags:
            if tag == "open_weights":
                img_df = img_df[img_df["open_weights"] == True]
            else:
                img_df = img_df[img_df["tags"].apply(lambda t: tag in t)]
    # No fallback — an empty result shows empty charts, which is honest.
    # Previously this silently reset to the full dataset, making filters appear
    # to work when they were actually ignored.
    return build_image_faceted(img_df), build_image_rankings(img_df)


# ── Video Gen tab ─────────────────────────────────────────────────────────────
@callback(
    Output("video-rankings-chart", "figure"),
    Output("video-scatter-chart",  "figure"),
    Input("video-provider-filter", "value"),
    Input("video-tag-filter",      "value"),
    prevent_initial_call=True,
)
def update_video_charts(providers, tags):
    vdf = get_video_df()
    if providers:
        vdf = vdf[vdf["provider"].isin(providers)]
    if tags:
        for tag in tags:
            if tag == "open-weights":
                vdf = vdf[vdf["open_weights"] == True]
            else:
                vdf = vdf[vdf["tags"].apply(lambda t: tag in t)]
    paid = vdf[vdf["price_per_sec"] > 0] if not vdf.empty else vdf
    return build_video_rankings(vdf), build_video_scatter(paid if not paid.empty else vdf)


# ── Embeddings tab ─────────────────────────────────────────────────────────────
@callback(
    Output("embedding-scatter-chart",  "figure"),
    Output("embedding-rankings-chart", "figure"),
    Input("embedding-provider-filter", "value"),
    Input("embedding-tag-filter",      "value"),
    prevent_initial_call=True,
)
def update_embedding_charts(providers, tags):
    edf = get_embedding_df()
    if providers:
        edf = edf[edf["provider"].isin(providers)]
    if tags:
        for tag in tags:
            if tag == "open-weights":
                edf = edf[edf["open_weights"] == True]
            else:
                edf = edf[edf["tags"].apply(lambda t: tag in t)]
    return build_embedding_scatter(edf), build_embedding_rankings(edf)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(debug=debug, port=8050)
