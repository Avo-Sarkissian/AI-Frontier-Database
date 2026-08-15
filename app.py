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
from data.local_scraper import start_background_local_scraper
from data.local_models import (
    get_local_df, get_gpu_options, GPU_BY_NAME, QUANT_LEVELS,
    DEFAULT_VRAM_GB, DEFAULT_GPU_COUNT, DEFAULT_BANDWIDTH_GBPS,
)
from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR, canonical_provider
from components.charts.pareto               import build_pareto_scatter
from components.charts.quadrant             import build_quadrant
from components.charts.treemap              import build_treemap
from components.charts.rankings             import build_rankings
from components.charts.radar                import build_radar
from components.charts.cost_calc            import build_cost_calc
from components.charts.provider_leaderboard import build_provider_leaderboard
from components.charts.local_scatter        import build_local_scatter
from components.charts.local_compat         import build_local_compat
from components.charts.image_scatter        import build_image_faceted
from components.charts.video_chart          import build_video_rankings, build_video_scatter
from components.stack_recommender           import build_stack_cards
from data.image_models                      import get_image_df, get_image_providers, get_image_tags, PROVIDER_COLORS as IMG_PROVIDER_COLORS
from data.video_models                      import get_video_df, get_video_providers, get_video_tags
from components.charts.bump_chart          import build_bump_chart, build_value_leaders

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
    start_background_local_scraper(interval_s=3600)

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

from captions import CAPTIONS
from static_helpers import (
    apply_filters,
    coerce_number as _coerce_number,
    cap_compare_selection as _cap_compare_selection,
    quality_options as _quality_options,
    export_frame_for_tab as _export_frame_for_tab_shared,
    TABS_WITHOUT_GLOBAL_FILTERS as _TABS_WITHOUT_GLOBAL_FILTERS,
    compute_diverse5 as _compute_diverse5,
    ctx_to_k as _ctx_to_k,
    quality_label as _quality_label,
    provider_options as _provider_options,
    model_options as _model_options,
)


_DIVERSE5    = _compute_diverse5(df)


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


def _apply_filters(providers, min_quality, search: str = "") -> pd.DataFrame:
    return apply_filters(df, providers, min_quality, search)


def _export_frame_for_tab(tab, providers, min_quality, search):
    return _export_frame_for_tab_shared(tab, df, providers, min_quality, search)


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
    "responsive": True,
}
# For charts whose layout.height is authoritative (e.g. tall bar lists where
# row count determines height). responsive=True would make Plotly fit the
# container and ignore layout.height, crushing N-row charts.
_GRAPH_CONFIG_FIXED = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"],
    "responsive": False,
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
    dcc.Store(id="resize-sink"),
    dcc.Store(id="table-data-store", data=(lambda _df: _df.assign(
        value=_df.apply(lambda r: r["quality"] / r["price"] if r["price"] > 0 else None, axis=1)
    ).sort_values("quality", ascending=False)[
        ["model", "provider", "quality", "value", "price", "speed", "latency", "context"]
    ].to_dict("records"))(df)),
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
            # Includes the exact preset percentiles — writing 42.1 into a
            # round-ladder dropdown left the control visually blank.
            options=_quality_options(_P75, _P90),
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
        html.Button("Reset filters", id="preset-all", className="preset-btn"),
        html.Button("Top 25%",  id="preset-strong", className="preset-btn"),
        html.Button("Top 10%",  id="preset-elite",  className="preset-btn"),
    ], className="filters", id="global-filters"),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dcc.Tabs(id="tabs", value="overview", className="tabs", children=[

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
            html.Div(id="overview-desc", children=[_desc(CAPTIONS["overview_price"])]),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="pareto-chart", figure=build_pareto_scatter(df),
                          config=_GRAPH_CONFIG, style={"height": "620px"}),
            ])], className="chart-card"),

        ]),

        # Agent Stack ──────────────────────────────────────────────────────────
        dcc.Tab(label="Agent Stack", value="recommend",
                className="tab", selected_className="tab--selected", children=[
            _desc(CAPTIONS["recommend"]),
            # Row 1: workflow mode
            html.Div([
                html.Span("WORKFLOW", className="filter-label"),
                dcc.RadioItems(
                    id="recommend-mode",
                    options=[
                        {"label": "API Only",                    "value": "api"},
                        {"label": "Hybrid — Fast local",         "value": "hybrid"},
                        {"label": "Hybrid — Fast + Balanced local", "value": "hybrid2"},
                        {"label": "Local Only",                  "value": "local"},
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
                        # Artificial Analysis renamed xAI to SpaceXAI; the label
                        # follows the data (every chart legend says SpaceXAI),
                        # and canonical_provider still resolves old ?p=xAI links.
                        {"label": "SpaceXAI",   "value": "SpaceXAI"},
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
            _desc(CAPTIONS["landscape_treemap"]),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="treemap-chart", figure=build_treemap(df),
                          config=_GRAPH_CONFIG, style={"height": "600px"}),
            ])], className="chart-card"),
            _desc(CAPTIONS["landscape_leaderboard"]),
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
            _desc(CAPTIONS["rankings_intelligence"]),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="rankings-chart", figure=build_rankings(df, top_n=25),
                          config=_GRAPH_CONFIG, style={"height": "750px"}),
            ])], className="chart-card"),

            # ── Value Leaders ─────────────────────────────────────────────────
            _desc(CAPTIONS["rankings_value"]),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(
                    id="value-leaders-chart",
                    figure=build_value_leaders(df),
                    config=_GRAPH_CONFIG,
                    style={"height": "540px"},
                ),
            ])], className="chart-card"),
        ]),

        # Compare ──────────────────────────────────────────────────────────────
        dcc.Tab(label="Compare", value="compare",
                className="tab", selected_className="tab--selected", children=[
            _desc(CAPTIONS["compare"]),
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
                dcc.Graph(id="radar-chart", figure=build_radar(df, _DIVERSE5, full_df=df),
                          config=_GRAPH_CONFIG, style={"height": "560px"}),
            ])], className="chart-card"),
            html.Div(id="compare-raw-table", className="chart-card",
                     style={"padding": "0"},
                     children=_build_raw_table(df, _DIVERSE5)),
        ]),

        # Budget ───────────────────────────────────────────────────────────────
        dcc.Tab(label="Budget", value="budget",
                className="tab", selected_className="tab--selected", children=[
            _desc(CAPTIONS["budget"]),
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
            _desc(CAPTIONS["table"]),
            html.Div([
                html.Span("SORT BY", className="filter-label"),
                dcc.Dropdown(
                    id="table-sort-col",
                    options=[
                        {"label": "Intelligence", "value": "quality"},
                        {"label": "Value",        "value": "value"},
                        {"label": "Price",        "value": "price"},
                        {"label": "Speed",        "value": "speed"},
                        {"label": "Latency",      "value": "latency"},
                        {"label": "Context",      "value": "context"},
                        {"label": "Model",        "value": "model"},
                        {"label": "Provider",     "value": "provider"},
                    ],
                    value="quality",
                    clearable=False,
                    style={"width": "160px"},
                ),
                dcc.Dropdown(
                    id="table-sort-dir",
                    options=[
                        {"label": "↓ High → Low", "value": "desc"},
                        {"label": "↑ Low → High", "value": "asc"},
                    ],
                    value="desc",
                    clearable=False,
                    style={"width": "140px"},
                ),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div([
                dash_table.DataTable(
                    id="model-table",
                    columns=[
                        {"name": "Model",            "id": "model",      "type": "text"},
                        {"name": "Provider",          "id": "provider",   "type": "text"},
                        {"name": "Intelligence",      "id": "quality",    "type": "numeric",
                         "format": {"specifier": ".1f"}},
                        {"name": "Value (score/$)",   "id": "value",      "type": "numeric",
                         "format": {"specifier": ".2f"}},
                        {"name": "Price ($/M tok)",   "id": "price",      "type": "numeric",
                         "format": {"specifier": ".4f"}},
                        {"name": "Speed (tok/s)",     "id": "speed",      "type": "numeric",
                         "format": {"specifier": ".0f"}},
                        {"name": "Latency (s)",       "id": "latency",    "type": "numeric",
                         "format": {"specifier": ".2f"}},
                        {"name": "Context",           "id": "context"},
                    ],
                    sort_action="none",
                    filter_action="none",
                    page_action="none",
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
                        "cursor": "pointer",
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
                        {"if": {"column_id": "quality"},   "color": "#f2f2f2", "textAlign": "right"},
                        {"if": {"column_id": "value"},     "color": "#34d399", "textAlign": "right"},
                        {"if": {"column_id": "price"},     "textAlign": "right"},
                        {"if": {"column_id": "speed"},     "textAlign": "right"},
                        {"if": {"column_id": "latency"},   "textAlign": "right"},
                        {"if": {"column_id": "context"},
                         "textAlign": "right", "minWidth": "90px", "width": "90px"},
                    ],
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#0a0a0a"},
                        {"if": {"state": "active"},
                         "backgroundColor": "rgba(0,212,255,0.06)",
                         "border": "1px solid rgba(0,212,255,0.2)"},
                        *[
                            {"if": {"filter_query": f'{{provider}} = "{p}"',
                                    "column_id": "provider"},
                             "color": c}
                            for p, c in PROVIDER_COLORS.items()
                        ],
                    ],
                    style_as_list_view=True,
                ),
            ], className="chart-card", style={"padding": "0"}),
        ]),

        # Run Local ────────────────────────────────────────────────────────────
        dcc.Tab(label="Run Local", value="local",
                className="tab", selected_className="tab--selected", children=[
            _desc(CAPTIONS["local"]),
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
                        # Must match the tag vocabulary in data/local_models.py:
                        # "multilingual" was never emitted, so selecting it
                        # matched zero models and blanked both Run Local charts.
                        {"label": "Code",      "value": "code"},
                        {"label": "Reasoning", "value": "reasoning"},
                        {"label": "Vision",    "value": "vision"},
                        {"label": "Audio",     "value": "audio"},
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
                    config=_GRAPH_CONFIG_FIXED,
                ),
            ])], className="chart-card chart-card--uncapped"),
        ]),

        # Image Gen ────────────────────────────────────────────────────────────
        dcc.Tab(label="Image Gen", value="image",
                className="tab", selected_className="tab--selected", children=[
            _desc(CAPTIONS["image"]),
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
                    # Built from the tags the pipeline actually emits — a
                    # hardcoded list drifts from the data and silently matches
                    # nothing (see data/image_models.get_image_tags).
                    options=get_image_tags(),
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
        ]),

        # Video Gen ────────────────────────────────────────────────────────────
        dcc.Tab(label="Video Gen", value="video",
                className="tab", selected_className="tab--selected", children=[
            _desc(CAPTIONS["video"]),
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
                    options=get_video_tags(),
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

# Must match the dcc.Tab values in the layout exactly. It previously listed
# "performance"/"imagegen"/"videogen" — none of which exist — while omitting the
# real "image" and "video", so a shared ?tab=image link silently reset to
# Overview.
_VALID_TABS = {
    "overview", "recommend", "landscape", "rankings", "compare",
    "budget", "table", "local", "image", "video",
}

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
        return "overview", [], 0
    params    = parse_qs(search.lstrip("?"))
    tab       = params.get("tab", ["overview"])[0]
    if tab not in _VALID_TABS:
        tab = "overview"
    raw_p     = params.get("p",   [""])[0]
    # Canonicalise: a ?p=xAI link predates AA's rename, and an unresolved name
    # matches no dropdown option — which reads as "no filter" and shows
    # everything rather than the one provider the link named.
    providers = [canonical_provider(x) for x in raw_p.split(",") if x] if raw_p else []
    try:
        # float, not int: the presets emit 42.1 / 52.7, so int() raised, the
        # error was swallowed, and the filter silently became 0 — a shared
        # "Top 25%" link opened as no filter at all.
        quality = float(params.get("q", [0])[0])
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

# ── Tab switch: force Plotly to re-measure charts that were hidden ────────────
# Dash renders every tab's content at startup; non-active tabs live inside a
# parent with display:none, so Plotly initialises their graphs with zero
# dimensions. Dispatching window.resize isn't enough — Plotly's ResizeObserver
# only fires when the element actually has a visible box, and the dispatch
# sometimes races the tab's visibility flip. Instead, after the tab switch we
# directly call Plotly.Plots.resize() on every .js-plotly-plot element that
# is now visible, at three increasing delays to catch both synchronous and
# late layout passes (font load, loader spinner teardown, etc.).
clientside_callback(
    """
    function(tab) {
        function resizeAll() {
            if (!window.Plotly) return;
            var plots = document.querySelectorAll('.js-plotly-plot');
            plots.forEach(function(el) {
                if (el.offsetParent !== null) {
                    try { window.Plotly.Plots.resize(el); } catch (e) {}
                }
            });
        }
        setTimeout(resizeAll,  60);
        setTimeout(resizeAll, 200);
        setTimeout(resizeAll, 500);
        return null;
    }
    """,
    Output("resize-sink", "data"),
    Input("tabs", "value"),
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
    """Quality presets touch quality. Only "Reset filters" touches the rest.

    All three used to return `[]` into filter-provider. The buttons are labelled
    purely in quality terms and sit in the same bar as the PROVIDER dropdown, so
    PROVIDER=Anthropic + "Top 10%" quietly became "top 10% of everything" — a
    different question from the one asked. They also disagreed with each other:
    "All" cleared SEARCH, the other two did not.
    """
    trigger = ctx.triggered_id
    if trigger == "preset-all":
        return 0, [], ""                      # relabelled "Reset filters"
    if trigger == "preset-strong":
        return _P75, no_update, no_update
    if trigger == "preset-elite":
        return _P90, no_update, no_update
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
    hw_style   = {"paddingTop": "0"} if mode in ("hybrid", "hybrid2", "local") \
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
    if mode in ("hybrid", "hybrid2", "local"):
        gpu_meta       = GPU_BY_NAME.get(gpu_preset or "", {})
        # 0 is a figure the user typed, not a blank box — see coerce_number.
        gpu_count      = int(_coerce_number(num_gpus, DEFAULT_GPU_COUNT, minimum=1))
        vram_gb        = _coerce_number(vram_per_gpu, DEFAULT_VRAM_GB, minimum=0.0) * gpu_count
        bandwidth_gbps = _coerce_number(gpu_meta.get("bandwidth_gbps"),
                                        DEFAULT_BANDWIDTH_GBPS, minimum=0.0)
        hw_type        = gpu_meta.get("hw_type", "nvidia")
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
        desc = _desc(CAPTIONS["overview_speed"])
        # `df` is the unfiltered catalogue: the median crosshairs that decide
        # "Fast · Smart" and the frontier are market-wide claims, so they must
        # not move when the user narrows the frame.
        return build_quadrant(filtered, full_df=df), desc
    desc = _desc(CAPTIONS["overview_price_dyn"])
    return build_pareto_scatter(filtered, full_df=df), desc


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

    capped = _cap_compare_selection(selected_models, filtered, triggered)

    raw_table = _build_raw_table(filtered, capped)
    return build_radar(filtered, capped, full_df=df), options, capped, raw_table


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
    # 0 tokens is a real query; a negative volume would invert the "cheapest" sort.
    tokens   = _coerce_number(monthly_tokens_m, default=1.0, minimum=0.0)
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
    # Shared defaults: this used to assume 8GB while the public site assumed 32,
    # so a cleared VRAM box answered "which models fit?" two different ways.
    gpu_count      = int(_coerce_number(num_gpus, DEFAULT_GPU_COUNT, minimum=1))
    vram_gb        = _coerce_number(vram_per_gpu, DEFAULT_VRAM_GB, minimum=0.0) * gpu_count
    bandwidth_gbps = _coerce_number((hw_meta or {}).get("bandwidth_gbps"),
                                    DEFAULT_BANDWIDTH_GBPS, minimum=0.0)
    hw_type        = (hw_meta or {}).get("hw_type", "nvidia")
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
    # The Overview tab renders BOTH the price and speed views into `pareto-chart`
    # (see update_overview), so there is no `quadrant-chart` component. Listening
    # for one raised a runtime ReferenceError that suppress_callback_exceptions
    # hid, killing this whole callback — clicking a bubble did nothing.
    Input("pareto-chart", "clickData"),
    Input("detail-close", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_detail_panel(pareto_click, _close):
    trigger = ctx.triggered_id

    if trigger == "detail-close":
        return "detail-panel", [], None

    click = pareto_click
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
    Input("filter-provider",  "value"),
    Input("filter-quality",   "value"),
    Input("model-search",     "value"),
    Input("table-sort-col",   "value"),
    Input("table-sort-dir",   "value"),
)
def update_table(providers, min_quality, search, sort_col, sort_dir):
    filtered = _apply_filters(providers, min_quality, search or "").copy()
    filtered["value"] = filtered.apply(
        lambda r: r["quality"] / r["price"] if r["price"] > 0 else None, axis=1
    )
    col = sort_col or "quality"
    asc = (sort_dir or "desc") == "asc"
    if col == "context":
        filtered["_ctx_k"] = filtered["context"].map(_ctx_to_k)
        filtered = filtered.sort_values("_ctx_k", ascending=asc, na_position="last")
    else:
        filtered = filtered.sort_values(col, ascending=asc, na_position="last")
    cols = ["model", "provider", "quality", "value", "price", "speed", "latency", "context"]
    return filtered[cols].to_dict("records")


# ── Export CSV ────────────────────────────────────────────────────────────────
@callback(
    Output("download-csv",   "data"),
    Input("btn-export",      "n_clicks"),
    State("tabs",            "value"),
    State("filter-provider", "value"),
    State("filter-quality",  "value"),
    State("model-search",    "value"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, tab, providers, min_quality, search):
    """Export what is on screen, not always the hosted-LLM table.

    The global filter bar sits outside dcc.Tabs and four tabs consume none of
    it, but ↓CSV always exported the LLM catalogue through those filters. From
    Image Gen, `export_csv(['Anthropic'], 50, '')` returned the LLM header and
    seven text models — a file describing a different dataset from the one the
    user was looking at.
    """
    if not n_clicks:
        return no_update
    frame, name = _export_frame_for_tab(tab, providers, min_quality, search)
    return dcc.send_data_frame(frame.to_csv, name, index=False)


# ── Hide the global filter bar where it does nothing ──────────────────────────
@callback(
    Output("global-filters", "style"),
    Input("tabs", "value"),
)
def toggle_global_filters(tab):
    """Agent Stack, Run Local, Image Gen and Video Gen read none of PROVIDER /
    MIN SCORE / SEARCH. Leaving the bar visible and live on those tabs meant it
    displayed "Anthropic / >= 45" over a chart plotting 72 models from a dozen
    other providers."""
    return {"display": "none"} if tab in _TABS_WITHOUT_GLOBAL_FILTERS else {}


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
    Input("image-provider-filter", "value"),
    Input("image-tag-filter",      "value"),
    prevent_initial_call=True,
)
def update_image_charts(providers, tags):
    full_img_df = get_image_df()
    img_df = full_img_df
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
    # Facet metrics are chosen against the whole arena, so a provider filter
    # cannot swap a facet onto a retired 2025 ELO column.
    return build_image_faceted(img_df, full_df=full_img_df)


# ── Video Gen tab ─────────────────────────────────────────────────────────────
@callback(
    Output("video-rankings-chart", "figure"),
    Output("video-scatter-chart",  "figure"),
    Input("video-provider-filter", "value"),
    Input("video-tag-filter",      "value"),
    prevent_initial_call=True,
)
def update_video_charts(providers, tags):
    full_vdf = get_video_df()
    vdf = full_vdf
    if providers:
        vdf = vdf[vdf["provider"].isin(providers)]
    if tags:
        for tag in tags:
            if tag == "open-weights":
                vdf = vdf[vdf["open_weights"] == True]
            else:
                vdf = vdf[vdf["tags"].apply(lambda t: tag in t)]
    paid = vdf[vdf["price_per_sec"] > 0] if not vdf.empty else vdf
    full_paid = full_vdf[full_vdf["price_per_sec"] > 0] if not full_vdf.empty else full_vdf
    return (
        build_video_rankings(vdf),
        build_video_scatter(paid if not paid.empty else vdf, full_df=full_paid),
    )



# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(debug=debug, port=8050)
