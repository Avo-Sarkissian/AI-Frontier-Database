"""
AI Frontier — Main Dash Application
Interactive dashboard comparing 100+ LLMs on cost, speed, and quality.

Architecture: all dcc.Graph components live inside dcc.Tab.children so their
IDs are always in the DOM regardless of active tab. This prevents Dash 4's
"nonexistent object" callback errors.
"""
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs

import dash
from dash import ctx, dcc, html, dash_table, Input, Output, State, callback, clientside_callback, no_update
import pandas as pd

from data.ingest import get_models
from data.scraper import start_background_scraper
from data.image_scraper import start_background_image_scraper
from data.local_scraper import start_background_local_scraper
from data.video_scraper import start_background_video_scraper
from data.local_models import (
    get_local_df, get_gpu_options, GPU_BY_NAME, QUANT_LEVELS, quant_options,
    DEFAULT_VRAM_GB, DEFAULT_GPU_COUNT, DEFAULT_BANDWIDTH_GBPS,
    DEFAULT_CONTEXT_TOKENS, DEFAULT_SPEED_MODE, DEFAULT_FP16_TFLOPS,
    context_options, speed_mode_options, tflops_for_gpu,
    effective_bandwidth, memory_note,
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
from data.video_models                      import (
    get_video_df, get_video_providers, get_video_tags, filter_video_df,
    VIDEO_MODES, DEFAULT_MODE,
)
from components.charts.bump_chart          import build_value_leaders
from data.workload_mix                      import (
    apply_mix, resolve_mix, mix_options as _mix_options, fmt_price, fmt_floor,
    floor_label as _floor_label, table_price_header as _table_price_header,
    DEFAULT_MIX_KEY, CUSTOM_KEY, mix_summary as _mix_summary,
    cache_label as _cache_label, mark_price as _mark_price,
    filled_flags as _filled_flags, FILL_MARK, FILL_SHORT,
)
from data.task_cost                         import (
    median_token_burn, detail_rows as _task_detail_rows, detail_note as _task_detail_note,
)

# ── Data ─────────────────────────────────────────────────────────────────────
df = get_models()
# load_history() used to run here and on every cache change — 90 CSVs,
# 23,742 rows, ~7 MB — with no consumer anywhere in the app. The history
# directory is still written by the scraper; nothing reads it yet.

# ── Background scrapers — started by the SERVER, never by an import ──────────
# `import app` used to start four live AA scrapes as a side effect. Anything
# that imported the module for a layout check or a one-off call — two review
# verifiers, a REPL, a build script — rewrote data/raw/ and froze a history
# snapshot, leaving uncommitted scrape output behind. Importing is now inert;
# the scrapers start when something is actually going to serve the app:
#
#   * `python app.py`          — the __main__ block below;
#   * gunicorn (`app:server`)  — detected at import, since a gunicorn worker
#                                imports this module precisely to serve it;
#   * any other WSGI server    — the first request it handles;
#   * AI_FRONTIER_SCRAPE=1     — forces a start at import; =0 disables them.
#
# Werkzeug's debug reloader runs two processes — the watchdog
# (WERKZEUG_RUN_MAIN unset) and the worker child (WERKZEUG_RUN_MAIN=true). Only
# the child serves, so only the child starts them; both used to race to write
# the cache on every file-save reload.
#
# Never under pytest: a short test process killed the daemon threads
# mid-flight and wrote ok=false for every scraper into
# data/raw/scrape_status.json, so the live freshness badge showed a warning
# because someone had run the tests.
_SCRAPE_FLAG      = os.environ.get("AI_FRONTIER_SCRAPE", "").strip().lower()
_scrapers_lock    = threading.Lock()
_scrapers_started = False


def _scrapers_allowed() -> bool:
    if _SCRAPE_FLAG in ("0", "false", "no", "off"):
        return False
    return not ("PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules)


def start_scrapers() -> bool:
    """Start the four hourly scrapers once per process. True if this call did."""
    global _scrapers_started
    if not _scrapers_allowed():
        return False
    with _scrapers_lock:
        if _scrapers_started:
            return False
        _scrapers_started = True
    start_background_scraper(interval_s=3600)
    start_background_image_scraper(interval_s=3600)
    start_background_local_scraper(interval_s=3600)
    start_background_video_scraper(interval_s=3600)
    return True


_CACHE_PATH  = Path(__file__).parent / "data" / "raw" / "aa_models.csv"
_data_lock   = threading.Lock()
_cache_mtime = _CACHE_PATH.stat().st_mtime if _CACHE_PATH.exists() else 0.0


def _cache_ts() -> str:
    try:
        return datetime.fromtimestamp(_CACHE_PATH.stat().st_mtime).strftime("%b %d  %H:%M")
    except Exception:
        return "—"


def _reload_if_stale():
    """Re-read df if the cache file has changed on disk."""
    global df, _cache_mtime
    try:
        mtime = _CACHE_PATH.stat().st_mtime
        if mtime != _cache_mtime:
            with _data_lock:
                # Double-check inside the lock to avoid redundant reloads
                # if two callbacks both saw the stale mtime simultaneously.
                if mtime != _cache_mtime:
                    df           = get_models()
                    _cache_mtime = mtime
    except Exception:
        pass

from captions import CAPTIONS
from static_helpers import (
    index_label,
    apply_filters,
    task_axis_frames,
    coerce_number as _coerce_number,
    cap_compare_selection as _cap_compare_selection,
    COMPARE_MAX,
    csv_safe as _csv_safe,
    quality_options as _quality_options,
    export_frame_for_tab as _export_frame_for_tab_shared,
    ranking_frame,
    effort_options as _effort_options,
    weights_options as _weights_options,
    TABS_WITHOUT_GLOBAL_FILTERS as _TABS_WITHOUT_GLOBAL_FILTERS,
    compute_diverse5 as _compute_diverse5,
    ctx_to_k as _ctx_to_k,
    quality_label as _quality_label,
    provider_options as _provider_options,
    model_options as _model_options,
)
from data.open_weights import valid_weights as _valid_weights, weights_label as _weights_label
from static_helpers import local_frame as _local_frame, with_local_rent_cost as _with_local_rent_cost
from data import serving_cost as _serving_cost
from data import gpu_tco as _gpu_tco
from components.charts.econ_serve import build_econ_serve
from components.charts.econ_compute import build_econ_compute


_DIVERSE5    = _compute_diverse5(df)

# Read once for the layout's first paint. Both video figures and the callback
# use the same frame, so the pre-rendered chart and the first interaction agree
# on the axis anchor and the frontier's reference set.
_VIDEO_DEFAULT_DF = get_video_df(DEFAULT_MODE)


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


@server.before_request
def _start_scrapers_on_first_request():
    # A flag read, then nothing, on every request after the first.
    if not _scrapers_started:
        start_scrapers()


if _SCRAPE_FLAG in ("1", "true", "yes", "on") or "gunicorn" in sys.modules:
    start_scrapers()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _stat(value: str, label: str, accent: bool = False) -> html.Div:
    return html.Div([
        html.Div(value, className="stat-value",
                 style={"color": "#00d4ff" if accent else "#f2f2f2"}),
        html.Div(label, className="stat-label"),
    ], className="stat")


def _mix(key=None, ratio=None, cache=None):
    """The MIX control's three values as a workload_mix.Mix (never raises)."""
    return resolve_mix([key, ratio, cache])


def _base(mix=None) -> pd.DataFrame:
    """The catalogue priced at `mix`; the default is `df` itself. Mirrors
    static_api._base: the mix is applied before any filter, and this unfiltered
    frame is what the builders get as `full_df`."""
    return apply_mix(df, resolve_mix(mix))


def _apply_filters(providers, min_quality, search: str = "",
                   effort: str | None = None, mix=None, weights=None) -> pd.DataFrame:
    return apply_filters(_base(mix), providers, min_quality, search, effort, weights)


def _export_frame_for_tab(tab, providers, min_quality, search,
                          effort=None, local_args=None, mix=None, weights=None,
                          econ_args=None):
    return _export_frame_for_tab_shared(tab, df, providers, min_quality, search,
                                        effort=effort, local_args=local_args, mix=mix,
                                        weights=weights, econ_args=econ_args)


def _mix_option_items() -> list[dict]:
    """dcc.Dropdown options for MIX, each with its source in the hover title."""
    return [{"label": o["label"], "value": o["value"], "title": o["title"]}
            for o in _mix_options()]


def _desc(text: str) -> html.Div:
    return html.Div(text, className="chart-caption")


# Economics › Serve control options and the Run Local $/GPU-HR tooltip: one
# source (data/serving_cost.serve_options), shared with the static site.
_ECON_OPTS = _serving_cost.serve_options()
_LOCAL_RATE_TITLE = (
    f"Your cost per GPU-hour. Picking {_serving_cost.priced_cards_phrase()} "
    "fills its large-hyperscaler-volume owning (Own) or renting (Rent) rate; "
    "other cards have no published rate — type your own. The hover then prices "
    "this tab's max-throughput figure per 1M output tokens at full use (decode "
    "only, uncalibrated).")


def _econ_serve_figure(gpu=None, basis=None, util=None):
    """Economics › Serve, the same call static_api.update_econ_serve makes."""
    frame, meta = _serving_cost.serving_data(gpu, basis, util, hosted=df)
    return build_econ_serve(frame, meta, axis_range=_serving_cost.axis_range(df))


# Economics › Compute: options, placeholders and tooltips from one source
# (data/gpu_tco.compute_options), shared with the static site.
_COMPUTE_OPTS = _gpu_tco.compute_options()
_COMPUTE_BOXES = (  # (id suffix, label, step, width px)
    ("life", "LIFE YR", 1, 56), ("rate", "RATE %", 0.1, 64), ("capex", "CAPEX $K", 1, 72),
    ("kw", "kW", 0.01, 64), ("pue", "PUE", 0.01, 56), ("kwh", "$/kWh", 0.001, 68),
    ("colo", "COLO $/kW-MO", 1, 64),
)


def _compute_placeholders(gpu=None, preset=None) -> dict:
    """{box suffix: (placeholder, title)} for the GPU and preset on screen."""
    o = _COMPUTE_OPTS
    g = o["defaults"].get(_gpu_tco.coerce_gpu(gpu), {})
    p = o["preset_defaults"].get(_gpu_tco.coerce_preset(preset), {})
    f = o["facility"]
    show = lambda v: "—" if v is None else f"{v:g}"
    return {
        "life": (show(p.get("life_years")), p.get("life_title", "")),
        "rate": (show(p.get("rate_pct")), p.get("rate_title", "")),
        "capex": (show(g.get("capex_k")), g.get("capex_title", "")),
        "kw": (show(g.get("it_kw")), g.get("kw_title", "")),
        "pue": (show(f["pue"]["value"]), f["pue"]["title"]),
        "kwh": (show(f["usd_per_kwh"]["value"]), f["usd_per_kwh"]["title"]),
        "colo": (show(f["colo_usd_kw_month"]["value"]), f["colo_usd_kw_month"]["title"]),
    }


def _compute_args(gpu=None, preset=None, util=None, life=None, rate=None, capex_k=None,
                  kw=None, pue=None, kwh=None, colo=None) -> list:
    """update_econ_compute's argument list; the CAPEX box is in $K."""
    capex = None if capex_k in (None, "") else float(capex_k) * 1000.0
    return [gpu, preset, util, life, rate, capex, kw, pue, kwh, colo]


def _econ_compute_figures(*args):
    """The three Compute figures, the same call static_api.update_econ_compute makes."""
    figs = build_econ_compute(_gpu_tco.compute(*args))
    return figs["econ_tco"], figs["econ_ladder"], figs["econ_rental_series"]


def _build_raw_table(dataframe: pd.DataFrame, selected_models: list[str],
                     mix=None) -> html.Div:
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
        html.Th(_table_price_header(mix), style=header_style),
        html.Th("Speed (tok/s)",   style=header_style),
        html.Th("Latency (TTFT)",  style=header_style),
        html.Th("Context",         style=header_style),
    ])

    table_rows = []
    for _, r in rows.sort_values("quality", ascending=False, kind="mergesort").iterrows():
        pcolor    = PROVIDER_COLORS.get(r["provider"], DEFAULT_COLOR)
        price_str = fmt_price(r["price"]) if pd.notna(r["price"]) and r["price"] > 0 else "—"
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
    # The FIRST PAINT reads this, so it must carry every field the model needs.
    # A literal figure here is a second copy of a constant that lives in
    # data/local_models.py — hence the names, not the numbers.
    dcc.Store(id="local-hw-meta",
              data={"bandwidth_gbps": DEFAULT_BANDWIDTH_GBPS, "hw_type": "nvidia",
                    "fp16_tflops": DEFAULT_FP16_TFLOPS}),
    dcc.Store(id="detail-model-name", data=None),
    # The MIX value before the latest change, so Custom can start from the
    # preset that was showing (docs/app.js keeps it on the element).
    dcc.Store(id="mix-prev", data=None),
    dcc.Store(id="refresh-sink"),
    dcc.Store(id="share-sink"),
    dcc.Store(id="resize-sink"),
    dcc.Store(id="econ-click-sink"),
    dcc.Store(id="table-data-store", data=(lambda _df: _df.assign(
        value=_df.apply(lambda r: r["quality"] / r["price"] if r["price"] > 0 else None, axis=1)
    ).sort_values("quality", ascending=False, kind="mergesort")[
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
                     children=fmt_floor(df["price"].min()),
                     className="stat-value", style={"color": "#00d4ff"}),
            html.Div(_floor_label(None), id="stat-floor-label", className="stat-label"),
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
        html.Span("EFFORT", className="filter-label"),
        dcc.Dropdown(
            id="filter-effort",
            options=_effort_options(),
            value="",
            clearable=False,
            style={"width": "150px"},
        ),
        html.Div(className="filter-sep"),
        # Open or closed weights (data/open_weights.py): AA's flag, else a
        # name match to its open-weights list. Hollow marks = open.
        html.Span("WEIGHTS", className="filter-label",
                  title="Open or closed weights, from Artificial Analysis's flag"),
        dcc.Dropdown(
            id="filter-weights",
            options=_weights_options(),
            value="",
            clearable=False,
            style={"width": "110px"},
        ),
        html.Div(className="filter-sep"),
        # Workload MIX (data/workload_mix): the input:output ratio and cache-hit
        # rate every hosted price is computed at. Each option's title carries
        # its source and date. Custom reveals two boxes.
        html.Span("MIX", className="filter-label",
                  title="Workload mix: input:output ratio and cache-hit rate every price is computed at"),
        dcc.Dropdown(
            id="filter-mix",
            options=_mix_option_items(),
            value=DEFAULT_MIX_KEY,
            clearable=False,
            style={"width": "190px"},
        ),
        html.Span([
            dcc.Input(id="mix-ratio", type="number", min=0, max=1000, step="any",
                      debounce=True, className="search-input", style={"width": "64px"}),
            html.Span(":1 in:out", className="filter-note"),
            dcc.Input(id="mix-cache", type="number", min=0, max=100, step=1,
                      debounce=True, className="search-input", style={"width": "56px"}),
            html.Span("% cached", className="filter-note"),
        ], id="mix-custom", style={"display": "none"}),
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
                        {"label": "Cost / task", "value": "task"},
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
                html.Span(id="recommend-vram-note", className="filter-note"),
                html.Div(className="filter-sep"),
                html.Span("GPUs", className="filter-label", title="Extra GPUs pool VRAM so a bigger model fits, and pooled VRAM raises how many concurrent sessions the KV cache allows. They do not increase single-stream tokens/sec: under a layer split the cards run in sequence, so throughput is set by one card's memory bandwidth. A tensor-parallel vLLM deployment does better and these figures do not model it."),
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
                    options=quant_options(),
                    value="Q4", clearable=False,
                    style={"width": "132px"},
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
                dcc.Graph(id="rankings-chart",
                          figure=build_rankings(ranking_frame(df), top_n=25),
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
                        {"name": _table_price_header(None), "id": "price", "type": "numeric",
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
                        # No cursor:pointer — sort_action is "none" and the
                        # headers are inert, so the pointer invited a click that
                        # does nothing. Sorting is done with the two dropdowns
                        # above the table.
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
                html.Span(id="local-vram-note", className="filter-note"),
                html.Div(className="filter-sep"),
                html.Span("GPUs", className="filter-label", title="Extra GPUs pool VRAM so a bigger model fits, and pooled VRAM raises how many concurrent sessions the KV cache allows. They do not increase single-stream tokens/sec: under a layer split the cards run in sequence, so throughput is set by one card's memory bandwidth. A tensor-parallel vLLM deployment does better and these figures do not model it."),
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
                    options=quant_options(),
                    value="Q4", clearable=False,
                    style={"width": "132px"},
                ),
                html.Div(className="filter-sep"),
                html.Span("CONTEXT", className="filter-label", title="VRAM and speed both move with context: the KV cache is read in full every token and is not affected by quantising the weights. Capped per model at its own maximum."),
                dcc.Dropdown(
                    id="local-context",
                    options=context_options(),
                    value=DEFAULT_CONTEXT_TOKENS, clearable=False,
                    style={"width": "84px"},
                ),
                html.Div(className="filter-sep"),
                html.Span("SPEED", className="filter-label", title="Single stream is one conversation at a time. Max throughput is the total across as many concurrent sessions as the KV cache fits in VRAM while each one still decodes at 10 tok/s or better — MLPerf Inference's Llama-3.1-8B Server floor. The two are 2-4x apart, so the chart shows one at a time."),
                dcc.Dropdown(
                    id="local-speed-mode",
                    options=speed_mode_options(),
                    value=DEFAULT_SPEED_MODE, clearable=False,
                    style={"width": "156px"},
                ),
                html.Div(className="filter-sep"),
                html.Span("$/GPU-HR", className="filter-label", title=_LOCAL_RATE_TITLE),
                dcc.Input(
                    id="local-gpu-rate", type="number", min=0, step=0.01, debounce=True,
                    value=None, placeholder="—",
                    style={
                        "background": "var(--bg-card)", "border": "1px solid var(--border)",
                        "borderRadius": "4px", "color": "#f2f2f2",
                        "fontFamily": "Inter, sans-serif", "fontSize": "13px",
                        "padding": "6px 10px", "width": "72px", "outline": "none",
                    },
                ),
                dcc.RadioItems(
                    id="local-rate-basis",
                    options=[{"label": o["label"], "value": o["value"]}
                             for o in _serving_cost.serve_options()["bases"]],
                    value=_serving_cost.DEFAULT_LOCAL_RATE_BASIS, inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "12px", "cursor": "pointer",
                                "fontSize": "12px", "color": "#aaa"},
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
                        get_local_df(quant="Q4", vram_gb=DEFAULT_VRAM_GB,
                                     bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS,
                                     hw_type="nvidia",
                                     ctx_tokens=DEFAULT_CONTEXT_TOKENS,
                                     fp16_tflops=DEFAULT_FP16_TFLOPS),
                        vram_gb=DEFAULT_VRAM_GB, quant="Q4",
                        ctx_tokens=DEFAULT_CONTEXT_TOKENS,
                    ),
                    config=_GRAPH_CONFIG, style={"height": "640px"},
                ),
            ])], className="chart-card"),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(
                    id="local-compat-chart",
                    figure=build_local_compat(
                        get_local_df(quant="Q4", vram_gb=DEFAULT_VRAM_GB,
                                     bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS,
                                     hw_type="nvidia",
                                     ctx_tokens=DEFAULT_CONTEXT_TOKENS,
                                     fp16_tflops=DEFAULT_FP16_TFLOPS),
                        quant="Q4", vram_gb=DEFAULT_VRAM_GB,
                        ctx_tokens=DEFAULT_CONTEXT_TOKENS,
                    ),
                    config=_GRAPH_CONFIG_FIXED,
                ),
            ])], className="chart-card chart-card--uncapped"),
        ]),

        # Economics ────────────────────────────────────────────────────────────
        dcc.Tab(label="Economics", value="economics",
                className="tab", selected_className="tab--selected", children=[
            html.Div([
                html.Span("VIEW", className="filter-label"),
                dcc.RadioItems(
                    id="econ-view",
                    options=[{"label": "Serve", "value": "serve"},
                             {"label": "Compute", "value": "compute"},
                             {"label": "Labs", "value": "labs"}],
                    value="serve", inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "20px", "cursor": "pointer",
                                "fontSize": "12px", "color": "#aaa"},
                ),
                html.Span(id="econ-shared-controls", children=[
                    html.Div(className="filter-sep"),
                    html.Span("USE", className="filter-label",
                              title=_ECON_OPTS["util_title"]),
                    html.Div(dcc.Slider(
                        id="econ-util", min=_ECON_OPTS["util_min"],
                        max=_ECON_OPTS["util_max"], step=_ECON_OPTS["util_step"],
                        value=_ECON_OPTS["default_util_pct"], marks=None,
                        updatemode="mouseup",
                    ), style={"width": "170px"}),
                    html.Span(id="econ-util-value", children=f"{_ECON_OPTS['default_util_pct']}%",
                              style={"fontSize": "13px", "color": "#f2f2f2"}),
                ], style={"display": "inline-flex", "alignItems": "center", "gap": "8px"}),
                html.Span(id="econ-serve-controls", children=[
                    html.Div(className="filter-sep"),
                    html.Span("GPU", className="filter-label"),
                    dcc.Dropdown(
                        id="econ-serve-gpu",
                        options=[{"label": o["label"], "value": o["value"], "title": o["title"]}
                                 for o in _ECON_OPTS["gpus"]],
                        value=_ECON_OPTS["default_gpu"], clearable=False,
                        style={"minWidth": "260px"},
                    ),
                    html.Div(className="filter-sep"),
                    html.Span("$/GPU-HR", className="filter-label",
                              title=_ECON_OPTS["basis_title"]),
                    dcc.RadioItems(
                        id="econ-serve-basis",
                        options=[{"label": html.Span(o["label"], title=o["title"]),
                                  "value": o["value"]}
                                 for o in _ECON_OPTS["bases"]],
                        value=_ECON_OPTS["default_basis"], inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "12px", "cursor": "pointer",
                                    "fontSize": "12px", "color": "#aaa"},
                    ),
                ], style={"display": "inline-flex", "alignItems": "center", "gap": "8px"}),
                html.Span(id="econ-compute-controls", children=[
                    html.Div(className="filter-sep"),
                    html.Span("GPU", className="filter-label"),
                    dcc.Dropdown(
                        id="econ-compute-gpu",
                        options=[{"label": o["label"], "value": o["value"], "title": o["title"]}
                                 for o in _COMPUTE_OPTS["gpus"]],
                        value=_COMPUTE_OPTS["default_gpu"], clearable=False,
                        style={"minWidth": "140px"},
                    ),
                    html.Div(className="filter-sep"),
                    html.Span("PRESET", className="filter-label"),
                    dcc.Dropdown(
                        id="econ-compute-preset",
                        options=[{"label": o["label"], "value": o["value"], "title": o["title"]}
                                 for o in _COMPUTE_OPTS["presets"]],
                        value=_COMPUTE_OPTS["default_preset"], clearable=False,
                        style={"minWidth": "230px"},
                    ),
                    html.Div(className="filter-sep"),
                    *[c for key, label, step, width in _COMPUTE_BOXES for c in (
                        html.Span(label, id=f"econ-compute-{key}-label", className="filter-label",
                                  title=_compute_placeholders()[key][1]),
                        # dcc.Input takes no title: the tooltip rides a wrapper.
                        html.Span(dcc.Input(id=f"econ-compute-{key}", type="number", min=0,
                                            step=step, debounce=True,
                                            placeholder=_compute_placeholders()[key][0],
                                            style={"width": f"{width}px"}),
                                  id=f"econ-compute-{key}-wrap",
                                  title=_compute_placeholders()[key][1]),
                    )],
                ], style={"display": "none"}),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div(id="econ-view-serve", children=[
                _desc(CAPTIONS["economics_serve"]),
                html.Div([dcc.Loading(**_LOADING, children=[
                    dcc.Graph(id="econ-serve-chart", figure=_econ_serve_figure(),
                              config=_GRAPH_CONFIG, style={"height": "640px"}),
                ])], className="chart-card"),
            ]),
            html.Div(id="econ-view-compute", style={"display": "none"}, children=[
                _desc(CAPTIONS["economics_compute"]),
                *[html.Div([dcc.Loading(**_LOADING, children=[
                    dcc.Graph(id=gid, config=_GRAPH_CONFIG, style={"minHeight": "460px"}),
                ])], className="chart-card")
                  for gid in ("econ-tco-chart", "econ-ladder-chart", "econ-rental-series-chart")],
            ]),
            html.Div(id="econ-view-labs", style={"display": "none"}, children=[
                _desc(CAPTIONS["economics_open_gap"]),
                *[html.Div([dcc.Loading(**_LOADING, children=[
                    dcc.Graph(id=f"{gid}-chart", config=_GRAPH_CONFIG,
                              style={"minHeight": "460px"}),
                ])], className="chart-card")
                  for gid in ("econ-open-gap-history", "econ-open-gap-lag",
                              "econ-open-gap-price")],
            ]),
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
                html.Span("MODE", className="filter-label"),
                dcc.Dropdown(
                    id="video-mode-filter",
                    options=VIDEO_MODES,
                    value=DEFAULT_MODE,
                    clearable=False,
                    style={"minWidth": "180px"},
                ),
                html.Div(className="filter-sep"),
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
                          # full_df matches what the callback passes, so the
                          # axis does not shift on the first interaction.
                          figure=build_video_rankings(_VIDEO_DEFAULT_DF,
                                                      full_df=_VIDEO_DEFAULT_DF),
                          config=_GRAPH_CONFIG, style={"minHeight": "400px"}),
            ])], className="chart-card"),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="video-scatter-chart",
                          figure=build_video_scatter(_VIDEO_DEFAULT_DF,
                                                     full_df=_VIDEO_DEFAULT_DF),
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
    "budget", "table", "local", "economics", "image", "video",
}

# ── URL: restore state on load ────────────────────────────────────────────────
@callback(
    Output("tabs",            "value"),
    Output("filter-provider", "value"),
    Output("filter-quality",  "value"),
    Output("filter-effort",   "value"),
    Input("url", "search"),
    prevent_initial_call=False,
)
def init_from_url(search: str):
    if not search:
        return "overview", [], 0, ""
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
    # Unknown efforts fall back to "all variants" rather than raising: this is a
    # URL, so the value is user input, and select_effort ignores anything it does
    # not recognise anyway.
    effort = params.get("e", [""])[0]
    if effort not in {o["value"] for o in _effort_options()}:
        effort = ""
    return tab, providers, quality, effort


# ── URL: the MIX setting (?m=agentic, ?m=custom&mr=20&mh=50) ─────────────────
# A separate callback, deliberately: init_from_url's 4-tuple is destructured by
# existing tests and shared-link code, and widening it would break them all.
@callback(
    Output("filter-mix", "value"),
    Output("mix-ratio",  "value"),
    Output("mix-cache",  "value"),
    Input("url", "search"),
    prevent_initial_call=False,
)
def init_mix_from_url(search: str):
    """A URL is user input: an unknown key falls back to the default mix, and
    the custom numbers are clamped by workload_mix.resolve_mix when used."""
    if not search:
        return DEFAULT_MIX_KEY, None, None
    params = parse_qs(search.lstrip("?"))
    key = params.get("m", [DEFAULT_MIX_KEY])[0]
    if key not in {o["value"] for o in _mix_options()}:
        return DEFAULT_MIX_KEY, None, None
    if key != CUSTOM_KEY:
        return key, None, None
    m = resolve_mix([CUSTOM_KEY, params.get("mr", [None])[0], params.get("mh", [None])[0]])
    return CUSTOM_KEY, m.ratio, m.cache_hit * 100.0


@callback(
    Output("mix-custom", "style"),
    Output("mix-ratio",  "value", allow_duplicate=True),
    Output("mix-cache",  "value", allow_duplicate=True),
    Output("mix-prev",   "data"),
    Input("filter-mix",  "value"),
    State("mix-prev",    "data"),
    prevent_initial_call="initial_duplicate",
)
def toggle_mix_custom(key, prev):
    """The ratio and cache boxes exist only for Custom; presets carry their own.

    Choosing Custom from a preset fills the boxes with THAT preset's numbers,
    as the static site does: left blank they silently meant 1:3 at 0%, a mix
    nobody chose. The first call (page load, after init_mix_from_url) has no
    previous value, so a shared ?m=custom&mr=20 link keeps its own numbers."""
    style = ({"display": "inline-flex", "alignItems": "center", "gap": "6px"}
             if key == CUSTOM_KEY else {"display": "none"})
    ratio = cache = no_update
    if key == CUSTOM_KEY and prev and prev != CUSTOM_KEY:
        opt = next((o for o in _mix_options() if o["value"] == prev), None)
        if opt and opt["ratio"] is not None:
            ratio, cache = float(f'{opt["ratio"]:.4g}'), opt["cache_pct"]
    return style, ratio, cache, key


@callback(
    Output("stat-floor-price", "children", allow_duplicate=True),
    Output("stat-floor-label", "children"),
    Output("model-table",      "columns"),
    Input("filter-mix", "value"),
    Input("mix-ratio",  "value"),
    Input("mix-cache",  "value"),
    State("model-table", "columns"),
    prevent_initial_call=True,
)
def update_mix_labels(mix_key, mix_ratio, mix_cache, columns):
    """The header stat and the table's price header name the active mix. The
    floor is the cheapest effective price over the UNFILTERED catalogue, as it
    always was — at the default it is the same string the layout renders."""
    basis = _mix(mix_key, mix_ratio, mix_cache)
    value, label = _floor_stat(basis)
    cols = [dict(c, name=_table_price_header(basis)) if c.get("id") == "price" else c
            for c in (columns or [])]
    return value, label, cols


def _floor_stat(basis):
    """(stat-floor-price children, stat-floor-label children) for `basis`.

    Mirrors docs/app.js refreshMixInfo: workload_mix.mix_summary decides the
    floor, and a floor model with no AA cache price carries the '*' plus a
    VISIBLE key in the label (a title alone is invisible on a touch screen).
    At the default mix both are the plain strings the layout renders."""
    info = _mix_summary(df, basis)
    if not info["floor_filled"]:
        return info["floor_price"], info["floor_label"]
    value = html.Span(info["floor_price"] + "*",
                      title=f'{info["floor_model"]} · {info["fill_key"]}')
    return value, f'{info["floor_label"]} · {info["fill_key"]}'


@callback(
    Output("filter-mix", "value", allow_duplicate=True),
    Input("preset-all",  "n_clicks"),
    prevent_initial_call=True,
)
def reset_mix(_n):
    """'Reset filters' clears everything, MIX included; the quality presets do not."""
    return DEFAULT_MIX_KEY


# ── URL: the WEIGHTS setting (?w=open | ?w=closed) ───────────────────────────
# Separate for the same reason as init_mix_from_url: init_from_url's 4-tuple is
# destructured by existing tests.
@callback(
    Output("filter-weights", "value"),
    Input("url", "search"),
    prevent_initial_call=False,
)
def init_weights_from_url(search: str):
    """A URL is user input: anything but 'open' / 'closed' means All ('')."""
    if not search:
        return ""
    return _valid_weights(parse_qs(search.lstrip("?")).get("w", [""])[0])


@callback(
    Output("filter-weights", "value", allow_duplicate=True),
    Input("preset-all",      "n_clicks"),
    prevent_initial_call=True,
)
def reset_weights(_n):
    """'Reset filters' clears WEIGHTS too; the quality presets do not."""
    return ""


# ── URL: write state on change ────────────────────────────────────────────────
clientside_callback(
    """
    function(tab, providers, quality, effort, mix, mixRatio, mixCache, weights) {
        var params = new URLSearchParams();
        if (tab)                           params.set('tab', tab);
        if (providers && providers.length) params.set('p', providers.join(','));
        if (quality > 0)                   params.set('q', quality);
        if (effort)                        params.set('e', effort);
        if (weights)                       params.set('w', weights);
        if (mix && mix !== '""" + DEFAULT_MIX_KEY + """') params.set('m', mix);
        if (mix === '""" + CUSTOM_KEY + """') {
            if (mixRatio !== null && mixRatio !== undefined && mixRatio !== '') params.set('mr', mixRatio);
            if (mixCache !== null && mixCache !== undefined && mixCache !== '') params.set('mh', mixCache);
        }
        var qs = params.toString();
        history.replaceState(null, '', qs ? '?' + qs : window.location.pathname);
        return window.location.href;
    }
    """,
    Output("url-sync", "data"),
    Input("tabs",            "value"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("filter-effort",   "value"),
    Input("filter-mix",      "value"),
    Input("mix-ratio",       "value"),
    Input("mix-cache",       "value"),
    Input("filter-weights",  "value"),
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

# ── Economics › Serve: a measured point opens its InferenceX run ──────────────
# The caption and every measured hover promise "click to open the run"; the
# static site does it in docs/app.js attachEconServeClick, and this is the same
# rule for Dash: customdata[1] must be an InferenceX GitHub Actions run URL
# (prefix + digits), opened with noopener. Anything else does nothing.
clientside_callback(
    """
    function(clickData) {
        var pts = clickData && clickData.points;
        var cd = pts && pts.length ? pts[0].customdata : null;
        var url = Array.isArray(cd) ? String(cd[1] || '') : '';
        var prefix = '""" + _serving_cost.IX_RUNS_PREFIX + """';
        if (url.indexOf(prefix) === 0 && /^\\d+\\/?$/.test(url.slice(prefix.length))) {
            window.open(url, '_blank', 'noopener');
            return url;
        }
        return null;
    }
    """,
    Output("econ-click-sink", "data"),
    Input("econ-serve-chart", "clickData"),
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
        eff_bw         = effective_bandwidth(bandwidth_gbps, gpu_count)
        local_df = get_local_df(
            quant=quant or "Q4",
            vram_gb=vram_gb,
            bandwidth_gbps=eff_bw,
            hw_type=hw_type,
            # Pinned to the shared default rather than left to get_local_df's
            # own, so this tab and Run Local price the same hardware the same
            # way. The Agent Stack has no context control of its own; if it
            # ever gains one, it goes here, not into a second default.
            ctx_tokens=DEFAULT_CONTEXT_TOKENS,
            fp16_tflops=tflops_for_gpu(gpu_preset or ""),
            gpu_count=gpu_count,
        )
        # Local picks on a datacenter card are priced, not free (static_helpers).
        local_df = _with_local_rent_cost(local_df, gpu_preset, gpu_count)

    cards = build_stack_cards(df, providers, mode=mode, local_df=local_df)
    return cards, prov_style, hw_style


# ── Chart update callbacks (all respond to global filters + search) ───────────
@callback(
    Output("pareto-chart",  "figure"),
    Output("overview-desc", "children"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("filter-effort",   "value"),
    Input("model-search",    "value"),
    Input("overview-xaxis",  "value"),
    Input("data-version",    "data"),
    Input("filter-mix",      "value"),
    Input("mix-ratio",       "value"),
    Input("mix-cache",       "value"),
    Input("filter-weights",  "value"),
    prevent_initial_call=True,
)
def update_overview(providers, min_quality, effort, search, xaxis, _v,
                    mix_key=None, mix_ratio=None, mix_cache=None,
                    weights=None):
    basis = _mix(mix_key, mix_ratio, mix_cache)
    base = _base(basis)
    filtered = apply_filters(base, providers, min_quality, search or "", effort, weights)
    if xaxis == "speed":
        desc = _desc(CAPTIONS["overview_speed"])
        # `base` is the unfiltered catalogue at this mix: the median crosshairs
        # that decide "Fast · Smart" and the frontier are market-wide claims,
        # so they must not move when the user narrows the frame.
        return build_quadrant(filtered, full_df=base, basis=basis), desc
    if xaxis == "task":
        # AA's cost per task: MIX reaches only the hover's price line. EFFORT
        # chooses among the tiers AA has costed (static_helpers.task_axis_frames).
        costed, every = task_axis_frames(base, providers, min_quality,
                                         search or "", effort, weights)
        return (build_pareto_scatter(costed, full_df=base, basis=basis, x="task",
                                     count_df=every),
                _desc(CAPTIONS["overview_task"]))
    desc = _desc(CAPTIONS["overview_price_dyn"])
    return build_pareto_scatter(filtered, full_df=base, basis=basis), desc


@callback(
    Output("treemap-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("filter-effort",   "value"),
    Input("model-search",    "value"),
    Input("data-version",    "data"),
    Input("filter-mix",      "value"),
    Input("mix-ratio",       "value"),
    Input("mix-cache",       "value"),
    Input("filter-weights",  "value"),
    prevent_initial_call=True,
)
def update_treemap(providers, min_quality, effort, search, _v,
                   mix_key=None, mix_ratio=None, mix_cache=None,
                   weights=None):
    basis = _mix(mix_key, mix_ratio, mix_cache)
    return build_treemap(_apply_filters(providers, min_quality, search or "", effort, basis,
                                        weights),
                         basis=basis)


@callback(
    Output("rankings-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("filter-effort",   "value"),
    Input("model-search",    "value"),
    Input("rankings-sort",   "value"),
    Input("data-version",    "data"),
    Input("filter-mix",      "value"),
    Input("mix-ratio",       "value"),
    Input("mix-cache",       "value"),
    Input("filter-weights",  "value"),
    prevent_initial_call=True,
)
def update_rankings(providers, min_quality, effort, search, sort_by, _v,
                    mix_key=None, mix_ratio=None, mix_cache=None,
                    weights=None):
    # Ranked on intelligence, so the open-weight models no host sells belong
    # here even though they cannot appear on any price or speed axis.
    basis = _mix(mix_key, mix_ratio, mix_cache)
    filtered = apply_filters(ranking_frame(_base(basis)), providers, min_quality,
                             search or "", effort, weights)
    return build_rankings(filtered, top_n=min(25, len(filtered)),
                          metric=sort_by or "intelligence", basis=basis)


@callback(
    Output("value-leaders-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("filter-effort",   "value"),
    Input("model-search",    "value"),
    Input("data-version",    "data"),
    Input("filter-mix",      "value"),
    Input("mix-ratio",       "value"),
    Input("mix-cache",       "value"),
    Input("filter-weights",  "value"),
    prevent_initial_call=True,
)
def update_value_leaders(providers, min_quality, effort, search, _v,
                         mix_key=None, mix_ratio=None, mix_cache=None,
                         weights=None):
    # Value Leaders sits under the same visible filter bar as the Rankings chart
    # above it, and was the one chart there with no callback: PROVIDER=Anthropic
    # narrowed Rankings and left this listing every provider, and the hourly
    # data-version bump never reached it, so its prices were frozen at boot.
    # The hosted frame, not ranking_frame: the score divides by price.
    basis = _mix(mix_key, mix_ratio, mix_cache)
    return build_value_leaders(_apply_filters(providers, min_quality, search or "", effort, basis,
                                              weights),
                               basis=basis)



@callback(
    Output("provider-leaderboard-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("filter-effort",   "value"),
    Input("model-search",    "value"),
    Input("filter-mix",      "value"),
    Input("mix-ratio",       "value"),
    Input("mix-cache",       "value"),
    Input("filter-weights",  "value"),
    prevent_initial_call=True,
)
def update_provider_leaderboard(providers, min_quality, effort, search,
                                mix_key=None, mix_ratio=None, mix_cache=None,
                                weights=None):
    basis = _mix(mix_key, mix_ratio, mix_cache)
    return build_provider_leaderboard(
        _apply_filters(providers, min_quality, search or "", effort, basis, weights),
        basis=basis)


@callback(
    Output("radar-chart",        "figure"),
    Output("radar-model-select", "options"),
    Output("radar-model-select", "value"),
    Output("compare-raw-table",  "children"),
    Input("filter-provider",     "value"),
    Input("filter-quality",      "value"),
    Input("filter-effort",   "value"),
    Input("model-search",        "value"),
    Input("radar-model-select",  "value"),
    Input("filter-mix",      "value"),
    Input("mix-ratio",       "value"),
    Input("mix-cache",       "value"),
    Input("filter-weights",  "value"),
    prevent_initial_call=True,
)
def update_compare(providers, min_quality, effort, search, selected_models,
                   mix_key=None, mix_ratio=None, mix_cache=None,
                   weights=None):
    basis     = _mix(mix_key, mix_ratio, mix_cache)
    base      = _base(basis)
    filtered  = apply_filters(base, providers, min_quality, search or "", effort, weights)
    options   = _model_options(filtered)
    triggered = ctx.triggered_id

    capped = _cap_compare_selection(selected_models, filtered, triggered)

    raw_table = _build_raw_table(filtered, capped, basis)
    return (build_radar(filtered, capped, full_df=base, basis=basis),
            options, capped, raw_table)


@callback(
    Output("cost-calc-chart", "figure"),
    Input("budget-tokens",   "value"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("filter-effort",   "value"),
    Input("model-search",    "value"),
    Input("filter-mix",      "value"),
    Input("mix-ratio",       "value"),
    Input("mix-cache",       "value"),
    Input("filter-weights",  "value"),
    prevent_initial_call=True,
)
def update_cost_calc(monthly_tokens_m, providers, min_quality, effort, search,
                     mix_key=None, mix_ratio=None, mix_cache=None,
                     weights=None):
    basis    = _mix(mix_key, mix_ratio, mix_cache)
    filtered = _apply_filters(providers, min_quality, search or "", effort, basis, weights)
    # 0 tokens is a real query; a negative volume would invert the "cheapest" sort.
    tokens   = _coerce_number(monthly_tokens_m, default=1.0, minimum=0.0)
    return build_cost_calc(filtered, monthly_tokens_m=tokens, basis=basis)


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


# ── "usable of N GB" beside both VRAM boxes ───────────────────────────────────
# The rule and the words live in data/local_models.memory_note; docs/app.js
# syncVramNote makes the same box-equals-preset test in the browser.
def _vram_note(vram, gpu_name):
    note = memory_note(gpu_name, vram if vram is not None else "")
    return (note["text"], note["title"]) if note else ("", "")


@callback(
    Output("local-vram-note", "children"),
    Output("local-vram-note", "title"),
    Input("local-vram",       "value"),
    Input("local-gpu-preset", "value"),
)
def update_local_vram_note(vram, gpu_name):
    return _vram_note(vram, gpu_name)


@callback(
    Output("recommend-vram-note", "children"),
    Output("recommend-vram-note", "title"),
    Input("recommend-vram",       "value"),
    Input("recommend-gpu-preset", "value"),
)
def update_recommend_vram_note(vram, gpu_name):
    return _vram_note(vram, gpu_name)


# ── Local tab ─────────────────────────────────────────────────────────────────
def _preset_rate(gpu_name, basis):
    """InferenceX's own/rent $/GPU-hr for a preset, or None (consumer cards)."""
    r = _serving_cost.gpu_rates().get(gpu_name or "")
    b = basis if basis in _serving_cost.BASES else _serving_cost.DEFAULT_LOCAL_RATE_BASIS
    return r[b] if r else None


@callback(
    Output("local-gpu-rate", "value", allow_duplicate=True),
    Input("local-rate-basis", "value"),
    State("local-gpu-preset", "value"),
    prevent_initial_call=True,
)
def update_local_rate_basis(basis, gpu_name):
    return _preset_rate(gpu_name, basis)


@callback(
    Output("local-vram",    "value"),
    Output("local-hw-meta", "data"),
    Output("local-gpu-rate", "value"),
    Input("local-gpu-preset", "value"),
    State("local-rate-basis", "value"),
    prevent_initial_call=True,
)
def update_local_hw(gpu_name: str, basis=None):
    gpu = GPU_BY_NAME.get(gpu_name)
    if not gpu:
        return no_update, no_update, no_update
    # fp16_tflops travels WITH the other two. There are four copies of this
    # shape (here, static_api.local_hw_for_gpu, the dcc.Store default, and
    # refreshLocal in docs/app.js); a field added to fewer than all four means
    # one render path computes a roofline and the other a bandwidth-only
    # estimate for the same card.
    return (gpu["vram_gb"], {"bandwidth_gbps": gpu["bandwidth_gbps"],
                             "hw_type": gpu["hw_type"],
                             "fp16_tflops": tflops_for_gpu(gpu_name)},
            _preset_rate(gpu_name, basis))


@callback(
    Output("local-scatter",      "figure"),
    Output("local-compat-chart", "figure"),
    Input("local-vram",     "value"),
    Input("local-num-gpus", "value"),
    Input("local-quant",    "value"),
    Input("local-context",  "value"),
    Input("local-speed-mode", "value"),
    Input("local-hw-meta",  "data"),
    Input("local-tags",     "value"),
    Input("local-gpu-rate", "value"),
    prevent_initial_call=True,
)
def update_local_charts(vram_per_gpu, num_gpus, quant, ctx_tokens, speed_mode, hw_meta, tags,
                        gpu_rate=None):
    # Shared defaults: this used to assume 8GB while the public site assumed 32,
    # so a cleared VRAM box answered "which models fit?" two different ways.
    gpu_count      = int(_coerce_number(num_gpus, DEFAULT_GPU_COUNT, minimum=1))
    vram_gb        = _coerce_number(vram_per_gpu, DEFAULT_VRAM_GB, minimum=0.0) * gpu_count
    bandwidth_gbps = _coerce_number((hw_meta or {}).get("bandwidth_gbps"),
                                    DEFAULT_BANDWIDTH_GBPS, minimum=0.0)
    hw_type        = (hw_meta or {}).get("hw_type", "nvidia")
    eff_bw         = effective_bandwidth(bandwidth_gbps, gpu_count)
    ctx            = int(_coerce_number(ctx_tokens, DEFAULT_CONTEXT_TOKENS, minimum=0))
    # A cleared preset leaves no compute figure. None is the honest value —
    # decode_roofline drops the compute roof and reports bound="memory?" —
    # rather than a stand-in that would imply a ceiling was checked.
    tflops         = (hw_meta or {}).get("fp16_tflops")

    # static_helpers.local_frame: the same frame (and $/GPU-HR readout) the
    # static site's update_local and both ↓CSVs build.
    local_df, vram_gb, ctx, _q = _local_frame(
        vram_per_gpu, num_gpus, quant or "Q4", bandwidth_gbps, hw_type, tags or None,
        ctx, tflops, gpu_rate)
    mode = speed_mode or DEFAULT_SPEED_MODE
    return (
        build_local_scatter(local_df, vram_gb=vram_gb, quant=quant or "Q4",
                            ctx_tokens=ctx, speed_mode=mode),
        build_local_compat(local_df, quant=quant or "Q4", vram_gb=vram_gb,
                           ctx_tokens=ctx, speed_mode=mode),
    )


# ── Economics tab ─────────────────────────────────────────────────────────────
@callback(
    Output("econ-serve-chart", "figure"),
    Output("econ-util-value",  "children"),
    Input("econ-serve-gpu",   "value"),
    Input("econ-serve-basis", "value"),
    Input("econ-util",        "value"),
    prevent_initial_call=True,
)
def update_econ_serve(gpu, basis, util):
    used = _serving_cost.coerce_util(util)
    return _econ_serve_figure(gpu, basis, util), f"{used:g}%"


@callback(
    Output("econ-tco-chart",           "figure"),
    Output("econ-ladder-chart",        "figure"),
    Output("econ-rental-series-chart", "figure"),
    Input("econ-compute-gpu",    "value"),
    Input("econ-compute-preset", "value"),
    Input("econ-util",           "value"),
    *[Input(f"econ-compute-{key}", "value") for key, *_ in _COMPUTE_BOXES],
)
def update_econ_compute(gpu, preset, util, life=None, rate=None, capex_k=None, kw=None,
                        pue=None, kwh=None, colo=None):
    """Economics › Compute: the same three figures as the static site's
    update_econ_compute, from the same data/gpu_tco.compute call."""
    return _econ_compute_figures(*_compute_args(gpu, preset, util, life, rate, capex_k,
                                                kw, pue, kwh, colo))


@callback(
    Output("econ-compute-capex", "value"),
    Output("econ-compute-kw",    "value"),
    Output("econ-compute-capex", "placeholder"),
    Output("econ-compute-kw",    "placeholder"),
    Output("econ-compute-capex-wrap", "title"),
    Output("econ-compute-kw-wrap",    "title"),
    Input("econ-compute-gpu", "value"),
    prevent_initial_call=True,
)
def sync_econ_compute_gpu(gpu):
    """A GPU change clears the per-GPU boxes, so a typed CAPEX or kW never
    carries over to a part it was not typed for, and shows that part's
    sourced defaults as placeholders."""
    ph = _compute_placeholders(gpu)
    return None, None, ph["capex"][0], ph["kw"][0], ph["capex"][1], ph["kw"][1]


@callback(
    Output("econ-compute-life", "value"),
    Output("econ-compute-rate", "value"),
    Output("econ-compute-life", "placeholder"),
    Output("econ-compute-rate", "placeholder"),
    Output("econ-compute-life-wrap", "title"),
    Output("econ-compute-rate-wrap", "title"),
    Input("econ-compute-preset", "value"),
    prevent_initial_call=True,
)
def sync_econ_compute_preset(preset):
    """A preset change clears LIFE and RATE and shows the preset's own."""
    ph = _compute_placeholders(None, preset)
    return None, None, ph["life"][0], ph["rate"][0], ph["life"][1], ph["rate"][1]


@callback(
    Output("econ-view-serve",      "style"),
    Output("econ-view-labs",       "style"),
    Output("econ-serve-controls",  "style"),
    Output("econ-shared-controls", "style"),
    Output("econ-open-gap-history-chart", "figure"),
    Output("econ-open-gap-lag-chart",     "figure"),
    Output("econ-open-gap-price-chart",   "figure"),
    Output("econ-view-compute",     "style"),
    Output("econ-compute-controls", "style"),
    Input("econ-view", "value"),
)
def toggle_econ_view(view):
    """Serve and Compute show the shared USE slider and their own controls;
    Labs shows neither. Labs is built from the same open-gap builders as the
    static pre-render, only when first opened (the score history reads every
    daily snapshot)."""
    shown = {"display": "inline-flex", "alignItems": "center", "gap": "8px",
             "flexWrap": "wrap"}
    hidden = {"display": "none"}
    if view == "compute":
        return (hidden, hidden, hidden, shown, no_update, no_update, no_update, {}, shown)
    if view != "labs":
        return ({}, hidden, shown, shown, no_update, no_update, no_update, hidden, hidden)
    from data.ingest import load_history
    from data.open_gap import (gap_history, release_lag, iso_quality_ratios,
                               snapshot_dates, score_dates)
    from components.charts.open_gap import (build_gap_history, build_release_lag,
                                            build_iso_price)
    hist = load_history()
    dates = snapshot_dates(hist)
    asof = dates[-1] if dates else None
    return ({"display": "none"}, {}, hidden, hidden,
            build_gap_history(*gap_history(hist)),
            build_release_lag(*release_lag(df, asof, score_dates(hist, df))),
            build_iso_price(*iso_quality_ratios(df, None)), hidden, hidden)


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
    # MIX is an Input, not a State: the panel states a price at a mix, and an
    # open panel left at the old mix contradicted every chart behind it.
    Input("filter-mix",   "value"),
    Input("mix-ratio",    "value"),
    Input("mix-cache",    "value"),
    State("detail-model-name", "data"),
    prevent_initial_call=True,
)
def toggle_detail_panel(pareto_click, _close, mix_key=None, mix_ratio=None, mix_cache=None,
                        open_model=None):
    trigger = ctx.triggered_id

    if trigger == "detail-close":
        return "detail-panel", [], None

    if trigger in ("filter-mix", "mix-ratio", "mix-cache"):
        # Re-render the open panel at the new mix; a closed one stays closed.
        if not open_model:
            return no_update, no_update, no_update
        model_name, provider = open_model, None
    else:
        click = pareto_click
        if not click or not click.get("points"):
            return no_update, no_update, no_update

        pt         = click["points"][0]
        customdata = pt.get("customdata", [])
        if not customdata or len(customdata) < 2:
            return no_update, no_update, no_update

        # customdata[0] and [1] are HTML-escaped for Plotly's hover text; [5]
        # and [6] carry the raw strings. Looking up the escaped name meant a
        # model called "R&D Coder" matched no row and its panel silently never
        # opened, and a provider with "&" printed as a literal "&amp;".
        model_name = customdata[5] if len(customdata) > 6 else customdata[0]
        provider   = customdata[6] if len(customdata) > 6 else customdata[1]

    basis = _mix(mix_key, mix_ratio, mix_cache)
    base  = _base(basis)
    rows = base[base["model"] == model_name]
    if rows.empty:
        return no_update, no_update, no_update
    row = rows.iloc[0]
    if provider is None:
        provider = str(row["provider"])
    cache_str = _cache_label(row)   # shared with static_api

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

    # AA's 2026-09 overhaul. Mirrors static_api._detail_html — the Dash app and
    # the published static site must state the same thing about the same model.
    estimated = bool(row.get("quality_estimated", False))

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
                html.Span(f"{quality:.0f}  ·  {qlabel}{'  ·  est.' if estimated else ''}",
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
            html.Div(("AA estimated this score — not the full evaluation suite"
                      if estimated else f"Top {100 - pct}% of all models"),
                     style={"fontSize": "10px", "color": "var(--text-3)",
                            "marginBottom": "16px"}),
        ]),

        html.Div(className="detail-panel-divider"),
        _metric("Price",   f"{_mark_price(row['price'], row.get('cache_price_filled'))}"
                           f" / 1M  ·  {basis.short}"),
        _metric("Cached in", cache_str),
        # Cost per task and token burn: shared with static_api via data/task_cost.
        *[_metric(lbl, val) for lbl, val in _task_detail_rows(row, median_token_burn(df))],
        *([html.Div(_task_detail_note(row), style={"fontSize": "10px", "color": "var(--text-3)",
                                                   "marginTop": "2px"})]
          if _task_detail_note(row) else []),
        _metric("Speed",   speed_str),
        _metric("Latency", latency_str),
        _metric("Context", str(row.get("context", "N/A")) or "N/A"),
        _metric("Weights", _weights_label(row.get("open_weights", False),
                                          row.get("weights_source"))),
        html.Div(className="detail-panel-divider"),
        _metric("Coding",      index_label(row.get("coding"))),
        _metric("Agentic",     index_label(row.get("agentic"))),
        _metric("Omniscience", index_label(row.get("omniscience"))),
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
    """Add the model the user is looking at, evicting the OLDEST pick.

    `[:5]` kept the first five and threw away the one just clicked, so with a
    full selection — which is the default state, since both renderings start
    with exactly five — this button did nothing while still switching tabs.
    """
    if not n_clicks or not model_name:
        return no_update, no_update
    current = list(current_selection or [])
    if model_name in current:
        return no_update, "compare"        # already there; just show it
    return (current + [model_name])[-COMPARE_MAX:], "compare"


# ── Table view ────────────────────────────────────────────────────────────────
@callback(
    Output("model-table", "data"),
    Output("model-table", "tooltip_data"),
    Input("filter-provider",  "value"),
    Input("filter-quality",   "value"),
    Input("filter-effort",   "value"),
    Input("model-search",     "value"),
    Input("table-sort-col",   "value"),
    Input("table-sort-dir",   "value"),
    Input("filter-mix",      "value"),
    Input("mix-ratio",       "value"),
    Input("mix-cache",       "value"),
    Input("filter-weights",  "value"),
)
def update_table(providers, min_quality, effort, search, sort_col, sort_dir,
                 mix_key=None, mix_ratio=None, mix_cache=None,
                 weights=None):
    basis    = _mix(mix_key, mix_ratio, mix_cache)
    filtered = _apply_filters(providers, min_quality, search or "", effort, basis,
                              weights).copy()
    filtered["value"] = filtered.apply(
        lambda r: r["quality"] / r["price"] if r["price"] > 0 else None, axis=1
    )
    col = sort_col or "quality"
    asc = (sort_dir or "desc") == "asc"
    if col == "context":
        filtered["_ctx_k"] = filtered["context"].map(_ctx_to_k)
        filtered = filtered.sort_values("_ctx_k", ascending=asc, na_position="last",
                                        kind="mergesort")
    else:
        filtered = filtered.sort_values(col, ascending=asc, na_position="last",
                                        kind="mergesort")
    cols = ["model", "provider", "quality", "value", "price", "speed", "latency", "context"]
    # Mirrors the static table's '*': a row whose price rests on the
    # no-cache-price fill-in says so on its price cell.
    flags = _filled_flags(filtered)
    key = f"{FILL_MARK} {FILL_SHORT}"
    tips = [{"price": key} if f else {} for f in flags]
    return filtered[cols].to_dict("records"), tips


# ── Export CSV ────────────────────────────────────────────────────────────────
@callback(
    Output("download-csv",   "data"),
    Input("btn-export",      "n_clicks"),
    State("tabs",            "value"),
    State("filter-provider", "value"),
    State("filter-quality",  "value"),
    State("model-search",    "value"),
    State("filter-effort",   "value"),
    State("local-vram",      "value"),
    State("local-num-gpus",  "value"),
    State("local-quant",     "value"),
    State("local-context",   "value"),
    State("local-hw-meta",   "data"),
    State("local-tags",      "value"),
    State("filter-mix",      "value"),
    State("mix-ratio",       "value"),
    State("mix-cache",       "value"),
    State("filter-weights",  "value"),
    State("local-gpu-rate",  "value"),
    State("econ-view",       "value"),
    State("econ-serve-gpu",  "value"),
    State("econ-serve-basis", "value"),
    State("econ-util",       "value"),
    State("econ-compute-gpu",    "value"),
    State("econ-compute-preset", "value"),
    *[State(f"econ-compute-{key}", "value") for key, *_ in _COMPUTE_BOXES],
    prevent_initial_call=True,
)
def export_csv(n_clicks, tab, providers, min_quality, search, effort=None,
               vram_per_gpu=None, num_gpus=None, quant=None, ctx_tokens=None,
               hw_meta=None, tags=None, mix_key=None, mix_ratio=None, mix_cache=None,
               weights=None, gpu_rate=None, econ_view=None, econ_gpu=None,
               econ_basis=None, econ_util=None, compute_gpu=None, compute_preset=None,
               compute_life=None, compute_rate=None, compute_capex_k=None, compute_kw=None,
               compute_pue=None, compute_kwh=None, compute_colo=None):
    """Export what is on screen, not always the hosted-LLM table.

    The global filter bar sits outside dcc.Tabs and four tabs consume none of
    it, but ↓CSV always exported the LLM catalogue through those filters. From
    Image Gen, `export_csv(['Anthropic'], 50, '')` returned the LLM header and
    seven text models — a file describing a different dataset from the one the
    user was looking at.
    """
    if not n_clicks:
        return no_update
    # EFFORT and the Run Local hardware ride along too: the LLM export ignored
    # EFFORT (195 rows exported against 113 on screen), and the Run Local one
    # described the default 32 GB / Q4 card whatever hardware was picked.
    hw = hw_meta or {}
    local_args = {
        "vram_per_gpu": vram_per_gpu, "num_gpus": num_gpus, "quant": quant,
        "bandwidth_gbps": hw.get("bandwidth_gbps"), "hw_type": hw.get("hw_type"),
        "tags": tags or None, "ctx_tokens": ctx_tokens,
        "fp16_tflops": hw.get("fp16_tflops"), "gpu_rate": gpu_rate,
    }
    if econ_view == "compute":
        econ_args = ["compute", *_compute_args(compute_gpu, compute_preset, econ_util,
                                               compute_life, compute_rate, compute_capex_k,
                                               compute_kw, compute_pue, compute_kwh,
                                               compute_colo)]
    else:
        econ_args = [econ_view or "serve", econ_gpu, econ_basis, econ_util]
    frame, name = _export_frame_for_tab(tab, providers, min_quality, search,
                                        effort=effort, local_args=local_args,
                                        mix=_mix(mix_key, mix_ratio, mix_cache),
                                        weights=weights, econ_args=econ_args)
    return dcc.send_data_frame(_csv_safe(frame).to_csv, name, index=False)


# ── Hide the global filter bar where it does nothing ──────────────────────────
@callback(
    Output("global-filters", "className"),
    Input("tabs", "value"),
)
def toggle_global_filters(tab):
    """Agent Stack, Run Local, Image Gen and Video Gen read none of PROVIDER /
    MIN SCORE / SEARCH. Leaving the bar visible and live on those tabs meant it
    displayed "Anthropic / >= 45" over a chart plotting 72 models from a dozen
    other providers.

    Only the filters are hidden (.export-only in assets/style.css): ↓CSV lives
    in the same bar, and hiding all of it left those tabs' exports — each its
    own dataset — with no button to reach them."""
    return "filters export-only" if tab in _TABS_WITHOUT_GLOBAL_FILTERS else "filters"


# ── Auto data refresh — drives ALL stat bar values and data-version ───────────
@callback(
    Output("stat-model-count",    "children"),
    Output("stat-provider-count", "children"),
    Output("stat-floor-price",    "children"),
    Output("stat-floor-label",    "children", allow_duplicate=True),
    Output("stat-peak-quality",   "children"),
    Output("data-version",        "data"),
    Input("data-refresh-interval", "n_intervals"),
    State("data-version",          "data"),
    State("filter-mix",            "value"),
    State("mix-ratio",             "value"),
    State("mix-cache",             "value"),
    prevent_initial_call=True,
)
def auto_refresh_data(_, current_version, mix_key=None, mix_ratio=None, mix_cache=None):
    prev_mtime = _cache_mtime
    _reload_if_stale()
    # Content-addressed, not a counter. _cache_mtime is a module global while
    # data-version is per-session, and _reload_if_stale clears the "changed"
    # flag as a side effect — so whichever session ticked first consumed the
    # signal and the others got a fresh stat bar above stale charts. The mtime
    # is the same value in every session, so they all converge.
    new_version = _cache_mtime
    return (
        str(len(df)),
        str(df["provider"].nunique()),
        # At the visitor's MIX, so an hourly refresh cannot reset the stat to
        # the default basis under a label that still names their mix. The
        # label is refreshed with it, so the '*' and its key never part.
        *_floor_stat(_mix(mix_key, mix_ratio, mix_cache)),
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
    Input("video-mode-filter",     "value"),
    Input("video-provider-filter", "value"),
    Input("video-tag-filter",      "value"),
    prevent_initial_call=True,
)
def update_video_charts(mode, providers, tags):
    # filter_video_df is shared with static_api.update_video so the Dash tab and
    # the published site cannot drift apart on what a tag means.
    full_vdf = get_video_df(mode)
    vdf = filter_video_df(full_vdf, providers, tags)
    return (
        build_video_rankings(vdf, full_df=full_vdf, mode=mode),
        build_video_scatter(vdf, full_df=full_vdf, mode=mode),
    )



# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.getenv("DEBUG", "false").lower() == "true"
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_scrapers()
    app.run(debug=debug, port=8050)
