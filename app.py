"""
AI Frontier — Main Dash Application
Interactive dashboard comparing 100+ LLMs on cost, speed, and quality.

Architecture: all dcc.Graph components live inside dcc.Tab.children so their
IDs are always in the DOM regardless of active tab. This prevents Dash 4's
"nonexistent object" callback errors.
"""
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs

import dash
from dash import ctx, dcc, html, Input, Output, State, callback, clientside_callback, no_update
import pandas as pd

from data.ingest import get_models, load_history
from data.local_models import get_local_df, get_gpu_options, GPU_BY_NAME, QUANT_LEVELS
from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR
from components.charts.pareto               import build_pareto_scatter
from components.charts.quadrant             import build_quadrant
from components.charts.treemap              import build_treemap
from components.charts.rankings             import build_rankings
from components.charts.value                import build_value_chart
from components.charts.radar                import build_radar
from components.charts.cost_calc            import build_cost_calc
from components.charts.trends               import build_trends
from components.charts.animated_pareto      import build_animated_pareto
from components.charts.context_chart        import build_context_chart
from components.charts.price_timeline       import build_price_timeline
from components.charts.provider_leaderboard import build_provider_leaderboard
from components.charts.local_scatter        import build_local_scatter
from components.charts.local_compat         import build_local_compat

# ── Data ─────────────────────────────────────────────────────────────────────
df         = get_models()
history_df = load_history()

_CACHE_PATH = Path(__file__).parent / "data" / "raw" / "aa_models.csv"

def _cache_ts() -> str:
    try:
        return datetime.fromtimestamp(_CACHE_PATH.stat().st_mtime).strftime("%b %d  %H:%M")
    except Exception:
        return "—"

_TOP5 = (
    df[df["quality"] > 0]
    .sort_values("quality", ascending=False)
    .head(5)["model"]
    .tolist()
)

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
    top = dataframe.sort_values("quality", ascending=False).head(40)
    return [{"label": f"{r['model']} ({r['provider']})", "value": r["model"]}
            for _, r in top.iterrows()]


def _apply_filters(providers, min_quality, search: str = "") -> pd.DataFrame:
    filtered = df.copy()
    if providers:
        filtered = filtered[filtered["provider"].isin(providers)]
    if min_quality and min_quality > 0:
        filtered = filtered[filtered["quality"] >= min_quality]
    if search and search.strip():
        pat = search.strip()
        mask = (
            filtered["model"].str.contains(pat, case=False, na=False) |
            filtered["provider"].str.contains(pat, case=False, na=False)
        )
        filtered = filtered[mask]
    return filtered


def _quality_label(q: float) -> str:
    if q >= 85: return "Exceptional"
    if q >= 70: return "Strong"
    if q >= 55: return "Capable"
    if q >= 40: return "Average"
    return "Limited"


def _desc(text: str) -> html.Div:
    return html.Div([html.P(text, className="chart-desc")], className="chart-desc-wrap")


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
    dcc.Download(id="download-csv"),

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
            html.Span("LIVE DATA", className="header-badge"),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], className="header"),

    # ── Stat bar ──────────────────────────────────────────────────────────────
    html.Div([
        _stat(str(len(df)),                                         "Models tracked"),
        _stat(str(df["provider"].nunique()),                        "Providers"),
        _stat(f"${df['price'].min():.3f}",                         "Floor price / 1M", accent=True),
        _stat(str(int(df["quality"].max())),                        "Peak intelligence"),
        _stat(f"{int(df['speed'].replace(0, pd.NA).max()):,}",     "Max speed tok/s"),
        _stat(_cache_ts(),                                          "Data updated"),
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
        html.Span("MIN IQ", className="filter-label"),
        dcc.Dropdown(
            id="filter-quality",
            options=[{"label": f"≥ {v}", "value": v}
                     for v in [0, 10, 20, 30, 40, 50, 60, 70, 75, 80]],
            value=0,
            clearable=False,
            style={"width": "88px"},
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
    ], className="filters"),

    # ── Preset quick-filters ──────────────────────────────────────────────────
    html.Div([
        html.Span("QUICK FILTER", className="filter-label"),
        html.Button("All Models", id="preset-all",    className="preset-btn"),
        html.Button("Strong ≥50", id="preset-strong", className="preset-btn"),
        html.Button("Elite ≥75",  id="preset-elite",  className="preset-btn"),
    ], className="presets-bar"),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dcc.Tabs(id="tabs", value="overview", className="tabs", children=[

        # Overview ─────────────────────────────────────────────────────────────
        dcc.Tab(label="Overview", value="overview",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Each bubble is one model. X-axis = cost per 1M tokens (log scale). "
                "Y-axis = intelligence score. Bubble size = generation speed. "
                "The dotted line traces the Pareto frontier — best quality for the price. "
                "Click any bubble to open a full model detail panel."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="pareto-chart", figure=build_pareto_scatter(df),
                          config=_GRAPH_CONFIG, style={"height": "620px"}),
            ])], className="chart-card"),
            _desc(
                "Frontier evolution — watch how intelligence vs. price has shifted across "
                "every saved daily snapshot. Press ▶ Play to animate. "
                "Dotted cyan line = Pareto frontier at that point in time."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="animated-pareto-chart",
                          figure=build_animated_pareto(history_df),
                          config=_GRAPH_CONFIG, style={"height": "580px"}),
            ])], className="chart-card"),
        ]),

        # Performance ──────────────────────────────────────────────────────────
        dcc.Tab(label="Performance", value="performance",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Speed vs. intelligence. Models in the top-right are both fast and smart. "
                "Bubble size is inversely proportional to price — larger = cheaper. "
                "Click any bubble to see full model details."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="quadrant-chart", figure=build_quadrant(df),
                          config=_GRAPH_CONFIG, style={"height": "620px"}),
            ])], className="chart-card"),
        ]),

        # Value ────────────────────────────────────────────────────────────────
        dcc.Tab(label="Value", value="value",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Intelligence points earned per dollar spent. "
                "Models at the top give you the most cognitive output for your budget."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="value-chart", figure=build_value_chart(df),
                          config=_GRAPH_CONFIG, style={"height": "900px"}),
            ])], className="chart-card"),
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
            _desc(
                "Top models ranked by AA Intelligence Index. "
                "Hover for price and speed details."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="rankings-chart", figure=build_rankings(df, top_n=25),
                          config=_GRAPH_CONFIG, style={"height": "750px"}),
            ])], className="chart-card"),
            _desc(
                "Context window vs. intelligence. Bubble size = cheaper. "
                "Find long-context models that don't sacrifice quality."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="context-chart", figure=build_context_chart(df),
                          config=_GRAPH_CONFIG, style={"height": "560px"}),
            ])], className="chart-card"),
        ]),

        # Compare ──────────────────────────────────────────────────────────────
        dcc.Tab(label="Compare", value="compare",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Radar chart comparing up to 5 models across 4 normalized dimensions. "
                "Select models below, or click → Compare in any model detail panel."
            ),
            html.Div([
                html.Span("SELECT MODELS", className="filter-label"),
                dcc.Dropdown(
                    id="radar-model-select",
                    options=_model_options(df),
                    value=_TOP5,
                    multi=True,
                    placeholder="Select up to 5 models…",
                    style={"minWidth": "500px"},
                ),
                html.Span("max 5", className="filter-label",
                          style={"color": "#333333", "paddingLeft": "8px"}),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="radar-chart", figure=build_radar(df, _TOP5),
                          config=_GRAPH_CONFIG, style={"height": "560px"}),
            ])], className="chart-card"),
        ]),

        # Budget ───────────────────────────────────────────────────────────────
        dcc.Tab(label="Budget", value="budget",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Estimate your monthly API spend. Enter your expected monthly token volume "
                "and see projected costs across all models. Sorted cheapest-first."
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
                          style={"color": "#444444", "fontSize": "11px", "paddingLeft": "6px"}),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="cost-calc-chart",
                          figure=build_cost_calc(df, monthly_tokens_m=1.0),
                          config=_GRAPH_CONFIG, style={"height": "1100px"}),
            ])], className="chart-card"),
        ]),

        # Trends ───────────────────────────────────────────────────────────────
        dcc.Tab(label="Trends", value="trends",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Intelligence score history for the top models. "
                "A new snapshot is saved each time data is refreshed."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="trends-chart", figure=build_trends(history_df),
                          config=_GRAPH_CONFIG, style={"height": "600px"}),
            ])], className="chart-card"),
            _desc(
                "Price history for the top models — USD per million tokens, log scale. "
                "The relentless decline in AI inference costs, quantified."
            ),
            html.Div([dcc.Loading(**_LOADING, children=[
                dcc.Graph(id="price-timeline-chart",
                          figure=build_price_timeline(history_df),
                          config=_GRAPH_CONFIG, style={"height": "560px"}),
            ])], className="chart-card"),
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
        return "overview", [], 0
    params    = parse_qs(search.lstrip("?"))
    tab       = params.get("tab", ["overview"])[0]
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
        return 50, [], no_update
    if trigger == "preset-elite":
        return 75, [], no_update
    return no_update, no_update, no_update


# ── Chart update callbacks (all respond to global filters + search) ───────────
@callback(
    Output("pareto-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    prevent_initial_call=True,
)
def update_pareto(providers, min_quality, search):
    return build_pareto_scatter(_apply_filters(providers, min_quality, search or ""))


@callback(
    Output("quadrant-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    prevent_initial_call=True,
)
def update_quadrant(providers, min_quality, search):
    return build_quadrant(_apply_filters(providers, min_quality, search or ""))


@callback(
    Output("value-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    prevent_initial_call=True,
)
def update_value(providers, min_quality, search):
    return build_value_chart(_apply_filters(providers, min_quality, search or ""))


@callback(
    Output("treemap-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    prevent_initial_call=True,
)
def update_treemap(providers, min_quality, search):
    return build_treemap(_apply_filters(providers, min_quality, search or ""))


@callback(
    Output("rankings-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    prevent_initial_call=True,
)
def update_rankings(providers, min_quality, search):
    filtered = _apply_filters(providers, min_quality, search or "")
    return build_rankings(filtered, top_n=min(25, len(filtered)))


@callback(
    Output("context-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    Input("model-search",    "value"),
    prevent_initial_call=True,
)
def update_context(providers, min_quality, search):
    return build_context_chart(_apply_filters(providers, min_quality, search or ""))


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
        capped = (
            filtered[filtered["quality"] > 0]
            .sort_values("quality", ascending=False)
            .head(5)["model"]
            .tolist()
        )
    else:
        capped = (selected_models or [])[:5]

    return build_radar(filtered, capped), options, capped


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

    color   = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
    quality = float(row["quality"])
    qlabel  = _quality_label(quality)

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
                    "width":        f"{min(quality, 100):.1f}%",
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
                     style={"fontSize": "10px", "color": "#333333",
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


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(debug=debug, port=8050)
