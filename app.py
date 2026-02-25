"""
AI Frontier — Main Dash Application
Interactive dashboard comparing 100+ LLMs on cost, speed, and quality.

Architecture note: all dcc.Graph components live in the static layout (inside
dcc.Tab.children) so their IDs are always in the DOM. This lets callbacks
reference them reliably, including clickData for the detail panel.
"""
import os
from urllib.parse import parse_qs

import dash
from dash import ctx, dcc, html, Input, Output, callback, clientside_callback, no_update
import pandas as pd

from data.ingest import get_models, load_history
from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR
from components.charts.pareto    import build_pareto_scatter
from components.charts.quadrant  import build_quadrant
from components.charts.treemap   import build_treemap
from components.charts.rankings  import build_rankings
from components.charts.value     import build_value_chart
from components.charts.radar     import build_radar
from components.charts.cost_calc import build_cost_calc
from components.charts.trends    import build_trends

# ── Data ────────────────────────────────────────────────────────────────────
df         = get_models()
history_df = load_history()

_TOP5 = (
    df[df["quality"] > 0]
    .sort_values("quality", ascending=False)
    .head(5)["model"]
    .tolist()
)

# ── App ─────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="AI Frontier",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server


# ── Helpers ─────────────────────────────────────────────────────────────────
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


def _apply_filters(providers, min_quality) -> pd.DataFrame:
    filtered = df.copy()
    if providers:
        filtered = filtered[filtered["provider"].isin(providers)]
    if min_quality and min_quality > 0:
        filtered = filtered[filtered["quality"] >= min_quality]
    return filtered


def _desc(text: str) -> html.Div:
    return html.Div([html.P(text, className="chart-desc")], className="chart-desc-wrap")


_GRAPH_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"],
}


# ── Layout ──────────────────────────────────────────────────────────────────
# All dcc.Graph components are inside dcc.Tab.children so their IDs always
# exist in the DOM regardless of which tab is active.
app.layout = html.Div([

    dcc.Location(id="url", refresh=False),
    dcc.Store(id="url-sync"),   # dummy store — clientside callback writes URL here

    # ── Header
    html.Div([
        html.Div([
            html.H1("AI FRONTIER"),
            html.Span("LLM comparison dashboard", className="subtitle"),
        ], className="header-left"),
        html.Span("LIVE DATA", className="header-badge"),
    ], className="header"),

    # ── Stat bar
    html.Div([
        _stat(str(len(df)),                                             "Models tracked"),
        _stat(str(df["provider"].nunique()),                            "Providers"),
        _stat(f"${df['price'].min():.3f}",                             "Floor price / 1M", accent=True),
        _stat(str(int(df["quality"].max())),                            "Peak intelligence"),
        _stat(f"{int(df['speed'].replace(0, pd.NA).max()):,}",         "Max speed tok/s"),
    ], className="stat-bar"),

    # ── Global filters
    html.Div([
        html.Span("PROVIDER", className="filter-label"),
        dcc.Dropdown(
            id="filter-provider",
            options=_provider_options(df),
            multi=True,
            placeholder="All providers",
            style={"minWidth": "260px"},
        ),
        html.Div(className="filter-sep"),
        html.Span("MIN INTELLIGENCE", className="filter-label"),
        dcc.Dropdown(
            id="filter-quality",
            options=[{"label": f"≥ {v}", "value": v} for v in [0, 10, 20, 30, 40, 50]],
            value=0,
            clearable=False,
            style={"width": "100px"},
        ),
    ], className="filters"),

    # ── Tabs — children embedded so all graph IDs are always in DOM
    dcc.Tabs(id="tabs", value="overview", className="tabs", children=[

        dcc.Tab(label="Overview", value="overview",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Each bubble is one model. X-axis = cost per 1M tokens (log scale). "
                "Y-axis = intelligence score. Bubble size = generation speed. "
                "The dotted line traces the Pareto frontier — best quality for the price. "
                "Click any bubble to see full model details."
            ),
            html.Div([
                dcc.Graph(id="pareto-chart", figure=build_pareto_scatter(df),
                          config=_GRAPH_CONFIG, style={"height": "620px"}),
            ], className="chart-card"),
        ]),

        dcc.Tab(label="Performance", value="performance",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Speed vs. intelligence. Models in the top-right are both fast and smart. "
                "Bubble size is inversely proportional to price — larger = cheaper. "
                "Click any bubble to see full model details."
            ),
            html.Div([
                dcc.Graph(id="quadrant-chart", figure=build_quadrant(df),
                          config=_GRAPH_CONFIG, style={"height": "620px"}),
            ], className="chart-card"),
        ]),

        dcc.Tab(label="Value", value="value",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Intelligence points earned per dollar spent. "
                "Models at the top give you the most cognitive output for your budget."
            ),
            html.Div([
                dcc.Graph(id="value-chart", figure=build_value_chart(df),
                          config=_GRAPH_CONFIG, style={"height": "900px"}),
            ], className="chart-card"),
        ]),

        dcc.Tab(label="Landscape", value="landscape",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "AI ecosystem by provider. Tile area = number of models in the dataset. "
                "Color intensity = average intelligence score."
            ),
            html.Div([
                dcc.Graph(id="treemap-chart", figure=build_treemap(df),
                          config=_GRAPH_CONFIG, style={"height": "600px"}),
            ], className="chart-card"),
        ]),

        dcc.Tab(label="Rankings", value="rankings",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Top models ranked by AA Intelligence Index. "
                "Hover for price and speed details."
            ),
            html.Div([
                dcc.Graph(id="rankings-chart", figure=build_rankings(df, top_n=25),
                          config=_GRAPH_CONFIG, style={"height": "750px"}),
            ], className="chart-card"),
        ]),

        dcc.Tab(label="Compare", value="compare",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Radar chart comparing up to 5 models across 4 normalized dimensions. "
                "Select models below to compare."
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
            html.Div([
                dcc.Graph(id="radar-chart", figure=build_radar(df, _TOP5),
                          config=_GRAPH_CONFIG, style={"height": "560px"}),
            ], className="chart-card"),
        ]),

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
                    type="number",
                    value=1.0,
                    min=0.001,
                    step=0.5,
                    debounce=True,
                    placeholder="e.g. 10",
                    style={
                        "background": "var(--bg-card)", "border": "1px solid var(--border)",
                        "borderRadius": "4px", "color": "#f2f2f2", "fontFamily": "Inter, sans-serif",
                        "fontSize": "13px", "padding": "6px 10px", "width": "100px",
                        "outline": "none",
                    },
                ),
                html.Span("million tokens / month", className="budget-unit",
                          style={"color": "#444444", "fontSize": "11px", "paddingLeft": "6px"}),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div([
                dcc.Graph(id="cost-calc-chart", figure=build_cost_calc(df, monthly_tokens_m=1.0),
                          config=_GRAPH_CONFIG, style={"height": "1100px"}),
            ], className="chart-card"),
        ]),

        dcc.Tab(label="Trends", value="trends",
                className="tab", selected_className="tab--selected", children=[
            _desc(
                "Intelligence score history for the top models. "
                "A new snapshot is saved each time data is refreshed. "
                "Trend lines appear once multiple snapshots have accumulated."
            ),
            html.Div([
                dcc.Graph(id="trends-chart", figure=build_trends(history_df),
                          config=_GRAPH_CONFIG, style={"height": "600px"}),
            ], className="chart-card"),
        ]),

    ]),

    # ── Model detail panel (slide-in from right on scatter point click)
    html.Div(id="detail-panel", className="detail-panel", children=[
        html.Button("✕", id="detail-close", className="detail-panel-close"),
        html.Div(id="detail-panel-body"),
    ]),

    # ── Footer
    html.Div([
        html.Span("Source: Artificial Analysis · Updated live"),
        html.Span("AI Frontier"),
    ], className="footer"),

], style={"minHeight": "100vh", "background": "#0a0a0a"})


# ── Sync filters from URL on page load ──────────────────────────────────────
@callback(
    Output("tabs",            "value"),
    Output("filter-provider", "value"),
    Output("filter-quality",  "value"),
    Input("url", "search"),
    prevent_initial_call=False,
)
def init_from_url(search: str):
    """Read ?tab=&p=&q= from URL and restore state on page load."""
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


# ── Write state to URL via history.replaceState (no page reload) ─────────────
clientside_callback(
    """
    function(tab, providers, quality) {
        const params = new URLSearchParams();
        if (tab)                           params.set('tab', tab);
        if (providers && providers.length) params.set('p', providers.join(','));
        if (quality > 0)                   params.set('q', quality);
        const qs = params.toString();
        history.replaceState(null, '', qs ? '?' + qs : window.location.pathname);
        return window.location.href;
    }
    """,
    Output("url-sync", "data"),   # dummy store — side effect is the URL update
    Input("tabs",            "value"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    prevent_initial_call=True,
)


# ── Chart update callbacks ────────────────────────────────────────────────────
@callback(
    Output("pareto-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    prevent_initial_call=True,
)
def update_pareto(providers, min_quality):
    return build_pareto_scatter(_apply_filters(providers, min_quality))


@callback(
    Output("quadrant-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    prevent_initial_call=True,
)
def update_quadrant(providers, min_quality):
    return build_quadrant(_apply_filters(providers, min_quality))


@callback(
    Output("value-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    prevent_initial_call=True,
)
def update_value(providers, min_quality):
    return build_value_chart(_apply_filters(providers, min_quality))


@callback(
    Output("treemap-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    prevent_initial_call=True,
)
def update_treemap(providers, min_quality):
    return build_treemap(_apply_filters(providers, min_quality))


@callback(
    Output("rankings-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality",  "value"),
    prevent_initial_call=True,
)
def update_rankings(providers, min_quality):
    filtered = _apply_filters(providers, min_quality)
    return build_rankings(filtered, top_n=min(25, len(filtered)))


@callback(
    Output("radar-chart",        "figure"),
    Output("radar-model-select", "options"),
    Output("radar-model-select", "value"),
    Input("filter-provider",     "value"),
    Input("filter-quality",      "value"),
    Input("radar-model-select",  "value"),
    prevent_initial_call=True,
)
def update_compare(providers, min_quality, selected_models):
    filtered  = _apply_filters(providers, min_quality)
    options   = _model_options(filtered)
    triggered = ctx.triggered_id

    if triggered in ("filter-provider", "filter-quality"):
        # Re-default to top-5 of filtered dataset
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
    Input("budget-tokens",    "value"),
    Input("filter-provider",  "value"),
    Input("filter-quality",   "value"),
    prevent_initial_call=True,
)
def update_cost_calc(monthly_tokens_m, providers, min_quality):
    filtered = _apply_filters(providers, min_quality)
    tokens   = float(monthly_tokens_m) if monthly_tokens_m else 1.0
    return build_cost_calc(filtered, monthly_tokens_m=tokens)


# ── Model detail panel ────────────────────────────────────────────────────────
@callback(
    Output("detail-panel",      "className"),
    Output("detail-panel-body", "children"),
    Input("pareto-chart",   "clickData"),
    Input("quadrant-chart", "clickData"),
    Input("detail-close",   "n_clicks"),
    prevent_initial_call=True,
)
def toggle_detail_panel(pareto_click, quadrant_click, _close):
    trigger = ctx.triggered_id

    if trigger == "detail-close":
        return "detail-panel", []

    click = pareto_click if trigger == "pareto-chart" else quadrant_click
    if not click or not click.get("points"):
        return no_update, no_update

    pt         = click["points"][0]
    customdata = pt.get("customdata", [])
    if not customdata or len(customdata) < 2:
        return no_update, no_update

    model_name = customdata[0]
    provider   = customdata[1]

    rows = df[df["model"] == model_name]
    if rows.empty:
        return no_update, no_update
    row = rows.iloc[0]

    color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)

    def _metric(label, value, accent=False):
        return html.Div([
            html.Span(label, className="detail-metric-label"),
            html.Span(value, className=f"detail-metric-value{' accent' if accent else ''}"),
        ], className="detail-metric")

    speed_val   = row["speed"]
    speed_str   = f"{int(speed_val):,} tok/s" if pd.notna(speed_val) and speed_val > 0 else "N/A"
    latency_val = row.get("latency", float("nan"))
    latency_str = f"{latency_val:.2f}s" if pd.notna(latency_val) and latency_val > 0 else "N/A"

    body = [
        html.Div(provider,   className="detail-panel-provider", style={"color": color}),
        html.Div(model_name, className="detail-panel-model"),
        html.Div(className="detail-panel-divider"),
        _metric("Intelligence", f"{row['quality']:.0f}",             accent=True),
        _metric("Price",        f"${row['price']:.4f} / 1M tokens"),
        _metric("Speed",        speed_str),
        _metric("Latency",      latency_str),
        _metric("Context",      str(row.get("context", "N/A")) or "N/A"),
    ]

    return "detail-panel open", body


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(debug=debug, port=8050)
