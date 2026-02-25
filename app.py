"""
AI Frontier — Main Dash Application
Interactive dashboard comparing 100+ LLMs on cost, speed, and quality.
"""
import dash
from dash import dcc, html, Input, Output, callback, no_update
import pandas as pd

from data.ingest import get_models
from components.charts.pareto   import build_pareto_scatter
from components.charts.quadrant import build_quadrant
from components.charts.treemap  import build_treemap
from components.charts.rankings import build_rankings
from components.charts.value    import build_value_chart
from components.charts.radar    import build_radar

# ── Data ────────────────────────────────────────────────────────────────────
df = get_models()

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


def _tab(label: str, value: str) -> dcc.Tab:
    return dcc.Tab(
        label=label,
        value=value,
        className="tab",
        selected_className="tab--selected",
    )


# ── Layout ──────────────────────────────────────────────────────────────────
app.layout = html.Div([

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
        _stat(str(len(df)),                                              "Models tracked"),
        _stat(str(df["provider"].nunique()),                             "Providers"),
        _stat(f"${df['price'].min():.3f}",                              "Floor price / 1M", accent=True),
        _stat(str(int(df["quality"].max())),                             "Peak intelligence"),
        _stat(f"{int(df['speed'].replace(0, pd.NA).max()):,}",          "Max speed tok/s"),
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

    # ── Tab navigation
    dcc.Tabs(
        id="tabs",
        value="overview",
        className="tabs",
        children=[
            _tab("Overview",    "overview"),
            _tab("Performance", "performance"),
            _tab("Value",       "value"),
            _tab("Landscape",   "landscape"),
            _tab("Rankings",    "rankings"),
            _tab("Compare",     "compare"),
        ],
    ),

    # ── Tab content
    html.Div(id="tab-content", className="tab-content"),

    # ── Footer
    html.Div([
        html.Span("Source: Artificial Analysis · Updated live"),
        html.Span("AI Frontier"),
    ], className="footer"),

], style={"minHeight": "100vh", "background": "#0a0a0a"})


# ── Tab content router ────────────────────────────────────────────────────────
@callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("filter-provider", "value"),
    Input("filter-quality", "value"),
)
def render_tab(tab: str, providers: list[str] | None, min_quality: int):
    filtered = df.copy()
    if providers:
        filtered = filtered[filtered["provider"].isin(providers)]
    if min_quality and min_quality > 0:
        filtered = filtered[filtered["quality"] >= min_quality]

    def _chart_card(graph_id: str, height: int = 600) -> html.Div:
        return html.Div([
            dcc.Graph(
                id=graph_id,
                config={"displayModeBar": True, "displaylogo": False,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"]},
                style={"height": f"{height}px"},
            ),
        ], className="chart-card")

    if tab == "overview":
        return html.Div([
            html.Div([
                html.P(
                    "Each bubble is one model. X-axis = cost per 1M tokens (log scale). "
                    "Y-axis = intelligence score. Bubble size = generation speed. "
                    "The dotted line traces the Pareto frontier — models with the best "
                    "quality for their price.",
                    className="chart-desc",
                ),
            ], className="chart-desc-wrap"),
            html.Div([
                dcc.Graph(
                    id="pareto-chart",
                    figure=build_pareto_scatter(filtered),
                    config={"displayModeBar": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"]},
                    style={"height": "620px"},
                ),
            ], className="chart-card"),
        ])

    elif tab == "performance":
        return html.Div([
            html.Div([
                html.P(
                    "Speed vs. intelligence. Models in the top-right are both fast and smart. "
                    "Bubble size is inversely proportional to price — larger = cheaper.",
                    className="chart-desc",
                ),
            ], className="chart-desc-wrap"),
            html.Div([
                dcc.Graph(
                    id="quadrant-chart",
                    figure=build_quadrant(filtered),
                    config={"displayModeBar": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"]},
                    style={"height": "620px"},
                ),
            ], className="chart-card"),
        ])

    elif tab == "value":
        return html.Div([
            html.Div([
                html.P(
                    "Intelligence points earned per dollar spent. "
                    "Models at the top of this chart give you the most cognitive output for your budget.",
                    className="chart-desc",
                ),
            ], className="chart-desc-wrap"),
            html.Div([
                dcc.Graph(
                    id="value-chart",
                    figure=build_value_chart(filtered),
                    config={"displayModeBar": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"]},
                    style={"height": "900px", "overflowY": "auto"},
                ),
            ], className="chart-card"),
        ])

    elif tab == "landscape":
        return html.Div([
            html.Div([
                html.P(
                    "AI ecosystem by provider. Tile area = number of models in the dataset. "
                    "Color intensity = average intelligence score.",
                    className="chart-desc",
                ),
            ], className="chart-desc-wrap"),
            html.Div([
                dcc.Graph(
                    id="treemap-chart",
                    figure=build_treemap(filtered),
                    config={"displayModeBar": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"]},
                    style={"height": "600px"},
                ),
            ], className="chart-card"),
        ])

    elif tab == "rankings":
        top_n = min(25, len(filtered))
        return html.Div([
            html.Div([
                html.P(
                    f"Top {top_n} models ranked by AA Intelligence Index. "
                    "Hover for price and speed details.",
                    className="chart-desc",
                ),
            ], className="chart-desc-wrap"),
            html.Div([
                dcc.Graph(
                    id="rankings-chart",
                    figure=build_rankings(filtered, top_n=top_n),
                    config={"displayModeBar": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"]},
                    style={"height": f"{max(400, top_n * 28)}px"},
                ),
            ], className="chart-card"),
        ])

    elif tab == "compare":
        # Default to top-5 models
        default_models = (
            filtered[filtered["quality"] > 0]
            .sort_values("quality", ascending=False)
            .head(5)["model"]
            .tolist()
        )
        return html.Div([
            html.Div([
                html.P(
                    "Radar chart comparing up to 5 models across 4 normalized dimensions. "
                    "Select models below to compare.",
                    className="chart-desc",
                ),
            ], className="chart-desc-wrap"),
            html.Div([
                html.Span("SELECT MODELS", className="filter-label"),
                dcc.Dropdown(
                    id="radar-model-select",
                    options=_model_options(filtered),
                    value=default_models[:5],
                    multi=True,
                    placeholder="Select up to 5 models…",
                    style={"minWidth": "500px"},
                ),
            ], className="filters", style={"borderTop": "none", "paddingTop": "0"}),
            html.Div([
                dcc.Graph(
                    id="radar-chart",
                    figure=build_radar(filtered, default_models[:5]),
                    config={"displayModeBar": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"]},
                    style={"height": "560px"},
                ),
            ], className="chart-card"),
        ])

    return html.Div("Select a tab.", style={"color": "#555", "padding": "32px"})


# ── Radar model selector callback ─────────────────────────────────────────────
@callback(
    Output("radar-chart", "figure"),
    Input("radar-model-select", "value"),
    Input("filter-provider", "value"),
    Input("filter-quality", "value"),
    prevent_initial_call=True,
)
def update_radar(selected_models, providers, min_quality):
    filtered = df.copy()
    if providers:
        filtered = filtered[filtered["provider"].isin(providers)]
    if min_quality and min_quality > 0:
        filtered = filtered[filtered["quality"] >= min_quality]
    return build_radar(filtered, selected_models or [])


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
