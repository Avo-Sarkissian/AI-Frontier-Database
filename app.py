"""
AI Frontier — Main Dash Application
Interactive dashboard comparing 100+ LLMs on cost, speed, and quality.
"""
import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd

from data.ingest import get_models
from components.charts.pareto import build_pareto_scatter, PROVIDER_COLORS

# ── Data ────────────────────────────────────────────────────────────────────
df = get_models()   # load from cache

# ── App ─────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="AI Frontier",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # for gunicorn

# ── Helpers ─────────────────────────────────────────────────────────────────
def _stat(value: str, label: str) -> html.Div:
    return html.Div([
        html.Div(value, className="stat-value"),
        html.Div(label, className="stat-label"),
    ], className="stat")


def _provider_options(df: pd.DataFrame) -> list[dict]:
    providers = sorted(df["provider"].unique())
    return [{"label": p, "value": p} for p in providers]


# ── Layout ──────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Header
    html.Div([
        html.H1("AI FRONTIER"),
        html.Span("100+ LLMs · cost · speed · intelligence", className="subtitle"),
    ], className="header"),

    # Stat bar
    html.Div([
        _stat(str(len(df)), "Models"),
        _stat(str(df["provider"].nunique()), "Providers"),
        _stat(f"${df['price'].min():.3f}", "Lowest Price /1M"),
        _stat(str(int(df["quality"].max())), "Top Intelligence Score"),
        _stat(f"{int(df['speed'].max()):,}", "Peak Speed (tok/s)"),
    ], className="stat-bar"),

    # Filters
    html.Div([
        html.Span("Provider", className="filter-label"),
        dcc.Dropdown(
            id="filter-provider",
            options=_provider_options(df),
            multi=True,
            placeholder="All providers",
            style={"minWidth": "260px", "fontSize": "13px"},
            className="dark-select",
        ),
        html.Span("Min Quality", className="filter-label"),
        dcc.Dropdown(
            id="filter-quality",
            options=[{"label": f"≥ {v}", "value": v} for v in [0, 10, 20, 30, 40, 50]],
            value=0,
            clearable=False,
            style={"width": "110px", "fontSize": "13px"},
        ),
    ], className="filters", style={"gap": "12px", "alignItems": "center"}),

    # Chart
    html.Div([
        dcc.Graph(
            id="pareto-chart",
            config={"displayModeBar": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
            style={"height": "580px"},
        ),
    ], className="chart-card"),

    # Footer
    html.Div([
        html.Span("Data: Artificial Analysis · Scraped live"),
        html.Span("AI Frontier · Built with Plotly Dash"),
    ], className="footer"),

], style={"minHeight": "100vh", "background": "#0f1117"})


# ── Callbacks ───────────────────────────────────────────────────────────────
@callback(
    Output("pareto-chart", "figure"),
    Input("filter-provider", "value"),
    Input("filter-quality", "value"),
)
def update_chart(providers: list[str] | None, min_quality: int):
    filtered = df.copy()

    if providers:
        filtered = filtered[filtered["provider"].isin(providers)]

    if min_quality and min_quality > 0:
        filtered = filtered[filtered["quality"] >= min_quality]

    return build_pareto_scatter(filtered)


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
