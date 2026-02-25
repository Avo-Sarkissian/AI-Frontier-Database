"""
AI Frontier — Main Dash Application
Interactive dashboard comparing 100+ LLMs on cost, speed, and quality.
"""
import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd

from data.ingest import get_models
from components.charts.pareto import build_pareto_scatter

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
        _stat(str(len(df)),                         "Models tracked"),
        _stat(str(df["provider"].nunique()),         "Providers"),
        _stat(f"${df['price'].min():.3f}",           "Floor price / 1M", accent=True),
        _stat(str(int(df["quality"].max())),         "Peak intelligence"),
        _stat(f"{int(df['speed'].replace(0, pd.NA).max()):,}", "Max speed tok/s"),
    ], className="stat-bar"),

    # ── Filters
    html.Div([
        html.Span("PROVIDER", className="filter-label"),
        dcc.Dropdown(
            id="filter-provider",
            options=_provider_options(df),
            multi=True,
            placeholder="All providers",
            style={"minWidth": "280px"},
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

    # ── Chart
    html.Div([
        dcc.Graph(
            id="pareto-chart",
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"],
            },
            style={"height": "600px"},
        ),
    ], className="chart-card"),

    # ── Footer
    html.Div([
        html.Span("Source: Artificial Analysis · Updated live"),
        html.Span("AI Frontier"),
    ], className="footer"),

], style={"minHeight": "100vh", "background": "#0a0a0a"})


# ── Callbacks ────────────────────────────────────────────────────────────────
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


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
