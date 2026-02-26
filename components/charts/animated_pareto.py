"""
Animated Pareto frontier chart.

Animates the cost/quality scatter over every saved snapshot.
- ≥ 2 snapshots: full Plotly-frames animation with play/pause + time slider.
- 1 snapshot   : static scatter with pareto frontier line drawn.
- 0 snapshots  : empty-state card.

The Pareto frontier (dotted cyan line) traces models that are strictly
optimal — no other model is both cheaper AND smarter.
"""
import math
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)

_MIN_FOR_ANIM = 2


# ── Pareto helpers ────────────────────────────────────────────────────────────
def _pareto_frontier(sub: pd.DataFrame) -> pd.DataFrame:
    """Return Pareto-optimal rows, sorted by price ascending."""
    sub = sub[(sub["price"] > 0) & (sub["quality"] > 0)].sort_values("price")
    pareto, max_q = [], 0.0
    for _, row in sub.iterrows():
        if row["quality"] > max_q:
            pareto.append(row)
            max_q = float(row["quality"])
    return pd.DataFrame(pareto) if pareto else pd.DataFrame(columns=sub.columns)


def _bubble_sizes(speed_series: pd.Series) -> list[float]:
    return [max(6, math.sqrt(max(s, 1)) * 2.2) for s in speed_series.fillna(1)]


# ── Per-frame trace builders ──────────────────────────────────────────────────
def _scatter_trace(day_df: pd.DataFrame) -> go.Scatter:
    colors = [PROVIDER_COLORS.get(p, DEFAULT_COLOR) for p in day_df["provider"]]
    return go.Scatter(
        x=day_df["price"],
        y=day_df["quality"],
        mode="markers",
        marker=dict(
            size=_bubble_sizes(day_df["speed"]),
            color=colors,
            opacity=0.82,
            line=dict(width=0.5, color="rgba(255,255,255,0.10)"),
        ),
        customdata=day_df[["model", "provider", "speed"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "Price: $%{x:.4f}/M tokens<br>"
            "Intelligence: %{y:.0f}<br>"
            "Speed: %{customdata[2]:.0f} tok/s<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    )


def _frontier_trace(pareto_df: pd.DataFrame) -> go.Scatter:
    if pareto_df.empty:
        return go.Scatter(x=[], y=[], mode="lines", showlegend=False, hoverinfo="skip")
    return go.Scatter(
        x=pareto_df["price"],
        y=pareto_df["quality"],
        mode="lines+markers",
        line=dict(color="rgba(0,212,255,0.50)", width=1.8, dash="dot"),
        marker=dict(color="rgba(0,212,255,0.70)", size=7,
                    line=dict(width=0, color="rgba(0,0,0,0)")),
        showlegend=False,
        hoverinfo="skip",
    )


# ── Public API ────────────────────────────────────────────────────────────────
def build_animated_pareto(history_df: pd.DataFrame) -> go.Figure:
    """Animated pareto frontier over all saved daily snapshots."""
    if history_df.empty:
        return _empty("No snapshot data available.")

    h = history_df.copy()
    h["scraped_at"] = pd.to_datetime(h["scraped_at"])
    h = h[(h["price"] > 0) & (h["quality"] > 0)].sort_values("scraped_at")

    dates = sorted(h["scraped_at"].unique())

    if len(dates) < _MIN_FOR_ANIM:
        return _static(h[h["scraped_at"] == dates[-1]])

    # ── Build animation frames ─────────────────────────────────────────────
    frames = []
    for dt in dates:
        day  = h[h["scraped_at"] == dt]
        par  = _pareto_frontier(day)
        dstr = pd.Timestamp(dt).strftime("%Y-%m-%d")
        frames.append(go.Frame(
            data=[_scatter_trace(day), _frontier_trace(par)],
            name=dstr,
            layout=go.Layout(title_text=(
                "AI Frontier Evolution"
                f"  <span style='font-size:11px;color:#3d3d3d;font-weight:400'>"
                f"  ·  {dstr}  ·  {len(day)} models</span>"
            )),
        ))

    latest  = h[h["scraped_at"] == dates[-1]]
    par_lat = _pareto_frontier(latest)

    fig = go.Figure(
        data=[_scatter_trace(latest), _frontier_trace(par_lat)],
        frames=frames,
    )

    # Slider steps
    steps = [
        {"args": [[f.name],
                  {"frame": {"duration": 600, "redraw": True},
                   "mode": "immediate",
                   "transition": {"duration": 200}}],
         "label": f.name,
         "method": "animate"}
        for f in frames
    ]

    _apply_layout(fig, len(dates), animated=True)
    fig.layout.sliders = [dict(
        active=len(dates) - 1,
        currentvalue=dict(
            prefix="Snapshot: ",
            font=dict(color="#555555", size=10, family=_FONT),
        ),
        pad=dict(b=10, t=50),
        x=0.10, len=0.88,
        bgcolor="rgba(255,255,255,0.02)",
        bordercolor="rgba(255,255,255,0.06)",
        tickcolor="rgba(255,255,255,0.06)",
        font=dict(color="#444444", size=9, family=_FONT),
        steps=steps,
    )]
    fig.layout.updatemenus = [dict(
        type="buttons",
        showactive=False,
        x=0.0, y=-0.13,
        xanchor="left", yanchor="top",
        bgcolor="#161616",
        bordercolor="rgba(255,255,255,0.07)",
        font=dict(color="#777777", size=11, family=_FONT),
        buttons=[
            dict(
                label="▶  Play",
                method="animate",
                args=[None, {"frame": {"duration": 900, "redraw": True},
                             "fromcurrent": True,
                             "transition": {"duration": 300, "easing": "linear"}}],
            ),
            dict(
                label="⏸",
                method="animate",
                args=[[None], {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate",
                               "transition": {"duration": 0}}],
            ),
        ],
    )]
    return fig


# ── Fallbacks ────────────────────────────────────────────────────────────────
def _static(day_df: pd.DataFrame) -> go.Figure:
    pareto  = _pareto_frontier(day_df)
    dstr    = pd.Timestamp(day_df["scraped_at"].iloc[0]).strftime("%Y-%m-%d")
    n_days  = 1

    fig = go.Figure(data=[_scatter_trace(day_df), _frontier_trace(pareto)])
    _apply_layout(fig, n_days, animated=False, date_str=dstr)
    return fig


def _apply_layout(
    fig: go.Figure,
    n_snapshots: int,
    animated: bool,
    date_str: str = "",
) -> None:
    subtitle = (
        f"  ·  {n_snapshots} snapshots  ·  dotted line = Pareto frontier"
        if animated else
        f"  ·  snapshot {date_str}  ·  animation unlocks once multiple snapshots exist"
    )
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                "AI Frontier Evolution"
                f"  <span style='font-size:11px;color:#3d3d3d;font-weight:400'>{subtitle}</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Price / 1M tokens (USD)", font=dict(color=_AXIS, size=11), standoff=12),
            type="log",
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            tickprefix="$",
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=11), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        margin=dict(l=56, r=40, t=52, b=130 if animated else 52),
        height=580,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#555555"),
        annotations=[dict(x=0.5, y=0.5, xref="paper", yref="paper",
                          text=msg, showarrow=False,
                          font=dict(color="#444444", size=13, family=_FONT))],
        margin=dict(l=40, r=40, t=60, b=40), height=400,
    )
    return fig
