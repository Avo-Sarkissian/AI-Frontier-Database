"""
Intelligence trend chart.
Shows how the top models' quality scores have evolved over time
as daily snapshots accumulate.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)

_TOP_N = 15   # track this many models in the trend view


def build_trends(history_df: pd.DataFrame) -> go.Figure:
    """
    Line chart of quality score over time for the top N models.
    If only one snapshot exists, renders a 'baseline established' state.
    """
    if history_df.empty:
        return _empty_state("No snapshot data found.")

    history_df = history_df.copy()
    history_df["scraped_at"] = pd.to_datetime(history_df["scraped_at"])
    history_df = history_df.sort_values("scraped_at")

    snapshots = history_df["scraped_at"].nunique()

    if snapshots < 2:
        return _single_snapshot(history_df)

    # Identify models that appear in the most recent snapshot
    latest_date = history_df["scraped_at"].max()
    latest = history_df[history_df["scraped_at"] == latest_date]
    top_models = (
        latest[latest["quality"] > 0]
        .sort_values("quality", ascending=False)
        .head(_TOP_N)["model"]
        .tolist()
    )

    fig = go.Figure()

    for model in top_models:
        mdf = history_df[history_df["model"] == model].sort_values("scraped_at")
        if mdf.empty:
            continue
        provider = mdf.iloc[-1].get("provider", "")
        color    = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)

        fig.add_trace(go.Scatter(
            x=mdf["scraped_at"],
            y=mdf["quality"],
            mode="lines+markers",
            name=model[:28] + ("…" if len(model) > 28 else ""),
            line=dict(color=color, width=1.5),
            marker=dict(color=color, size=5),
            hovertemplate=(
                f"<b>{model}</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Intelligence: %{y:.0f}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Intelligence Over Time"
                "  <span style='font-size:11px;color:#666666;font-weight:400'>"
                f"  ·  top {_TOP_N} models  ·  {snapshots} snapshots</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Date", font=dict(color=_AXIS, size=11), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=11), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
            font=dict(color="#888888", size=9, family=_FONT),
            x=1.01, y=1, xanchor="left",
        ),
        margin=dict(l=56, r=160, t=52, b=52),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )

    return fig


def _single_snapshot(history_df: pd.DataFrame) -> go.Figure:
    """
    When only one day of data exists, show a ranked bar chart as the baseline,
    with a message that trends will appear as snapshots accumulate.
    """
    snap = history_df.copy()
    snap_date = snap["scraped_at"].iloc[0]
    if hasattr(snap_date, "strftime"):
        date_str = snap_date.strftime("%Y-%m-%d")
    else:
        date_str = str(snap_date)[:10]

    top = (
        snap[snap["quality"] > 0]
        .sort_values("quality", ascending=False)
        .head(_TOP_N)
        .iloc[::-1]
        .reset_index(drop=True)
    )
    colors = [PROVIDER_COLORS.get(p, DEFAULT_COLOR) for p in top["provider"]]
    short  = top["model"].apply(lambda m: m[:28] + "…" if len(m) > 28 else m)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=short,
        x=top["quality"],
        orientation="h",
        marker=dict(color=colors, opacity=0.7, line=dict(width=0)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Intelligence: %{x:.0f}<br>"
            "Provider: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
        customdata=top[["model", "provider"]].values,
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Intelligence Baseline"
                f"  <span style='font-size:11px;color:#666666;font-weight:400'>"
                f"  ·  snapshot {date_str}  ·  trends appear as data accumulates</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=11), standoff=12),
            gridcolor=_GRID,
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            tickfont=dict(color="#888888", size=10, family=_FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        ),
        margin=dict(l=20, r=40, t=52, b=36),
        height=500,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
        annotations=[dict(
            x=0.5, y=-0.12, xref="paper", yref="paper",
            text=(
                "📅  A new snapshot is saved each day data is refreshed. "
                "Trend lines will appear once multiple snapshots exist."
            ),
            showarrow=False,
            font=dict(color="#777777", size=10, family=_FONT),
            align="center",
        )],
    )

    return fig


def _empty_state(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        annotations=[dict(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=msg, showarrow=False,
            font=dict(color="#777777", size=13, family=_FONT),
        )],
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    return fig
