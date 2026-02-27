"""
Price-over-time line chart.

Shows how API pricing ($/1M tokens) has evolved across snapshots for the
top N models. Complements the quality-over-time chart in the Trends tab,
telling the story of AI becoming simultaneously smarter and cheaper.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)

_TOP_N = 15


def build_price_timeline(history_df: pd.DataFrame) -> go.Figure:
    """
    Line chart of price/M tokens over time for top N models.
    Falls back to a ranked bar if only one snapshot exists.
    """
    if history_df.empty:
        return _empty("No snapshot data found.")

    h = history_df.copy()
    h["scraped_at"] = pd.to_datetime(h["scraped_at"])
    h = h.sort_values("scraped_at")
    h = h[(h["price"] > 0) & (h["quality"] > 0)]

    snapshots = h["scraped_at"].nunique()
    if snapshots < 2:
        return _single_snapshot(h)

    # Track models present in the most recent snapshot
    latest_date = h["scraped_at"].max()
    latest = h[h["scraped_at"] == latest_date]
    tracked = (
        latest
        .sort_values("quality", ascending=False)
        .head(_TOP_N)["model"]
        .tolist()
    )

    fig = go.Figure()

    for model in tracked:
        mdf = h[h["model"] == model].sort_values("scraped_at")
        if mdf.empty:
            continue
        provider = mdf.iloc[-1].get("provider", "")
        color    = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)

        fig.add_trace(go.Scatter(
            x=mdf["scraped_at"],
            y=mdf["price"],
            mode="lines+markers",
            name=model[:28] + ("…" if len(model) > 28 else ""),
            line=dict(color=color, width=1.5),
            marker=dict(color=color, size=5),
            hovertemplate=(
                f"<b>{model}</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Price: $%{y:.4f}/M tokens<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Price Over Time"
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
            title=dict(text="Price / 1M tokens (USD)", font=dict(color=_AXIS, size=11), standoff=12),
            type="log",
            tickprefix="$",
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


def _single_snapshot(h: pd.DataFrame) -> go.Figure:
    snap_date = h["scraped_at"].iloc[0]
    date_str  = pd.Timestamp(snap_date).strftime("%Y-%m-%d") if hasattr(snap_date, "strftime") else str(snap_date)[:10]

    top = (
        h[h["price"] > 0]
        .sort_values("price")
        .head(_TOP_N)
        .iloc[::-1]
        .reset_index(drop=True)
    )
    colors = [PROVIDER_COLORS.get(p, DEFAULT_COLOR) for p in top["provider"]]
    short  = top["model"].apply(lambda m: m[:28] + "…" if len(m) > 28 else m)

    fig = go.Figure(go.Bar(
        y=short,
        x=top["price"],
        orientation="h",
        marker=dict(color=colors, opacity=0.7, line=dict(width=0)),
        customdata=top[["model", "provider", "price"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "Price: $%{customdata[2]:.4f}/M tokens<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Price Baseline"
                f"  <span style='font-size:11px;color:#666666;font-weight:400'>"
                f"  ·  snapshot {date_str}  ·  cheapest {_TOP_N} models</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left", pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Price / 1M tokens (USD)", font=dict(color=_AXIS, size=11), standoff=12),
            gridcolor=_GRID, tickprefix="$",
            tickfont=dict(color=_TICK, size=10, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            tickfont=dict(color="#888888", size=10, family=_FONT),
            showgrid=False, showline=False, ticks="", automargin=True,
        ),
        margin=dict(l=20, r=40, t=52, b=36),
        height=460,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )
    return fig


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888"),
        annotations=[dict(x=0.5, y=0.5, xref="paper", yref="paper",
                          text=msg, showarrow=False,
                          font=dict(color="#777777", size=13, family=_FONT))],
        margin=dict(l=40, r=40, t=60, b=40), height=400,
    )
    return fig
