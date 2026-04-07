"""
Frontier Price Tracker.

Shows how the median API price of the top-10 intelligence models has
evolved across daily snapshots — demonstrating AI price compression.
Individual model price lines are drawn with a bold median trend line.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)

_TOP_N = 10  # top models by intelligence to track


def build_bump_chart(history_df: pd.DataFrame) -> go.Figure:
    """
    Price tracker: median price trend of top-N models over time,
    plus individual model lines showing the price compression story.
    """
    if history_df.empty:
        return _empty("No snapshot data available.")

    h = history_df.copy()
    h["scraped_at"] = pd.to_datetime(h["scraped_at"]).dt.date
    h = h[(h["quality"] > 0) & (h["price"] > 0)]

    dates = sorted(h["scraped_at"].unique())
    if len(dates) < 2:
        return _empty("Need at least 2 daily snapshots.")

    # Track models in the top _TOP_N by quality on the latest snapshot
    latest = h[h["scraped_at"] == dates[-1]]
    tracked = (
        latest.sort_values("quality", ascending=False)
        .head(_TOP_N)["model"]
        .tolist()
    )
    prov_map = latest.set_index("model")["provider"].to_dict()

    date_strs = [d.strftime("%b %d") for d in dates]

    fig = go.Figure()

    # --- Per-model price lines (thin, semi-transparent) ---
    for model_name in tracked:
        mdf = h[h["model"] == model_name].sort_values("scraped_at")
        if mdf.empty:
            continue

        model_dates = [d.strftime("%b %d") for d in mdf["scraped_at"]]
        provider = prov_map.get(model_name, "")
        color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
        short = model_name[:26] + ("…" if len(model_name) > 26 else "")

        fig.add_trace(go.Scatter(
            x=model_dates,
            y=mdf["price"],
            mode="lines",
            name=short,
            line=dict(width=1.2, color=color),
            opacity=0.4,
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                "Date: %{x}<br>"
                "Price: $%{y:.4f}/M tokens<br>"
                f"Provider: {provider}<br>"
                "<extra></extra>"
            ),
            showlegend=True,
        ))

    # --- Median price trend of the top-N (bold overlay) ---
    median_prices = []
    for dt in dates:
        day = h[h["scraped_at"] == dt]
        top_day = day[day["model"].isin(tracked)]
        if not top_day.empty:
            median_prices.append(top_day["price"].median())
        else:
            median_prices.append(None)

    fig.add_trace(go.Scatter(
        x=date_strs,
        y=median_prices,
        mode="lines+markers",
        name=f"Median (top {_TOP_N})",
        line=dict(width=3, color="#00d4ff"),
        marker=dict(size=6, color="#00d4ff"),
        hovertemplate=(
            "<b>Median price (top 10)</b><br>"
            "Date: %{x}<br>"
            "Median: $%{y:.4f}/M tokens<br>"
            "<extra></extra>"
        ),
    ))

    # --- Compute % change annotation ---
    valid_medians = [m for m in median_prices if m is not None]
    if len(valid_medians) >= 2:
        pct_change = (valid_medians[-1] - valid_medians[0]) / valid_medians[0] * 100
        direction = "↓" if pct_change < 0 else "↑"
        fig.add_annotation(
            x=date_strs[-1], y=valid_medians[-1],
            text=f"  {direction} {abs(pct_change):.0f}%",
            showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(
                color="#00d4ff" if pct_change < 0 else "#ff6b6b",
                size=13, family=_FONT, weight=700,
            ),
        )

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                f"Frontier Price Tracker  —  Top {_TOP_N} Models"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  bold line = median price  ·  thinner lines = individual models</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Snapshot Date", font=dict(color=_AXIS, size=12), standoff=12),
            gridcolor=_GRID,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            title=dict(text="Price  (USD / 1M tokens)", font=dict(color=_AXIS, size=12), standoff=12),
            type="log",
            gridcolor=_GRID,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            tickprefix="$",
            showgrid=True, showline=False, ticks="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(color="#999999", size=10, family=_FONT),
            x=1.01, y=1, xanchor="left",
        ),
        margin=dict(l=56, r=240, t=52, b=52),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
        height=560,
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
