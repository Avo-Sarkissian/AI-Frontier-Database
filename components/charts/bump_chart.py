"""
Frontier Price Change Tracker.

Shows how API pricing has changed (as % from first snapshot) for the top
intelligence models — making price compression immediately legible.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)

_TOP_N = 10


def build_bump_chart(history_df: pd.DataFrame) -> go.Figure:
    """
    % change in price from first snapshot for top-N models by quality.
    Lines below 0% = cheaper. Bold cyan = median of tracked models.
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

    # Zero reference line
    fig.add_hline(
        y=0,
        line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
    )

    # Per-model % change lines
    model_pct_by_date: dict[str, list] = {}  # for median computation

    for model_name in tracked:
        mdf = h[h["model"] == model_name].sort_values("scraped_at")
        if mdf.empty or len(mdf) < 2:
            continue

        base_price = mdf.iloc[0]["price"]
        if base_price <= 0:
            continue

        pct_series = ((mdf["price"] - base_price) / base_price * 100).tolist()
        model_dates = [d.strftime("%b %d") for d in mdf["scraped_at"]]

        # Store for median
        for d_str, pct in zip(model_dates, pct_series):
            model_pct_by_date.setdefault(d_str, []).append(pct)

        provider = prov_map.get(model_name, "")
        color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
        short = model_name[:26] + ("…" if len(model_name) > 26 else "")
        final_pct = pct_series[-1]

        fig.add_trace(go.Scatter(
            x=model_dates,
            y=pct_series,
            mode="lines",
            name=short,
            line=dict(width=1.5, color=color),
            opacity=0.55,
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                "Date: %{x}<br>"
                "Change: %{y:+.1f}%<br>"
                f"Provider: {provider}<br>"
                "<extra></extra>"
            ),
        ))

        # End-label for notable movers
        if abs(final_pct) >= 5:
            fig.add_annotation(
                x=model_dates[-1], y=final_pct,
                text=f" {final_pct:+.0f}%",
                showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(
                    color="#22c55e" if final_pct < 0 else "#f87171",
                    size=10, family=_FONT,
                ),
            )

    # Median % change line (bold cyan)
    median_pcts = []
    for d_str in date_strs:
        vals = model_pct_by_date.get(d_str, [])
        median_pcts.append(float(np.median(vals)) if vals else None)

    fig.add_trace(go.Scatter(
        x=date_strs,
        y=median_pcts,
        mode="lines+markers",
        name=f"Median (top {_TOP_N})",
        line=dict(width=3, color="#00d4ff"),
        marker=dict(size=6, color="#00d4ff"),
        hovertemplate=(
            "<b>Median (top 10)</b><br>"
            "Date: %{x}<br>"
            "Change: %{y:+.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    # Final % annotation on median
    valid_m = [m for m in median_pcts if m is not None]
    if valid_m:
        final_m = valid_m[-1]
        direction = "↓" if final_m < 0 else "↑"
        fig.add_annotation(
            x=date_strs[-1], y=final_m,
            text=f"  {direction} {abs(final_m):.0f}%",
            showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(
                color="#00d4ff" if final_m < 0 else "#ff6b6b",
                size=13, family=_FONT, weight=700,
            ),
        )

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                f"Price Change Since First Snapshot  —  Top {_TOP_N} by Intelligence"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  bold line = median  ·  below 0% = cheaper</span>"
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
            title=dict(text="Price Change (%)", font=dict(color=_AXIS, size=12), standoff=12),
            ticksuffix="%",
            gridcolor=_GRID,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(color="#999999", size=10, family=_FONT),
            x=1.01, y=1, xanchor="left",
        ),
        margin=dict(l=56, r=200, t=52, b=52),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
        height=540,
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
