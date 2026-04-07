"""
Rank evolution bump chart.

Shows how the top models' intelligence rankings have shifted over the
daily snapshot window. Each line is one model; rank 1 = highest quality.
Rising lines = improving rank, falling lines = losing ground.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)

_TOP_N = 12   # models to track


def build_bump_chart(history_df: pd.DataFrame) -> go.Figure:
    """
    Bump chart: intelligence rank over time for the top _TOP_N models
    (ranked by quality on the most recent snapshot).
    """
    if history_df.empty:
        return _empty("No snapshot data available — bump chart requires daily history.")

    h = history_df.copy()
    h["scraped_at"] = pd.to_datetime(h["scraped_at"]).dt.date
    h = h[(h["quality"] > 0) & (h["price"] > 0)]

    dates = sorted(h["scraped_at"].unique())
    if len(dates) < 2:
        return _empty("Need at least 2 daily snapshots to show rank evolution.")

    # Models to track: top _TOP_N by quality on the latest date
    latest   = h[h["scraped_at"] == dates[-1]]
    tracked  = (
        latest.sort_values("quality", ascending=False)
        .head(_TOP_N)["model"]
        .tolist()
    )

    # Build a provider lookup from the latest snapshot
    prov_map = latest.set_index("model")["provider"].to_dict()

    # For each date, compute rank of every model (quality descending)
    rank_records: dict[str, dict] = {m: {} for m in tracked}
    for dt in dates:
        day = h[h["scraped_at"] == dt].copy()
        day = day.sort_values("quality", ascending=False).reset_index(drop=True)
        day["rank"] = day.index + 1
        for model in tracked:
            row = day[day["model"] == model]
            rank_records[model][dt] = int(row["rank"].iloc[0]) if not row.empty else None

    date_strs = [d.strftime("%b %d") for d in dates]
    date_isos  = [d.isoformat() for d in dates]

    fig = go.Figure()

    for model_name in tracked:
        ranks = [rank_records[model_name].get(dt) for dt in dates]

        # Skip models that never appeared
        if all(r is None for r in ranks):
            continue

        provider = prov_map.get(model_name, "")
        color    = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
        short    = model_name[:26] + ("…" if len(model_name) > 26 else "")

        # Final rank for annotation
        final_rank = ranks[-1]

        fig.add_trace(go.Scatter(
            x=date_strs,
            y=ranks,
            mode="lines+markers",
            name=short,
            line=dict(width=1.8, color=color),
            marker=dict(
                size=7,
                color=color,
                line=dict(width=1, color="rgba(0,0,0,0.4)"),
            ),
            connectgaps=False,
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                "Date: %{x}<br>"
                "Rank: #%{y}<br>"
                f"Provider: {provider}<br>"
                "<extra></extra>"
            ),
        ))

        # Annotate the final rank at the right edge
        if final_rank is not None:
            fig.add_annotation(
                x=date_strs[-1],
                y=final_rank,
                text=f"<span style='color:{color}'>{short}</span>",
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(size=10, family=_FONT, color=color),
            )

    # Y-axis: rank 1 at top.
    # Cap at _TOP_N + 4 so early-snapshot outliers (rank 30+) don't stretch
    # the axis and turn most of the chart into empty whitespace.
    all_ranks = [
        r for m in tracked for r in rank_records[m].values() if r is not None
    ]
    y_max = min(max(all_ranks) + 1, _TOP_N + 4) if all_ranks else _TOP_N + 2

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                f"Intelligence Rank Evolution  —  Top {_TOP_N} Models"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  rank 1 = highest AA Intelligence Index  ·  rising line = improving</span>"
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
            title=dict(text="Rank  (1 = best)", font=dict(color=_AXIS, size=12), standoff=12),
            range=[y_max, 0.5],   # reversed: low rank numbers (best) at top
            dtick=1,
            gridcolor=_GRID,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
            tickprefix="#",
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
