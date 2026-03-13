"""
Image generation: Cost vs Quality scatter.
X = price per 1,000 images (log scale)
Y = ELO quality score
Bubble size = speed (inverted gen time — larger = faster)
Color = provider
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import BG, GRID, TICK, AXIS, FONT
from data.image_models import PROVIDER_COLORS, DEFAULT_COLOR


def _pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    pareto = []
    for _, row in df.iterrows():
        dominated = df[
            (df["elo"] >= row["elo"]) &
            (df["price_per_1k"] <= row["price_per_1k"]) &
            ~((df["elo"] == row["elo"]) & (df["price_per_1k"] == row["price_per_1k"]))
        ]
        if dominated.empty:
            pareto.append(row)
    return pd.DataFrame(pareto).sort_values("price_per_1k")


def build_image_scatter(df: pd.DataFrame) -> go.Figure:
    plot_df = df[(df["price_per_1k"] > 0) & (df["elo"] > 0)].copy()

    # Bubble size: inversely proportional to gen_time (faster = bigger)
    max_t = plot_df["gen_time_s"].replace(0, np.nan).max() or 1
    plot_df["size"] = plot_df["gen_time_s"].apply(
        lambda t: 8 + (1 - t / max_t) * 28 if pd.notna(t) and t > 0 else 8
    )

    fig = go.Figure()

    for provider in sorted(plot_df["provider"].unique()):
        pdf = plot_df[plot_df["provider"] == provider]
        color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)

        fig.add_trace(go.Scatter(
            x=pdf["price_per_1k"],
            y=pdf["elo"],
            mode="markers",
            name=provider,
            marker=dict(
                color=color,
                size=pdf["size"],
                opacity=0.80,
                line=dict(width=0),
            ),
            customdata=list(zip(
                pdf["model"], pdf["provider"],
                pdf["elo"],
                pdf["price_per_1k"],
                pdf["gen_time_s"],
                pdf["tags_str"],
                pdf["open_weights"].map({True: "Yes", False: "No"}),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Provider: %{customdata[1]}<br>"
                "ELO Quality: %{customdata[2]}<br>"
                "Price: $%{customdata[3]:.1f} / 1k images<br>"
                "Gen time: %{customdata[4]:.0f}s<br>"
                "Tags: %{customdata[5]}<br>"
                "Open weights: %{customdata[6]}<br>"
                "<extra></extra>"
            ),
        ))

    # Pareto frontier
    pareto_df = _pareto_frontier(plot_df)
    if not pareto_df.empty:
        fig.add_trace(go.Scatter(
            x=pareto_df["price_per_1k"],
            y=pareto_df["elo"],
            mode="lines",
            name="Pareto Frontier",
            line=dict(color="rgba(0,212,255,0.45)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=True,
        ))

        # Spaced pareto labels
        spaced_rows, spaced_pos = [], []
        last_log_price = -math.inf
        for _, prow in pareto_df.iterrows():
            log_p = math.log10(prow["price_per_1k"])
            if log_p - last_log_price >= 0.4:
                spaced_rows.append(prow)
                spaced_pos.append("top center" if len(spaced_rows) % 2 == 1 else "bottom center")
                last_log_price = log_p
        if spaced_rows:
            label_df = pd.DataFrame(spaced_rows)
            fig.add_trace(go.Scatter(
                x=label_df["price_per_1k"],
                y=label_df["elo"],
                mode="text",
                text=label_df["model"].apply(lambda m: m[:22] + "…" if len(m) > 22 else m),
                textposition=spaced_pos,
                textfont=dict(color="rgba(0,212,255,0.65)", size=9, family=FONT),
                hoverinfo="skip",
                showlegend=False,
            ))

    _zero = "rgba(255,255,255,0.06)"

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Cost vs. Quality"
                "  <span style='font-size:11px;color:#666666;font-weight:400'>"
                "  ·  bubble size = speed (larger = faster)</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Price  (USD / 1,000 images)", font=dict(color=AXIS, size=11), standoff=12),
            type="log",
            gridcolor=GRID,
            zerolinecolor=_zero, zerolinewidth=1,
            tickfont=dict(color=TICK, size=10, family=FONT),
            tickformat="$~g",
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            title=dict(text="ELO Quality Score", font=dict(color=AXIS, size=11), standoff=12),
            gridcolor=GRID,
            zerolinecolor=_zero, zerolinewidth=1,
            tickfont=dict(color=TICK, size=10, family=FONT),
            showgrid=True, showline=False, ticks="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1,
            font=dict(color="#888888", size=11, family=FONT),
            orientation="v", x=1.01, y=1, xanchor="left",
        ),
        margin=dict(l=20, r=160, t=52, b=36),
        height=600,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
        ),
    )

    return fig


def build_image_rankings(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart ranked by ELO, annotated with price."""
    plot_df = df.sort_values("elo", ascending=True).reset_index(drop=True)

    short_name = plot_df["model"].apply(lambda n: n[:34] + "…" if len(n) > 34 else n)
    colors = plot_df["provider"].map(PROVIDER_COLORS).fillna(DEFAULT_COLOR).tolist()
    max_elo = plot_df["elo"].max() or 1

    fig = go.Figure()

    # Ghost track
    fig.add_trace(go.Bar(
        y=short_name, x=[max_elo] * len(plot_df),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))

    fig.add_trace(go.Bar(
        y=short_name,
        x=plot_df["elo"],
        orientation="h",
        marker=dict(color=colors, opacity=0.80, line=dict(width=0)),
        customdata=list(zip(
            plot_df["model"], plot_df["provider"],
            plot_df["elo"], plot_df["price_per_1k"], plot_df["gen_time_s"],
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "ELO: %{customdata[2]}<br>"
            "Price: $%{customdata[3]:.1f} / 1k images<br>"
            "Gen time: %{customdata[4]:.0f}s<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # Right-side annotations: price + gen time
    for i, row in plot_df.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        fig.add_annotation(
            x=max_elo + 2,
            y=short_name[i],
            text=f"${row['price_per_1k']:.0f}/1k  ·  {row['gen_time_s']:.0f}s",
            showarrow=False, xanchor="left",
            font=dict(size=9, family=FONT, color=color),
            xref="x", yref="y",
        )

    height = max(400, len(plot_df) * 22 + 80)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Rankings by Quality"
                "  <span style='font-size:11px;color:#666666;font-weight:400'>"
                "  ·  ELO score from blind human comparisons</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="ELO Score", font=dict(color=AXIS, size=11), standoff=12),
            range=[min(plot_df["elo"]) - 20, max_elo * 1.30],
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=TICK, size=10, family=FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            tickfont=dict(color="#888888", size=10, family=FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        ),
        barmode="overlay",
        margin=dict(l=20, r=180, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
        ),
    )

    return fig
