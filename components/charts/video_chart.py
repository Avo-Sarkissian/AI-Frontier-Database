"""
Video generation: quality rankings with speed and price annotations.
Primary: horizontal bar ranked by quality score.
Secondary: price vs quality scatter with gen-time as bubble size.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    BG, GRID, TICK, AXIS, FONT, right_gutter, fit_text, ANNOTATED_AXIS_HEADROOM,
)
from data.video_models import PROVIDER_COLORS, DEFAULT_COLOR


def build_video_rankings(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar ranked by quality, annotated with price/sec and gen time."""
    plot_df = df.sort_values("quality", ascending=True).reset_index(drop=True)

    short_name = plot_df["model"].apply(lambda n: n[:34] + "…" if len(n) > 34 else n)
    colors = plot_df["provider"].map(PROVIDER_COLORS).fillna(DEFAULT_COLOR).tolist()
    max_q  = plot_df["quality"].max() or 1

    fig = go.Figure()

    # Ghost track
    fig.add_trace(go.Bar(
        y=short_name, x=[max_q] * len(plot_df),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))

    # Quality bars
    fig.add_trace(go.Bar(
        y=short_name,
        x=plot_df["quality"],
        orientation="h",
        marker=dict(color=colors, opacity=0.82, line=dict(width=0)),
        customdata=list(zip(
            plot_df["model"], plot_df["provider"],
            plot_df["quality"], plot_df["price_per_sec"],
            plot_df["gen_time_s"], plot_df["max_res"],
            plot_df["max_duration_s"],
            plot_df["open_weights"].map({True: "Yes", False: "No"}),
            plot_df["tags_str"],
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "Quality: %{customdata[2]}/100<br>"
            "Price: $%{customdata[3]:.3f}/sec of video<br>"
            "Gen time: ~%{customdata[4]:.0f}s<br>"
            "Max resolution: %{customdata[5]}<br>"
            "Max duration: %{customdata[6]}s<br>"
            "Open weights: %{customdata[7]}<br>"
            "Tags: %{customdata[8]}<br>"
            "<extra></extra>"
        ),
        showlegend=False,
        text=plot_df["quality"].apply(lambda q: f"{q:.0f}"),
        textposition="inside",
        textfont=dict(color="rgba(255,255,255,0.55)", size=11, family=FONT),
    ))

    _gutter = right_gutter(
        f"${p:.3f}/sec  ·  {g:.0f}s gen  ·  {r}  ·  open"
        for p, g, r in zip(plot_df["price_per_sec"], plot_df["gen_time_s"], plot_df["max_res"])
    )
    # Right-side annotations
    for i, row in plot_df.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        ow_tag = "  ·  open" if row["open_weights"] else ""
        fig.add_annotation(
            x=1.01,
            y=short_name[i],
            text=fit_text(
                f"${row['price_per_sec']:.3f}/sec  ·  {row['gen_time_s']:.0f}s gen"
                f"  ·  {row['max_res']}{ow_tag}", _gutter
            ),
            showarrow=False, xanchor="left",
            font=dict(size=10, family=FONT, color=color),
            xref="paper", yref="y",
        )

    height = max(380, len(plot_df) * 24 + 80)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Video Generation — Ranked by Quality"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  human preference score 0–100  ·  annotations: price/sec · gen time · resolution</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Quality Score (0–100)", font=dict(color=AXIS, size=12), standoff=12),
            # From 0, because bar length is the ONLY encoding here. The axis
            # started at 40 under a title that says 0–100: with data spanning
            # 58–93 that made SVD:Veo3 read as (58-40)/(93-40) = 0.34 of the
            # leader when the truthful ratio is 58/93 = 0.62 — 1.8x overstated,
            # in the default view of the tab, with no filtering required.
            # A truncated baseline is legitimate for dots or lines; for bars it
            # is a lie about proportion.
            range=[0, max_q * ANNOTATED_AXIS_HEADROOM],
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=TICK, size=11, family=FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            tickfont=dict(color="#999999", size=11, family=FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        ),
        barmode="overlay",
        margin=dict(l=20, r=_gutter, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
        ),
    )

    return fig


def build_video_scatter(df: pd.DataFrame, full_df: pd.DataFrame | None = None) -> go.Figure:
    """Price/sec vs quality scatter. Bubble size = speed (larger = faster).

    ``full_df`` is the unfiltered video catalogue; the frontier is a claim about
    the whole market, so filtering must be able to hide a frontier point but
    never promote one the market already dominates.
    """
    plot_df = df[(df["price_per_sec"] > 0) & (df["quality"] > 0)].copy()
    ref_df = plot_df if full_df is None else \
        full_df[(full_df["price_per_sec"] > 0) & (full_df["quality"] > 0)].copy()

    max_t = plot_df["gen_time_s"].replace(0, np.nan).max() or 1
    plot_df["size"] = plot_df["gen_time_s"].apply(
        lambda t: 8 + (1 - t / max_t) * 28 if pd.notna(t) and t > 0 else 8
    )

    fig = go.Figure()

    for provider in sorted(plot_df["provider"].unique()):
        pdf = plot_df[plot_df["provider"] == provider]
        color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)

        fig.add_trace(go.Scatter(
            x=pdf["price_per_sec"],
            y=pdf["quality"],
            mode="markers",
            name=provider,
            marker=dict(color=color, size=pdf["size"], opacity=0.82, line=dict(width=0)),
            customdata=list(zip(
                pdf["model"], pdf["provider"],
                pdf["quality"], pdf["price_per_sec"],
                pdf["gen_time_s"], pdf["max_res"],
                pdf["open_weights"].map({True: "Yes", False: "No"}),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Provider: %{customdata[1]}<br>"
                "Quality: %{customdata[2]}/100<br>"
                "Price: $%{customdata[3]:.3f}/sec<br>"
                "Gen time: ~%{customdata[4]:.0f}s<br>"
                "Max resolution: %{customdata[5]}<br>"
                "Open weights: %{customdata[6]}<br>"
                "<extra></extra>"
            ),
        ))

    # Pareto frontier
    _add_pareto(fig, plot_df, ref_df)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Cost vs. Quality"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  bubble size = speed (larger = faster to generate)</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Price (USD / second of video)", font=dict(color=AXIS, size=12), standoff=12),
            type="log",
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=TICK, size=11, family=FONT),
            tickformat="$~g",
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            title=dict(text="Quality Score (0–100)", font=dict(color=AXIS, size=12), standoff=12),
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=TICK, size=11, family=FONT),
            showgrid=True, showline=False, ticks="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1,
            font=dict(color="#999999", size=11, family=FONT),
            x=1.01, y=1, xanchor="left",
        ),
        margin=dict(l=20, r=110, t=52, b=36),
        height=520,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
        ),
    )

    return fig


def _add_pareto(fig: go.Figure, df: pd.DataFrame, ref_df: pd.DataFrame | None = None):
    """Draw the frontier of ``ref_df`` (the whole market), clipped to ``df``.

    Testing dominance only within the plotted frame meant a two-provider filter
    could put a model on the frontier that a model 2.5x cheaper at identical
    quality already dominated, under a legend that says only "Pareto Frontier".
    """
    ref = df if ref_df is None or ref_df.empty else ref_df
    frontier_models = []
    for _, row in ref.iterrows():
        dominated = ref[
            (ref["quality"] >= row["quality"]) &
            (ref["price_per_sec"] <= row["price_per_sec"]) &
            ~((ref["quality"] == row["quality"]) & (ref["price_per_sec"] == row["price_per_sec"]))
        ]
        if dominated.empty:
            frontier_models.append(row["model"])
    pareto = [row for _, row in df.iterrows() if row["model"] in set(frontier_models)]
    if not pareto:
        return
    pf = pd.DataFrame(pareto).sort_values("price_per_sec")
    fig.add_trace(go.Scatter(
        x=pf["price_per_sec"], y=pf["quality"],
        mode="lines",
        name="Pareto Frontier",
        line=dict(color="rgba(0,212,255,0.45)", width=1, dash="dot"),
        hoverinfo="skip", showlegend=True,
    ))
    spaced, pos = [], []
    last = -math.inf
    for _, r in pf.iterrows():
        lp = math.log10(r["price_per_sec"])
        if lp - last >= 0.35:
            spaced.append(r)
            pos.append("top center" if len(spaced) % 2 == 1 else "bottom center")
            last = lp
    if spaced:
        ld = pd.DataFrame(spaced)
        fig.add_trace(go.Scatter(
            x=ld["price_per_sec"], y=ld["quality"],
            mode="text",
            text=ld["model"].apply(lambda m: m[:20] + "…" if len(m) > 20 else m),
            textposition=pos,
            textfont=dict(color="rgba(0,212,255,0.65)", size=10, family=FONT),
            hoverinfo="skip", showlegend=False,
        ))
