"""
Image generation charts.

build_image_faceted  — 3-column rankings: Photorealistic / Artistic / Text & Type
build_image_rankings — full ELO rankings bar (all models, secondary view)
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.charts.constants import BG, GRID, TICK, AXIS, FONT
from data.image_models import PROVIDER_COLORS, DEFAULT_COLOR

_CATEGORIES = [
    {"key": "photorealistic", "label": "Photorealistic",    "accent": "#60a5fa"},
    {"key": "artistic",       "label": "Artistic",          "accent": "#c084fc"},
    {"key": "text",           "label": "Text & Typography", "accent": "#f472b6"},
]

_TOP_N = 12  # models shown per column


def build_image_faceted(df: pd.DataFrame) -> go.Figure:
    """
    3-column horizontal bar chart: one column per style category.
    Each column shows the top _TOP_N models for that style, ranked by ELO.
    Bars are colored by provider. Right annotations show gen time.
    """
    # Pre-filter per category (best models at the top → sort ascending for plotly)
    col_dfs = []
    for cat in _CATEGORIES:
        cdf = df[df["tags"].apply(lambda t: cat["key"] in t)] \
                .sort_values("elo", ascending=False) \
                .head(_TOP_N) \
                .sort_values("elo", ascending=True) \
                .reset_index(drop=True)
        col_dfs.append(cdf)

    n_rows = max(len(cdf) for cdf in col_dfs)

    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=False,
        horizontal_spacing=0.06,
        subplot_titles=[c["label"] for c in _CATEGORIES],
    )

    for col_idx, (cat, cdf) in enumerate(zip(_CATEGORIES, col_dfs), start=1):
        if cdf.empty:
            continue

        short_name = cdf["model"].apply(lambda n: n[:28] + "…" if len(n) > 28 else n)
        colors = cdf["provider"].map(PROVIDER_COLORS).fillna(DEFAULT_COLOR).tolist()
        max_elo = cdf["elo"].max() or 1
        xaxis_key = f"xaxis{col_idx}" if col_idx > 1 else "xaxis"

        # Ghost track for visual alignment
        fig.add_trace(go.Bar(
            y=short_name, x=[max_elo] * len(cdf),
            orientation="h",
            marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=col_idx)

        # Quality bars
        fig.add_trace(go.Bar(
            y=short_name,
            x=cdf["elo"],
            orientation="h",
            marker=dict(color=colors, opacity=0.82, line=dict(width=0)),
            customdata=list(zip(
                cdf["model"], cdf["provider"],
                cdf["elo"], cdf["price_per_1k"], cdf["gen_time_s"],
                cdf["open_weights"].map({True: "Yes", False: "No"}),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Provider: %{customdata[1]}<br>"
                "ELO: %{customdata[2]}<br>"
                "Price: $%{customdata[3]:.1f}/1k images<br>"
                "Gen time: %{customdata[4]:.0f}s<br>"
                "Open weights: %{customdata[5]}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ), row=1, col=col_idx)

        # Gen-time annotations on the right of each bar
        for _, row in cdf.iterrows():
            color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
            fig.add_annotation(
                x=row["elo"] + (max_elo * 0.015),
                y=short_name[_],
                text=f"{row['gen_time_s']:.0f}s",
                showarrow=False,
                xanchor="left",
                font=dict(size=8, family=FONT, color=color),
                xref=f"x{col_idx}" if col_idx > 1 else "x",
                yref=f"y{col_idx}" if col_idx > 1 else "y",
            )

    height = max(380, n_rows * 26 + 100)

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, color="#888888", size=11),
        barmode="overlay",
        margin=dict(l=10, r=10, t=60, b=20),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
        ),
    )

    # Style each subplot's axes
    for i in range(1, 4):
        xref = f"xaxis{i}" if i > 1 else "xaxis"
        yref = f"yaxis{i}" if i > 1 else "yaxis"
        all_elos = col_dfs[i - 1]["elo"] if not col_dfs[i - 1].empty else pd.Series([1000, 1300])
        x_min = max(900, all_elos.min() - 30)
        x_max = all_elos.max() + (all_elos.max() - x_min) * 0.18

        fig.layout[xref].update(
            range=[x_min, x_max],
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=TICK, size=9, family=FONT),
            showgrid=True, showline=False, ticks="",
        )
        fig.layout[yref].update(
            tickfont=dict(color="#aaaaaa", size=9, family=FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        )

    # Style subplot title text
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(color="#f2f2f2", size=13, family=FONT), y=1.04)

    return fig


def build_image_rankings(df: pd.DataFrame) -> go.Figure:
    """Full ELO rankings — all models, horizontal bars, annotated with speed + price."""
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
            "Price: $%{customdata[3]:.1f}/1k images<br>"
            "Gen time: %{customdata[4]:.0f}s<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    for i, row in plot_df.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        fig.add_annotation(
            x=max_elo + 2,
            y=short_name[i],
            text=f"{row['gen_time_s']:.0f}s  ·  ${row['price_per_1k']:.0f}/1k",
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
                "All Models — Ranked by Quality"
                "  <span style='font-size:11px;color:#666666;font-weight:400'>"
                "  ·  ELO from blind human comparisons  ·  annotations: gen time · price</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="ELO Score", font=dict(color=AXIS, size=11), standoff=12),
            range=[min(plot_df["elo"]) - 20, max_elo * 1.25],
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
        margin=dict(l=20, r=200, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
        ),
    )

    return fig
