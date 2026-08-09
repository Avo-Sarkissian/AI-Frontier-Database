"""
Image generation charts.

build_image_faceted  — 3-column rankings: Photorealistic / Artistic / Text & Type
build_image_rankings — full ELO rankings bar (all models, secondary view)
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.charts.constants import BG, GRID, TICK, AXIS, FONT, empty_figure
from data.image_models import PROVIDER_COLORS, DEFAULT_COLOR

# Artificial Analysis migrated its arena to a new category taxonomy, and the
# retired categories are no longer scored for new models: of the current top 12
# by global ELO, only 1-2 carry an elo_general_photorealistic /
# elo_cartoon_illustration / elo_text_typography value. Ranking on those columns
# therefore filtered out every current flagship — GPT Image 2, Reve 2.1, all the
# Nano Banana models — and the tab quietly showed a 2025-era leaderboard.
#
# Each column now names its successor first and keeps the retired column as a
# fallback, so an older cached CSV (or an upstream revert) still renders.
_CATEGORIES = [
    {"key": "photorealistic", "label": "Photorealistic",         "accent": "#60a5fa",
     "elo_cols": ["elo_live_action_film", "elo_general_photorealistic"]},
    {"key": "artistic",       "label": "Illustration & Animation", "accent": "#c084fc",
     "elo_cols": ["elo_animation_gaming", "elo_cartoon_illustration"]},
    {"key": "text",           "label": "Text Rendering",          "accent": "#f472b6",
     "elo_cols": ["elo_text_rendering", "elo_text_typography"]},
]

_TOP_N = 12  # models shown per column


def _pick_elo_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """First candidate column that actually carries values in this frame."""
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _price_label(value, short: bool = False) -> str:
    """Price per 1k images.

    data/image_models.py fills a missing price with 0.0, so "no published
    price" and "genuinely free" are indistinguishable downstream — printing
    "free" for both asserted a commercial fact the data does not support.
    """
    if pd.isna(value) or value <= 0:
        return "n/a"
    return f"${value:.0f}/1k" if short else f"${value:.1f}/1k"


def build_image_faceted(df: pd.DataFrame) -> go.Figure:
    """
    3-column horizontal bar chart: one column per style category.
    Uses per-category ELO from live AA arena data when available.
    Falls back to tag-based filtering + global ELO if category columns missing.
    """
    if df.empty:
        # Reachable: most provider x tag filter combinations select nothing,
        # and this used to raise KeyError: 'elo' out of the fallback branch.
        return empty_figure("No image models match these filters")

    col_dfs = []
    for cat in _CATEGORIES:
        elo_col = _pick_elo_column(df, cat["elo_cols"])
        if elo_col:
            # Live data: rank by category-specific ELO
            cdf = df[df[elo_col].notna()].copy() \
                    .sort_values(elo_col, ascending=False) \
                    .head(_TOP_N) \
                    .sort_values(elo_col, ascending=True) \
                    .reset_index(drop=True)
            cdf["_elo_display"] = cdf[elo_col]
        else:
            # Static fallback: filter by tag, rank by global ELO
            tags_col = df["tags"]
            mask = tags_col.apply(
                lambda t: (isinstance(t, list) and cat["key"] in t)
                          or (isinstance(t, str) and cat["key"] in t)
            )
            cdf = df[mask] \
                    .sort_values("elo", ascending=False) \
                    .head(_TOP_N) \
                    .sort_values("elo", ascending=True) \
                    .reset_index(drop=True)
            cdf["_elo_display"] = cdf["elo"] if "elo" in cdf.columns else 0
        col_dfs.append(cdf)

    n_rows = max(len(cdf) for cdf in col_dfs)

    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=False,
        horizontal_spacing=0.14,
        subplot_titles=[c["label"] for c in _CATEGORIES],
    )

    for col_idx, (cat, cdf) in enumerate(zip(_CATEGORIES, col_dfs), start=1):
        if cdf.empty:
            continue

        short_name = cdf["model"].apply(lambda n: n[:22] + "…" if len(n) > 22 else n)
        colors = cdf["provider"].map(PROVIDER_COLORS).fillna(DEFAULT_COLOR).tolist()
        elo_vals = cdf["_elo_display"]
        max_elo = elo_vals.max() or 1
        has_gen_time = "gen_time_s" in cdf.columns and cdf["gen_time_s"].gt(0).any()

        # Ghost track for visual alignment
        fig.add_trace(go.Bar(
            y=short_name, x=[max_elo] * len(cdf),
            orientation="h",
            marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=col_idx)

        price_display = cdf["price_per_1k"].apply(_price_label)
        # Quality bars
        fig.add_trace(go.Bar(
            y=short_name,
            x=elo_vals,
            orientation="h",
            marker=dict(color=colors, opacity=0.82, line=dict(width=0)),
            customdata=list(zip(
                cdf["model"], cdf["provider"],
                elo_vals, price_display,
                cdf["open_weights"].map({True: "Yes", False: "No"}),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Provider: %{customdata[1]}<br>"
                "ELO: %{customdata[2]:.0f}<br>"
                "Price: %{customdata[3]}<br>"
                "Open weights: %{customdata[4]}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ), row=1, col=col_idx)

        # Right-side annotations: price (no gen time in live data)
        for idx, row in cdf.iterrows():
            color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
            price_str = _price_label(row["price_per_1k"], short=True)
            ann_text = (
                f"{row['gen_time_s']:.0f}s  ·  {price_str}"
                if has_gen_time and row.get("gen_time_s", 0) > 0
                else price_str
            )
            fig.add_annotation(
                x=row["_elo_display"] + (max_elo * 0.015),
                y=short_name[idx],
                text=ann_text,
                showarrow=False,
                xanchor="left",
                font=dict(size=10, family=FONT, color=color),
                xref=f"x{col_idx}" if col_idx > 1 else "x",
                yref=f"y{col_idx}" if col_idx > 1 else "y",
            )

    height = max(480, n_rows * 32 + 120)

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(
            text=("Image Models by Style"
                  "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                  "  ·  arena ELO within each category  ·  top 12 per column</span>"),
            font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left", pad=dict(l=20, t=16),
        ),
        barmode="overlay",
        margin=dict(l=10, r=10, t=96, b=20),
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
        cdf_i = col_dfs[i - 1]
        elo_series = cdf_i["_elo_display"] if not cdf_i.empty else pd.Series([1000, 1300])
        x_min = max(900, elo_series.min() - 30)
        x_max = elo_series.max() + (elo_series.max() - x_min) * 0.32

        fig.layout[xref].update(
            range=[x_min, x_max],
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color=TICK, size=11, family=FONT),
            showgrid=True, showline=False, ticks="",
        )
        fig.layout[yref].update(
            tickfont=dict(color="#aaaaaa", size=11, family=FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        )

    # Style subplot title text only — skip the price annotations we added above,
    # otherwise they all get y-forced to 1.04 and pile up on one row.
    _title_labels = {c["label"] for c in _CATEGORIES}
    for annotation in fig.layout.annotations:
        if annotation.text in _title_labels:
            annotation.update(font=dict(color="#f2f2f2", size=14, family=FONT), y=1.04)

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

    has_gen_time = "gen_time_s" in plot_df.columns and plot_df["gen_time_s"].gt(0).any()

    fig.add_trace(go.Bar(
        y=short_name,
        x=plot_df["elo"],
        orientation="h",
        marker=dict(color=colors, opacity=0.80, line=dict(width=0)),
        customdata=list(zip(
            plot_df["model"], plot_df["provider"],
            plot_df["elo"], plot_df["price_per_1k"],
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "ELO: %{customdata[2]:.0f}<br>"
            "Price: $%{customdata[3]:.1f}/1k images<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    for i, row in plot_df.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        price_str = _price_label(row["price_per_1k"], short=True)
        ann_text = (
            f"{row['gen_time_s']:.0f}s  ·  {price_str}"
            if has_gen_time and row.get("gen_time_s", 0) > 0
            else price_str
        )
        fig.add_annotation(
            x=max_elo + 2,
            y=short_name[i],
            text=ann_text,
            showarrow=False, xanchor="left",
            font=dict(size=10, family=FONT, color=color),
            xref="x", yref="y",
        )

    height = max(400, len(plot_df) * 22 + 80)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(
            text=(
                "All Models — Ranked by Quality"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  ELO from AA Image Arena blind comparisons  ·  119 models</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="ELO Score", font=dict(color=AXIS, size=12), standoff=12),
            range=[min(plot_df["elo"]) - 20, max_elo * 1.25],
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
        margin=dict(l=20, r=200, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
        ),
    )

    return fig
