"""
Image generation charts.

build_image_faceted  — 3-column rankings: Photorealistic / Artistic / Text & Type
build_image_rankings — full ELO rankings bar (all models, secondary view)
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.charts.constants import BG, GRID, TICK, AXIS, FONT, empty_figure, plot_text
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


# Phone layout (layout.meta.narrow, applied by docs/app.js below this width):
# three facets side by side left each ~70 px, so the facets stack instead.
NARROW_PX = 600
_NARROW_ROW_PX = 20          # per model row in a stacked facet
_NARROW_GAP_PX = 64          # x ticks of one facet + the title of the next
_DOT_PX = 10


def _shared_elo_range(ref_df: pd.DataFrame, col_cols: list, col_dfs: list) -> list[float]:
    """One x range for all three facets.

    Each facet used to take its own range from its own column, so the same Elo
    sat at a different horizontal position in each column (600-1300 in one,
    660-1300 in the next) and a reader comparing across facets compared
    nothing. The range is the union of what the unfiltered arena puts in each
    facet's top _TOP_N (so a filter cannot rescale it) and of the dots actually
    drawn (so a filter that surfaces a low-ranked model still shows it) — the
    video ranked view's rule. The right side carries headroom for the price
    label beside each dot.
    """
    market, shown = [], []
    for col, cdf in zip(col_cols, col_dfs):
        if col and col in ref_df.columns:
            market.append(pd.to_numeric(ref_df[col], errors="coerce").dropna()
                          .sort_values(ascending=False).head(_TOP_N))
        if not cdf.empty:
            shown.append(pd.to_numeric(cdf["_elo_display"], errors="coerce").dropna())
    market = [p for p in market if not p.empty]
    shown = [p for p in shown if not p.empty]
    ref = pd.concat(market) if market else (pd.concat(shown) if shown
                                           else pd.Series([1000.0, 1300.0]))
    lo, hi = float(ref.min()), float(ref.max())
    if not hi > lo:
        lo, hi = lo - 20, hi + 20
    # Padding is sized from the MARKET's span, so a filter that extends the
    # range downward to show its own dot does not also move the top.
    span = hi - lo
    lo, hi = lo - span * 0.06, hi + span * 0.30
    if shown:
        drawn = pd.concat(shown)
        lo = min(lo, float(drawn.min()) - span * 0.06)
        hi = max(hi, float(drawn.max()) + span * 0.30)
    return [lo, hi]


def build_image_faceted(df: pd.DataFrame, full_df: pd.DataFrame | None = None) -> go.Figure:
    """
    3-column Cleveland dot plot: one column per style category, ranked by that
    category's arena ELO, all three on ONE shared x range.

    Uses per-category ELO from live AA arena data when available.
    Falls back to tag-based filtering + global ELO if category columns missing.

    Dots, not bars: ELO has no meaningful zero (see video_chart's docstring),
    and the bars here started at ~600, so their length overstated every gap.
    Position on a shared interval axis is the honest encoding. Hollow dot =
    open weights, as everywhere else on the site.

    ``full_df`` is the unfiltered image catalogue. Each facet picks the first
    candidate column that carries data, and every category keeps a *retired*
    2025 column as its fallback — so choosing from the filtered frame let a
    single-provider filter silently switch a facet to a different metric, and
    models scored only on the current column disappeared entirely.
    """
    if df.empty:
        # Reachable: most provider x tag filter combinations select nothing,
        # and this used to raise KeyError: 'elo' out of the fallback branch.
        return empty_figure("No image models match these filters")

    ref_df = df if full_df is None or full_df.empty else full_df

    col_dfs = []
    col_cols = []   # the ELO column each facet ended up using, for the axis range
    for cat in _CATEGORIES:
        elo_col = _pick_elo_column(ref_df, cat["elo_cols"])
        if elo_col and elo_col not in df.columns:
            elo_col = None
        col_cols.append(elo_col)
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
    x_range = _shared_elo_range(ref_df, col_cols, col_dfs)
    any_open = False

    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=False,
        horizontal_spacing=0.14,
        subplot_titles=[c["label"] for c in _CATEGORIES],
    )

    # Price labels for all three facets, collected as plain dicts and appended
    # in one assignment after the loop: one fig.add_annotation per row
    # re-validates the whole annotations tuple on every call, which makes the
    # build quadratic in rows.
    annotations = []
    for col_idx, (cat, cdf) in enumerate(zip(_CATEGORIES, col_dfs), start=1):
        if cdf.empty:
            continue

        short_name = cdf["model"].apply(
            lambda n: plot_text(n[:22] + "…" if len(n) > 22 else n))
        colors = cdf["provider"].map(PROVIDER_COLORS).fillna(DEFAULT_COLOR).tolist()
        elo_vals = cdf["_elo_display"]
        has_gen_time = "gen_time_s" in cdf.columns and cdf["gen_time_s"].gt(0).any()
        is_open = cdf["open_weights"].eq(True)
        any_open = any_open or bool(is_open.any())

        price_display = cdf["price_per_1k"].apply(_price_label)
        fig.add_trace(go.Scatter(
            y=short_name,
            x=elo_vals,
            mode="markers",
            marker=dict(
                color=colors, size=_DOT_PX, opacity=0.95,
                symbol=["circle-open" if o else "circle" for o in is_open],
                line=dict(color=colors, width=[2 if o else 0 for o in is_open]),
            ),
            customdata=list(zip(
                cdf["model"].map(plot_text), cdf["provider"].map(plot_text),
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

        # Price beside each dot (no gen time in live data).
        for idx, row in cdf.iterrows():
            color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
            price_str = _price_label(row["price_per_1k"], short=True)
            ann_text = (
                f"{row['gen_time_s']:.0f}s  ·  {price_str}"
                if has_gen_time and row.get("gen_time_s", 0) > 0
                else price_str
            )
            annotations.append(dict(
                x=row["_elo_display"],
                y=short_name[idx],
                text=ann_text,
                showarrow=False,
                xanchor="left",
                xshift=_DOT_PX - 2,
                font=dict(size=10, family=FONT, color=color),
                xref=f"x{col_idx}" if col_idx > 1 else "x",
                yref=f"y{col_idx}" if col_idx > 1 else "y",
            ))

    # Subplot titles: make_subplots placed them centred over each column;
    # restyle those only (the price labels above are appended after).
    _title_labels = {c["label"] for c in _CATEGORIES}
    titles = [a for a in fig.layout.annotations if a.text in _title_labels]
    for a in titles:
        a.update(font=dict(color="#f2f2f2", size=14, family=FONT), y=1.04)
    fig.layout.annotations = tuple(titles) + tuple(annotations)

    height = max(480, n_rows * 32 + 120)

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(
            text=("Image Models by Style"
                  "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                  f"  ·  arena ELO within each category, one shared scale"
                  f"  ·  top {_TOP_N} per column"
                  + ("  ·  hollow = open weights" if any_open else "")
                  + "</span>"),
            font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left", pad=dict(l=20, t=16),
        ),
        margin=dict(l=10, r=10, t=96, b=20),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
        ),
    )

    for i in range(1, 4):
        xref = f"xaxis{i}" if i > 1 else "xaxis"
        yref = f"yaxis{i}" if i > 1 else "yaxis"
        fig.layout[xref].update(
            range=x_range,
            gridcolor=GRID, zeroline=False,
            tickfont=dict(color=TICK, size=11, family=FONT),
            showgrid=True, showline=False, ticks="",
        )
        fig.layout[yref].update(
            tickfont=dict(color="#aaaaaa", size=11, family=FONT),
            # The leader line from label to dot, as on the video ranking.
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            showline=False, ticks="",
            automargin=True,
        )

    fig.update_layout(meta=_narrow_meta(col_dfs, titles, annotations))
    return fig


def _narrow_meta(col_dfs: list, titles: list, annotations: list) -> dict:
    """Stack the three facets on a phone: each takes the full width, one under
    the other, with its title above it and its x ticks below."""
    heights = [max(len(c), 1) * _NARROW_ROW_PX for c in col_dfs]
    plot_h = sum(heights) + _NARROW_GAP_PX * (len(heights) - 1)
    top_m, bottom_m = 64, 28
    layout: dict = {"height": plot_h + top_m + bottom_m,
                    "margin": {"t": top_m, "b": bottom_m, "l": 8, "r": 8}}
    y = plot_h
    title_ann = []
    for i, (h, t) in enumerate(zip(heights, titles), start=1):
        dom = [(y - h) / plot_h, y / plot_h]
        sfx = "" if i == 1 else str(i)
        layout[f"xaxis{sfx}"] = {"domain": [0, 1]}
        layout[f"yaxis{sfx}"] = {"domain": dom,
                                 "tickfont": {"color": "#aaaaaa", "size": 10, "family": FONT}}
        title_ann.append(dict(t.to_plotly_json(), x=0, xanchor="left", xref="paper",
                              y=dom[1], yref="paper", yanchor="bottom", yshift=6,
                              font=dict(color="#f2f2f2", size=12, family=FONT)))
        y -= h + _NARROW_GAP_PX
    small = [dict(a, font=dict(a["font"], size=9)) for a in annotations]
    layout["annotations"] = title_ann + small
    return {"narrow": {"max_width": NARROW_PX, "layout": layout}}


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

    # Collected as plain dicts and set once in update_layout below: one
    # fig.add_annotation per row re-validates the whole annotations tuple on
    # every call, which makes the build quadratic in rows (2 s for 161 rows).
    annotations = []
    for i, row in plot_df.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        price_str = _price_label(row["price_per_1k"], short=True)
        ann_text = (
            f"{row['gen_time_s']:.0f}s  ·  {price_str}"
            if has_gen_time and row.get("gen_time_s", 0) > 0
            else price_str
        )
        annotations.append(dict(
            x=max_elo + 2,
            y=short_name[i],
            text=ann_text,
            showarrow=False, xanchor="left",
            font=dict(size=10, family=FONT, color=color),
            xref="x", yref="y",
        ))

    height = max(400, len(plot_df) * 22 + 80)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(
            text=(
                "All Models — Ranked by Quality"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                f"  ·  ELO from AA Image Arena blind comparisons  ·  {len(plot_df)} models</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        annotations=annotations,
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
