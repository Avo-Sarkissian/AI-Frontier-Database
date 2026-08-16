"""
Video generation charts, over live Artificial Analysis Video Arena Elo.

build_video_rankings — Cleveland dot plot ranked by arena Elo
build_video_scatter  — price per minute vs Elo, with the market's Pareto frontier

WHY THIS IS A DOT PLOT AND NOT A BAR CHART
------------------------------------------
The previous version drew bars, and an audit found its axis starting at 40 under
a title reading "0–100": bar length was the only encoding, so a 58-vs-93 gap
rendered as 0.34 of the leader when the truthful ratio is 0.62. The fix at the
time was to force the baseline to zero, which is right — for a ratio scale.

Elo is not one. It is Bradley-Terry log-odds rescaled for readability, so its
zero is an artefact of that rescaling and "twice the Elo" means nothing. Drawn
from zero, the live spread of 940–1330 puts every model between 71% and 100% of
the axis: 78 bars of visually identical length, which destroys the comparison
the tab exists to make. Drawn from a truncated baseline, bar length would
actively lie about proportion — the exact defect the audit caught.

A dot plot resolves it rather than trading one error for the other. Position
carries the value, and position on a truncated interval scale is legitimate
precisely because no one reads a dot's distance from the axis edge as a
magnitude. The bars are gone, so the zero-baseline rule no longer applies, and
the y-axis gridlines give the eye the leader it needs to travel from label to
dot.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    BG, GRID, TICK, AXIS, FONT, right_gutter, fit_text,
    empty_figure, plot_text, unique_labels, log_ticks,
)
from data.video_models import (
    PROVIDER_COLORS, DEFAULT_COLOR, DEFAULT_MODE, mode_label,
)

# The ranked view shows this many models before it stops. 78 current
# text-to-video models is 1,800px of scrolling, and past the first thirty the
# ranking answers nobody's question. It is a disclosed cap, not a silent one:
# the subtitle prints "top 30 of 78", and any provider or tag filter narrows the
# field below the cap so the models a reader asked for are always all shown.
_TOP_N = 30

# Legend entries in the scatter. The arena carries 31 creators; a vertical
# legend of 31 clips its own plot. Marks keep their provider colour either way —
# only the legend is capped, and the tail is counted rather than hidden.
_MAX_LEGEND_PROVIDERS = 8


def _price_label(value, short: bool = False) -> str:
    """USD per minute of generated video.

    Seven of 89 text-to-video models have no published rate, and data/video_
    scraper._price keeps that null rather than zero-filling it. Printing "free"
    for an unpriced model would assert a commercial fact the source does not
    make — see components/charts/image_scatter.py:43-52, where exactly that
    happened.
    """
    if value is None or pd.isna(value) or value <= 0:
        return "n/a"
    return f"${value:,.2f}/min" if not short else f"${value:,.2f}"


def _clip_cost(value) -> str:
    """What a five-second clip costs, which is the unit anyone actually buys."""
    if value is None or pd.isna(value) or value <= 0:
        return "n/a"
    return f"${value * 5 / 60:,.2f}"


def _gen_time_label(row) -> str:
    secs = row.get("gen_time_s")
    if secs is None or pd.isna(secs) or secs <= 0:
        return ""
    return f"{secs:.0f}s"


def _elo_bounds(ref, *also, fallback=(1000.0, 1330.0)) -> tuple[float, float]:
    """Axis bounds anchored to the unfiltered market, never to the selection.

    Deriving them from what is on screen means dot-to-dot distance silently
    rescales whenever a user narrows a filter, so the same two models appear
    closer or further apart depending on who else is displayed.
    components/charts/image_scatter.py:202-228 records the sharper version of
    this bug: the filtered-frame version produced 38 axes that ran backwards.

    ``also`` carries series that must stay inside the axis whatever the anchor
    says — the plotted values themselves. The ranked view anchors on the
    market's top ``_TOP_N`` rather than on all of it, because a chart showing
    thirty models between 1181 and 1321 should not spend two thirds of its width
    on the 763-Elo tail it is not drawing; unioning the plotted values back in
    means a filter that surfaces a low-ranked model still shows it.
    """
    parts = [pd.to_numeric(s, errors="coerce").dropna()
             for s in (ref, *also) if s is not None]
    series = pd.concat([p for p in parts if not p.empty]) if parts else pd.Series(dtype=float)
    series = series[series.notna()]
    if series.empty:
        lo, hi = fallback
    else:
        lo, hi = float(series.min()), float(series.max())
    if not (hi > lo):
        lo, hi = lo - 20, hi + 20
    pad = (hi - lo) * 0.06
    return lo - pad, hi + pad


def build_video_rankings(df: pd.DataFrame, full_df: pd.DataFrame | None = None,
                         mode: str = DEFAULT_MODE) -> go.Figure:
    """Dot plot ranked by arena Elo, annotated with price and measured speed."""
    if df is None or df.empty:
        # Reachable: tags are AND-ed, so most provider x tag combinations select
        # nothing, and this used to render full axis furniture over empty traces
        # — scaffolding that reads as a broken page.
        return empty_figure("No video models match these filters")

    ref_df = df if full_df is None or full_df.empty else full_df
    total = len(df)
    plot_df = (df.sort_values("elo", ascending=False)
                 .head(_TOP_N)
                 .sort_values("elo", ascending=True)
                 .reset_index(drop=True))

    # unique_labels keeps two models whose names share their first 34 characters
    # from collapsing onto one categorical row, where one of them would be drawn
    # under the other and could never be hovered. AA's video names collide
    # readily — "Kling 3.0 Omni 1080p (Pro)" and friends.
    short_name = unique_labels([
        plot_text(n[:34] + "…" if len(n) > 34 else n) for n in plot_df["model"]
    ])
    colors = plot_df["provider"].map(PROVIDER_COLORS).fillna(DEFAULT_COLOR).tolist()

    ref_top = (pd.to_numeric(ref_df["elo"], errors="coerce")
                 .dropna().sort_values(ascending=False).head(_TOP_N))
    lo, hi = _elo_bounds(ref_top, plot_df["elo"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=short_name,
        x=plot_df["elo"],
        mode="markers+text",
        marker=dict(color=colors, size=10, opacity=0.95, line=dict(width=0)),
        text=[f"{v:.0f}" for v in plot_df["elo"]],
        textposition="middle right",
        textfont=dict(color="#8a8a8a", size=10, family=FONT),
        cliponaxis=False,
        customdata=list(zip(
            plot_df["model"].map(plot_text),
            plot_df["provider"].map(plot_text),
            plot_df["elo"],
            plot_df["price_per_min"].apply(_price_label),
            plot_df["price_per_min"].apply(_clip_cost),
            plot_df.apply(lambda r: _gen_time_label(r) or "not measured", axis=1),
            plot_df["open_weights"].map({True: "Yes", False: "No"}),
            plot_df["audio"].map({True: "Yes", False: "No"}),
            plot_df["tags_str"].map(plot_text),
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "Arena Elo: %{customdata[2]:.0f}<br>"
            "Price: %{customdata[3]}<br>"
            "5s clip: %{customdata[4]}<br>"
            "Generation time: %{customdata[5]}<br>"
            "Open weights: %{customdata[6]}<br>"
            "Generates audio: %{customdata[7]}<br>"
            "Strong at: %{customdata[8]}<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    def _annotation(row) -> str:
        parts = [_price_label(row["price_per_min"])]
        secs = _gen_time_label(row)
        if secs:
            parts.append(secs)
        if row["open_weights"]:
            parts.append("open")
        return "  ·  ".join(parts)

    texts = [_annotation(r) for _, r in plot_df.iterrows()]
    _gutter = right_gutter(texts)
    for i, text in enumerate(texts):
        fig.add_annotation(
            x=1.01, y=short_name[i],
            text=fit_text(text, _gutter),
            showarrow=False, xanchor="left",
            font=dict(size=10, family=FONT,
                      color=PROVIDER_COLORS.get(plot_df.loc[i, "provider"],
                                                DEFAULT_COLOR)),
            xref="paper", yref="y",
        )

    shown = len(plot_df)
    scope = f"top {shown} of {total}" if shown < total else f"all {total}"
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(
            text=(
                f"Video Generation — Ranked by Arena Elo"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                f"  ·  {plot_text(mode_label(mode).lower())}  ·  {scope}"
                "  ·  annotations: price · measured generation time</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Arena Elo", font=dict(color=AXIS, size=12), standoff=12),
            # Truncated deliberately, and legitimately: see the module docstring.
            # Elo has no meaningful zero, and nothing here encodes magnitude by
            # length, so the baseline carries no claim about proportion.
            range=[lo, hi],
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=TICK, size=11, family=FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            tickfont=dict(color="#999999", size=11, family=FONT),
            # The leader line from label to dot. Without it the eye cannot
            # travel 700px of whitespace and land on the right row.
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            showline=False, ticks="", automargin=True,
        ),
        margin=dict(l=20, r=_gutter, t=52, b=36),
        height=max(380, shown * 22 + 90),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT), namelength=-1,
        ),
    )
    return fig


def _legend_names(ref_df: pd.DataFrame) -> list[str]:
    """Providers that get their own legend entry, densest first.

    Ordering is a total order, not value_counts' order: pandas does not define
    how counts tie, and components/charts/constants.spotlight_split records the
    consequence — CI and the browser produced different legend orders from
    identical data, so entries visibly swapped a few seconds after page load.
    """
    counts = ref_df["provider"].value_counts()
    ordered = sorted(counts.index, key=lambda p: (-int(counts[p]), str(p)))
    return ordered[:_MAX_LEGEND_PROVIDERS]


def build_video_scatter(df: pd.DataFrame, full_df: pd.DataFrame | None = None,
                        mode: str = DEFAULT_MODE) -> go.Figure:
    """Price per minute vs arena Elo, with the whole market's Pareto frontier.

    ``full_df`` is the unfiltered catalogue for this mode; the frontier is a
    claim about the market, so filtering must be able to hide a frontier point
    but never promote one the market already dominates.
    """
    if df is None or df.empty:
        return empty_figure("No video models match these filters")

    def _plottable(frame):
        price = pd.to_numeric(frame["price_per_min"], errors="coerce")
        elo = pd.to_numeric(frame["elo"], errors="coerce")
        return frame[(price > 0) & elo.notna()].copy()

    plot_df = _plottable(df)
    if plot_df.empty:
        return empty_figure("No priced video models match these filters")
    ref_df = plot_df if full_df is None or full_df.empty else _plottable(full_df)
    if ref_df.empty:
        ref_df = plot_df

    named = _legend_names(ref_df)
    fig = go.Figure()

    for provider in sorted(plot_df["provider"].unique()):
        pdf = plot_df[plot_df["provider"] == provider]
        fig.add_trace(go.Scatter(
            x=pdf["price_per_min"],
            y=pdf["elo"],
            mode="markers",
            name=plot_text(provider),
            marker=dict(color=PROVIDER_COLORS.get(provider, DEFAULT_COLOR),
                        size=11, opacity=0.85, line=dict(width=0)),
            showlegend=provider in named,
            customdata=list(zip(
                pdf["model"].map(plot_text), pdf["provider"].map(plot_text),
                pdf["elo"],
                pdf["price_per_min"].apply(_price_label),
                pdf["price_per_min"].apply(_clip_cost),
                pdf.apply(lambda r: _gen_time_label(r) or "not measured", axis=1),
                pdf["open_weights"].map({True: "Yes", False: "No"}),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Provider: %{customdata[1]}<br>"
                "Arena Elo: %{customdata[2]:.0f}<br>"
                "Price: %{customdata[3]}<br>"
                "5s clip: %{customdata[4]}<br>"
                "Generation time: %{customdata[5]}<br>"
                "Open weights: %{customdata[6]}<br>"
                "<extra></extra>"
            ),
        ))

    # The tail keeps its colours and its hover; only its legend rows are pooled,
    # and the count says how many were pooled rather than letting them vanish.
    hidden = sorted(set(plot_df["provider"]) - set(named))
    if hidden:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=DEFAULT_COLOR, size=11, opacity=0.85),
            name=f"+{len(hidden)} more studios",
            hoverinfo="skip", showlegend=True,
        ))

    _add_pareto(fig, plot_df, ref_df)

    prices = pd.to_numeric(ref_df["price_per_min"], errors="coerce").dropna()
    tickvals, ticktext = log_ticks(float(prices.min()), float(prices.max()),
                                   fmt=lambda v: f"${v:g}")
    lo, hi = _elo_bounds(ref_df["elo"])

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Cost vs. Arena Elo"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                f"  ·  {plot_text(mode_label(mode).lower())}"
                "  ·  the dotted line is the market's price-quality frontier</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Price (USD / minute of video)",
                       font=dict(color=AXIS, size=12), standoff=12),
            type="log",
            # Explicit 1-2-5 ticks: a bare log axis makes Plotly fall back to
            # "D2" minor ticks, which render as a bare-mantissa run whose
            # leading label reads as a value and misstates the axis by orders of
            # magnitude. See constants.log_ticks.
            tickvals=tickvals, ticktext=ticktext,
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=TICK, size=11, family=FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            title=dict(text="Arena Elo", font=dict(color=AXIS, size=12), standoff=12),
            range=[lo, hi],
            gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=TICK, size=11, family=FONT),
            showgrid=True, showline=False, ticks="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1,
            font=dict(color="#999999", size=11, family=FONT),
            x=1.01, y=1, xanchor="left",
        ),
        margin=dict(l=20, r=150, t=52, b=36),
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
    frontier = []
    for _, row in ref.iterrows():
        dominated = ref[
            (ref["elo"] >= row["elo"]) &
            (ref["price_per_min"] <= row["price_per_min"]) &
            ~((ref["elo"] == row["elo"]) &
              (ref["price_per_min"] == row["price_per_min"]))
        ]
        if dominated.empty:
            frontier.append(row["model"])
    pareto = [row for _, row in df.iterrows() if row["model"] in set(frontier)]
    if not pareto:
        return
    pf = pd.DataFrame(pareto).sort_values("price_per_min")
    fig.add_trace(go.Scatter(
        x=pf["price_per_min"], y=pf["elo"],
        mode="lines",
        name="Pareto Frontier",
        line=dict(color="rgba(0,212,255,0.45)", width=1, dash="dot"),
        hoverinfo="skip", showlegend=True,
    ))

    # Label spacing in log units, sized to the span actually on screen rather
    # than to a constant: per-minute pricing spans about one decade where the
    # old per-second figures spanned 1.7, and the previous fixed 0.35-decade
    # threshold would have left at most three labels on the whole frontier.
    span = math.log10(float(pf["price_per_min"].max()) /
                      float(pf["price_per_min"].min())) if len(pf) > 1 else 0.0
    min_gap = max(span / 6.0, 0.04) if span > 0 else 0.0

    spaced, pos = [], []
    last = -math.inf
    for _, r in pf.iterrows():
        lp = math.log10(r["price_per_min"])
        if lp - last >= min_gap:
            spaced.append(r)
            pos.append("top center" if len(spaced) % 2 == 1 else "bottom center")
            last = lp
    if spaced:
        ld = pd.DataFrame(spaced)
        fig.add_trace(go.Scatter(
            x=ld["price_per_min"], y=ld["elo"],
            mode="text",
            text=ld["model"].apply(
                lambda m: plot_text(m[:20] + "…" if len(m) > 20 else m)),
            textposition=pos,
            textfont=dict(color="rgba(0,212,255,0.65)", size=10, family=FONT),
            hoverinfo="skip", showlegend=False,
        ))
