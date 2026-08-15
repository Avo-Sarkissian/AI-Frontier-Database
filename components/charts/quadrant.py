"""
Speed vs Quality quadrant chart.
Divides models into 4 zones based on median speed and quality.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    empty_figure,
    PROVIDER_COLORS, PROVIDER_SHAPES, DEFAULT_COLOR, DEFAULT_SHAPE,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
    dedupe_to_best_variant, BUBBLE_PRICE_REF, BUBBLE_PRICE_FLOOR, bubble_size, safe_corr,
    marker_outline, legend_below, CHART_MARGIN, spotlight_split, log_ticks,
)

_ZONE = "rgba(255,255,255,0.02)"


def _plottable(df: pd.DataFrame) -> pd.DataFrame:
    """The rows this chart draws: measured speed, scored, one row per family."""
    sub = df[
        (df["speed"] > 0) &
        (df["quality"] > 0) &
        df["speed"].notna() &
        df["quality"].notna()
    ].copy()
    return dedupe_to_best_variant(sub)


def build_quadrant(df: pd.DataFrame, full_df: pd.DataFrame | None = None) -> go.Figure:
    """Speed vs Quality quadrant.

    ``full_df`` is the unfiltered catalogue. "Fast · Smart" reads as an absolute
    statement about a model, so the median crosshairs that decide which zone a
    model lands in must come from the whole market. Taken from the filtered
    frame, a provider filter moved the thresholds and flipped models between
    zones — the same model, the same numbers, the opposite verdict — and a
    quality-only filter moved the *speed* threshold.
    """
    plot_df = _plottable(df)
    ref_df = plot_df if full_df is None else _plottable(full_df)

    if plot_df.empty:
        # NOT a bare go.Figure(): that serialises Plotly's LIGHT default template
        # (paper_bgcolor 'white'), and docs/app.js hands layout straight to
        # Plotly.react — so a bright white card landed in the middle of the dark
        # dashboard, under a caption still saying "Click any bubble for details".
        # Reachable from 24 of 32 providers at some MIN SCORE setting.
        return empty_figure("No models match these filters")
    if ref_df.empty:
        ref_df = plot_df

    med_speed   = ref_df["speed"].median()
    med_quality = ref_df["quality"].median()

    # Bubble size encodes affordability (cheaper = bigger) on a fixed price
    # reference. Normalising against the plotted frame's own max meant a model
    # grew or shrank whenever the user narrowed a filter, even though its price
    # had not changed.
    # Log-scaled: price spans decades, so a linear mapping put 53 of 95 models
    # within 0.43 px of each other while every model over $20 drew identically.
    plot_df["size"] = bubble_size(plot_df["price"], BUBBLE_PRICE_REF, invert=True,
                                  log=True, floor=BUBBLE_PRICE_FLOOR).values

    # Correlation on log speed, matching the axis the reader is looking at.
    # (Correlation describes what is plotted, so it stays on plot_df.)
    _corr_r = safe_corr(np.log10(plot_df["speed"]), plot_df["quality"])
    _tickvals, _ticktext = log_ticks(
        ref_df["speed"].min(), ref_df["speed"].max(), fmt=lambda v: f"{v:g}"
    )

    fig = go.Figure()

    # Throughput spans ~9 to ~2200 tok/s but nearly every model sits under 400,
    # so on a linear axis two outliers stretched the scale and crushed the whole
    # catalogue into the leftmost fifth of the plot. Log spreads it out.
    # Axis bounds come from the full catalogue too, so a mark keeps the same
    # screen position — and the zone rectangles keep the same extent — however
    # the frame is filtered.
    x_min = ref_df["speed"].min() / 1.3
    x_max = ref_df["speed"].max() * 1.3
    y_max = ref_df["quality"].max() * 1.15
    lx0, lx1 = math.log10(x_min), math.log10(x_max)
    lmed = math.log10(med_speed)

    # Zone rectangles are drawn in data coordinates; on a log axis Plotly wants
    # the exponents, hence the log10() on every x.
    zones = [
        (lx0,  lmed, med_quality, y_max,       "Slow · Smart"),
        (lmed, lx1,  med_quality, y_max,       "Fast · Smart"),
        (lx0,  lmed, 0,           med_quality, "Slow · Weak"),
        (lmed, lx1,  0,           med_quality, "Fast · Weak"),
    ]

    for x0, x1, y0, y1, _label in zones:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=_ZONE, line=dict(width=0), layer="below")

    # Median crosshairs. The vertical one takes log10(median) because shapes on
    # a log axis are positioned in exponent space — passing the raw 103.8 put
    # the line at 10^103.8, i.e. far off the right of the plot.
    fig.add_hline(y=med_quality, line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"))
    fig.add_vline(x=lmed,        line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"))

    # Only spotlight providers get their own colour — the previous top-10-by-
    # count rule put whichever providers happened to be dense on screen
    # together, including pairs the palette validator never cleared.
    plot_df, providers = spotlight_split(plot_df)
    for provider in providers:
        pdf = plot_df[plot_df["display_provider"] == provider]
        color  = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
        symbol = PROVIDER_SHAPES.get(provider, DEFAULT_SHAPE)

        latency_str = pdf["latency"].apply(
            lambda l: f"{l:.2f}s" if pd.notna(l) and l > 0 else "N/A"
        )

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "Quality: %{y}<br>"
            "Speed: %{x:.0f} tok/s<br>"
            "Price: $%{customdata[2]:.3f}/M tokens  ·  blended 3:1 out:in<br>"
            "Rates: %{customdata[4]}<br>"
            "Latency (TTFT): %{customdata[3]}<br>"
            "<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=pdf["speed"],
            y=pdf["quality"],
            mode="markers",
            name=provider,
            marker=dict(
                color=color,
                symbol=symbol,
                size=pdf["size"],
                opacity=0.8,
                line=marker_outline(),
            ),
            customdata=list(zip(pdf["model"], pdf["provider"], pdf["price"], latency_str,
                                _rate_pair(pdf))),
            hovertemplate=hover,
        ))

    # Zone captions pinned to the outer corner of each quadrant. Centring them
    # put "Slow · Smart" and "Slow · Weak" straight through the dense low-speed
    # cluster and over the y-axis title.
    annotations = []
    for x0, x1, y0, y1, label in zones:
        outer_left = x0 == lx0
        annotations.append(dict(
            x=x0 + (x1 - x0) * (0.04 if outer_left else 0.96),
            y=y0 + (y1 - y0) * (0.94 if y1 == y_max else 0.08),
            text=label,
            showarrow=False,
            xanchor="left" if outer_left else "right",
            font=dict(color="rgba(255,255,255,0.3)", size=12, family=_FONT),
            xref="x", yref="y",
        ))

    # Correlation sits above the plot, not inside it — all four in-plot corners
    # are taken by zone captions, and it collided with "Slow · Smart".
    if _corr_r is not None:
        annotations.append(dict(
            x=1.0, y=1.04,
            xref="paper", yref="paper",
            text=f"r = {_corr_r:.2f}  (log speed vs quality)",
            showarrow=False,
            xanchor="right", yanchor="bottom",
            font=dict(color="#666666", size=10, family=_FONT),
        ))

    # Label top-right quadrant outliers (Fast · Smart).
    # Use 75th-percentile speed as threshold so labels stay in the genuinely
    # fast region, and cap at 5 to avoid label pile-ups.
    speed_p75 = ref_df["speed"].quantile(0.75)
    candidates = plot_df[
        (plot_df["speed"] > speed_p75) & (plot_df["quality"] > med_quality)
    ].sort_values("quality", ascending=False)

    # Take the best models that are far enough apart *on screen* to be legible.
    # Simply taking the top 5 stacked three labels on one another — the leaders
    # cluster tightly, so "Gemini 3.6 Flash" and "Gemini 3.5 Flash" landed on
    # the same pixels. Distance is measured in normalised plot space so it
    # tracks the log x axis rather than raw tok/s.
    kept_rows, kept_pos, positions = [], [], []
    x_span = (lx1 - lx0) or 1.0
    for _, row in candidates.iterrows():
        if len(kept_rows) >= 5:
            break
        nx = (math.log10(row["speed"]) - lx0) / x_span
        ny = row["quality"] / y_max if y_max else 0.0
        if any(math.dist((nx, ny), q) < 0.09 for q in kept_pos):
            continue
        kept_pos.append((nx, ny))
        kept_rows.append(row)
        positions.append("top center" if len(kept_rows) % 2 else "bottom center")

    if kept_rows:
        label_df = pd.DataFrame(kept_rows)
        fig.add_trace(go.Scatter(
            x=label_df["speed"],
            y=label_df["quality"],
            mode="text",
            text=label_df["model"].apply(lambda m: m[:20] + "…" if len(m) > 20 else m),
            textposition=positions,
            textfont=dict(color="rgba(255,255,255,0.5)", size=10, family=_FONT),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Speed vs. Intelligence"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  bubble size ∝ affordability  ·  shape = provider family</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Speed  (tokens / second, log scale)",
                       font=dict(color=_AXIS, size=12), standoff=12),
            type="log",
            tickmode="array" if _tickvals else "auto",
            tickvals=_tickvals or None,
            ticktext=_ticktext or None,
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
            range=[lx0, lx1],
        ),
        yaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=12), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
            range=[0, y_max],
        ),
        legend=legend_below(),
        margin=CHART_MARGIN,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
        annotations=annotations,
    )

    return fig


def _rate_pair(df) -> list[str]:
    """"$5.00 in · $25.00 out" per row, or "—" where the sides aren't known."""
    import pandas as _pd

    def _fmt(v):
        return f"${v:,.2f}" if _pd.notna(v) and v > 0 else None

    out = []
    for pin, pout in zip(df.get("price_in", []), df.get("price_out", [])):
        a, b = _fmt(pin), _fmt(pout)
        out.append(f"{a} in  ·  {b} out" if a and b else "—")
    return out
