"""
Pareto Frontier scatter: Cost (x) vs Quality (y), sized by speed.
Highlights the Pareto-optimal models (best quality for their price tier).
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, PROVIDER_SHAPES, DEFAULT_COLOR, DEFAULT_SHAPE,
    BG, GRID, TICK, AXIS, FONT, dedupe_to_best_variant, canonical_provider,
    MAX_LEGEND_PROVIDERS, SPOTLIGHT_PROVIDERS,
    BUBBLE_MIN_PX, BUBBLE_MAX_PX, BUBBLE_SPEED_REF,
)


def _bubble_px(speed) -> float:
    """Diameter for a throughput value.

    Scaled against a fixed reference rather than the plotted subset's maximum,
    so filtering the chart never resizes a model that is still on screen, and
    sqrt-scaled so bubble *area* is proportional to throughput.
    """
    if pd.isna(speed) or speed <= 0:
        return BUBBLE_MIN_PX
    frac = min(float(speed), BUBBLE_SPEED_REF) / BUBBLE_SPEED_REF
    return BUBBLE_MIN_PX + math.sqrt(frac) * (BUBBLE_MAX_PX - BUBBLE_MIN_PX)


def _log_price_ticks(lo: float, hi: float) -> tuple[list[float], list[str]]:
    """1-2-5 ticks across the decades the data spans, labelled as prices.

    A log axis with no explicit tickvals makes Plotly fall back to "D2" minor
    ticks, which render as the unreadable run "2 5 0.1 2 5 1 2 5 10 2 5 100" —
    the leftmost label reads "2" for a $0.02 model.
    """
    if not (lo > 0 and hi > 0):
        return [], []
    vals: list[float] = []
    decade = math.floor(math.log10(lo))
    while 10 ** decade <= hi * 10:
        for mantissa in (1, 2, 5):
            v = mantissa * (10 ** decade)
            if lo / 1.6 <= v <= hi * 1.6:
                vals.append(v)
        decade += 1
    text = [f"${v:.2f}" if v < 1 else f"${v:g}" for v in vals]
    return vals, text


def _pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    O(n log n) Pareto frontier.
    Sort by price ascending; keep a model only if its quality exceeds
    every cheaper model seen so far.
    """
    sub = df[(df["price"] > 0) & (df["quality"] > 0)].sort_values("price")
    pareto, max_q = [], 0.0
    for _, row in sub.iterrows():
        if row["quality"] > max_q:
            pareto.append(row)
            max_q = float(row["quality"])
    return pd.DataFrame(pareto) if pareto else pd.DataFrame(columns=sub.columns)


def _top_providers(df: pd.DataFrame, n: int = 10) -> set:
    """Return the top-n providers by model count; rest become 'Other'."""
    counts = df["provider"].value_counts()
    return set(counts.head(n).index)


def build_pareto_scatter(df: pd.DataFrame) -> go.Figure:
    """Build the Cost vs Quality Pareto scatter figure."""
    plot_df = df[
        (df["price"] > 0) &
        (df["quality"] > 0) &
        df["price"].notna() &
        df["quality"].notna()
    ].copy()
    plot_df = dedupe_to_best_variant(plot_df)

    plot_df["size"] = plot_df["speed"].apply(_bubble_px)

    # Pearson r between log10(price) and quality. Needs at least three points
    # to mean anything — with one row np.corrcoef returns nan, which used to
    # print a literal "r = nan" over the chart whenever a search narrowed to a
    # single model.
    _corr_r = None
    valid_c = plot_df[plot_df["price"] > 0]
    if len(valid_c) >= 3:
        try:
            r = np.corrcoef(np.log10(valid_c["price"]), valid_c["quality"])[0, 1]
            if np.isfinite(r):
                _corr_r = float(r)
        except Exception:
            pass

    fig = go.Figure()

    # --- Only spotlight providers earn their own series; the rest are "Other".
    # Restricting to that set is what keeps the colours on screen a subset the
    # palette validator has cleared for all-pairs separation. ---
    plot_df["canon_provider"] = plot_df["provider"].apply(canonical_provider)
    counts = plot_df["canon_provider"].value_counts()
    named = [p for p in counts.index if p in SPOTLIGHT_PROVIDERS][:MAX_LEGEND_PROVIDERS]
    named_set = set(named)
    plot_df["display_provider"] = plot_df["canon_provider"].apply(
        lambda p: p if p in named_set else "Other"
    )

    # --- Per-provider scatter traces, densest provider first so the legend
    # leads with the labs a reader is actually looking for ---
    providers = [p for p in named if (plot_df["display_provider"] == p).any()]
    if (plot_df["display_provider"] == "Other").any():
        providers.append("Other")
    for provider in providers:
        pdf = plot_df[plot_df["display_provider"] == provider]
        # Draw the biggest bubbles first so small ones land on top and stay
        # visible (and clickable) instead of disappearing underneath.
        pdf = pdf.sort_values("size", ascending=False)
        color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
        symbol = PROVIDER_SHAPES.get(provider, DEFAULT_SHAPE)

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "Quality: %{y}<br>"
            "Price: $%{x:.3f}/M tokens<br>"
            "Speed: %{customdata[2]}<br>"
            "Latency (TTFT): %{customdata[3]}<br>"
            "<extra></extra>"
        )

        speed_str = pdf["speed"].apply(
            lambda s: f"{s:.0f} tok/s" if pd.notna(s) and s > 0 else "N/A"
        )
        latency_str = pdf["latency"].apply(
            lambda l: f"{l:.2f}s" if pd.notna(l) and l > 0 else "N/A"
        )

        fig.add_trace(go.Scatter(
            x=pdf["price"],
            y=pdf["quality"],
            mode="markers",
            name=provider,
            marker=dict(
                color=color,
                symbol=symbol,
                size=pdf["size"],
                opacity=0.8,
                # A background-coloured rim separates overlapping bubbles in the
                # dense sub-$1 cluster. The old rgba(0,0,0,0) stroke was dead
                # config and left that region an undifferentiated blob.
                line=dict(width=0.8, color=BG),
            ),
            customdata=list(zip(pdf["model"], pdf["provider"], speed_str, latency_str)),
            hovertemplate=hover,
        ))

    # --- Pareto frontier line ---
    pareto_df = _pareto_frontier(plot_df)
    if not pareto_df.empty:
        fig.add_trace(go.Scatter(
            x=pareto_df["price"],
            y=pareto_df["quality"],
            mode="lines",
            name="Pareto Frontier",
            line=dict(color="rgba(0,212,255,0.5)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=True,
        ))

        # Label Pareto-optimal models — skip models too close in log-price space
        spaced_rows, spaced_pos = [], []
        last_log_price = -math.inf
        for _, prow in pareto_df.iterrows():
            log_p = math.log10(prow["price"])
            if log_p - last_log_price >= 0.4:
                spaced_rows.append(prow)
                spaced_pos.append("top center" if len(spaced_rows) % 2 == 1 else "bottom center")
                last_log_price = log_p
        if spaced_rows:
            label_df = pd.DataFrame(spaced_rows)
            fig.add_trace(go.Scatter(
                x=label_df["price"],
                y=label_df["quality"],
                mode="text",
                text=label_df["model"].apply(lambda m: m[:22] + "…" if len(m) > 22 else m),
                textposition=spaced_pos,
                textfont=dict(color="rgba(0,212,255,0.65)", size=10, family=FONT),
                hoverinfo="skip",
                showlegend=False,
            ))

    _zero = "rgba(255,255,255,0.06)"

    if not plot_df.empty:
        tickvals, ticktext = _log_price_ticks(
            float(plot_df["price"].min()), float(plot_df["price"].max())
        )
    else:
        tickvals, ticktext = [], []

    # Correlation annotation
    corr_text = (
        f"r = {_corr_r:.2f}  (log price vs quality)"
        if _corr_r is not None else ""
    )

    annotations = []
    if corr_text:
        annotations.append(dict(
            x=0.01, y=0.99,
            xref="paper", yref="paper",
            text=corr_text,
            showarrow=False,
            xanchor="left", yanchor="top",
            font=dict(color="#666666", size=10, family=FONT),
        ))

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Cost vs. Intelligence"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  bubble size = speed (tok/s)  ·  shape = provider family</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
            x=0.0,
            xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(
                text="Price  (USD / 1M tokens)",
                font=dict(color=AXIS, size=12),
                standoff=12,
            ),
            type="log",
            tickmode="array" if tickvals else "auto",
            tickvals=tickvals or None,
            ticktext=ticktext or None,
            gridcolor=GRID,
            zerolinecolor=_zero,
            zerolinewidth=1,
            tickfont=dict(color=TICK, size=11, family=FONT),
            showgrid=True,
            showline=False,
            ticks="",
        ),
        yaxis=dict(
            title=dict(
                text="AA Intelligence Index",
                font=dict(color=AXIS, size=12),
                standoff=12,
            ),
            gridcolor=GRID,
            zerolinecolor=_zero,
            zerolinewidth=1,
            tickfont=dict(color=TICK, size=11, family=FONT),
            showgrid=True,
            showline=False,
            ticks="",
        ),
        # Horizontal legend under the axis. The old vertical legend sat in a
        # fixed 172px right gutter, where 25 entries ran ~487px tall against a
        # ~300px plot area — clipped and scrollable — while squeezing the
        # scatter into 39% of the width on narrow viewports.
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(color="#999999", size=11, family=FONT),
            itemsizing="constant",
            orientation="h",
            x=0,
            y=-0.16,
            xanchor="left",
            yanchor="top",
            tracegroupgap=2,
        ),
        margin=dict(l=56, r=28, t=52, b=104),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=FONT),
            namelength=-1,
        ),
        annotations=annotations,
    )

    return fig
