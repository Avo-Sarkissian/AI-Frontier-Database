"""
Pareto Frontier scatter: Cost (x) vs Quality (y), sized by speed.
Highlights the Pareto-optimal models (best quality for their price tier).
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    plot_text,
    PROVIDER_COLORS, PROVIDER_SHAPES, DEFAULT_COLOR, DEFAULT_SHAPE,
    BG, GRID, TICK, AXIS, FONT, dedupe_to_best_variant,
    BUBBLE_SPEED_REF, bubble_size, safe_corr, marker_outline,
    legend_below, CHART_MARGIN, spotlight_split, log_ticks,
)


def _log_price_ticks(lo: float, hi: float) -> tuple[list[float], list[str]]:
    """1-2-5 ticks labelled as prices."""
    return log_ticks(lo, hi, fmt=lambda v: f"${v:.2f}" if v < 1 else f"${v:g}")


def _pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    O(n log n) Pareto frontier.
    Sort by price ascending; keep a model only if its quality exceeds
    every cheaper model seen so far.

    The secondary quality key is load-bearing, not cosmetic. A bare
    ``sort_values("price")`` is an unstable quicksort, so when two models share a
    price their row order decided which was seen first — and if the *worse* one
    came first it was appended, putting a strictly dominated model on a line
    labelled "Pareto Frontier". Sorting quality descending within a price tie
    means only the best model at that price can ever clear ``max_q``.
    """
    sub = df[(df["price"] > 0) & (df["quality"] > 0)].sort_values(
        ["price", "quality"], ascending=[True, False], kind="stable"
    )
    pareto, max_q = [], 0.0
    for _, row in sub.iterrows():
        if row["quality"] > max_q:
            pareto.append(row)
            max_q = float(row["quality"])
    return pd.DataFrame(pareto) if pareto else pd.DataFrame(columns=sub.columns)


def _plottable(df: pd.DataFrame) -> pd.DataFrame:
    """The rows this chart draws: priced, scored, one row per model family."""
    sub = df[
        (df["price"] > 0) &
        (df["quality"] > 0) &
        df["price"].notna() &
        df["quality"].notna()
    ].copy()
    return dedupe_to_best_variant(sub)


def _top_providers(df: pd.DataFrame, n: int = 10) -> set:
    """Return the top-n providers by model count; rest become 'Other'."""
    counts = df["provider"].value_counts()
    return set(counts.head(n).index)


def build_pareto_scatter(df: pd.DataFrame, full_df: pd.DataFrame | None = None) -> go.Figure:
    """Build the Cost vs Quality Pareto scatter figure.

    ``full_df`` is the unfiltered catalogue. The frontier is a claim about the
    *market* — "nothing beats this on price and quality" — so it must be derived
    from every model, not from whatever survived the user's filter. Computing it
    from the filtered frame let one click promote a model the market already
    dominates onto a line labelled "Pareto Frontier".
    """
    plot_df = _plottable(df)
    ref_df = plot_df if full_df is None else _plottable(full_df)

    plot_df["size"] = bubble_size(plot_df["speed"], BUBBLE_SPEED_REF).values

    valid_c = plot_df[plot_df["price"] > 0]
    _corr_r = safe_corr(np.log10(valid_c["price"]), valid_c["quality"])

    fig = go.Figure()

    plot_df, providers = spotlight_split(plot_df)
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
            "Price: $%{x:.3f}/M tokens  ·  blended 3:1 out:in<br>"
            "Rates: %{customdata[4]}<br>"
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
                line=marker_outline(),
            ),
            customdata=list(zip(pdf["model"].map(plot_text), pdf["provider"].map(plot_text),
                                speed_str, latency_str,
                                _rate_pair(pdf))),
            hovertemplate=hover,
        ))

    # --- Pareto frontier line ---
    # Membership is decided against the full catalogue, then intersected with
    # what is on screen: a filter may HIDE a frontier point, never invent one.
    frontier_models = set(_pareto_frontier(ref_df)["model"])
    pareto_df = plot_df[plot_df["model"].isin(frontier_models)].sort_values(
        ["price", "quality"], ascending=[True, False], kind="stable"
    )
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
                text=label_df["model"].apply(
                    lambda m: plot_text(m[:22] + "…" if len(m) > 22 else m)),
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
                text="Price  (USD / 1M tokens, blended 3:1 output:input)",
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
        legend=legend_below(),
        margin=CHART_MARGIN,
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
