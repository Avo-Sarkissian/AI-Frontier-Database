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
    BG, GRID, TICK, AXIS, FONT,
)


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

    # Normalize speed for bubble size (8–36px range)
    max_speed = plot_df["speed"].replace(0, np.nan).max()
    if pd.isna(max_speed) or max_speed == 0:
        max_speed = 1
    plot_df["size"] = plot_df["speed"].apply(
        lambda s: 8 + (s / max_speed) * 28 if pd.notna(s) and s > 0 else 8
    )

    # Pearson r between log10(price) and quality
    _corr_r = None
    try:
        valid_c = plot_df[plot_df["price"] > 0]
        _corr_r = np.corrcoef(np.log10(valid_c["price"]), valid_c["quality"])[0, 1]
    except Exception:
        pass

    fig = go.Figure()

    # --- Group small providers into "Other" to keep legend readable ---
    top_provs = _top_providers(plot_df, n=10)
    plot_df["display_provider"] = plot_df["provider"].apply(
        lambda p: p if p in top_provs else "Other"
    )

    # --- Per-provider scatter traces ---
    providers = sorted(plot_df["display_provider"].unique(), key=lambda p: (p == "Other", p))
    for provider in providers:
        pdf = plot_df[plot_df["display_provider"] == provider]
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
                opacity=0.75,
                line=dict(width=0.5, color="rgba(0,0,0,0)"),
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
                textfont=dict(color="rgba(0,212,255,0.65)", size=9, family=FONT),
                hoverinfo="skip",
                showlegend=False,
            ))

    _zero = "rgba(255,255,255,0.06)"

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
            font=dict(color="#666666", size=9, family=FONT),
        ))

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, color="#888888", size=12),
        title=dict(
            text=(
                "Cost vs. Intelligence"
                "  <span style='font-size:11px;color:#666666;font-weight:400'>"
                "  ·  bubble size = speed (tok/s)  ·  shape = provider family</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=FONT, weight=600),
            x=0.0,
            xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(
                text="Price  (USD / 1M tokens)",
                font=dict(color=AXIS, size=11),
                standoff=12,
            ),
            type="log",
            gridcolor=GRID,
            zerolinecolor=_zero,
            zerolinewidth=1,
            tickfont=dict(color=TICK, size=10, family=FONT),
            showgrid=True,
            showline=False,
            ticks="",
        ),
        yaxis=dict(
            title=dict(
                text="AA Intelligence Index",
                font=dict(color=AXIS, size=11),
                standoff=12,
            ),
            gridcolor=GRID,
            zerolinecolor=_zero,
            zerolinewidth=1,
            tickfont=dict(color=TICK, size=10, family=FONT),
            showgrid=True,
            showline=False,
            ticks="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(color="#888888", size=10, family=FONT),
            itemsizing="constant",
            orientation="v",
            x=1.01,
            y=1,
            xanchor="left",
            tracegroupgap=2,
        ),
        margin=dict(l=56, r=172, t=52, b=52),
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
