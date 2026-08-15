"""
Budget calculator chart.
Given a monthly token volume, shows projected monthly cost per model.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    right_gutter, ANNOTATED_AXIS_HEADROOM,
    PROVIDER_COLORS, DEFAULT_COLOR, clean_model_name, unique_labels,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
)


def _spendable(df: pd.DataFrame, monthly_tokens_m: float, min_quality: float) -> pd.DataFrame:
    """Priceable, scored models at or above `min_quality`, cheapest first."""
    out = df[
        (df["price"] > 0) &
        (df["quality"] > 0) &
        df["price"].notna()
    ].copy()
    if min_quality:
        out = out[out["quality"] >= float(min_quality)]
    out["monthly_cost"] = monthly_tokens_m * out["price"]
    # Rank on (cost, unit price, name), not cost alone. At a volume of 0 every
    # model costs $0.000, so a single-key sort is a no-op and "CHEAPEST MODEL
    # SCORING N+" named whatever happened to sit at row 0 — Claude Opus 5 at
    # $20/M instead of DeepSeek V4 Flash at $0.1575/M, a 127x error under a
    # label that says cheapest. Unit price breaks the tie the way the user
    # means it; the model name makes the order total, so a refresh that merely
    # reorders the catalogue cannot reorder the answer. mergesort because
    # pandas' default quicksort is unstable (the same defect once let the
    # Pareto frontier admit dominated models on a price tie).
    return (
        out.sort_values(["monthly_cost", "price", "model"],
                        ascending=True, kind="mergesort")
           .reset_index(drop=True)
    )


def cheapest_above(df: pd.DataFrame, min_quality: float = 0.0,
                   monthly_tokens_m: float = 1.0) -> dict | None:
    """The cheapest model scoring at least `min_quality`, or None if there is none.

    This is the Budget tab's actual question ("what is the cheapest model smarter
    than X"), so it is computed once here and reused by the chart and the callout
    rather than re-derived on the JS side.
    """
    ranked = _spendable(df, monthly_tokens_m, min_quality)
    if ranked.empty:
        return None
    row = ranked.iloc[0]
    return {
        "model": str(row["model"]),
        "provider": str(row["provider"]),
        "quality": float(row["quality"]),
        "price": float(row["price"]),
        "monthly_cost": float(row["monthly_cost"]),
        "n_qualifying": int(len(ranked)),
    }


def _no_match_figure(min_quality: float) -> go.Figure:
    """Explicit empty state. A blank chart reads as 'broken', and silently
    showing an unfiltered list would be worse — it would answer a question the
    user did not ask."""
    fig = go.Figure()
    fig.add_annotation(
        text=(
            f"No model scores {min_quality:g} or higher<br>"
            "<span style='font-size:12px;color:#777777'>"
            "Lower the minimum intelligence, or widen the provider filter</span>"
        ),
        showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
        font=dict(size=15, family=_FONT, color="#999999"), align="center",
    )
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=52, b=36), height=320,
        title=dict(
            text="Monthly API Cost",
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left", pad=dict(l=20, t=16),
        ),
    )
    return fig


def build_cost_calc(df: pd.DataFrame, monthly_tokens_m: float = 1.0, top_n: int = 30,
                    min_quality: float = 0.0) -> go.Figure:
    """Horizontal bar chart: monthly cost = monthly_tokens_m × price/M tokens.

    `min_quality` keeps only models scoring at least that on the AA Intelligence
    Index, so the top bar is the cheapest model smarter than the threshold.
    """
    plot_df = _spendable(df, monthly_tokens_m, min_quality)
    if plot_df.empty:
        return _no_match_figure(min_quality)

    plot_df = plot_df.head(top_n).reset_index(drop=True)

    colors = [PROVIDER_COLORS.get(p, DEFAULT_COLOR) for p in plot_df["provider"]]
    # Truncation can collapse two models onto one category (the Nemotron 3 Nano
    # "(Reasoning)" / "(Non-reasoning)" pair, for one). Plotly then stacks both
    # bars in the same row and draws both right-hand labels on top of each other.
    short_names = pd.Series(
        unique_labels(plot_df["model"].apply(clean_model_name).tolist()),
        index=plot_df.index,
    )

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "Provider: %{customdata[1]}<br>"
        "Monthly cost: $%{x:,.2f}<br>"
        "Price: $%{customdata[2]:.4f}/M tokens  ·  blended 3:1 out:in<br>"
        "Rates: %{customdata[4]}<br>"
        "Intelligence: %{customdata[3]:.0f}<br>"
        "<extra></extra>"
    )

    fig = go.Figure()

    # Background track
    fig.add_trace(go.Bar(
        y=short_names,
        x=[plot_df["monthly_cost"].max() * 1.05] * len(plot_df),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Cost bars. With a threshold set, the cheapest qualifying model is the
    # answer the user came for, so it carries full opacity and the rest recede.
    opacities = (
        [1.0] + [0.45] * (len(plot_df) - 1) if min_quality else [0.85] * len(plot_df)
    )
    fig.add_trace(go.Bar(
        y=short_names,
        x=plot_df["monthly_cost"],
        orientation="h",
        marker=dict(color=colors, opacity=opacities, line=dict(width=0)),
        customdata=list(zip(plot_df["model"], plot_df["provider"], plot_df["price"],
                            plot_df["quality"], _rate_pair(plot_df))),
        hovertemplate=hover,
        showlegend=False,
        text=plot_df["monthly_cost"].apply(
            lambda c: f"${c:,.2f}" if c >= 1 else f"${c:.3f}"
        ),
        textposition="inside",
        textfont=dict(color="rgba(255,255,255,0.6)", size=11, family=_FONT),
    ))

    _gutter = right_gutter(
        f"{p}  {q:.0f}pt" for p, q in zip(plot_df["provider"], plot_df["quality"])
    )
    # Provider label + intelligence score on right
    for i, row in plot_df.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        score_color = (
            "#00d4ff" if row["quality"] >= 45
            else "#7ecfaa" if row["quality"] >= 30
            else "#888888"
        )
        fig.add_annotation(
            x=1.01,
            y=short_names[i],
            text=(
                f"<span style='color:{color}'>{row['provider']}</span>"
                f"  <span style='color:{score_color}'>{row['quality']:.0f}pt</span>"
            ),
            showarrow=False,
            xanchor="left",
            font=dict(size=10, family=_FONT, color=color),
            xref="paper", yref="y",
        )

    height = max(500, top_n * 26)

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Monthly API Cost"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                + (
                    f"  ·  cheapest models scoring {min_quality:g}+"
                    if min_quality else
                    "  ·  cheapest models for your token budget"
                )
                + "</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="Estimated Monthly Cost  (USD)", font=dict(color=_AXIS, size=12), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            tickprefix="$",
            showgrid=True, showline=False, ticks="",
            range=[0, plot_df["monthly_cost"].max() * ANNOTATED_AXIS_HEADROOM],
        ),
        yaxis=dict(
            tickfont=dict(color="#999999", size=11, family=_FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
            autorange="reversed",  # cheapest at top
        ),
        barmode="overlay",
        margin=dict(l=20, r=_gutter, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
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
