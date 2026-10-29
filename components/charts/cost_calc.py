"""
Budget calculator chart.
Given a monthly token volume, shows projected monthly cost per model.
"""
import pandas as pd
import plotly.graph_objects as go

from data.workload_mix import (
    resolve_mix, fmt_price, fill_note, fill_hover, is_filled, price_marks,
    FILL_MARK, FILL_SHORT,
)

from components.charts.constants import (
    right_gutter, ANNOTATED_AXIS_HEADROOM,
    PROVIDER_COLORS, DEFAULT_COLOR, clean_model_name, unique_labels,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
    dedupe_to_best_variant, text_on, empty_state_figure,
    bar_text_positions, BAR_PLOT_PX_DESKTOP, BAR_PLOT_PX_PHONE, PHONE_PX,
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
    # One row per model family (its best-scoring tier). Every tier of a family
    # is billed at the same rate, so under "All variants" Claude Opus 5.5 took
    # three adjacent rows at one price and pushed real alternatives off the
    # list; the answer is the same either way.
    out = dedupe_to_best_variant(out)
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
    # The answer card states this price as THE cheapest, so it carries the
    # same flag every chart does when the model has no AA cache price.
    filled = is_filled(row.get("cache_price_filled"))
    return {
        "model": str(row["model"]),
        "provider": str(row["provider"]),
        "quality": float(row["quality"]),
        "price": float(row["price"]),
        "monthly_cost": float(row["monthly_cost"]),
        "n_qualifying": int(len(ranked)),
        "cache_price_filled": filled,
        "fill_key": f"{FILL_MARK} {FILL_SHORT}" if filled else "",
    }


def _no_match_figure(min_quality: float) -> go.Figure:
    """Explicit empty state. A blank chart reads as 'broken', and silently
    showing an unfiltered list would be worse — it would answer a question the
    user did not ask. The pattern every empty chart now shares
    (constants.empty_state_figure)."""
    if min_quality:
        return empty_state_figure(
            "Monthly API Cost", f"No model scores {min_quality:g} or higher",
            "Lower the minimum intelligence, or widen the provider filter")
    return empty_state_figure("Monthly API Cost", "No priced model matches these filters",
                              "Clear the search or widen the provider filter")


def build_cost_calc(df: pd.DataFrame, monthly_tokens_m: float = 1.0, top_n: int = 30,
                    min_quality: float = 0.0, basis=None,
                    full_df: pd.DataFrame | None = None) -> go.Figure:
    """Horizontal bar chart: monthly cost = monthly_tokens_m × price/M tokens.

    `min_quality` keeps only models scoring at least that on the AA Intelligence
    Index, so the top bar is the cheapest model smarter than the threshold.

    ``basis`` is the workload MIX `price` is at (data/workload_mix); the title
    states the token mix every bar assumes.

    ``full_df`` (default: `df`) is the unfiltered catalogue the score colours'
    bands are read from (data/quality_scale quantiles), so a
    filter cannot recolour a model.
    """
    basis = resolve_mix(basis)
    plot_df = _spendable(df, monthly_tokens_m, min_quality)
    if plot_df.empty:
        return _no_match_figure(min_quality)

    plot_df = plot_df.head(top_n).reset_index(drop=True)

    colors = [PROVIDER_COLORS.get(p, DEFAULT_COLOR) for p in plot_df["provider"]]
    # Same rule as every price on the page: cent-and-up values print exactly as
    # before ($1,234.56 / $0.123); below a cent, 4 significant digits.
    cost_text = plot_df["monthly_cost"].map(fmt_price).tolist()
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
        # customdata[5] is the cost as text (workload_mix.fmt_price): ",.2f"
        # hovered the agentic floor's $0.0041 / month as "$0.00".
        "Monthly cost: %{customdata[5]}<br>"
        "Price: %{customdata[2]}/M tokens  ·  " + basis.short + "<br>"
        "Rates: %{customdata[4]}%{customdata[6]}<br>"
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
    # Value labels: upright, at the bar's end, coloured for the bar actually
    # on screen (fill at its opacity over the background); a bar too short to
    # hold its label gets it just outside, in the axis grey.
    inside_colors = [text_on(c, o) for c, o in zip(colors, opacities)]
    _x_hi = plot_df["monthly_cost"].max() * ANNOTATED_AXIS_HEADROOM
    fig.add_trace(go.Bar(
        y=short_names,
        x=plot_df["monthly_cost"],
        orientation="h",
        marker=dict(color=colors, opacity=opacities, line=dict(width=0)),
        customdata=list(zip(plot_df["model"], plot_df["provider"],
                            price_marks(plot_df),
                            plot_df["quality"], _rate_pair(plot_df),
                            cost_text, fill_hover(plot_df))),
        hovertemplate=hover,
        showlegend=False,
        text=cost_text,
        textposition=bar_text_positions(plot_df["monthly_cost"], cost_text, _x_hi,
                                        BAR_PLOT_PX_DESKTOP),
        textangle=0,
        insidetextanchor="end",
        constraintext="none",
        insidetextfont=dict(color=inside_colors, size=11, family=_FONT),
        outsidetextfont=dict(color="#999999", size=11, family=_FONT),
        cliponaxis=False,
    ))

    _gutter = right_gutter(
        f"{p}  {q:.0f}pt" for p, q in zip(plot_df["provider"], plot_df["quality"])
    )
    # Provider label + intelligence score on right.
    # Collected as plain dicts and set once in update_layout below: one
    # fig.add_annotation per row re-validates the whole annotations tuple on
    # every call, which makes the build quadratic in rows.
    from data.quality_scale import floor as _q_floor
    _ref = df if full_df is None else full_df
    _strong, _mid = _q_floor(_ref, "budget_strong"), _q_floor(_ref, "budget_mid")
    annotations = []
    for i, row in plot_df.iterrows():
        color = PROVIDER_COLORS.get(row["provider"], DEFAULT_COLOR)
        score_color = (
            "#00d4ff" if row["quality"] >= _strong
            else "#7ecfaa" if row["quality"] >= _mid
            else "#888888"
        )
        annotations.append(dict(
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
        ))

    height = max(500, top_n * 26)

    # Phone (docs/app.js, below PHONE_PX): shorter row names — the full ones
    # took two thirds of a 356 px plot and ran off its left edge — with the
    # right-hand labels re-keyed to them, and labels placed for a ~140 px bar.
    phone_names = unique_labels(plot_df["model"].apply(
        lambda m: clean_model_name(m, max_len=16)).tolist())
    narrow = {"narrow": {
        "max_width": PHONE_PX,
        "data": {"0": {"y": phone_names},
                 "1": {"y": phone_names,
                       "textposition": bar_text_positions(
                           plot_df["monthly_cost"], cost_text, _x_hi, BAR_PLOT_PX_PHONE)}},
        "layout": {"annotations": [{**a, "y": phone_names[i]}
                                   for i, a in enumerate(annotations)]},
    }}

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Monthly API Cost"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                # The fill note leads: Plotly titles do not wrap, and a narrow
                # card clips the end of this long subtitle.
                + fill_note(plot_df, basis)
                + (
                    f"  ·  cheapest models scoring {min_quality:g}+"
                    if min_quality else
                    "  ·  cheapest models for your token budget"
                )
                # The entire cost model is `tokens * price`, and `price` is
                # the effective price at the MIX setting — at the default, the
                # 1:3 in:out blend, so every figure assumes 75% of your tokens
                # are generated. An input-heavy RAG workload costs far less
                # than that (Claude Opus 5 at 100M: $2000 charted, $500
                # all-input). Whatever the mix, the title says which.
                + f"  ·  assumes {basis.assumption}"
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
            # Tick labels in the same format as the bar labels (fmt_price):
            # Plotly's own "$0.1" sat under bars reading "$0.100".
            tickmode="array",
            tickvals=_money_ticks(plot_df["monthly_cost"].max())[0],
            ticktext=_money_ticks(plot_df["monthly_cost"].max())[1],
            showgrid=True, showline=False, ticks="",
            range=[0, plot_df["monthly_cost"].max() * ANNOTATED_AXIS_HEADROOM],
        ),
        yaxis=dict(
            tickfont=dict(color="#999999", size=11, family=_FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
            autorange="reversed",  # cheapest at top
        ),
        annotations=annotations,
        meta=narrow,
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


def _money_ticks(hi: float, n: int = 5) -> tuple[list[float], list[str]]:
    """About `n` round ticks from 0 to `hi`, labelled with fmt_price."""
    import math
    hi = float(hi) if hi and math.isfinite(float(hi)) and float(hi) > 0 else 1.0
    raw = hi / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    vals = [round(i * step, 10) for i in range(int(hi / step) + 1)]
    return vals, [fmt_price(v) for v in vals]


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
