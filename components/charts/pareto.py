"""
Pareto Frontier scatter: Cost (x) vs Quality (y), sized by speed.
Highlights the Pareto-optimal models (best quality for their price tier).

Two x axes share this builder (Overview's X AXIS radio):
  x="price": $/1M tokens at the workload MIX;
  x="task":  AA's cost per Intelligence Index task (data/task_cost.py). The
             axis is AA-measured, so MIX does not move it; rows AA has not
             costed are counted in the title, never imputed.
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    plot_text,
    PROVIDER_COLORS, PROVIDER_SHAPES, DEFAULT_COLOR, DEFAULT_SHAPE,
    BG, GRID, TICK, AXIS, FONT, dedupe_to_best_variant, base_model_name,
    BUBBLE_SPEED_REF, bubble_size, safe_corr, marker_outline,
    legend_below, CHART_MARGIN, spotlight_split, log_ticks,
    weights_symbol, row_is_open, UNTIMED_RIM,
    place_point_labels, family_label, legend_reference, size_note, NOTE_MARGIN_T, empty_state_figure,
)
from data.open_weights import weights_labels
from data.workload_mix import resolve_mix, fill_note, fill_hover, price_marks
from data.task_cost import (
    task_costs, token_burn, fmt_task_cost, burn_text, list_blend_short,
    rank_flip_share,
)

# Overview X AXIS values this builder draws, and the frame column each plots.
X_COLUMNS = {"price": "price", "task": "cost_per_task"}


def _log_price_ticks(lo: float, hi: float) -> tuple[list[float], list[str]]:
    """1-2-5 ticks labelled as prices.

    Below a cent the label is the value itself: an agentic MIX puts the floor
    near $0.004, and '.2f' printed a $0.005 tick as "$0.01". Cent-and-up ticks
    keep their old format, so the default figure is unchanged."""
    def _fmt(v):
        if v < 0.01:
            return f"${v:g}"
        return f"${v:.2f}" if v < 1 else f"${v:g}"
    return log_ticks(lo, hi, fmt=_fmt)


def _pareto_frontier(df: pd.DataFrame, xcol: str = "price") -> pd.DataFrame:
    """
    O(n log n) Pareto frontier.
    Sort by `xcol` (price, or cost per task) ascending; keep a model only if
    its quality exceeds every cheaper model seen so far.

    The secondary quality key is load-bearing, not cosmetic. A bare
    ``sort_values("price")`` is an unstable quicksort, so when two models share a
    price their row order decided which was seen first — and if the *worse* one
    came first it was appended, putting a strictly dominated model on a line
    labelled "Pareto Frontier". Sorting quality descending within a price tie
    means only the best model at that price can ever clear ``max_q``.
    """
    if xcol not in df.columns:
        return pd.DataFrame(columns=df.columns)
    sub = df[(df[xcol] > 0) & (df["quality"] > 0)].sort_values(
        [xcol, "quality"], ascending=[True, False], kind="stable"
    )
    pareto, max_q = [], 0.0
    for _, row in sub.iterrows():
        if row["quality"] > max_q:
            pareto.append(row)
            max_q = float(row["quality"])
    return pd.DataFrame(pareto) if pareto else pd.DataFrame(columns=sub.columns)


def _plottable(df: pd.DataFrame, xcol: str = "price") -> pd.DataFrame:
    """The rows this chart draws: priced, scored, one row per model family.

    On the task axis a row also needs AA's cost per task, and the filter runs
    BEFORE the family dedupe, so a family is drawn by its best-scoring variant
    that AA has costed rather than dropped because its top tier was not. With
    EFFORT set the tier is chosen upstream, so the callers select from costed
    rows only (static_helpers.task_axis_frames) for the same guarantee."""
    sub = df[
        (df["price"] > 0) &
        (df["quality"] > 0) &
        df["price"].notna() &
        df["quality"].notna()
    ].copy()
    if xcol == "cost_per_task":
        sub["cost_per_task"] = task_costs(sub)
        sub = sub[sub["cost_per_task"].notna()]
    return dedupe_to_best_variant(sub)


# Unmeasured-speed markers draw at the median of BUBBLE_MIN_PX..BUBBLE_MAX_PX
# area, so their size claims nothing about speed in either direction.
UNMEASURED_PX = 12.0


def _speed_measured(df: pd.DataFrame) -> pd.Series:
    """True where AA has published a throughput; it writes 0 until it has one."""
    return df["speed"].notna() & (df["speed"] > 0)


def _top_providers(df: pd.DataFrame, n: int = 10) -> set:
    """Return the top-n providers by model count; rest become 'Other'."""
    counts = df["provider"].value_counts()
    return set(counts.head(n).index)


def build_pareto_scatter(df: pd.DataFrame, full_df: pd.DataFrame | None = None,
                         basis=None, x: str = "price",
                         count_df: pd.DataFrame | None = None) -> go.Figure:
    """Build the Cost vs Quality Pareto scatter figure.

    ``full_df`` is the unfiltered catalogue. The frontier is a claim about the
    *market* — "nothing beats this on price and quality" — so it must be derived
    from every model, not from whatever survived the user's filter. Computing it
    from the filtered frame let one click promote a model the market already
    dominates onto a line labelled "Pareto Frontier".

    ``basis`` is the workload MIX (data/workload_mix.Mix) the frame was priced
    at; every price label names it. None = the dashboard's default blend.

    ``x`` is "price" (default) or "task": AA's cost per Intelligence Index
    task on the x axis. In task mode the MIX only reaches the hover's price
    line; the axis, frontier and counts are AA's numbers.

    ``count_df`` (task mode only) is the filtered frame BEFORE the costed-only
    restriction, i.e. every family on screen; the title counts it. None = `df`.
    """
    basis = resolve_mix(basis)
    task = x == "task"
    xcol = X_COLUMNS["task" if task else "price"]
    plot_df = _plottable(df, xcol)
    ref_df = plot_df if full_df is None else _plottable(full_df, xcol)
    if plot_df.empty:
        # A bare pair of axes read as "broken"; say what matched nothing and
        # what to change, the way Budget does (constants.empty_state_figure).
        n_on_screen = len(_plottable(df if count_df is None else count_df))
        if task and n_on_screen:
            return empty_state_figure(
                "Cost per Task vs. Intelligence",
                f"Artificial Analysis has no cost per task for the {n_on_screen} "
                f"model famil{'y' if n_on_screen == 1 else 'ies'} matching these filters",
                "Switch X AXIS to Price, or widen the filters")
        return empty_state_figure(
            "Cost per Task vs. Intelligence" if task else "Cost vs. Intelligence",
            "No models match these filters",
            "Clear the search, widen PROVIDER, or lower MIN SCORE")

    plot_df["size"] = bubble_size(plot_df["speed"], BUBBLE_SPEED_REF).values

    valid_c = plot_df[plot_df[xcol] > 0]
    _corr_r = safe_corr(np.log10(valid_c[xcol]), valid_c["quality"])

    fig = go.Figure()

    plot_df, providers = spotlight_split(
        plot_df, order_df=legend_reference(df if full_df is None else full_df))
    for provider in providers:
        pdf = plot_df[plot_df["display_provider"] == provider]
        # Draw the biggest bubbles first so small ones land on top and stay
        # visible (and clickable) instead of disappearing underneath.
        # Name breaks size ties: an unstable sort drew equal bubbles in a
        # different stacking order in the CI build and in the browser.
        pdf = pdf.sort_values(["size", "model"], ascending=[False, True],
                              kind="mergesort")
        color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
        symbol = PROVIDER_SHAPES.get(provider, DEFAULT_SHAPE)

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Provider: %{customdata[1]}<br>"
            "Intelligence: %{y:.1f}<br>"
            + ("Cost / task: %{customdata[8]}  ·  AA-measured<br>" if task else "")
            + "Price: %{customdata[7]}/M tokens  ·  " + basis.short + "<br>"
            + ("Token burn: %{customdata[9]}  ·  at " + list_blend_short() + "<br>"
               if task else "")
            + "Rates: %{customdata[4]}<br>"
            "Speed: %{customdata[2]}<br>"
            "Latency (TTFT): %{customdata[3]}<br>"
            "Weights: %{customdata[10]}<br>"
            "<extra></extra>"
        )

        speed_str = pdf["speed"].apply(
            lambda s: f"{s:.0f} tok/s" if pd.notna(s) and s > 0 else "not yet measured"
        )
        latency_str = pdf["latency"].apply(
            lambda l: f"{l:.2f}s" if pd.notna(l) and l > 0 else "N/A"
        )

        # [5] and [6] are the RAW model and provider for the click
        # handler's lookup; [0] and [1] are escaped for the hover text.
        # [7] is the price as text (workload_mix.fmt_price), so a sub-cent
        # agentic price never hovers as "$0.004"; it carries a '*' and [4]
        # a key line where the model has no AA cache price.
        # [8] AA's cost per task and [9] the token burn (data/task_cost), as
        # text, in BOTH x modes so every index after them means one thing.
        # [10] open or closed weights and where that came from
        # (data/open_weights.weights_label). [10] is the last index.
        rates = [r + h for r, h in zip(_rate_pair(pdf), fill_hover(pdf))]
        customdata = pd.Series(list(zip(
            pdf["model"].map(plot_text), pdf["provider"].map(plot_text),
            speed_str, latency_str, rates,
            pdf["model"].astype(str), pdf["provider"].astype(str),
            price_marks(pdf),
            task_costs(pdf).map(fmt_task_cost),
            token_burn(pdf).map(burn_text),
            weights_labels(pdf),
        )), index=pdf.index)

        # A model AA has not timed yet has no size to draw: the floor would
        # claim "slowest in the catalogue", and a 7px mark at the same price as
        # a filled bubble is buried under it (Gemini 4 Argon, #2 by score,
        # vanished beneath GPT-6.1 Sol at $8.00). Draw those at the median
        # size, with a centre dot and a light rim, on a layer above every
        # filled bubble.
        # HOLLOW = open weights (data/open_weights.py), filled = closed;
        # colour stays the provider. Closed before open within each layer, so
        # the provider's legend swatch is the filled form when it has one.
        timed = _speed_measured(pdf)
        is_open = row_is_open(pdf)
        groups = []
        for t_flag, layer in ((True, 0), (False, 1)):
            for o_flag in (False, True):
                part = pdf[(timed == t_flag) & (is_open == o_flag)]
                sym = weights_symbol(symbol, o_flag, timed=t_flag)
                if t_flag:
                    size = part["size"]
                    line = (dict(width=1.6, color=color) if o_flag
                            else marker_outline())
                    opacity = 1.0 if o_flag else 0.8
                else:
                    size = [UNMEASURED_PX] * len(part)
                    line = dict(width=1.6, color=color if o_flag else UNTIMED_RIM)
                    opacity = 1.0
                groups.append((part, dict(symbol=sym, size=size, opacity=opacity,
                                          line=line), layer))
        legend_shown = False
        for part, marker, zorder in groups:
            if part.empty:
                continue
            fig.add_trace(go.Scatter(
                x=part[xcol],
                y=part["quality"],
                mode="markers",
                name=provider,
                legendgroup=provider,
                showlegend=not legend_shown,
                zorder=zorder,
                marker=dict(color=color, **marker),
                customdata=customdata.loc[part.index].tolist(),
                hovertemplate=hover,
            ))
            legend_shown = True

    # --- Pareto frontier line ---
    # Membership is decided against the full catalogue, then intersected with
    # what is on screen: a filter may HIDE a frontier point, never invent one.
    frontier_models = set(_pareto_frontier(ref_df, xcol)["model"])
    pareto_df = plot_df[plot_df["model"].isin(frontier_models)].sort_values(
        [xcol, "quality"], ascending=[True, False], kind="stable"
    )
    if not pareto_df.empty:
        fig.add_trace(go.Scatter(
            x=pareto_df[xcol],
            y=pareto_df["quality"],
            mode="lines",
            name="Pareto Frontier",
            line=dict(color="rgba(0,212,255,0.5)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=True,
            # Leads the legend: trailing it, its dashed swatch ran into the
            # "Other" label before it and read as one entry.
            legendrank=1,
        ))

    _zero = "rgba(255,255,255,0.06)"

    # Axis ranges from the UNFILTERED catalogue (ref_df), like the frontier: a
    # mark keeps its screen position under a filter, and the label placement
    # below needs to know where every mark lands.
    rng_df = ref_df if not ref_df.empty else plot_df
    if not rng_df.empty:
        lo_x, hi_x = float(rng_df[xcol].min()), float(rng_df[xcol].max())
        lx0, lx1 = math.log10(lo_x), math.log10(hi_x)
        pad_x = max(0.08, (lx1 - lx0) * 0.06)
        x_range = [lx0 - pad_x, lx1 + pad_x]
        q_lo, q_hi = float(rng_df["quality"].min()), float(rng_df["quality"].max())
        pad_y = max(2.0, (q_hi - q_lo) * 0.08)
        y_range = [max(0.0, q_lo - pad_y), q_hi + pad_y]
    else:
        x_range = y_range = None

    # Label the frontier models on screen, cheapest first: offset plates that
    # never cover a mark or another label (constants.place_point_labels).
    annotations = []
    if not pareto_df.empty and x_range:
        markers = list(zip(plot_df[xcol], plot_df["quality"],
                           [s if _ok else UNMEASURED_PX for s, _ok in
                            zip(plot_df["size"], _speed_measured(plot_df))]))
        cands = [(r[xcol], r["quality"], family_label(r["model"]),
                  dict(zip(plot_df["model"], [m[2] for m in markers])).get(r["model"], 12))
                 for _, r in pareto_df.iterrows()]
        annotations = place_point_labels(cands, markers, x_range, y_range, x_log=True,
                                         color="rgba(0,212,255,0.85)")

    _zero = "rgba(255,255,255,0.06)"

    if x_range:
        tickvals, ticktext = _log_price_ticks(10 ** x_range[0], 10 ** x_range[1])
        inside = [i for i, v in enumerate(tickvals)
                  if x_range[0] <= math.log10(v) <= x_range[1]]
        tickvals = [tickvals[i] for i in inside]
        ticktext = [ticktext[i] for i in inside]
    else:
        tickvals, ticktext = [], []

    # The note above the plot: what bubble size means, then the correlation.
    # Same place and same order in all three X AXIS views (quadrant.py builds
    # it with the same helper): it used to sit inside the plot's top-left on
    # Price and Cost / task and above the plot's right on Speed, and size
    # silently changed meaning between them.
    parts = []
    if _corr_r is not None:
        parts.append(f"r = {_corr_r:.2f}  (log {'cost/task' if task else 'price'} vs quality)")
    if task and _corr_r is not None:
        # Why the axis exists: list price per token mis-orders cost per task.
        # Ranked at the LIST blend whatever the MIX (as the token burn is):
        # the hover's Price line follows the MIX, and at agentic the share
        # computed on that price is 23%, not 18%.
        flips = rank_flip_share(plot_df)
        if flips is not None:
            parts.append(f"{flips[0]:.0%} of {flips[1]:,} pairs cheaper per token but "
                         f"dearer per task (list price, {list_blend_short()})")
    annotations.append(size_note("size = speed (tok/s)", parts))

    # Each key appears only when its encoding is on screen. Terse on purpose:
    # Plotly titles do not wrap, and the longer "hollow = open weights  ·
    # centre dot = speed not yet measured" pushed the tail of every Overview
    # subtitle off the card at 1366 px. The keys come LAST, after the counts.
    hollow = (("  ·  hollow = open" if row_is_open(plot_df).any() else "")
              + ("  ·  centre dot = no speed yet"
                 if (~_speed_measured(plot_df)).any() else ""))
    if task:
        # The count leads (Plotly titles do not wrap): how many families on
        # screen AA has costed, and how many it has not. None is imputed.
        # Families on screen = those in the unrestricted filtered frame, plus
        # any drawn only because a costed tier passed a filter the selected
        # tier failed, so "N of M" never goes negative.
        fams = set(_plottable(df if count_df is None else count_df)["model"]
                   .map(base_model_name))
        n_fam = len(fams | set(plot_df["model"].map(base_model_name)))
        title_text = (
            "Cost per Task vs. Intelligence"
            "  <span style='font-size:12px;color:#777777;font-weight:400'>"
            f"{len(plot_df)} of {n_fam} model families have an AA cost per task"
            f" ({n_fam - len(plot_df)} lack one; not imputed)"
            "  ·  shape = provider"
            + hollow + "</span>"
        )
        # Two lines: on one it overflowed a 390 px plot at both ends.
        x_title = ("Cost per Intelligence Index task  (USD, log)"
                   "<br>AA-measured  ·  MIX does not apply")
        # A paper-relative legend sits a fixed fraction of the plot height
        # under the axis; at 390 px that put its first row on the title's
        # second line. Pinned to the figure bottom, automargin keeps it clear.
        legend = {**legend_below(), "yref": "container", "y": 0,
                  "yanchor": "bottom"}
    else:
        title_text = (
            "Cost vs. Intelligence"
            "  <span style='font-size:12px;color:#777777;font-weight:400'>"
            # The fill note leads: Plotly titles do not wrap, and at 1300 px
            # the tail of this subtitle ran off the card.
            + fill_note(plot_df, basis)
            + f"  ·  {len(plot_df)} of {len(df)} rows, one per model family"
            + "  ·  shape = provider"
            + hollow
            + "</span>"
        )
        x_title = f"Price  (USD / 1M tokens, log)  ·  {basis.short}"
        legend = legend_below()

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        title=dict(
            text=title_text,
            font=dict(size=15, color="#f2f2f2", family=FONT, weight=600),
            x=0.0,
            xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(
                text=x_title,
                font=dict(color=AXIS, size=12),
                standoff=12,
            ),
            type="log",
            range=x_range,
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
            range=y_range,
            gridcolor=GRID,
            zerolinecolor=_zero,
            zerolinewidth=1,
            tickfont=dict(color=TICK, size=11, family=FONT),
            showgrid=True,
            showline=False,
            ticks="",
        ),
        legend=legend,
        margin={**CHART_MARGIN, "t": NOTE_MARGIN_T},
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

    # A frame without the columns (pre-2026-08 history rows) must still yield
    # one entry per row: zip() against [] silently emptied the hover data.
    missing = _pd.Series(float("nan"), index=df.index)
    out = []
    for pin, pout in zip(df.get("price_in", missing), df.get("price_out", missing)):
        a, b = _fmt(pin), _fmt(pout)
        out.append(f"{a} in  ·  {b} out" if a and b else "—")
    return out
