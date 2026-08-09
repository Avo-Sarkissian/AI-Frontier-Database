"""
Top-N model rankings — horizontal bar chart with tier separators.
Bars within 5% of the top score are bracketed in the same tier.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR, clean_model_name,
    BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT,
    empty_figure, unique_labels,
)


def _assign_tiers(values: pd.Series, gap_pct: float = 0.05) -> list[int]:
    """
    Assign tier numbers (1 = best) based on relative gaps.
    A new tier starts when the drop from the previous model exceeds gap_pct
    of the overall range.
    """
    vals = values.tolist()
    top  = vals[-1]   # series is bottom-to-top (reversed)
    rng  = top - vals[0] if top != vals[0] else 1
    tiers, current = [], 1
    prev = top
    for v in reversed(vals):
        if prev > 0 and (prev - v) / rng > gap_pct:
            current += 1
        tiers.append(current)
        prev = v
    return list(reversed(tiers))


def build_rankings(df: pd.DataFrame, top_n: int = 25, metric: str = "intelligence") -> go.Figure:
    """Horizontal bar chart of top-N models by the chosen metric."""
    valid = df[df["quality"] > 0].copy()

    if metric == "value":
        valid = valid[valid["price"] > 0].copy()
        valid["_metric"] = valid["quality"] / valid["price"]
        x_col, x_label, title_metric = "_metric", "AA Score per $ / 1M tokens", "Value (score/$)"
    elif metric == "speed":
        valid = valid[valid["speed"] > 0].copy()
        valid["_metric"] = valid["speed"]
        x_col, x_label, title_metric = "_metric", "Speed (tokens / second)", "Speed"
    else:
        valid["_metric"] = valid["quality"]
        x_col, x_label, title_metric = "_metric", "AA Intelligence Index", "Intelligence"

    if valid.empty:
        # Reachable from the UI: any filter combination matching nothing used to
        # raise IndexError out of _assign_tiers rather than render.
        return empty_figure()

    ranked = valid.sort_values("_metric", ascending=False).head(top_n).copy()
    # Reverse for bottom-up bar display
    ranked = ranked.iloc[::-1].reset_index(drop=True)

    colors = [PROVIDER_COLORS.get(p, DEFAULT_COLOR) for p in ranked["provider"]]
    # clean_model_name truncates at 32 chars, which collapsed distinct models
    # onto one categorical row — Plotly then stacked both bars there and the
    # shorter one could never be hovered.
    short_names = pd.Series(
        unique_labels(ranked["model"].apply(clean_model_name)), index=ranked.index
    )

    speed_str = ranked["speed"].apply(
        lambda s: f"{s:.0f} tok/s" if pd.notna(s) and s > 0 else "N/A"
    )
    price_str = ranked["price"].apply(
        lambda p: f"${p:.4f}/M" if pd.notna(p) and p > 0 else "N/A"
    )
    latency_str = ranked["latency"].apply(
        lambda l: f"{l:.2f}s" if pd.notna(l) and l > 0 else "N/A"
    )

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "Provider: %{customdata[1]}<br>"
        f"{title_metric}: %{{x:.1f}}<br>"
        "Price: %{customdata[2]}<br>"
        "Speed: %{customdata[3]}<br>"
        "Latency (TTFT): %{customdata[4]}<br>"
        "<extra></extra>"
    )

    max_metric = ranked["_metric"].max() or 1
    fig = go.Figure()

    # Background track
    fig.add_trace(go.Bar(
        y=short_names,
        x=[max_metric * 1.05] * len(ranked),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.03)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Actual bars
    fig.add_trace(go.Bar(
        y=short_names,
        x=ranked["_metric"],
        orientation="h",
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(width=0),
        ),
        customdata=list(zip(ranked["model"], ranked["provider"], price_str, speed_str, latency_str)),
        hovertemplate=hover,
        showlegend=False,
        text=ranked["_metric"].apply(lambda q: f"{q:.0f}"),
        textposition="inside",
        textfont=dict(color="rgba(255,255,255,0.6)", size=11, family=_FONT),
    ))

    # Provider label on the right (with price sub-label). Collected into a list
    # rather than added with fig.add_annotation, because update_layout(
    # annotations=tier_annotations) below REPLACES the layout's annotation
    # tuple — every one of these 25 labels was being silently discarded and the
    # chart shipped with only the two tier tags.
    annotations = []
    for i, row in ranked.iterrows():
        price_tag = (
            f"  <span style='color:#555'>${row['price']:.4f}/M</span>"
            if pd.notna(row["price"]) and row["price"] > 0 else ""
        )
        annotations.append(dict(
            x=max_metric * 1.06,
            y=short_names[i],
            # Provider identity is the text itself; keeping it at the chart's
            # body grey rather than the provider hue puts this 10px label at
            # 9.9:1 instead of the 3:1 several palette entries sit at.
            text=f"{row['provider']}{price_tag}",
            showarrow=False,
            xanchor="left",
            font=dict(size=10, family=_FONT, color="#999999"),
            xref="x", yref="y",
        ))

    # ── Tier separator lines ──────────────────────────────────────────────────
    # Separate tiers: a new tier begins when the gap between adjacent models
    # exceeds 5% of the full range. Draw a subtle dashed line between tiers.
    tiers = _assign_tiers(ranked["_metric"], gap_pct=0.08)
    tier_changes = []
    for i in range(1, len(tiers)):
        if tiers[i] != tiers[i - 1]:
            # Line falls between index i-1 and i (in the reversed/bottom-up list)
            tier_changes.append(i - 0.5)

    for y_pos in tier_changes:
        fig.add_shape(
            type="line",
            x0=0, x1=max_metric * 1.25,
            y0=y_pos, y1=y_pos,
            line=dict(color="rgba(255,255,255,0.10)", width=1, dash="dot"),
            xref="x", yref="y",
        )
        # Tier label
        tier_num = tiers[int(y_pos + 0.5)]
        annotations.append(dict(
            x=max_metric * 1.27,
            y=y_pos,
            text=f"<span style='color:#444'>T{tier_num}</span>",
            showarrow=False,
            xanchor="left",
            font=dict(size=8, family=_FONT, color="#444"),
            xref="x", yref="y",
        ))

    height = max(400, top_n * 26)

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                f"Top {top_n} Models by {title_metric}"
                f"  <span style='font-size:12px;color:#777777;font-weight:400'>"
                f"  ·  {x_label}  ·  dashed lines = tier boundaries (8% gap)</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(color=_AXIS, size=12), standoff=12),
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
            range=[0, max_metric * 1.3],
        ),
        yaxis=dict(
            tickfont=dict(color="#999999", size=11, family=_FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        ),
        barmode="overlay",
        margin=dict(l=20, r=120, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
        annotations=annotations,
    )

    return fig
