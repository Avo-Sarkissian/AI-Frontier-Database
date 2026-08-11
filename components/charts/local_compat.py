"""
Local models — compatible models ranked bar chart.

Shows only models that fit in the user's VRAM, ranked by quality score.
Each bar is colored by model family.
Right-side annotations show estimated tok/s and VRAM required.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT, unique_labels, right_gutter, fit_text, ANNOTATED_AXIS_HEADROOM
from data.local_models import FAMILY_COLORS, DEFAULT_FAMILY_COLOR


def build_local_compat(df: pd.DataFrame, quant: str) -> go.Figure:
    """
    Horizontal bar chart of quality scores for models that fit the user's hardware.
    df is the output of data.local_models.get_local_df(), pre-filtered.
    """
    runnable = df[df["fits"].isin(["yes", "tight"])].copy()

    if runnable.empty:
        return _empty(
            "No models fit your current VRAM. "
            "Try lowering the quantization (e.g. Q4 → Q3) or adding more GPUs."
        )

    runnable = runnable.sort_values("quality", ascending=True).reset_index(drop=True)

    # Truncate long names, then force them distinct: several Nemotron variants
    # share a 34-character prefix, and two models on one category means Plotly
    # stacks both bars in the same row and buries one.
    runnable["short_name"] = unique_labels(
        runnable["name"].apply(lambda n: n[:34] + "…" if len(n) > 34 else n).tolist()
    )

    colors  = runnable["family"].map(FAMILY_COLORS).fillna(DEFAULT_FAMILY_COLOR).tolist()
    opacity = [0.85 if f == "yes" else 0.50 for f in runnable["fits"]]

    fig = go.Figure()

    # Background track (full-width ghost bar for visual alignment)
    max_q = runnable["quality"].max() or 1
    fig.add_trace(go.Bar(
        y=runnable["short_name"],
        x=[max_q] * len(runnable),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Quality bars
    fig.add_trace(go.Bar(
        y=runnable["short_name"],
        x=runnable["quality"],
        orientation="h",
        marker=dict(
            color=colors,
            opacity=opacity,
            line=dict(width=0),
        ),
        customdata=runnable[["name", "family", "vram_req_gb", "speed_tps",
                              "license", "context_k", "tags_str", "fits"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Family: %{customdata[1]}<br>"
            "Intelligence: %{x:.0f}<br>"
            "VRAM needed: %{customdata[2]:.1f} GB<br>"
            "Speed: %{customdata[3]:.0f} tok/s<br>"
            "License: %{customdata[4]}<br>"
            "Context: %{customdata[5]}k tokens<br>"
            "Tags: %{customdata[6]}<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    _gutter = right_gutter(
        f"{s:.0f} tok/s  ·  {v:.1f} GB  ⚠ tight"
        for s, v in zip(runnable["speed_tps"], runnable["vram_req_gb"])
    )
    # Right-side annotations: speed + VRAM
    for i, row in runnable.iterrows():
        color = FAMILY_COLORS.get(row["family"], DEFAULT_FAMILY_COLOR)
        speed_str = f"{row['speed_tps']:.0f} tok/s" if row["speed_tps"] > 0 else "–"
        vram_str  = f"{row['vram_req_gb']:.1f} GB"
        tight_tag = "  ⚠ tight" if row["fits"] == "tight" else ""

        fig.add_annotation(
            x=1.01,
            y=row["short_name"],
            text=fit_text(f"{speed_str}  ·  {vram_str}{tight_tag}", _gutter, size_px=11),
            showarrow=False,
            xanchor="left",
            font=dict(size=11, family=_FONT, color=color),
            xref="paper", yref="y",
        )

    height = max(480, len(runnable) * 42 + 80)

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                f"Runnable Models  "
                f"<span style='font-size:11px;color:#666666;font-weight:400'>"
                f"  ·  {len(runnable)} models fit your hardware  ·  ranked by intelligence</span>"
            ),
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=12), standoff=12),
            range=[0, max_q * ANNOTATED_AXIS_HEADROOM],
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            tickfont=dict(color="#aaaaaa", size=12, family=_FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        ),
        barmode="overlay",
        bargap=0.35,
        margin=dict(l=20, r=_gutter, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )

    return fig


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        annotations=[dict(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=msg, showarrow=False,
            font=dict(color="#777777", size=13, family=_FONT),
            align="center",
        )],
        margin=dict(l=40, r=40, t=60, b=40), height=300,
    )
    return fig
