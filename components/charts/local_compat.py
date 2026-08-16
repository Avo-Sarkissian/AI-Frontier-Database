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


def _vram_note(vram_gb) -> str:
    """"…fit 32 GB", not "…fit your hardware".

    The old wording asserted a fact about the reader's machine while the figure
    behind it could be a default the reader never chose: clear the VRAM box on
    an 8 GB preset and the chart claimed 49 models fit, using a global 32 GB
    constant that appeared nowhere on screen while the GPU dropdown still read
    "NVIDIA RTX 5060". Naming the number makes the substitution visible.
    """
    try:
        return f"in {float(vram_gb):.0f} GB"
    except (TypeError, ValueError):
        return "your hardware"


def build_local_compat(df: pd.DataFrame, quant: str, vram_gb=None) -> go.Figure:
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

    # Unscored models sort to the TOP of an ascending chart, which would read as
    # "worst". They are not worst — they are unmeasured, and the two must not
    # look alike. Split them out, rank the scored ones, then re-append the
    # unscored group above the ranking with an outline-only bar that makes no
    # length claim at all.
    pending = runnable[runnable.get("pending", False) & runnable["quality"].isna()] \
        if "pending" in runnable.columns else runnable.iloc[0:0]
    scored = runnable.drop(index=pending.index)
    scored = scored.sort_values("quality", ascending=True)
    runnable = pd.concat([scored, pending]).reset_index(drop=True)

    # Truncate long names, then force them distinct: several Nemotron variants
    # share a 34-character prefix, and two models on one category means Plotly
    # stacks both bars in the same row and buries one.
    runnable["short_name"] = unique_labels(
        runnable["name"].apply(lambda n: n[:34] + "…" if len(n) > 34 else n).tolist()
    )

    colors  = runnable["family"].map(FAMILY_COLORS).fillna(DEFAULT_FAMILY_COLOR).tolist()
    opacity = [0.85 if f == "yes" else 0.50 for f in runnable["fits"]]
    is_pending = (
        runnable["pending"].fillna(False).tolist()
        if "pending" in runnable.columns else [False] * len(runnable)
    )

    fig = go.Figure()

    # Background track (full-width ghost bar for visual alignment)
    _mq = runnable["quality"].max()
    max_q = float(_mq) if pd.notna(_mq) and _mq > 0 else 1.0
    fig.add_trace(go.Bar(
        y=runnable["short_name"],
        x=[max_q] * len(runnable),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Unscored models: a full-width OUTLINE, no fill. An outline reads as "this
    # row exists and fits your hardware" without encoding a magnitude, which is
    # the honest rendering of a number nobody has measured. A zero-length bar
    # would have said "scored zero"; a filled one would have invented a score.
    if any(is_pending):
        fig.add_trace(go.Bar(
            y=[n for n, p in zip(runnable["short_name"], is_pending) if p],
            x=[max_q for p in is_pending if p],
            orientation="h",
            marker=dict(color="rgba(0,0,0,0)",
                        line=dict(color="rgba(255,255,255,0.28)", width=1)),
            hovertemplate="<b>%{y}</b><br>Not yet scored by Artificial Analysis"
                          "<extra></extra>",
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
            "VRAM needed: %{customdata[2]:.1f} GB  ·  weights only<br>"
            "Speed: %{customdata[3]:.0f} tok/s<br>"
            "License: %{customdata[4]}<br>"
            "Max context: %{customdata[5]}k tokens  ·  KV cache not included above<br>"
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
                f"  ·  {len(runnable)} models fit {_vram_note(vram_gb)}"
                f"  ·  ranked by intelligence"
                + (f"  ·  {int(sum(is_pending))} not yet scored (outlined)"
                   if any(is_pending) else "")
                + "</span>"
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
