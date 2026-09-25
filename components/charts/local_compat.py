"""
Local models — compatible models ranked bar chart.

Shows only models that fit in the user's VRAM, ranked by quality score.
Each bar is colored by model family.
Right-side annotations show, on two lines: single-stream tok/s and the total
VRAM required at the selected context, then the concurrent sessions the same
hardware supports and the aggregate tok/s they produce.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT, unique_labels, right_gutter, fit_text, ANNOTATED_AXIS_HEADROOM
from data.local_models import (
    FAMILY_COLORS, DEFAULT_FAMILY_COLOR, DEFAULT_SPEED_MODE, DEFAULT_SLO,
    SLO_FLOORS_TPS, speed_columns,
)


# The fitted estimator's OUT-OF-SAMPLE error, as documented beside it in
# data/local_models.py ("mean 89%, median 33%, p90 272%, max 885%"). The hover
# said "±30%", which is roughly its median — the tail is where it goes wrong,
# and 3.5-10x misses on MLA and Mamba-hybrid rows were being presented as
# within a third. Quote the documented band, not a flattering midpoint.
# Shared with local_scatter.py so the two hovers cannot drift apart again.
ESTIMATED_KV_NOTE = "architecture estimated: median error 33%, p90 272%"
KV_NOTE = {"config": "published architecture",
           "hf": "published architecture",
           "estimated": ESTIMATED_KV_NOTE,
           "none": "no context priced"}


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


def _ctx_label(ctx_tokens) -> str:
    """"32k", not "32768" — and "the model's own max" when nothing was chosen."""
    try:
        n = int(ctx_tokens or 0)
    except (TypeError, ValueError):
        return "?"
    if n <= 0:
        return "0"
    return f"{n // 1024}k" if n >= 1024 else str(n)


def ctx_labels(df: pd.DataFrame, ctx_tokens) -> tuple[list[str], int]:
    """Per-row context label, and how many rows were priced below the selection.

    vram_breakdown caps each row's context at the model's own maximum, so on a
    128k selection Molmo 7B-D is priced at 4k — while the hover said "VRAM
    needed: 6.2 GB at 128k context" about a model that cannot take 5k tokens.
    The label now says what was priced: "4k (model max)".

    A row counts as capped only when its own figure reads smaller than the
    dropdown's. AA quotes context in decimal thousands (128k = 128,000) and the
    control in binary (128k = 131,072), so a 128k model on the 128k choice is
    priced 3,072 tokens short by unit alone; calling that "capped" would flag a
    third of the catalogue for a rounding difference.
    """
    sel = _ctx_label(ctx_tokens)
    try:
        sel_n = int(ctx_tokens or 0)
    except (TypeError, ValueError):
        sel_n = 0
    if "ctx_used" not in df.columns or sel_n <= 0:
        return [sel] * len(df), 0
    labels, capped = [], 0
    for used in pd.to_numeric(df["ctx_used"], errors="coerce").fillna(sel_n):
        used = int(used)
        own_k, sel_k = (used // 1000, sel_n // 1024) if sel_n >= 1024 else (used, sel_n)
        if used < sel_n and own_k < sel_k:
            capped += 1
            labels.append((f"{own_k}k" if sel_n >= 1024 and used >= 1000 else str(used))
                          + " (model max)")
        else:
            labels.append(sel)
    return labels, capped


def speed_notes(df: pd.DataFrame, speed_mode) -> list[str]:
    """The hover's speed line, one per row, leading with the metric chosen.

    Built here, not in the template, so a row with no session count renders a
    sentence instead of "Sessions: ×0 concurrent at 0 tok/s each → 0 tok/s
    total" — which is what the tooltip said for a model the chart had just
    listed as runnable. The hover leads with the metric the reader chose and
    offers the other one underneath, so the 2-4x gap between them is never a
    surprise and never an unlabelled second number.

    A single session has two different causes and they must not share a
    sentence. "kv_vram" means memory ran out after one; "latency" means even
    one stream misses the per-session floor — a 405B on 8x H100 decodes 8.6
    tok/s alone while its free VRAM would hold ~100 sessions, and "one session
    is all this fits" blamed the memory for what is a bandwidth limit.
    "latency" also covers a single stream that clears the floor when a second
    would not, so the sentence branches on the speed, not only the label.
    """
    n_rows = len(df)
    _single = df["speed_tps"] if "speed_tps" in df else [0] * n_rows
    _total = df["total_tps"] if "total_tps" in df else [0] * n_rows
    _sess = df["sessions"] if "sessions" in df else [0] * n_rows
    _bound = df["concurrency_bound"] if "concurrency_bound" in df else [""] * n_rows
    floor = SLO_FLOORS_TPS[DEFAULT_SLO]
    notes = []
    for s, t, n, b in zip(_single, _total, _sess, _bound):
        n = int(n or 0)
        if n > 1:
            notes.append(
                (f"Speed: {t:,.0f} tok/s across {n} sessions<br>"
                 f"       {s:,.0f} tok/s if you run one")
                if speed_mode == "throughput" else
                (f"Speed: {s:,.0f} tok/s single stream<br>"
                 f"       {t:,.0f} tok/s across {n} sessions"))
        elif b == "latency" and (s or 0) < floor:
            notes.append(f"Speed: {s:,.0f} tok/s — below the {floor:g} tok/s "
                         f"per-session floor even alone")
        elif b == "latency":
            # One stream clears the floor but two would not: the scan stopped
            # at B=1 on speed, not memory, so neither "even alone" (false —
            # this is above the floor) nor "all this fits" (blames VRAM) holds.
            notes.append(f"Speed: {s:,.0f} tok/s — a second session would drop "
                         f"each below the {floor:g} tok/s floor")
        else:
            notes.append(f"Speed: {s:,.0f} tok/s — one session is all this fits")
    return notes


def build_local_compat(df: pd.DataFrame, quant: str, vram_gb=None,
                       ctx_tokens=None, speed_mode=DEFAULT_SPEED_MODE) -> go.Figure:
    """
    Horizontal bar chart of quality scores for models that fit the user's hardware.
    df is the output of data.local_models.get_local_df(), pre-filtered.
    """
    runnable = df[df["fits"].isin(["yes", "tight"])].copy()

    if runnable.empty:
        return _empty(
            "No models fit your current VRAM. "
            "Try a shorter context, a lower quantization (e.g. Q4 → Q3), "
            "or adding more GPUs."
        )

    # Unscored models sort to the TOP of an ascending chart, which would read as
    # "worst". They are not worst — they are unmeasured, and the two must not
    # look alike. Split them out, rank the scored ones, then re-append the
    # unscored group above the ranking with an outline-only bar that makes no
    # length claim at all.
    pending = runnable[runnable.get("pending", False) & runnable["quality"].isna()] \
        if "pending" in runnable.columns else runnable.iloc[0:0]
    scored = runnable.drop(index=pending.index)
    # Stable and fully specified. The default quicksort ordered tied scores
    # differently on the CI build and in Pyodide, so the pre-rendered chart's
    # tied bars swapped places a second after load when the live render landed.
    scored = scored.sort_values(["quality", "name"], ascending=True, kind="mergesort")
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

    _speed_col, _speed_label = speed_columns(speed_mode)

    # Whether the KV figure two lines above came from a published config or the
    # fitted estimator. The reader has to be able to tell: an unlabelled
    # estimate beside an exact weights figure reads as though both were
    # measured. "config" is the hand-curated table, "hf" the model's own
    # config.json off HuggingFace. Both are published facts, so they read the
    # same; only the fitted estimate carries a warning, because it is the only
    # guess. See ESTIMATED_KV_NOTE for the band it quotes.
    runnable["kv_note"] = (runnable["kv_source"] if "kv_source" in runnable
                           else "none").map(KV_NOTE).fillna(ESTIMATED_KV_NOTE)
    runnable["ctx_label"], n_capped = ctx_labels(runnable, ctx_tokens)
    runnable["speed_note"] = speed_notes(runnable, speed_mode)
    for _c in ("weights_gb", "kv_gb", "sessions", "per_session_tps", "total_tps"):
        if _c not in runnable:
            runnable[_c] = 0

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
                              "license", "context_k", "tags_str", "fits",
                              "weights_gb", "kv_gb", "kv_note", "ctx_label",
                              "sessions", "per_session_tps", "total_tps",
                              "overhead_gb", "speed_note"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Family: %{customdata[1]}<br>"
            "Intelligence: %{x:.0f}<br>"
            "VRAM needed: %{customdata[2]:.1f} GB at %{customdata[11]} context<br>"
            "  ↳ weights %{customdata[8]:.1f} + KV cache %{customdata[9]:.1f}"
            " + runtime %{customdata[15]:.1f} GB  ·  %{customdata[10]}<br>"
            "%{customdata[16]}<br>"
            "License: %{customdata[4]}<br>"
            "Max context: %{customdata[5]}k tokens<br>"
            "Tags: %{customdata[6]}<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # ONE line, showing the metric the reader selected. It briefly showed two —
    # single-stream on top and aggregate underneath — which put two tok/s
    # figures 2-4x apart in the same 200 px gutter with nothing on screen saying
    # which was which.
    def _label(row):
        v = row[_speed_col]
        speed_str = f"{v:,.0f} tok/s" if v > 0 else "–"
        tight_tag = "  ⚠ tight" if row["fits"] == "tight" else ""
        # The session count rides directly after the throughput it qualifies,
        # on one space. The roomier "  (×24)" form pushed the label past the
        # 200 px gutter cap on 24 of 109 rows and got trimmed to "(×2…".
        sessions = ""
        if speed_mode == "throughput" and int(row.get("sessions") or 0) > 1:
            sessions = f" ×{int(row['sessions'])}"
        return f"{speed_str}{sessions}  ·  {row['vram_req_gb']:.1f} GB{tight_tag}"

    # One label per row, computed once. Records rather than iterrows: iterrows
    # builds a Series per row, which was a measurable share of a build that
    # runs on every Run Local control change.
    _rows = runnable.to_dict("records")
    _labels = [_label(r) for r in _rows]
    # size_px=11 to MATCH the fit_text call below. They disagreed — the gutter
    # was sized at 10 px/char and the labels trimmed at 11 — so every annotation
    # here lost ~10% of the width it had actually been given, and a throughput
    # label ended "816 tok/s · 22.3 GB (×2…" with room to spare beside it.
    _gutter = right_gutter(_labels, size_px=11)
    # Collected as plain dicts and set ONCE in the update_layout below. One
    # fig.add_annotation per row re-validated the whole annotations tuple on
    # every call, so the build was quadratic in runnable rows: 2.5 s natively
    # for 179 rows on 8x B200, and several times that in the Pyodide worker,
    # on every control change.
    annotations = [
        dict(
            x=1.01,
            y=row["short_name"],
            text=fit_text(text, _gutter, size_px=11),
            showarrow=False,
            xanchor="left",
            font=dict(size=11, family=_FONT,
                      color=FAMILY_COLORS.get(row["family"], DEFAULT_FAMILY_COLOR)),
            xref="paper", yref="y",
        )
        for row, text in zip(_rows, _labels)
    ]

    height = max(480, len(runnable) * 42 + 80)

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                f"Runnable Models  "
                f"<span style='font-size:11px;color:#666666;font-weight:400'>"
                f"  ·  {len(runnable)} models fit {_vram_note(vram_gb)} "
                f"at {_ctx_label(ctx_tokens)} context"
                # The count above includes models that were only ever priced at
                # their own shorter maximum; say how many, so "fit at 128k" is
                # not read as "run at 128k".
                + (f" ({n_capped} only up to their own shorter max)"
                   if n_capped else "")
                + "  ·  ranked by intelligence"
                f"  ·  tok/s = {_speed_label}"
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
        annotations=annotations,
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
