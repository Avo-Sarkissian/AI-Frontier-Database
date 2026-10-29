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


def cost_notes(df: pd.DataFrame) -> list[str]:
    """The $/GPU-HR readout appended to the speed line, one per row ("" when
    the row has no price: no rate typed, a consumer card, or no throughput).

    Priced at the tab's max-throughput figure (its layer-split aggregate at the
    10 tok/s floor) and full use, so it says so, and says it is uncalibrated:
    the calibrated tensor-parallel datacenter figure is on Economics › Serve,
    and for the same card the two can differ by 2x or more."""
    n = len(df)
    usd = df["usd_per_m_out"] if "usd_per_m_out" in df else [float("nan")] * n
    rate = df["gpu_rate_usd_hr"] if "gpu_rate_usd_hr" in df else [float("nan")] * n
    gpus = df["gpu_count"] if "gpu_count" in df else [1] * n
    out = []
    for u, r, g in zip(usd, rate, gpus):
        try:
            u, r, g = float(u), float(r), int(g or 1)
        except (TypeError, ValueError):
            out.append("")
            continue
        if not (u == u and r == r and u > 0):
            out.append("")
            continue
        price = f"${u:,.2f}" if u >= 0.1 else f"${u:.3g}"
        out.append(f"<br>Serving cost: {price} / M output tok at ${r:,.2f}/GPU-hr ×{g}, "
                   "full use, at the max-throughput figure<br>"
                   "       (decode only, uncalibrated; datacenter figures are on Economics)")
    return out


# Rows the ranked chart draws. Past this the chart was a 100+ bar scroll whose
# only x axis sat ~4,000 px below the title (8x B200: 109 rows at 42 px each),
# so the bars could not be read against it. The rest are counted on screen and
# every row stays in the ↓CSV export.
MAX_COMPAT_ROWS = 30
_ROW_PX = 30
_NOT_SCORED = "unscored"


def one_per_family(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model family, its best-scoring variant — the rule Overview
    and the hosted rankings use (constants.select_effort with no effort).

    AA benchmarks one open-weight model at several reasoning efforts and the
    weights are identical, so "GLM-5.3 (Max)" and "GLM-5.3 (Low)" are the
    same download at the same VRAM and speed: listing both spent a row on a
    setting, not a model. Families are keyed by the name with its trailing
    parenthetical removed; an unscored row only represents a family that has
    no scored one. Stable, so ties resolve the same in every pandas build."""
    from components.charts.constants import base_model_name

    if df.empty or "name" not in df.columns:
        return df
    w = df.copy()
    w["_family_key"] = w["name"].astype(str).map(base_model_name)
    w["_q"] = pd.to_numeric(w["quality"], errors="coerce")
    w = w.sort_values(["_q", "name"], ascending=[False, True], kind="mergesort",
                      na_position="last")
    return w.drop_duplicates("_family_key", keep="first").drop(columns=["_family_key", "_q"])


def tab_colors(families) -> list[str]:
    """Each family's on-screen colour (data/local_models.tab_family_colors):
    the families the tab names keep theirs, the rest share the Other grey."""
    from data.local_models import tab_family_colors
    named = tab_family_colors()
    return [named.get(f, DEFAULT_FAMILY_COLOR) for f in families]


def build_local_compat(df: pd.DataFrame, quant: str, vram_gb=None,
                       ctx_tokens=None, speed_mode=DEFAULT_SPEED_MODE) -> go.Figure:
    """
    Horizontal bar chart of quality scores for models that fit the user's hardware.
    df is the output of data.local_models.get_local_df(), pre-filtered.

    Scored models rank from the top; models AA has not scored follow at the
    BOTTOM with no bar at all (a bar's length is a score, and they have none)
    and "not scored" in their label. Every label leads with the score.
    """
    runnable = df[df["fits"].isin(["yes", "tight"])].copy() if "fits" in df.columns \
        else df.iloc[0:0]

    if df.empty:
        return empty_state("Runnable Models", "No model matches these tags",
                           "Clear a tag filter to see the catalogue")
    if runnable.empty:
        return empty_state(
            "Runnable Models", f"No model fits {_vram_note(vram_gb)}",
            "Try a shorter context or a lower quantization (e.g. Q4 → Q3),<br>"
            "or add GPUs")

    runnable = one_per_family(runnable)
    n_fit = len(runnable)
    q = pd.to_numeric(runnable["quality"], errors="coerce")
    unscored = runnable[q.isna() | (q <= 0)]
    scored = runnable.drop(index=unscored.index)
    # Best first, then the cap. Stable and fully specified: the default
    # quicksort ordered tied scores differently on the CI build and in Pyodide,
    # so the pre-rendered chart's tied bars swapped places a second after load
    # when the live render landed.
    scored = scored.sort_values(["quality", "name"], ascending=[False, True],
                                kind="mergesort")
    unscored = unscored.sort_values("name", kind="mergesort")
    shown_scored = scored.head(MAX_COMPAT_ROWS)
    shown_unscored = unscored.head(max(0, MAX_COMPAT_ROWS - len(shown_scored)))
    n_omitted = n_fit - len(shown_scored) - len(shown_unscored)
    # Plotly draws a horizontal bar chart bottom-up: unscored first (bottom),
    # then the scored ranking from weakest to best (top).
    runnable = pd.concat([shown_unscored.iloc[::-1], shown_scored.iloc[::-1]]) \
        .reset_index(drop=True)
    is_pending = (pd.to_numeric(runnable["quality"], errors="coerce").isna()
                  | (pd.to_numeric(runnable["quality"], errors="coerce") <= 0)).tolist()

    # Truncate long names, then force them distinct: several Nemotron variants
    # share a 34-character prefix, and two models on one category means Plotly
    # stacks both bars in the same row and buries one.
    runnable["short_name"] = unique_labels(
        runnable["name"].apply(lambda n: n[:34] + "…" if len(n) > 34 else n).tolist()
    )

    colors = tab_colors(runnable["family"])
    opacity = [0.85 if f == "yes" else 0.50 for f in runnable["fits"]]

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
    runnable["speed_note"] = [s + c for s, c in zip(speed_notes(runnable, speed_mode),
                                                   cost_notes(runnable))]
    runnable["score_text"] = [_NOT_SCORED if p else f"{float(v):.1f}"
                              for v, p in zip(runnable["quality"], is_pending)]
    runnable["score_hover"] = ["not yet scored by Artificial Analysis" if p
                               else f"{float(v):.1f}"
                               for v, p in zip(runnable["quality"], is_pending)]
    for _c in ("weights_gb", "kv_gb", "sessions", "per_session_tps", "total_tps"):
        if _c not in runnable:
            runnable[_c] = 0

    fig = go.Figure()

    _mq = pd.to_numeric(runnable["quality"], errors="coerce").max()
    max_q = float(_mq) if pd.notna(_mq) and _mq > 0 else 1.0
    scored_rows = [not p for p in is_pending]

    # Background track behind SCORED rows only: an unscored row has no length
    # to sit on a scale, so it gets no track either.
    fig.add_trace(go.Bar(
        y=[n for n, s in zip(runnable["short_name"], scored_rows) if s],
        x=[max_q] * sum(scored_rows),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Quality bars. An unscored row carries x=None: its category (and label)
    # is on the axis, but nothing is drawn — no length, so no claimed score.
    fig.add_trace(go.Bar(
        y=runnable["short_name"],
        x=[None if p else float(v) for v, p in zip(runnable["quality"], is_pending)],
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
                              "overhead_gb", "score_hover", "speed_note"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Family: %{customdata[1]}<br>"
            "Intelligence: %{customdata[16]}<br>"
            "VRAM needed: %{customdata[2]:.1f} GB at %{customdata[11]} context<br>"
            "  ↳ weights %{customdata[8]:.1f} + KV cache %{customdata[9]:.1f}"
            " + runtime %{customdata[15]:.1f} GB  ·  %{customdata[10]}<br>"
            "%{customdata[17]}<br>"
            "License: %{customdata[4]}<br>"
            "Max context: %{customdata[5]}k tokens<br>"
            "Tags: %{customdata[6]}<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # ONE speed figure, the metric the reader selected. It briefly showed two —
    # single-stream on top and aggregate underneath — which put two tok/s
    # figures 2-4x apart in the same 200 px gutter with nothing on screen saying
    # which was which. The score leads, so the ranking's own number is printed
    # rather than read off a scale.
    def _label(row, score):
        v = row[_speed_col]
        speed_str = f"{v:,.0f} tok/s" if v > 0 else "–"
        tight_tag = " ⚠ tight" if row["fits"] == "tight" else ""
        # The session count rides directly after the throughput it qualifies,
        # on one space. The roomier "  (×24)" form pushed the label past the
        # 200 px gutter cap on 24 of 109 rows and got trimmed to "(×2…".
        sessions = ""
        if speed_mode == "throughput" and int(row.get("sessions") or 0) > 1:
            sessions = f" ×{int(row['sessions'])}"
        gb = row["vram_req_gb"]
        gb_str = f"{gb:.0f} GB" if gb >= 100 else f"{gb:.1f} GB"
        return f"{score} · {speed_str}{sessions} · {gb_str}{tight_tag}"

    # One label per row, computed once. Records rather than iterrows: iterrows
    # builds a Series per row, which was a measurable share of a build that
    # runs on every Run Local control change.
    _rows = runnable.to_dict("records")
    _labels = [_label(r, r["score_text"]) for r in _rows]
    # size_px to MATCH the fit_text call below. They once disagreed — the
    # gutter sized at 10 px/char and the labels trimmed at 11 — so every
    # annotation lost ~10% of the width it had actually been given. 10 px, the
    # gutter labels' size on every other ranked chart, so the score-led label
    # fits the shared 200 px cap untrimmed.
    _gutter = right_gutter(_labels, size_px=10)
    # Collected as plain dicts and set ONCE in the update_layout below. One
    # fig.add_annotation per row re-validated the whole annotations tuple on
    # every call, so the build was quadratic in runnable rows.
    annotations = [
        dict(
            x=1.01,
            y=row["short_name"],
            text=fit_text(text, _gutter, size_px=10),
            showarrow=False,
            xanchor="left",
            font=dict(size=10, family=_FONT,
                      color="#777777" if pend else col),
            xref="paper", yref="y",
        )
        for row, text, col, pend in zip(_rows, _labels, colors, is_pending)
    ]
    # Phone: the same rows, with the gutter label cut to score and VRAM.
    _narrow_labels = [f"{r['score_text']}  ·  {r['vram_req_gb']:.0f} GB"
                      for r, p in zip(_rows, is_pending)]
    _narrow_gutter = right_gutter(_narrow_labels, size_px=10, pad_px=10)
    narrow_annotations = [dict(a, text=t)
                          for a, t in zip(annotations, _narrow_labels)]
    if n_omitted > 0:
        omitted = dict(
            x=0, y=0, xref="paper", yref="paper", xanchor="left", yanchor="top",
            yshift=-50, showarrow=False, align="left",
            text=(f"+ {n_omitted} more that fit, lower-scored, not shown"
                  "  ·  every row is in ↓CSV"),
            font=dict(size=11, family=_FONT, color="#888888"),
        )
        annotations.append(omitted)
        narrow_annotations.append(dict(omitted, text=f"+ {n_omitted} more not shown (↓CSV)",
                                       font=dict(omitted["font"], size=10)))

    n_rows = len(runnable)
    height = max(320, n_rows * _ROW_PX + 120 + (24 if n_omitted > 0 else 0))
    bottom = 44 + (26 if n_omitted > 0 else 0)

    # Phone names: the 34-character label took half a 356 px plot.
    _narrow_names = [n if len(n) <= 20 else n[:19] + "…" for n in runnable["short_name"]]

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        title=dict(
            text=(
                f"Runnable Models  "
                f"<span style='font-size:11px;color:#666666;font-weight:400'>"
                f"  ·  {n_fit} model{'s' if n_fit != 1 else ''} fit {_vram_note(vram_gb)} "
                f"at {_ctx_label(ctx_tokens)} context"
                # The count above includes models that were only ever priced at
                # their own shorter maximum; say how many, so "fit at 128k" is
                # not read as "run at 128k".
                + (f" ({n_capped} only up to their own shorter max)"
                   if n_capped else "")
                + (f"  ·  top {len(shown_scored)} shown" if n_omitted > 0 else "")
                + "  ·  ranked by intelligence, best variant per model"
                f"  ·  tok/s = {_speed_label}"
                + (f"  ·  {int(sum(is_pending))} not yet scored (listed last)"
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
            # Pinned: categories otherwise order by first appearance across
            # traces, and the track trace (scored rows only) would put the
            # unscored rows on TOP of a ranking they are not part of.
            categoryorder="array", categoryarray=list(runnable["short_name"]),
        ),
        annotations=annotations,
        barmode="overlay",
        bargap=0.3,
        margin=dict(l=20, r=_gutter, t=52, b=bottom),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
        # docs/app.js narrowFigure applies this below max_width (a phone).
        meta={"narrow": {
            "max_width": 500,
            "data": {},
            "layout": {
                "annotations": narrow_annotations,
                "margin": {"r": _narrow_gutter, "l": 8},
                "yaxis": {"tickvals": list(runnable["short_name"]),
                          "ticktext": _narrow_names,
                          "tickfont": {"color": "#aaaaaa", "size": 10, "family": _FONT}},
            },
        }},
    )

    return fig


def empty_state(title: str, headline: str, hint: str, height: int = 320) -> go.Figure:
    """Run Local's empty state, in the Budget tab's pattern: the chart keeps
    its title, the axes go (a bare grid reads as "broken", and Plotly drew
    default -1..6 ticks on it), and the card says what is empty and what to
    change. Shared by both Run Local charts."""
    fig = go.Figure()
    fig.add_annotation(
        text=(f"{headline}<br><span style='font-size:12px;color:#777777'>{hint}</span>"),
        showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
        font=dict(size=15, family=_FONT, color="#999999"), align="center",
    )
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=52, b=36), height=height,
        title=dict(
            text=title,
            font=dict(size=14, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left", pad=dict(l=20, t=16),
        ),
    )
    return fig
