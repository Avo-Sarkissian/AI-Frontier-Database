"""Browser bridge: ports the Dash callbacks to JSON-returning functions.
Runs inside Pyodide. No dash/requests/flask imports.

Task 5 extends this module (local/image/video/agent-stack).
"""
import html
import json
import pandas as pd

from data.ingest import get_models
from data.local_models import (
    get_local_df, get_gpu_options, GPU_BY_NAME, QUANT_LEVELS,
    quant_options as local_quant_options,
    context_options as local_context_options,
    speed_mode_options as local_speed_mode_options,
    DEFAULT_VRAM_GB, DEFAULT_GPU_COUNT, DEFAULT_BANDWIDTH_GBPS,
    DEFAULT_CONTEXT_TOKENS, DEFAULT_SPEED_MODE,
    effective_bandwidth, tflops_for_gpu,
)
from data.image_models import get_image_df
from data.video_models import (
    get_video_df, filter_video_df, VIDEO_MODES, DEFAULT_MODE,
)
from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR
from components.charts.local_scatter import build_local_scatter
from components.charts.local_compat import build_local_compat
from components.charts.image_scatter import build_image_faceted
from components.charts.video_chart import build_video_rankings, build_video_scatter
from components.stack_recommender import build_stack_cards_html
from components.charts.pareto import build_pareto_scatter
from components.charts.quadrant import build_quadrant
from components.charts.treemap import build_treemap
from components.charts.provider_leaderboard import build_provider_leaderboard
from components.charts.rankings import build_rankings
from components.charts.bump_chart import build_value_leaders
from components.charts.radar import build_radar
from components.charts.cost_calc import build_cost_calc, cheapest_above
from data.workload_mix import (
    apply_mix, resolve_mix, fmt_price, mix_summary, table_price_header,
    cache_label, filled_flags, mark_price,
)
from data.task_cost import median_token_burn, detail_rows, detail_note
from static_helpers import (
    index_label,
    apply_filters,
    task_axis_frames,
    coerce_number,
    cap_compare_selection,
    csv_safe,
    export_frame_for_tab,
    gpu_hw_meta,
    gpu_preset_options,
    local_frame,
    compute_diverse5,
    ctx_to_k,
    quality_label,
    model_options,
    provider_options,
)

_DF = get_models()


def _base(mix=None):
    """The catalogue priced at `mix` (data/workload_mix). The default returns
    _DF itself, so an untouched page computes exactly what it always did.

    The mix is applied BEFORE any filter, and the unfiltered result is what the
    builders get as `full_df`: the Pareto frontier, the medians and the floor
    stat are market-wide claims, and at an agentic mix the market is priced
    differently, not just the visible slice of it."""
    return apply_mix(_DF, resolve_mix(mix))


def _apply_filters(providers, min_quality, search="", effort=None, mix=None, weights=None):
    return apply_filters(_base(mix), providers, min_quality, search, effort, weights)


# ── HTML-string mirrors of app.py's dash-component builders ──────────────────

def _build_raw_table_html(df: pd.DataFrame, selected: list, mix=None) -> str:
    """Mirror of app.py _build_raw_table (lines ~130–189) as an HTML string."""
    _FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    rows = df[df["model"].isin(selected)].copy()
    if rows.empty:
        return "<div></div>"

    header_style = (
        "padding:8px 14px;font-size:9px;letter-spacing:0.08em;color:#555;"
        f"font-family:{_FONT};font-weight:600;text-transform:uppercase;"
        "text-align:right;white-space:nowrap;"
    )
    header_left_style = header_style.replace("text-align:right", "text-align:left")
    cell_base = (
        f"padding:7px 14px;font-size:11px;font-family:{_FONT};"
        "border-bottom:1px solid rgba(255,255,255,0.04);text-align:right;white-space:nowrap;"
    )

    def _th(label, align="right"):
        style = header_left_style if align == "left" else header_style
        return f'<th style="{style}">{label}</th>'

    def _td(text, color="#888", extra_style=""):
        style = f"{cell_base}color:{color};{extra_style}"
        return f'<td style="{style}">{text}</td>'

    header = (
        "<tr>"
        + _th("Model", "left")
        + _th("Provider", "left")
        + _th("Intelligence")
        + _th(html.escape(table_price_header(mix), quote=True))
        + _th("In / Out ($/M)")
        + _th("Speed (tok/s)")
        + _th("Latency (TTFT)")
        + _th("Context")
        + "</tr>"
    )

    table_rows = []
    for _, r in rows.sort_values("quality", ascending=False).iterrows():
        pcolor = PROVIDER_COLORS.get(r["provider"], DEFAULT_COLOR)
        price_str = fmt_price(r["price"]) if pd.notna(r["price"]) and r["price"] > 0 else "—"
        rates_str = (
            f"{r['price_in']:,.2f} / {r['price_out']:,.2f}"
            if pd.notna(r.get("price_in")) and pd.notna(r.get("price_out"))
            and r["price_out"] > 0 else "—"
        )
        speed_str = f"{int(r['speed']):,}" if pd.notna(r["speed"]) and r["speed"] > 0 else "—"
        lat_str = f"{r['latency']:.2f}s" if pd.notna(r["latency"]) and r["latency"] > 0 else "—"
        # Scraped values are third-party text: escape before interpolating into HTML.
        ctx_str = html.escape(str(r["context"]), quote=True) if pd.notna(r.get("context")) else "—"
        model_str = html.escape(str(r["model"]), quote=True)
        provider_str = html.escape(str(r["provider"]), quote=True)

        model_td_style = (
            f"{cell_base}text-align:left;color:#ccc;max-width:260px;"
            "overflow:hidden;text-overflow:ellipsis;"
        )
        row_html = (
            "<tr>"
            + f'<td style="{model_td_style}">{model_str}</td>'
            + f'<td style="{cell_base}text-align:left;color:{pcolor};">{provider_str}</td>'
            + _td(f"{r['quality']:.1f}", "#f2f2f2")
            + _td(price_str)
            + _td(rates_str)
            + _td(speed_str)
            + _td(lat_str)
            + _td(ctx_str)
            + "</tr>"
        )
        table_rows.append(row_html)

    label_style = (
        "font-size:9px;letter-spacing:0.1em;color:#444;"
        f"font-family:{_FONT};padding:14px 14px 6px;font-weight:600;"
    )
    table_style = "width:100%;border-collapse:collapse;overflow-x:auto;"
    return (
        f'<div><div style="{label_style}">RAW VALUES</div>'
        f'<table style="{table_style}">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        f"</table></div>"
    )


def _weights_text(row: pd.Series) -> str:
    """'Open · AA flag' etc. (data/open_weights.weights_label)."""
    from data.open_weights import annotate_open_weights, weights_label
    if "open_weights" not in row.index:
        row = annotate_open_weights(row.to_frame().T).iloc[0]
    return weights_label(row["open_weights"], row.get("weights_source"))


def _cache_label(row: pd.Series) -> str:
    """See workload_mix.cache_label (shared with app.py)."""
    return cache_label(row)


def _detail_html(row: pd.Series, provider: str, mix=None) -> str:
    """Mirror of app.py toggle_detail_panel body (lines ~1259–1318) as HTML string."""
    _FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
    quality = float(row["quality"])
    q_max = float(_DF["quality"].max()) or 1.0
    quality_pct = quality / q_max * 100
    qlabel = quality_label(quality_pct)

    speed_val = row["speed"]
    speed_str = f"{int(speed_val):,} tok/s" if pd.notna(speed_val) and speed_val > 0 else "N/A"
    latency_val = row.get("latency", float("nan"))
    latency_str = f"{latency_val:.2f}s" if pd.notna(latency_val) and latency_val > 0 else "N/A"

    n_total = len(_DF[_DF["quality"] > 0])
    n_below = int((_DF["quality"] < quality).sum())
    pct = round(n_below / n_total * 100) if n_total else 0

    def _metric(lbl, val, accent=False):
        val_class_extra = " accent" if accent else ""
        return (
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
            f'<span style="font-size:9px;letter-spacing:0.08em;color:#555;'
            f'font-family:{_FONT};text-transform:uppercase;">{lbl}</span>'
            f'<span style="font-size:11px;color:{"#00d4ff" if accent else "#888"};'
            f'font-family:{_FONT};">{val}</span>'
            f'</div>'
        )

    # Intelligence bar section
    bar_track = (
        'style="width:100%;height:3px;background:rgba(255,255,255,0.06);'
        'border-radius:2px;margin-bottom:4px;"'
    )
    bar_fill = (
        f'style="width:{quality_pct:.1f}%;height:3px;'
        'background:linear-gradient(90deg,#00d4ff,#4c9eff);'
        'border-radius:2px;transition:width 0.4s ease;"'
    )
    # AA's 2026-09 overhaul flags an index it extrapolated rather than measured
    # — true for 105 of the 198 models this catalogue publishes. The panel
    # states one number as "this model's intelligence", so it is the surface
    # that must say which kind of number it is.
    estimated = bool(row.get("quality_estimated", False))
    est_mark = (
        f'<span style="font-size:9px;color:#7a7a7a;font-family:{_FONT};"> · est.</span>'
        if estimated else ""
    )
    intel_row = (
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:baseline;margin-bottom:6px;">'
        f'<span style="font-size:9px;letter-spacing:0.08em;color:#555;'
        f'font-family:{_FONT};text-transform:uppercase;">INTELLIGENCE</span>'
        f'<span style="font-size:11px;color:#00d4ff;font-family:{_FONT};">'
        f'{quality:.0f}  ·  {qlabel}{est_mark}</span>'
        f'</div>'
    )
    percentile_note = (
        "AA estimated this score — not the full evaluation suite"
        if estimated else f"Top {100 - pct}% of all models"
    )
    percentile_div = (
        f'<div style="font-size:10px;color:var(--text-3);margin-bottom:16px;">'
        f'{percentile_note}</div>'
    )
    intel_section = (
        f'<div>{intel_row}'
        f'<div {bar_track}><div {bar_fill}></div></div>'
        f'{percentile_div}</div>'
    )

    divider = '<div style="height:1px;background:rgba(255,255,255,0.06);margin:8px 0;"></div>'

    ctx_raw = (str(row.get("context", "N/A")) or "N/A") if pd.notna(row.get("context")) else "N/A"
    # Scraped values are third-party text: escape before interpolating into HTML.
    ctx_str = html.escape(ctx_raw, quote=True)
    provider_str = html.escape(str(provider), quote=True)
    model_str = html.escape(str(row["model"]), quote=True)
    basis = resolve_mix(mix)
    price_str = html.escape(
        f"{mark_price(row['price'], row.get('cache_price_filled'))} / 1M  ·  {basis.short}",
        quote=True)
    p_in, p_out = row.get("price_in"), row.get("price_out")
    rates_str = (
        f"${p_in:,.2f} in  ·  ${p_out:,.2f} out"
        if pd.notna(p_in) and pd.notna(p_out) and p_out > 0 else "—"
    )
    # AA's cost per task and the token burn it implies (data/task_cost). The
    # burn divides by the LIST blend and is compared with the median over the
    # unmixed catalogue, so neither moves with MIX; the note names the blend.
    task_html = "".join(_metric(lbl, html.escape(val, quote=True))
                        for lbl, val in detail_rows(row, median_token_burn(_DF)))
    note = detail_note(row)
    task_note = (
        f'<div style="font-size:10px;color:var(--text-3);margin-top:2px;">'
        f'{html.escape(note, quote=True)}</div>' if note else ""
    )

    return (
        f'<div style="color:{color};font-size:10px;font-family:{_FONT};">{provider_str}</div>'
        f'<div style="font-size:16px;font-family:{_FONT};color:#f2f2f2;margin-bottom:12px;">'
        f'{model_str}</div>'
        + intel_section
        + divider
        + _metric("Price", price_str)
        + _metric("Rates", rates_str)
        + _metric("Cached in", html.escape(_cache_label(row), quote=True))
        + task_html
        + task_note
        + _metric("Speed", speed_str)
        + _metric("Latency", latency_str)
        + _metric("Context", ctx_str)
        + _metric("Weights", html.escape(_weights_text(row), quote=True))
        + divider
        + _metric("Coding", index_label(row.get("coding")))
        + _metric("Agentic", index_label(row.get("agentic")))
        + _metric("Omniscience", index_label(row.get("omniscience")))
    )


# ── Ported callbacks ──────────────────────────────────────────────────────────

# Every hosted function takes `effort`, then `mix`, then `weights` LAST, all
# optional. docs/pyworker.js spreads the browser's argument list into these
# with no arity check, so a parameter inserted mid-signature shifts every later
# argument and fails silently in someone's browser. docs/app.js sends
# ..., effort, readMix(), weights in that order.

def update_overview(providers, min_quality, search, xaxis, effort=None, mix=None,
                    weights=None):
    """xaxis: "price" | "speed" | "task". On "task" the x axis is AA's cost per
    Intelligence Index task, which MIX does not re-price: `mix` reaches only
    the hover's price line."""
    basis = resolve_mix(mix)
    base = _base(basis)
    # `base` is the unfiltered catalogue at this mix: medians, axis bounds and
    # frontier membership are market-wide claims and must not move with the
    # user's filter.
    if xaxis == "task":
        costed, every = task_axis_frames(base, providers, min_quality,
                                         search or "", effort, weights)
        return build_pareto_scatter(costed, full_df=base, basis=basis, x="task",
                                    count_df=every).to_json()
    f = apply_filters(base, providers, min_quality, search or "", effort, weights)
    if xaxis == "speed":
        fig = build_quadrant(f, full_df=base, basis=basis)
    else:
        fig = build_pareto_scatter(f, full_df=base, basis=basis)
    return fig.to_json()


def update_treemap(providers, min_quality, search, effort=None, mix=None, weights=None):
    return build_treemap(_apply_filters(providers, min_quality, search or "", effort, mix,
                                        weights),
                         basis=resolve_mix(mix)).to_json()


def update_provider_leaderboard(providers, min_quality, search, effort=None, mix=None,
                                weights=None):
    return build_provider_leaderboard(
        _apply_filters(providers, min_quality, search or "", effort, mix, weights),
        basis=resolve_mix(mix),
    ).to_json()


def update_rankings(providers, min_quality, search, sort_by, effort=None, mix=None,
                    weights=None):
    from static_helpers import ranking_frame
    basis = resolve_mix(mix)
    f = apply_filters(ranking_frame(_base(basis)), providers, min_quality, search or "",
                      effort, weights)
    return build_rankings(f, top_n=min(25, len(f)), metric=sort_by or "intelligence",
                          basis=basis).to_json()


def update_value_leaders(providers, min_quality, search, effort=None, mix=None,
                         weights=None):
    return build_value_leaders(
        _apply_filters(providers, min_quality, search or "", effort, mix, weights),
        basis=resolve_mix(mix),
    ).to_json()


def update_compare(providers, min_quality, search, selected_models, triggered, effort=None,
                   mix=None, weights=None):
    basis = resolve_mix(mix)
    base = _base(basis)
    f = apply_filters(base, providers, min_quality, search or "", effort, weights)
    options = model_options(f)
    capped = cap_compare_selection(selected_models, f, triggered)
    return json.dumps({
        "figure": json.loads(build_radar(f, capped, full_df=base, basis=basis).to_json()),
        "options": options,
        "value": capped,
        "raw_table_html": _build_raw_table_html(f, capped, basis),
    })


def update_cost_calc(monthly_tokens_m, providers, min_quality, search, min_intelligence=0,
                     effort=None, mix=None, weights=None):
    """`min_intelligence` is the Budget tab's own floor. It composes with the
    global MIN SCORE filter rather than replacing it, so the effective floor is
    whichever is higher — the same as any two filters intersecting.

    The monthly cost is tokens x the price at `mix`, so the cheapest answer
    moves with the workload; `basis` names the mix for the answer card."""
    basis = resolve_mix(mix)
    f = _apply_filters(providers, min_quality, search or "", effort, basis, weights)
    # 0 is a volume the user typed; only a blank box means "not given". Negative
    # input is clamped — the HTML min attribute is a hint, not a guard.
    tokens = coerce_number(monthly_tokens_m, default=1.0, minimum=0.0)
    floor = coerce_number(min_intelligence, default=0.0, minimum=0.0)
    fig = build_cost_calc(f, monthly_tokens_m=tokens, min_quality=floor, basis=basis)
    return json.dumps({
        "figure": json.loads(fig.to_json()),
        "best": cheapest_above(f, min_quality=floor, monthly_tokens_m=tokens),
        "floor": floor,
        "basis": basis.short,
    })


def update_table(providers, min_quality, search, sort_col, sort_dir, effort=None, mix=None,
                 weights=None):
    f = _apply_filters(providers, min_quality, search or "", effort, mix, weights).copy()
    f["value"] = f.apply(
        lambda r: r["quality"] / r["price"] if r["price"] > 0 else None, axis=1
    )
    col = sort_col or "quality"
    asc = (sort_dir or "desc") == "asc"
    if col == "context":
        f["_ctx_k"] = f["context"].map(ctx_to_k)
        f = f.sort_values("_ctx_k", ascending=asc, na_position="last")
    else:
        f = f.sort_values(col, ascending=asc, na_position="last")
    cols = ["model", "provider", "quality", "value", "price", "speed", "latency", "context"]
    # Off the default mix, each row says whether its price rests on the
    # no-cache-price fill-in, so the table can mark it like every chart does.
    if "cache_price_filled" in f.columns:
        cols.append("cache_price_filled")
        f["cache_price_filled"] = filled_flags(f)
    return json.dumps(f[cols].to_dict("records"))


def export_csv(providers, min_quality, search, tab=None, effort=None, local_args=None,
               mix=None, weights=None):
    """See static_helpers.export_frame_for_tab — ↓CSV must export the dataset
    on screen, not always the hosted-LLM table.

    `effort`, `local_args`, `mix` and `weights` are APPENDED, like
    update_local's late parameters: pyworker.js spreads the argument list with
    no arity check.
    `local_args` is the same list docs/app.js sends to update_local."""
    frame, _name = export_frame_for_tab(tab, _DF, providers, min_quality, search,
                                        effort=effort, local_args=local_args, mix=mix,
                                        weights=weights)
    return csv_safe(frame).to_csv(index=False)


def export_csv_filename(tab=None):
    _frame, name = export_frame_for_tab(tab, _DF.head(0), None, 0, "")
    return name


def model_detail(model_name, provider, mix=None):
    """Mirror toggle_detail_panel body as an HTML string, priced at `mix`."""
    basis = resolve_mix(mix)
    base = _base(basis)
    rows = base[base["model"] == model_name]
    if rows.empty:
        return ""
    return _detail_html(rows.iloc[0], provider, basis)


def mix_info(mix=None):
    """The header stat, its label and the table's price header for `mix`.

    The floor is the minimum effective price over the UNFILTERED catalogue —
    the same market-wide meaning the stat always had — and at the default it is
    the manifest's floor_price string exactly, so nothing jumps on boot.
    `floor_filled` is true when that model has no AA cache price (its cache
    hits are billed at the full input price), and `fill_key` is the visible
    key for the '*' that marks it."""
    return json.dumps(mix_summary(_DF, mix))


# ── Task 5: Local / Image / Video / Agent-Stack callbacks ─────────────────────

def update_local(vram_per_gpu, num_gpus, quant, bandwidth_gbps, hw_type, tags,
                 ctx_tokens=None, speed_mode=None, fp16_tflops=None):
    """Mirror app.py update_local_charts, returns {"scatter":.., "compat":..}.

    The three new parameters are APPENDED with None defaults, deliberately.
    docs/pyworker.js calls staticApi[fn](*args) with no whitelist, no arity
    check and no type check, so inserting a parameter mid-signature shifts every
    later argument and surfaces only as a console line in someone's browser.
    """
    # static_helpers.local_frame is shared with the Run Local ↓CSV, so the
    # export is computed for exactly the hardware this chart shows.
    ldf, vram_gb, ctx, quant = local_frame(vram_per_gpu, num_gpus, quant, bandwidth_gbps,
                                           hw_type, tags, ctx_tokens, fp16_tflops)
    mode = speed_mode or DEFAULT_SPEED_MODE
    return json.dumps({
        "scatter": json.loads(build_local_scatter(ldf, vram_gb=vram_gb, quant=quant,
                                                  ctx_tokens=ctx, speed_mode=mode).to_json()),
        "compat":  json.loads(build_local_compat(ldf, quant=quant, vram_gb=vram_gb,
                                                 ctx_tokens=ctx, speed_mode=mode).to_json()),
    })


def local_hw_for_gpu(gpu_name):
    """Mirror app.py update_local_hw — returns hw metadata for a GPU preset.

    The browser no longer calls this on a preset change (gpu_options ships the
    same dict per preset at boot); it stays as the single-preset lookup.
    """
    # fp16_tflops MUST be in the dict. It is the only route a compute figure
    # takes to the browser — app.py reads GPU_BY_NAME server-side and does not
    # need it — so omitting it makes the deployed site compute bandwidth-only
    # speeds while Dash computes roofline speeds for the same card.
    return json.dumps(gpu_hw_meta(gpu_name))


def context_options():
    """Return JSON list of {label, value} for the CONTEXT control."""
    return json.dumps(local_context_options())


def speed_mode_options():
    """Return JSON list of {label, value} for the SPEED control.

    Single-stream or aggregate-at-optimal-concurrency: the two are 2-4x apart,
    so the tab shows one at a time and this control names which.
    """
    return json.dumps(local_speed_mode_options())


def gpu_options():
    """Return JSON list of GPU option dicts for the preset dropdowns.

    Each option carries its preset's vram_gb / bandwidth_gbps / hw_type /
    fp16_tflops, so a preset change resolves in the browser in the same tick
    instead of queueing a lookup behind a multi-second update_local.
    """
    return json.dumps(gpu_preset_options())


def quant_options():
    """Return JSON list of {label, value} for the quantization control.

    Labels, not bare strings: Q3/Q2 carry a "(lossy)" marker, and the marker has
    to reach the published site — see data/local_models.QUANT_LOSSY for why the
    cost is disclosed here rather than subtracted from the score.
    """
    return json.dumps(local_quant_options())


def update_image(providers, tags):
    """Mirror app.py update_image_charts — returns image faceted figure JSON."""
    full = get_image_df()
    d = full
    if providers:
        d = d[d["provider"].isin(list(providers))]
    if tags:
        for tag in tags:
            if tag == "open_weights":
                d = d[d["open_weights"] == True]
            else:
                d = d[d["tags"].apply(lambda t: tag in t)]
    # Which ELO column each facet reads is decided against the whole arena, so a
    # provider filter cannot silently swap a facet onto a retired 2025 metric.
    return build_image_faceted(d, full_df=full).to_json()


def update_video(providers=None, tags=None, mode=None):
    """Mirror app.py update_video_charts — returns {"rankings":.., "scatter":..}.

    ``mode`` is last and optional so the two-argument call sites that predate the
    text-to-video / image-to-video split keep working.
    """
    mode = mode or DEFAULT_MODE
    full = get_video_df(mode)
    d = filter_video_df(full, providers, tags)
    # Both charts anchor on the unfiltered arena: the ranked axis so distances
    # do not rescale under a filter, the frontier so filtering can hide a point
    # but never promote one the market already dominates.
    return json.dumps({
        "rankings": json.loads(
            build_video_rankings(d, full_df=full, mode=mode).to_json()),
        "scatter":  json.loads(
            build_video_scatter(d, full_df=full, mode=mode).to_json()),
    })


def video_modes():
    """Return JSON list of {label, value} for the video mode control."""
    return json.dumps(list(VIDEO_MODES))


def update_recommend(selected, mode, gpu_preset, vram_per_gpu, num_gpus, quant):
    """Mirror app.py update_recommend — returns {"cards_html":.., "show_providers":bool, "show_hw":bool}."""
    mode = mode or "api"
    show_providers = mode != "local"
    show_hw = mode in ("hybrid", "hybrid2", "local")

    # Resolve providers for API tiers
    if mode == "local":
        providers = None
    elif not selected:
        providers = []
    elif "__all__" in selected:
        providers = None
    else:
        providers = list(selected)

    # Build local_df when needed (hybrid / local modes)
    local_df = None
    if show_hw:
        meta = GPU_BY_NAME.get(gpu_preset or "", {})
        gpu_count = int(coerce_number(num_gpus, default=DEFAULT_GPU_COUNT, minimum=1))
        vram_gb = coerce_number(vram_per_gpu, default=DEFAULT_VRAM_GB, minimum=0.0) * gpu_count
        bw = coerce_number(meta.get("bandwidth_gbps"), default=DEFAULT_BANDWIDTH_GBPS, minimum=0.0)
        eff_bw = effective_bandwidth(bw, gpu_count)
        local_df = get_local_df(
            quant=quant or "Q4",
            vram_gb=vram_gb,
            bandwidth_gbps=eff_bw,
            hw_type=meta.get("hw_type", "nvidia"),
            # Same pinned default as app.py's update_recommend, so the two tabs
            # cannot price the same hardware differently.
            ctx_tokens=DEFAULT_CONTEXT_TOKENS,
            fp16_tflops=tflops_for_gpu(gpu_preset or ""),
            gpu_count=gpu_count,
        )

    cards = build_stack_cards_html(_DF, providers, mode=mode, local_df=local_df)
    return json.dumps({"cards_html": cards, "show_providers": show_providers, "show_hw": show_hw})


def weights_options():
    """Return JSON list of {label, value} for the WEIGHTS control."""
    from static_helpers import weights_options as _opts
    return json.dumps(_opts())


def effort_options():
    """Return JSON list of {label, value} for the EFFORT control."""
    from static_helpers import effort_options as _opts
    return json.dumps(_opts())
