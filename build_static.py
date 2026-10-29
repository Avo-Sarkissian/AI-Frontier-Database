"""Pre-render default figures + bundle Python for the static Pyodide site."""
import hashlib, json, shutil, sys, zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from data.ingest import get_models, load_history
from data.local_models import (
    get_local_df, DEFAULT_VRAM_GB, DEFAULT_BANDWIDTH_GBPS,
    DEFAULT_CONTEXT_TOKENS, DEFAULT_FP16_TFLOPS,
)
from data.image_models import get_image_df, get_image_providers, get_image_tags
from data.video_models import (
    get_video_df, get_video_providers, get_video_tags, VIDEO_MODES,
)
from components.charts.pareto import build_pareto_scatter
from components.charts.quadrant import build_quadrant
from components.charts.treemap import build_treemap
from components.charts.provider_leaderboard import build_provider_leaderboard
from components.charts.rankings import build_rankings
from components.charts.bump_chart import build_value_leaders
from components.charts.radar import build_radar
from components.charts.cost_calc import build_cost_calc
from components.charts.local_scatter import build_local_scatter
from components.charts.local_compat import build_local_compat
from components.charts.image_scatter import build_image_faceted
from components.charts.video_chart import build_video_rankings, build_video_scatter
from components.charts.open_gap import (
    build_gap_history, build_release_lag, build_iso_price,
)
from data.open_gap import (
    gap_history, release_lag, iso_quality_ratios, snapshot_dates, score_dates,
)
from data import serving_cost, econ_common, gpu_tco
from data import open_gap as open_gap_data
from components.charts.econ_serve import build_econ_serve
from components.charts.econ_compute import build_econ_compute
from components.charts.lab_economics import build_labs_figures, labs_ledger_html
from data import lab_economics
from data.quality_scale import slider_max

from captions import CAPTIONS, method_payload
from data import scrape_status
from components.charts.constants import PROVIDER_ALIASES, AUTO_PROVIDER_COLORS
from static_helpers import (
    compute_diverse5, provider_options, model_options, quality_options,
    ranking_frame, effort_options, weights_options,
)
from data.workload_mix import (
    mix_options, DEFAULT_MIX_KEY, floor_label, fmt_floor, table_price_header,
)


def _load_coverage() -> dict:
    """What the scraper could not carry, so the stat tile can disclose it.
    Empty on a checkout that has not scraped since coverage was added."""
    path = Path(__file__).parent / "data" / "raw" / "coverage.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
FIG  = DOCS / "figures"

# The browser's Python arrives as two zips. pycode.zip (project modules plus
# vendored plotly, ~4 MB) changes only when code does, so it is keyed by a hash
# of its contents and cached by the worker. pydata.zip (the CSVs, ~100 KB) is
# rebuilt by every hourly refresh. One zip used to carry both, and the hourly
# version bump made every returning visitor re-download 4 MB of unchanged code.
CODE_BUNDLE = "pycode.zip"
DATA_BUNDLE = "pydata.zip"

DATA_CSVS = [
    "data/raw/aa_models.csv",
    "data/raw/aa_local_models.csv",
    # Attention geometry for the Run Local KV-cache term. It rides with the
    # other data files rather than the code because it is refreshed by the same
    # hourly bot, so a --data-only rebuild picks up newly resolved models
    # without a full rebuild.
    "data/raw/aa_local_arch.csv",
    "data/raw/aa_image_models.csv",
    "data/raw/aa_video_models.csv",
    # Automatic company colours (components/charts/auto_palette.py). Rides with
    # the data because the hourly bot adds to it whenever AA lists a new
    # company, and constants.py reads it at import in the browser too.
    "data/carried/aa_creator_colors.csv",
    # Open-weights name registry (data/open_weights.py): classifies a hosted
    # row that carries no AA flag of its own. Grown hourly by the bot.
    "data/carried/open_weight_names.csv",
    # SemiAnalysis InferenceX $/GPU-hr marks (13 rows, archived by the bot's
    # econ step): Economics › Serve and the Run Local $/GPU-HR presets.
    "data/carried/econ/inferencex_rates.csv",
    # Economics › Compute: hand-curated TCO inputs, financing presets and dated
    # rental price points, each value with source_url / as_of / status
    # (data/curated/), and the hyperscaler on-demand list prices the bot
    # fetches weekly (~8 rows).
    *gpu_tco.CURATED_FILES,
    gpu_tco.HYPERSCALER_FILE,
    # Economics › Labs: the hand-curated ledger (filings, press and analyst
    # figures, each with source_url / as_of / badge / status; ~20 KB).
    lab_economics.LEDGER_FILE,
]

# Build-time slices of the economics archives, written straight into
# pydata.zip (never committed; data/derived/ is git-ignored). The browser reads
# each through data/econ_common.read_derived, which computes the same CSV text
# from the archives wherever the file is absent (local runs, Dash, tests).
# The archives themselves never ship: the InferenceX points alone are ~700 KB.
# ORDER MATTERS: a later slice may read an earlier one (the calibration is
# fitted on the points slice), and shipped_derived() computes them in order.
DERIVED_DATA: dict[str, Callable[[], str]] = {
    serving_cost.DERIVED_POINTS: serving_cost.derived_points_csv,
    serving_cost.DERIVED_CALIBRATION: serving_cost.calibration_csv,
    # Economics › Labs ↓CSV: the rows of its three figures (the history they
    # come from never ships).
    open_gap_data.DERIVED_EXPORT: open_gap_data.export_csv_text,
    # Economics › Compute: GetDeploying's four weekly series for the nine GPUs
    # (last 104 weeks; the archive keeps every week), the newest SOFR fixing,
    # and the nine Epoch AI hardware rows (TDP, chip release price).
    gpu_tco.DERIVED_WEEKLY: gpu_tco.slim_weekly_csv,
    gpu_tco.DERIVED_SOFR: gpu_tco.sofr_latest_csv,
    gpu_tco.DERIVED_EPOCH: gpu_tco.epoch_specs_csv,
    # Economics › Labs: Epoch AI's annualised revenue rows and its
    # tokens-per-day rows, cut from the bot's Epoch mirrors (rows outside the
    # unit ranges are dropped with a warning, never raised).
    lab_economics.DERIVED_REVENUE: lab_economics.revenue_csv,
    lab_economics.DERIVED_TOKENS: lab_economics.tokens_csv,
}
# Each derived file, and the whole data zip, stay small: every visitor
# downloads pydata.zip on every hourly version. The Labs export grows by one
# score-gap row (~150 B) per daily snapshot, so it gets its own budget
# (~1.5 years of headroom; it deflates ~8x in the zip).
MAX_DERIVED_BYTES = 40_000
DERIVED_MAX_BYTES: dict[str, int] = {
    open_gap_data.DERIVED_EXPORT: 120_000,
    # 9 GPUs x 4 series x up to 104 weeks, ~35 B a row: ~45 KB with the 54
    # weeks archived on 2026-10-27, bounded by SERIES_WEEKS at ~131 KB if every
    # series were listed every week (it deflates ~6x in the zip).
    gpu_tco.DERIVED_WEEKLY: 140_000,
}
MAX_DATA_BUNDLE_BYTES = 150_000

# Do NOT import app.py — it starts background scrapers at import time. Shared
# pure logic lives in static_helpers (imported above).


def open_gap_figures(df) -> dict:
    """Economics › Labs: the open-vs-closed gap (data/open_gap.py).

    PRE-RENDER ONLY. The score-gap history reads every daily snapshot in
    data/raw/history/ (~4 MB), which never ships to the browser: it is
    aggregated here, at build time, into the figure. The release lag and the
    iso-quality price ratio use the current catalogue at the default workload
    mix — the Economics tab hides the global filter bar, MIX included, so no
    visitor setting could move them.
    """
    hist = load_history()
    # Every data/open_gap function treats a missing or empty history (a bare
    # pd.DataFrame()) as no snapshots, so this never aborts the build.
    dates = snapshot_dates(hist)
    asof = dates[-1] if dates else None
    return {
        "econ_open_gap_history": build_gap_history(*gap_history(hist)),
        "econ_open_gap_lag":     build_release_lag(*release_lag(df, asof,
                                                                score_dates(hist, df))),
        "econ_open_gap_price":   build_iso_price(*iso_quality_ratios(df, None)),
    }


def export_default_figures(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = get_models()
    diverse5 = compute_diverse5(df)
    local_df = get_local_df(quant="Q4", vram_gb=DEFAULT_VRAM_GB,
                            bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS,
                            hw_type="nvidia",
                            # Must match app.py's initial figures and the static
                            # shell's defaults. When they diverge the pre-rendered
                            # figure visibly changes a few seconds after every
                            # page load, as the live render replaces it.
                            ctx_tokens=DEFAULT_CONTEXT_TOKENS,
                            fp16_tflops=DEFAULT_FP16_TFLOPS)
    img_df = get_image_df()
    vdf = get_video_df()
    figures = {
        "pareto":               build_pareto_scatter(df),
        "quadrant":             build_quadrant(df),
        # Overview's "Cost / task" x axis; equals update_overview(..., "task").
        "pareto_task":          build_pareto_scatter(df, full_df=df, x="task"),
        "treemap":              build_treemap(df),
        "provider_leaderboard": build_provider_leaderboard(df),
        "rankings":             build_rankings(ranking_frame(df), top_n=25, metric="intelligence"),
        "value_leaders":        build_value_leaders(df),
        "radar":                build_radar(df, diverse5, full_df=df),
        "cost_calc":            build_cost_calc(df, monthly_tokens_m=1.0),
        "local_scatter":        build_local_scatter(local_df, vram_gb=DEFAULT_VRAM_GB, quant="Q4",
                                                    ctx_tokens=DEFAULT_CONTEXT_TOKENS),
        "local_compat":         build_local_compat(local_df, quant="Q4", vram_gb=DEFAULT_VRAM_GB,
                                                   ctx_tokens=DEFAULT_CONTEXT_TOKENS),
        "image_faceted":        build_image_faceted(img_df),
        # full_df on both, matching static_api.update_video. Without it the
        # pre-rendered default figure anchored its axis and its frontier on a
        # different reference set than the in-browser re-render, and the
        # worker's ready message re-renders with no user action — so the chart
        # visibly changed a few seconds after every page load.
        "video_rankings":       build_video_rankings(vdf, full_df=vdf),
        "video_scatter":        build_video_scatter(vdf, full_df=vdf),
        **open_gap_figures(df),
        # Economics › Serve at its defaults; equals static_api.update_econ_serve().
        "econ_serve":           build_econ_serve(*serving_cost.serving_data(hosted=df),
                                                 axis_range=serving_cost.axis_range(df)),
        # Economics › Compute at its defaults (econ_tco, econ_ladder,
        # econ_rental_series); equals static_api.update_econ_compute().
        **build_econ_compute(gpu_tco.compute()),
        # Economics › Labs (econ_labs_runrate, _margins, _commitments, _tokens,
        # _stack); equals static_api.update_econ_labs(). Labs has no controls.
        **build_labs_figures(),
    }
    written = []
    for fid, fig in figures.items():
        (out_dir / f"{fid}.json").write_text(fig.to_json())
        written.append(f"{fid}.json")
    # The Labs ledger is HTML, escaped in Python; app.js sets it as innerHTML.
    (out_dir / "econ_labs_ledger.json").write_text(json.dumps({"html": labs_ledger_html()}))
    written.append("econ_labs_ledger.json")

    _now = datetime.now(timezone.utc)
    manifest = {
        "model_count":      int(len(df)),
        "coverage":         _load_coverage(),
        "provider_count":   int(df["provider"].nunique()),
        # fmt_floor is the old f"${:.3f}" for any price above a cent, so the
        # stat is byte-identical at the default mix; static_api.mix_info
        # recomputes it (and its label) when the visitor changes MIX.
        "floor_price":      fmt_floor(df["price"].min()),
        "floor_price_label": floor_label(None),
        "table_price_header": table_price_header(None),
        # Workload MIX presets, each with its source and date in `title`
        # (data/workload_mix.MIX_PRESETS). The first is the default.
        "mix_options":      mix_options(),
        "default_mix":      DEFAULT_MIX_KEY,
        "peak_quality":     f"{df['quality'].max():.1f}",
        "diverse5":         diverse5,
        "provider_options": provider_options(df),
        "model_options":    model_options(df),
        "p75":              round(float(df["quality"].quantile(0.75)), 1),
        "p90":              round(float(df["quality"].quantile(0.90)), 1),
        "image_providers":  get_image_providers(),
        "video_providers":  get_video_providers(),
        # Tag options ship with the data so a control can never offer a tag the
        # pipeline stopped emitting (see data/image_models.get_image_tags).
        "image_tags":       get_image_tags(),
        "video_tags":       get_video_tags(),
        # The video arenas are separate Elo pools with separate prices, so the
        # mode control is data, not a hand-written pair of <option>s.
        "video_modes":      list(VIDEO_MODES),
        # Derived, not hand-copied: docs/app.js needs the retired->current
        # provider spellings to resolve an old ?p= share link, and a second
        # copy of this map in JS is exactly how palettes and labels drifted
        # before. Shipping it keeps PROVIDER_ALIASES the only source.
        "provider_aliases": dict(PROVIDER_ALIASES),
        # Colours for companies nobody hand-picked one for (auto_palette.py).
        # docs/app.js hand-mirrors only the hand-picked palette, so a company
        # AA adds reaches the JS-rendered views through here, without a code
        # change.
        "auto_provider_colors": dict(sorted(AUTO_PROVIDER_COLORS.items())),
        # One source for the prose too — the public site used to ship 1 of 14.
        "captions":         dict(CAPTIONS),
        "caption_methods":  method_payload(),
        # Per-dataset scrape status. `generated_iso` below is the BUILD time and
        # is kept only for cache-busting — the badge must report when the DATA
        # was last fetched, per dataset, or one succeeding scraper resets the
        # clock for all three. See data/scrape_status.py.
        "datasets":         scrape_status.load(),
        "data_fetched_iso": scrape_status.oldest_successful_fetch(),
        "stale_datasets":   scrape_status.stale_datasets(),
        # Shipped, not duplicated in JS. The browser re-checks staleness live
        # (a page left open must age honestly), which means it needs this
        # number — and a second copy of it in app.js is exactly how the palette
        # and the labels drifted. See data/scrape_status.STALE_AFTER_HOURS.
        "stale_after_hours": scrape_status.STALE_AFTER_HOURS,
        # MIN SCORE options, including the exact preset percentiles, so the
        # <select> can hold the value a preset sets instead of snapping it.
        # Reasoning effort: AA publishes the same model at several efforts and
        # the spread reaches 18.5 points, so which tier represents a model is a
        # user choice rather than a fixed one.
        "effort_options":  effort_options(),
        # WEIGHTS: All | Open | Closed (data/open_weights.WEIGHTS_OPTIONS).
        "weights_options": weights_options(),
        # Economics › Serve controls: GPU presets with their InferenceX rates,
        # the Own/Rent default and the USE slider's default/min/max/step. The
        # browser declares none of these numbers itself.
        "econ_serve_options": serving_cost.serve_options(),
        # Economics › Compute: GPU and financing-preset options, and per GPU /
        # preset the resolved default of each input box (its placeholder) with
        # its source, as-of and status (its tooltip).
        "econ_compute_options": gpu_tco.compute_options(),
        "quality_options": quality_options(
            round(float(df["quality"].quantile(0.75)), 1),
            round(float(df["quality"].quantile(0.90)), 1),
            peak=float(df["quality"].max())),
        # The Budget MIN INTELLIGENCE slider's max: the current peak, rounded
        # up (data/quality_scale.slider_max), not index v4's 65.
        "budget_slider_max": slider_max(float(df["quality"].max())),
        "generated":        _now.strftime("%b %d  %H:%M"),
        "version":          _now.strftime("%Y%m%dT%H%M%SZ"),
        "generated_iso":    _now.isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return written


def copy_css(docs: Path | None = None):
    """Copy assets/style.css into docs/, with a banner saying it is generated.

    The copy is git-tracked, byte-identical to its source, and overwritten on
    every build INCLUDING the hourly data-only one — while its sibling
    docs/assets/shell.css is hand-maintained and index.html loads both the same
    way. An edit to the generated one survives until the next hourly run and
    then vanishes. That already happened once; the response was to add a second
    file rather than label the first.
    """
    docs = DOCS if docs is None else docs
    (docs / "assets").mkdir(parents=True, exist_ok=True)
    banner = ("/* GENERATED by build_static.py — DO NOT EDIT.\n"
              "   Edit assets/style.css instead; this copy is overwritten on\n"
              "   every build, including the hourly data-only refresh.\n"
              "   docs/assets/shell.css is hand-maintained and is NOT generated. */\n")
    (docs / "assets" / "style.css").write_text(
        banner + (ROOT / "assets" / "style.css").read_text()
    )


# A first-time visitor downloads pycode.zip before the dashboard becomes interactive,
# so its size is a user-facing latency budget, not an implementation detail.
MAX_BUNDLE_MB = 6.0
_MAX_VALIDATOR_FILES = 500


def _preflight_environment() -> None:
    """Everything that can make a build unpublishable, checked before any write.

    Called at the top of main(). The plotly check used to run only inside
    build_pybundle — the last step — so a bad interpreter got as far as
    replacing every figure and the manifest before raising.
    """
    import importlib.util

    # Say WHICH environment is building. `source ai-frontier/bin/activate` on
    # this repo's path silently activates nothing — the path contains a space,
    # the prepended bin/ does not exist — yet the script exits 0 and still sets
    # the (ai-frontier) prompt, so you get the system interpreter believing you
    # are in a venv. That is how a build once ran under plotly 6.0 and tripled
    # the bundle. Printing the versions makes a wrong-but-lean environment
    # visible instead of silent.
    import plotly as _plotly
    import pandas as _pandas
    print(f"  building with {sys.executable}")
    print(f"  python {sys.version.split()[0]}  ·  plotly {_plotly.__version__}"
          f"  ·  pandas {_pandas.__version__}")
    if sys.prefix == sys.base_prefix:
        print("  NOTE: not running inside a virtualenv "
              "(sys.prefix == sys.base_prefix)")

    spec = importlib.util.find_spec("plotly")
    if spec and spec.submodule_search_locations:
        _assert_lean_plotly(Path(list(spec.submodule_search_locations)[0]))


def _assert_lean_plotly(pkg_dir: Path) -> None:
    """Refuse to vendor a plotly that ships the per-attribute validator tree.

    plotly < 6.1 generates one module per chart attribute — 13,329 files, ~7MB
    compressed. Building against such an install once tripled pybundle.zip from
    4.0MB to 11.4MB, which every visitor then re-downloaded before the page
    became interactive. 6.1+ collapses them into a single _validators.json.
    """
    validators = pkg_dir / "validators"
    n = sum(1 for f in validators.rglob("*.py")) if validators.is_dir() else 0
    if n > _MAX_VALIDATOR_FILES:
        raise RuntimeError(
            f"plotly at {pkg_dir} ships {n} generated validator modules "
            f"(>{_MAX_VALIDATOR_FILES}), which would bloat the code bundle to ~11MB "
            f"and slow every page load. Build against plotly>=6.1 instead:\n"
            f"    pip install 'plotly>=6.1'"
        )


def build_pybundle(docs: Path | None = None):
    """Build pycode.zip: project Python + plotly/tenacity vendored from venv.

    The data CSVs are NOT in here; build_data_bundle ships them separately.
    """
    import importlib.util, site

    docs = DOCS if docs is None else docs
    docs.mkdir(parents=True, exist_ok=True)
    bundle = docs / CODE_BUNDLE
    include = [
        "static_api.py", "static_helpers.py", "captions.py",
        "components/__init__.py", "components/stack_recommender.py",
        # Explicit list, not the whole directory. Globbing shipped 1,275 lines of
        # unreferenced builders to every visitor — including a context parser
        # with no 'M' case that would have silently dropped every long-context
        # model from any chart wired to it.
        "components/charts/__init__.py", "components/charts/constants.py",
        "components/charts/auto_palette.py",
        "components/charts/pareto.py", "components/charts/quadrant.py",
        "components/charts/treemap.py", "components/charts/rankings.py",
        "components/charts/radar.py", "components/charts/cost_calc.py",
        "components/charts/provider_leaderboard.py", "components/charts/bump_chart.py",
        "components/charts/local_scatter.py", "components/charts/local_compat.py",
        "components/charts/image_scatter.py", "components/charts/video_chart.py",
        "data/__init__.py", "data/ingest.py", "data/local_models.py",
        "data/image_models.py", "data/video_models.py",
        "data/pending_models.py", "data/scrape_status.py",
        "data/workload_mix.py", "data/task_cost.py", "data/quality_scale.py",
        "data/open_weights.py", "data/open_gap.py",
        "components/charts/open_gap.py",
        "data/econ_common.py", "data/serving_cost.py",
        "components/charts/econ_serve.py",
        "data/gpu_tco.py", "components/charts/econ_compute.py",
        "data/lab_economics.py", "components/charts/lab_economics.py",
    ]

    # Resolve site-packages so we can vendor plotly + tenacity
    site_pkgs = []
    for sp in site.getsitepackages():
        p = Path(sp)
        if p.is_dir():
            site_pkgs.append(p)
    # Also check importlib for the venv's site-packages
    try:
        spec = importlib.util.find_spec("plotly")
        if spec and spec.submodule_search_locations:
            plotly_root = Path(list(spec.submodule_search_locations)[0]).parent
            if plotly_root not in site_pkgs:
                site_pkgs.insert(0, plotly_root)
    except Exception:
        pass

    def find_pkg(name):
        for sp in site_pkgs:
            candidate = sp / name
            if candidate.is_dir():
                return candidate
        return None

    vendor_pkgs = []
    for pkg_name in ("plotly", "_plotly_utils", "tenacity"):
        pkg_dir = find_pkg(pkg_name)
        if pkg_dir:
            if pkg_name == "plotly":
                _assert_lean_plotly(pkg_dir)
            vendor_pkgs.append((pkg_name, pkg_dir))
            print(f"  vendoring {pkg_name} from {pkg_dir}")
        else:
            if pkg_name in ("plotly", "_plotly_utils"):
                raise RuntimeError(
                    f"{pkg_name} not found in site-packages — cannot build a working "
                    f"Pyodide bundle. Install it in this environment (pip install plotly) and rebuild."
                )
            print(f"  NOTE: optional {pkg_name} not vendored")

    # Build into a temp file and swap. Writing straight to the published path
    # meant an interrupt anywhere in the ~1,665-member loop left a truncated zip
    # at the URL every visitor fetches (pyworker.js then throws and the page
    # reads "interactivity unavailable"), and the size budget was checked only
    # AFTER the file was already in place — a rejected 4 MB bundle stayed
    # published. swap_bundle_csvs below already demonstrates the right pattern.
    staged = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as z:
        # Project files
        for rel in include:
            p = ROOT / rel
            if p.is_dir():
                for f in p.rglob("*.py"):
                    z.write(f, f.relative_to(ROOT))
            else:
                z.write(p, rel)
        # Vendor plotly + tenacity
        for pkg_name, pkg_dir in vendor_pkgs:
            for f in pkg_dir.rglob("*"):
                # Only .py + .json (plot schema / templates) + extensionless files are
                # needed for figure construction via fig.to_json(). Deliberately EXCLUDE
                # plotly's bundled .js (plotly.min.js is ~3.5MB) and HTML-export assets —
                # the browser renders with its own Plotly.js, so those would only bloat the zip.
                if f.is_file() and f.suffix in (".py", ".json", ""):
                    arc = pkg_name / f.relative_to(pkg_dir)
                    z.write(f, str(arc))

    size_mb = staged.stat().st_size / 1e6
    print(f"{CODE_BUNDLE}: {size_mb:.1f} MB")
    if size_mb > MAX_BUNDLE_MB:
        staged.unlink(missing_ok=True)      # reject BEFORE publishing
        raise RuntimeError(
            f"{CODE_BUNDLE} would be {size_mb:.1f}MB, over the {MAX_BUNDLE_MB}MB budget — "
            f"every visitor downloads this before the dashboard goes interactive. "
            f"Check what got vendored before publishing."
        )
    staged.replace(bundle)


@contextmanager
def shipped_derived():
    """Compute every DERIVED_DATA text ONCE, through derived_text (so a failed
    or oversized slice is "" here exactly as in the zip), and make read_derived
    return those texts while the block runs. The pre-rendered figures and
    pydata.zip are then built from the same bytes, and a slice shipped empty
    pre-renders empty too instead of from the full archive. Yields
    {rel: text}; clears the overrides and the serving caches on exit."""
    econ_common.clear_derived_overrides()
    serving_cost.reset_cache()
    gpu_tco.reset_cache()
    lab_economics.reset_cache()
    texts: dict[str, str] = {}
    try:
        for rel, compute in DERIVED_DATA.items():
            texts[rel] = derived_text(rel, compute)
            econ_common.set_derived_override(rel, texts[rel])
            serving_cost.reset_cache()      # the next slice reads this one
            gpu_tco.reset_cache()
            lab_economics.reset_cache()
        yield texts
    finally:
        econ_common.clear_derived_overrides()
        serving_cost.reset_cache()
        gpu_tco.reset_cache()
        lab_economics.reset_cache()


def build_data_bundle(docs: Path | None = None, derived: dict[str, str] | None = None):
    """Build pydata.zip from the data CSVs, staged and swapped like the code zip.

    Built from scratch every time, so unlike the old in-place CSV swap it needs
    no existing bundle and cannot carry a stale member forward. `derived` is
    shipped_derived()'s texts (the ones the figures were pre-rendered from);
    without it they are computed here the same way.
    """
    if derived is None:
        with shipped_derived() as texts:
            return build_data_bundle(docs, texts)
    docs = DOCS if docs is None else docs
    docs.mkdir(parents=True, exist_ok=True)
    bundle = docs / DATA_BUNDLE
    staged = bundle.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in DATA_CSVS:
                z.write(ROOT / rel, rel)
            for rel in DERIVED_DATA:
                z.writestr(rel, derived[rel])
        staged.replace(bundle)
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    print(f"{DATA_BUNDLE}: {bundle.stat().st_size / 1e3:.0f} KB")


def derived_text(rel: str, compute: Callable[[], str]) -> str:
    """One DERIVED_DATA file's CSV text. A failure or an oversized slice is
    printed and shipped empty (read_derived then shows an empty state): it must
    never block the hourly AA publish."""
    try:
        text = compute() or ""
    except Exception as exc:
        print(f"  WARNING: derived {rel} failed: {exc}")
        return ""
    limit = DERIVED_MAX_BYTES.get(rel, MAX_DERIVED_BYTES)
    if len(text.encode()) > limit:
        print(f"  WARNING: derived {rel} is {len(text.encode())} B "
              f"(> {limit}); shipped empty")
        return ""
    return text


def code_version(docs: Path | None = None) -> str:
    """Content hash of pycode.zip, the key the worker caches it under.

    Hashes member names, CRCs and sizes rather than the zip's bytes, because
    zip headers carry mtimes: two builds of identical code must share a key or
    every full build would evict every visitor's cached copy for nothing.
    """
    docs = DOCS if docs is None else docs
    h = hashlib.sha256()
    with zipfile.ZipFile(docs / CODE_BUNDLE) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            h.update(f"{info.filename}\0{info.CRC:08x}\0{info.file_size}\n".encode())
    return h.hexdigest()[:16]


def stamp_code_version(docs: Path | None = None) -> str:
    """Write code_version into the manifest the browser reads at boot.

    export_default_figures rewrites the manifest on every build, data-only
    included, so this runs after it each time rather than living in it.
    """
    docs = DOCS if docs is None else docs
    path = docs / "figures" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["code_version"] = code_version(docs)
    path.write_text(json.dumps(manifest))
    return manifest["code_version"]


def stale_bundle_modules(docs: Path | None = None) -> list[str]:
    """Project .py members of pycode.zip that differ from the working tree.

    export_default_figures imports the chart builders from the TREE, while
    a data-only refresh never rebuilds pycode.zip — so it would publish figures
    built from new code against a bundle still running the old code. A chart fix pushed without a full build looks
    right on load and visibly reverts the moment a visitor touches a filter.
    """
    docs = DOCS if docs is None else docs
    bundle = docs / CODE_BUNDLE
    if not bundle.exists():
        return []
    vendored = ("plotly/", "_plotly_utils/", "tenacity/")
    stale = []
    with zipfile.ZipFile(bundle) as z:
        for name in z.namelist():
            if not name.endswith(".py") or name.startswith(vendored):
                continue
            src = ROOT / name
            if src.exists() and z.read(name) != src.read_bytes():
                stale.append(name)
    return stale


def rebuild_data_only(docs: Path | None = None):
    """Data-only refresh for the hourly bot: figures + manifest + pydata.zip.

    Escalates to a full build when the code bundle is missing or has fallen
    behind the tree, because shipping figures from new code beside old
    in-browser code is a site that contradicts itself on first interaction.
    """
    docs = DOCS if docs is None else docs
    if not (docs / CODE_BUNDLE).exists():
        print(f"{CODE_BUNDLE} missing — escalating to a full build")
        main(docs)
        return
    stale = stale_bundle_modules(docs)
    if stale:
        print(f"{CODE_BUNDLE} is behind the tree ({len(stale)} modules: "
              f"{', '.join(stale[:4])}{'…' if len(stale) > 4 else ''})")
        print("escalating to a full build so the browser runs the same code as the figures")
        main(docs)
        return
    with shipped_derived() as derived:
        export_default_figures(docs / "figures")
        copy_css(docs)
        build_data_bundle(docs, derived)
    stamp_code_version(docs)
    print("Data-only rebuild complete →", docs)


def main(docs: Path | None = None):
    docs = DOCS if docs is None else docs
    # Validate first, write second. _assert_lean_plotly used to live inside
    # build_pybundle, the LAST step, so a build under the wrong interpreter
    # replaced all 13 figures and the CSS, bumped the manifest, and only then
    # raised — leaving fresh figures and a fresh manifest against a stale
    # bundle, which is exactly the half-state the manifest version tells every
    # client to go and fetch.
    _preflight_environment()
    with shipped_derived() as derived:
        export_default_figures(docs / "figures")
        copy_css(docs)
        (docs / ".nojekyll").write_text("")
        build_pybundle(docs)
        build_data_bundle(docs, derived)
    stamp_code_version(docs)
    print("Static build complete →", docs)


def _out_dir_from_argv(argv: list[str]) -> Path | None:
    """`--out DIR` redirects the whole build away from the published docs/ tree.

    Without it the only way to exercise a build was to overwrite the live site,
    so running the test suite republished the dashboard.
    """
    if "--out" not in argv:
        return None
    i = argv.index("--out")
    if i + 1 >= len(argv):
        raise SystemExit("--out requires a directory argument")
    return Path(argv[i + 1]).resolve()


if __name__ == "__main__":
    _out = _out_dir_from_argv(sys.argv)
    if "--data-only" in sys.argv:
        rebuild_data_only(_out)
    else:
        main(_out)
