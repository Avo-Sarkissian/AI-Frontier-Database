"""Pre-render default figures + bundle Python for the static Pyodide site."""
import json, shutil, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.ingest import get_models
from data.local_models import get_local_df, DEFAULT_VRAM_GB, DEFAULT_BANDWIDTH_GBPS
from data.image_models import get_image_df, get_image_providers, get_image_tags
from data.video_models import get_video_df, get_video_providers, get_video_tags
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

from captions import CAPTIONS
from data import scrape_status
from components.charts.constants import PROVIDER_ALIASES
from static_helpers import compute_diverse5, provider_options, model_options, quality_options


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

DATA_CSVS = [
    "data/raw/aa_models.csv",
    "data/raw/aa_local_models.csv",
    "data/raw/aa_image_models.csv",
]

# Do NOT import app.py — it starts background scrapers at import time. Shared
# pure logic lives in static_helpers (imported above).


def export_default_figures(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = get_models()
    diverse5 = compute_diverse5(df)
    local_df = get_local_df(quant="Q4", vram_gb=DEFAULT_VRAM_GB,
                            bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS, hw_type="nvidia")
    img_df = get_image_df()
    vdf = get_video_df()
    vpaid = vdf[vdf["price_per_sec"] > 0] if not vdf.empty else vdf
    figures = {
        "pareto":               build_pareto_scatter(df),
        "quadrant":             build_quadrant(df),
        "treemap":              build_treemap(df),
        "provider_leaderboard": build_provider_leaderboard(df),
        "rankings":             build_rankings(df, top_n=25, metric="intelligence"),
        "value_leaders":        build_value_leaders(df),
        "radar":                build_radar(df, diverse5, full_df=df),
        "cost_calc":            build_cost_calc(df, monthly_tokens_m=1.0),
        "local_scatter":        build_local_scatter(local_df, vram_gb=DEFAULT_VRAM_GB, quant="Q4"),
        "local_compat":         build_local_compat(local_df, quant="Q4", vram_gb=DEFAULT_VRAM_GB),
        "image_faceted":        build_image_faceted(img_df),
        "video_rankings":       build_video_rankings(vdf),
        "video_scatter":        build_video_scatter(vpaid if not vpaid.empty else vdf),
    }
    written = []
    for fid, fig in figures.items():
        (out_dir / f"{fid}.json").write_text(fig.to_json())
        written.append(f"{fid}.json")

    _now = datetime.now(timezone.utc)
    manifest = {
        "model_count":      int(len(df)),
        "coverage":         _load_coverage(),
        "provider_count":   int(df["provider"].nunique()),
        "floor_price":      f"${df['price'].min():.3f}",
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
        # Derived, not hand-copied: docs/app.js needs the retired->current
        # provider spellings to resolve an old ?p= share link, and a second
        # copy of this map in JS is exactly how palettes and labels drifted
        # before. Shipping it keeps PROVIDER_ALIASES the only source.
        "provider_aliases": dict(PROVIDER_ALIASES),
        # One source for the prose too — the public site used to ship 1 of 14.
        "captions":         dict(CAPTIONS),
        # Per-dataset scrape status. `generated_iso` below is the BUILD time and
        # is kept only for cache-busting — the badge must report when the DATA
        # was last fetched, per dataset, or one succeeding scraper resets the
        # clock for all three. See data/scrape_status.py.
        "datasets":         scrape_status.load(),
        "data_fetched_iso": scrape_status.oldest_successful_fetch(),
        "stale_datasets":   scrape_status.stale_datasets(),
        # MIN SCORE options, including the exact preset percentiles, so the
        # <select> can hold the value a preset sets instead of snapping it.
        "quality_options": quality_options(
            round(float(df["quality"].quantile(0.75)), 1),
            round(float(df["quality"].quantile(0.90)), 1)),
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


# Every visitor downloads pybundle.zip before the dashboard becomes interactive,
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
            f"(>{_MAX_VALIDATOR_FILES}), which would bloat pybundle.zip to ~11MB "
            f"and slow every page load. Build against plotly>=6.1 instead:\n"
            f"    pip install 'plotly>=6.1'"
        )


def build_pybundle(docs: Path | None = None):
    """Build pybundle.zip: project Python + plotly/tenacity vendored from venv."""
    import importlib.util, site

    docs = DOCS if docs is None else docs
    docs.mkdir(parents=True, exist_ok=True)
    bundle = docs / "pybundle.zip"
    include = [
        "static_api.py", "static_helpers.py", "captions.py",
        "components/__init__.py", "components/stack_recommender.py",
        "components/charts",                      # whole dir
        "data/__init__.py", "data/ingest.py", "data/local_models.py",
        "data/image_models.py", "data/video_models.py", "data/embedding_models.py",
        "data/raw/aa_models.csv", "data/raw/aa_local_models.csv", "data/raw/aa_image_models.csv",
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
    print(f"pybundle.zip: {size_mb:.1f} MB")
    if size_mb > MAX_BUNDLE_MB:
        staged.unlink(missing_ok=True)      # reject BEFORE publishing
        raise RuntimeError(
            f"pybundle.zip would be {size_mb:.1f}MB, over the {MAX_BUNDLE_MB}MB budget — "
            f"every visitor downloads this before the dashboard goes interactive. "
            f"Check what got vendored before publishing."
        )
    staged.replace(bundle)


def swap_bundle_csvs(docs: Path | None = None):
    """Replace the 3 data CSVs inside docs/pybundle.zip without re-vendoring plotly."""
    docs = DOCS if docs is None else docs
    bundle = docs / "pybundle.zip"
    if not bundle.exists():
        raise RuntimeError("pybundle.zip missing — run a full `python build_static.py` first.")
    tmp = bundle.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(bundle) as zin, \
             zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename in DATA_CSVS:
                    continue                                    # drop stale copy
                zout.writestr(item, zin.read(item.filename))    # pass everything else through
            for rel in DATA_CSVS:
                zout.write(ROOT / rel, rel)                     # add fresh copy
        tmp.replace(bundle)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print("swapped CSVs into pybundle.zip")


def stale_bundle_modules(docs: Path | None = None) -> list[str]:
    """Project .py members of pybundle.zip that differ from the working tree.

    export_default_figures imports the chart builders from the TREE, while
    swap_bundle_csvs passes every .py member through byte-for-byte — so a
    data-only refresh publishes figures built from new code against a bundle
    still running the old code. A chart fix pushed without a full build looks
    right on load and visibly reverts the moment a visitor touches a filter.
    """
    docs = DOCS if docs is None else docs
    bundle = docs / "pybundle.zip"
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
    """Data-only refresh for the hourly bot: figures + manifest + CSV swap.

    Escalates to a full build when the bundle's code has fallen behind the
    tree, because shipping figures from new code beside old in-browser code is
    a site that contradicts itself on first interaction.
    """
    docs = DOCS if docs is None else docs
    stale = stale_bundle_modules(docs)
    if stale:
        print(f"pybundle.zip is behind the tree ({len(stale)} modules: "
              f"{', '.join(stale[:4])}{'…' if len(stale) > 4 else ''})")
        print("escalating to a full build so the browser runs the same code as the figures")
        main(docs)
        return
    export_default_figures(docs / "figures")
    copy_css(docs)
    swap_bundle_csvs(docs)
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
    export_default_figures(docs / "figures")
    copy_css(docs)
    (docs / ".nojekyll").write_text("")
    build_pybundle(docs)
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
