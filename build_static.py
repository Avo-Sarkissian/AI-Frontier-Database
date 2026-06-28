"""Pre-render default figures + bundle Python for the static Pyodide site."""
import json, shutil, zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.ingest import get_models
from data.local_models import get_local_df
from data.image_models import get_image_df, get_image_providers
from data.video_models import get_video_df, get_video_providers
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

from static_helpers import compute_diverse5, provider_options, model_options

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
FIG  = DOCS / "figures"

# Do NOT import app.py — it starts background scrapers at import time. Shared
# pure logic lives in static_helpers (imported above).


def export_default_figures(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = get_models()
    diverse5 = compute_diverse5(df)
    local_df = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=1792, hw_type="nvidia")
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
        "radar":                build_radar(df, diverse5),
        "cost_calc":            build_cost_calc(df, monthly_tokens_m=1.0),
        "local_scatter":        build_local_scatter(local_df, vram_gb=32, quant="Q4"),
        "local_compat":         build_local_compat(local_df, quant="Q4"),
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
        "generated":        _now.strftime("%b %d  %H:%M"),
        "version":          _now.strftime("%Y%m%dT%H%M%SZ"),
        "generated_iso":    _now.isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return written


def copy_css():
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "assets" / "style.css", DOCS / "assets" / "style.css")


def build_pybundle():
    """Build pybundle.zip: project Python + plotly/tenacity vendored from venv."""
    import importlib.util, site

    bundle = DOCS / "pybundle.zip"
    include = [
        "static_api.py", "static_helpers.py",
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
            vendor_pkgs.append((pkg_name, pkg_dir))
            print(f"  vendoring {pkg_name} from {pkg_dir}")
        else:
            if pkg_name in ("plotly", "_plotly_utils"):
                raise RuntimeError(
                    f"{pkg_name} not found in site-packages — cannot build a working "
                    f"Pyodide bundle. Install it in this environment (pip install plotly) and rebuild."
                )
            print(f"  NOTE: optional {pkg_name} not vendored")

    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
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

    print("pybundle.zip:", bundle.stat().st_size // 1024, "KB")


def main():
    export_default_figures(FIG)
    copy_css()
    (DOCS / ".nojekyll").write_text("")
    build_pybundle()
    print("Static build complete →", DOCS)


if __name__ == "__main__":
    main()
