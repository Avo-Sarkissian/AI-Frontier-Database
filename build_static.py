"""Pre-render default figures + bundle Python for the static Pyodide site."""
import json, shutil, zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.ingest import get_models
from data.local_models import get_local_df
from data.image_models import get_image_df
from data.video_models import get_video_df
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
        "generated":        datetime.now().strftime("%b %d  %H:%M"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return written


def copy_css():
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "assets" / "style.css", DOCS / "assets" / "style.css")


def main():
    export_default_figures(FIG)
    copy_css()
    (DOCS / ".nojekyll").write_text("")
    # build_pybundle() added in Task 9.
    print("Static build complete →", DOCS)


if __name__ == "__main__":
    main()
