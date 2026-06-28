# tests/test_build_static.py
import json, subprocess, sys, re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

def test_build_produces_figures_and_manifest(tmp_path):
    # Build writes into docs/figures by default; assert key artifacts exist & parse.
    subprocess.run([sys.executable, "build_static.py"], cwd=ROOT, check=True)
    figdir = ROOT / "docs" / "figures"
    for fid in ["pareto", "treemap", "rankings", "radar", "cost_calc",
                "local_scatter", "image_faceted", "video_rankings"]:
        p = figdir / f"{fid}.json"
        assert p.exists(), f"missing {p}"
        fig = json.loads(p.read_text())
        assert "data" in fig and "layout" in fig
    manifest = json.loads((figdir / "manifest.json").read_text())
    assert int(manifest["model_count"]) > 0
    assert manifest["provider_options"] and manifest["diverse5"]

def test_manifest_has_version_and_iso():
    subprocess.run([sys.executable, "build_static.py"], cwd=ROOT, check=True)
    manifest = json.loads((ROOT / "docs" / "figures" / "manifest.json").read_text())
    assert re.fullmatch(r"\d{8}T\d{6}Z", manifest.get("version", "")), manifest.get("version")
    # generated_iso must parse as ISO-8601
    datetime.fromisoformat(manifest["generated_iso"])
