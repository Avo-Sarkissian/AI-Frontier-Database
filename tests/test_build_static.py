# tests/test_build_static.py
import json, subprocess, sys, re, zipfile
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

DATA_CSVS = ["data/raw/aa_models.csv", "data/raw/aa_local_models.csv", "data/raw/aa_image_models.csv"]

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

def test_data_only_swaps_csvs_and_preserves_plotly(tmp_path):
    # Full build first so a bundle exists.
    subprocess.run([sys.executable, "build_static.py"], cwd=ROOT, check=True)
    bundle = ROOT / "docs" / "pybundle.zip"
    with zipfile.ZipFile(bundle) as z:
        before = {i.filename: z.read(i.filename) for i in z.infolist()}
    sample_py = next(n for n in before if n.endswith(".py") and not n.startswith("data/raw/"))

    # Data-only rebuild.
    subprocess.run([sys.executable, "build_static.py", "--data-only"], cwd=ROOT, check=True)
    with zipfile.ZipFile(bundle) as z:
        after = set(z.namelist())
        # membership unchanged
        assert after == set(before)
        # a plotly/source member is byte-identical (no re-vendor drift)
        assert z.read(sample_py) == before[sample_py]
        # the 3 CSVs match the live data/raw files
        for csv in DATA_CSVS:
            assert csv in after
            assert z.read(csv) == (ROOT / csv).read_bytes()
