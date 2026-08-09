# tests/test_build_static.py
import json, subprocess, sys, re, zipfile
from pathlib import Path
from datetime import datetime

import pytest
import build_static

ROOT = Path(__file__).resolve().parent.parent

DATA_CSVS = ["data/raw/aa_models.csv", "data/raw/aa_local_models.csv", "data/raw/aa_image_models.csv"]


def _ambient_plotly_is_lean() -> bool:
    """build_pybundle refuses to vendor a pre-6.1 plotly (13k validator modules,
    ~11MB bundle). Interpreters with such an install can't run the full-build
    tests — CI installs plotly==6.5.2 explicitly, so they run there."""
    import plotly
    validators = Path(plotly.__file__).parent / "validators"
    n = sum(1 for _ in validators.rglob("*.py")) if validators.is_dir() else 0
    return n <= build_static._MAX_VALIDATOR_FILES


needs_lean_plotly = pytest.mark.skipif(
    not _ambient_plotly_is_lean(),
    reason="ambient plotly ships the pre-6.1 validator tree; "
           "build_static refuses to vendor it (pip install 'plotly>=6.1')",
)

@needs_lean_plotly
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

@needs_lean_plotly
def test_manifest_has_version_and_iso():
    subprocess.run([sys.executable, "build_static.py"], cwd=ROOT, check=True)
    manifest = json.loads((ROOT / "docs" / "figures" / "manifest.json").read_text())
    assert re.fullmatch(r"\d{8}T\d{6}Z", manifest.get("version", "")), manifest.get("version")
    # generated_iso must parse as ISO-8601
    datetime.fromisoformat(manifest["generated_iso"])

@needs_lean_plotly
def test_data_only_swaps_csvs_and_preserves_plotly():
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

def test_swap_raises_if_bundle_missing(tmp_path, monkeypatch):
    # Point the module's DOCS at an empty dir so pybundle.zip is absent.
    monkeypatch.setattr(build_static, "DOCS", tmp_path)
    with pytest.raises(RuntimeError, match="pybundle.zip missing"):
        build_static.swap_bundle_csvs()
