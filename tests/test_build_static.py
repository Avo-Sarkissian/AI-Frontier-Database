# tests/test_build_static.py
import hashlib, json, subprocess, sys, re, zipfile
from pathlib import Path
from datetime import datetime

import pytest
import build_static

ROOT = Path(__file__).resolve().parent.parent

DATA_CSVS = ["data/raw/aa_models.csv", "data/raw/aa_local_models.csv", "data/raw/aa_image_models.csv"]


def _tree_digest(root: Path) -> dict[str, str]:
    """sha256 per file, so a test can prove it did not touch the published site."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


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
def test_build_writes_only_into_the_out_dir(tmp_path):
    """Running the build must never touch the published site.

    build_static hardcoded DOCS = ROOT/"docs", and these tests shelled out with
    cwd=ROOT, so a plain `pytest` run rebuilt and dirtied docs/ — which, under
    the repo's auto-push mandate, republished the live site as a side effect of
    running tests.
    """
    before = _tree_digest(ROOT / "docs")
    subprocess.run([sys.executable, "build_static.py", "--out", str(tmp_path)],
                   cwd=ROOT, check=True)
    assert (tmp_path / "figures" / "manifest.json").exists()
    assert (tmp_path / "pybundle.zip").exists()
    assert _tree_digest(ROOT / "docs") == before, "build mutated docs/ despite --out"


@needs_lean_plotly
def test_build_produces_figures_and_manifest(tmp_path):
    subprocess.run([sys.executable, "build_static.py", "--out", str(tmp_path)],
                   cwd=ROOT, check=True)
    figdir = tmp_path / "figures"
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
def test_manifest_has_version_and_iso(tmp_path):
    subprocess.run([sys.executable, "build_static.py", "--out", str(tmp_path)],
                   cwd=ROOT, check=True)
    manifest = json.loads((tmp_path / "figures" / "manifest.json").read_text())
    assert re.fullmatch(r"\d{8}T\d{6}Z", manifest.get("version", "")), manifest.get("version")
    # generated_iso must parse as ISO-8601
    datetime.fromisoformat(manifest["generated_iso"])

@needs_lean_plotly
def test_data_only_swaps_csvs_and_preserves_plotly(tmp_path):
    # Full build first so a bundle exists.
    subprocess.run([sys.executable, "build_static.py", "--out", str(tmp_path)],
                   cwd=ROOT, check=True)
    bundle = tmp_path / "pybundle.zip"
    with zipfile.ZipFile(bundle) as z:
        before = {i.filename: z.read(i.filename) for i in z.infolist()}
    sample_py = next(n for n in before if n.endswith(".py") and not n.startswith("data/raw/"))

    # Data-only rebuild.
    subprocess.run([sys.executable, "build_static.py", "--data-only", "--out", str(tmp_path)],
                   cwd=ROOT, check=True)
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


# ── Coverage: what the scraper could not carry ───────────────────────────────

def test_scraper_counts_what_it_discards():
    """A model with no score or no price is dropped. Counting the drops is the
    point: "148 tracked" silently meant "162 upstream minus 14"."""
    from data.scraper import _parse_api_response, _last_coverage

    payload = {"hostModels": [
        {  # kept
            "model": {"name": "Good Model", "intelligence_index": 50.0,
                      "model_creators": {"name": "Anthropic"}},
            "price_1m_blended_3_to_1": 10.0,
            "price_1m_input_tokens": 5.0, "price_1m_output_tokens": 11.67,
        },
        {  # dropped: no intelligence score
            "model": {"name": "Unscored Model", "intelligence_index": None,
                      "model_creators": {"name": "Meta"}},
            "price_1m_blended_3_to_1": 2.0,
        },
        {  # dropped: no price at all
            "model": {"name": "Free Model", "intelligence_index": 30.0,
                      "model_creators": {"name": "Mistral"}},
        },
    ]}
    rows = _parse_api_response(payload)
    assert len(rows) == 1
    assert _last_coverage["kept"] == 1
    assert _last_coverage["skipped_no_score"] == ["Unscored Model"]
    assert _last_coverage["skipped_no_price"] == ["Free Model"]


def test_coverage_reconciles_with_the_live_catalog():
    """kept + dropped must account for every distinct upstream model, so the
    disclosure cannot drift from what was actually discarded."""
    import json
    from pathlib import Path
    from data.ingest import get_models

    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "coverage.json"
    if not path.exists():
        import pytest
        pytest.skip("no coverage.json — repo has not scraped since it was added")
    cov = json.loads(path.read_text())
    assert cov["kept"] == len(get_models())
    assert cov["kept"] > 0 and cov["upstream_records"] >= cov["kept"]


def test_manifest_carries_the_coverage_block():
    from build_static import _load_coverage
    cov = _load_coverage()
    assert set(cov) >= {"kept", "skipped_no_score", "skipped_no_price"}
