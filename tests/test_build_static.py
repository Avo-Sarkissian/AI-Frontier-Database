# tests/test_build_static.py
import hashlib, json, subprocess, sys, re, zipfile
from pathlib import Path
from datetime import datetime

import pytest
import build_static

ROOT = Path(__file__).resolve().parent.parent

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
    assert (tmp_path / build_static.CODE_BUNDLE).exists()
    assert (tmp_path / build_static.DATA_BUNDLE).exists()
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
def test_data_only_rebuilds_the_data_zip_and_leaves_the_code_zip_alone(tmp_path):
    """The split exists so the hourly refresh does not touch pycode.zip: a
    byte-identical code zip and an unchanged code_version are what let a
    returning visitor's cached copy survive the refresh."""
    subprocess.run([sys.executable, "build_static.py", "--out", str(tmp_path)],
                   cwd=ROOT, check=True)
    code = tmp_path / build_static.CODE_BUNDLE
    code_before = code.read_bytes()
    version_before = json.loads(
        (tmp_path / "figures" / "manifest.json").read_text())["code_version"]

    subprocess.run([sys.executable, "build_static.py", "--data-only", "--out", str(tmp_path)],
                   cwd=ROOT, check=True)
    assert code.read_bytes() == code_before, "a data-only rebuild rewrote pycode.zip"
    manifest = json.loads((tmp_path / "figures" / "manifest.json").read_text())
    assert manifest["code_version"] == version_before == build_static.code_version(tmp_path)
    with zipfile.ZipFile(tmp_path / build_static.DATA_BUNDLE) as z:
        assert sorted(z.namelist()) == sorted(build_static.DATA_CSVS)
        for csv in build_static.DATA_CSVS:
            assert z.read(csv) == (ROOT / csv).read_bytes()


@needs_lean_plotly
def test_the_code_zip_carries_no_data(tmp_path):
    """Data inside pycode.zip would change its hash every hour and defeat the
    cache the split was made for."""
    subprocess.run([sys.executable, "build_static.py", "--out", str(tmp_path)],
                   cwd=ROOT, check=True)
    with zipfile.ZipFile(tmp_path / build_static.CODE_BUNDLE) as z:
        leaked = [n for n in z.namelist() if n.startswith("data/raw/")]
    assert not leaked, f"data files leaked into the code zip: {leaked}"


def test_the_data_zip_builds_without_a_code_zip(tmp_path):
    build_static.build_data_bundle(tmp_path)
    assert not (tmp_path / build_static.CODE_BUNDLE).exists()
    with zipfile.ZipFile(tmp_path / build_static.DATA_BUNDLE) as z:
        assert sorted(z.namelist()) == sorted(build_static.DATA_CSVS)
    assert not (tmp_path / "pydata.zip.tmp").exists()


def _zip_with(path: Path, members: dict[str, bytes], mtime: tuple) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(zipfile.ZipInfo(name, date_time=mtime), data)


def test_code_version_follows_content_not_zip_timestamps(tmp_path):
    """Zip headers carry mtimes. Hashing the raw bytes would give identical
    code a new key on every full build and evict every visitor's cache."""
    a, b, c = (tmp_path / d for d in ("a", "b", "c"))
    for d in (a, b, c):
        d.mkdir()
    members = {"static_api.py": b"x = 1\n", "plotly/__init__.py": b""}
    _zip_with(a / "pycode.zip", members, (2026, 1, 1, 0, 0, 0))
    _zip_with(b / "pycode.zip", members, (2026, 9, 25, 12, 0, 0))
    _zip_with(c / "pycode.zip", {**members, "static_api.py": b"x = 2\n"}, (2026, 1, 1, 0, 0, 0))
    assert build_static.code_version(a) == build_static.code_version(b)
    assert build_static.code_version(a) != build_static.code_version(c)


def test_a_data_only_rebuild_without_a_code_zip_escalates(tmp_path, monkeypatch):
    """A fresh --out directory has no pycode.zip; publishing figures and data
    with nothing to run them would boot to "interactivity unavailable"."""
    calls = []
    monkeypatch.setattr(build_static, "main", lambda docs=None: calls.append(docs))
    build_static.rebuild_data_only(tmp_path)
    assert calls == [tmp_path]


# ── Coverage: what the scraper could not carry ───────────────────────────────

def test_scraper_counts_what_it_discards():
    """A model with no score or no price is dropped. Counting the drops is the
    point: "148 tracked" silently meant "162 upstream minus 14"."""
    from data.scraper import _parse_api_response, _last_coverage

    payload = [
        {  # kept
            "name": "Good Model", "intelligenceIndex": 50.0,
            "modelCreatorName": "Anthropic", "deprecated": False,
            "price1mInputTokens": 5.0, "price1mOutputTokens": 11.67,
        },
        {  # dropped: no intelligence score
            "name": "Unscored Model", "intelligenceIndex": None,
            "modelCreatorName": "Meta", "deprecated": False,
            "price1mInputTokens": 1.0, "price1mOutputTokens": 2.0,
        },
        {  # dropped: no price at all
            "name": "Free Model", "intelligenceIndex": 30.0,
            "modelCreatorName": "Mistral", "deprecated": False,
        },
    ]
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
    assert cov["kept"] > 0 and cov["upstream_host_model_rows"] >= cov["kept"]
    # The denominator the UI quotes is DISTINCT models, not host rows.
    assert cov["distinct_upstream_models"] >= cov["kept"]
    assert cov["distinct_upstream_models"] == (
        cov["kept"] + len(cov["skipped_no_score"]) + len(cov["skipped_no_price"]))


def test_manifest_carries_the_coverage_block():
    from build_static import _load_coverage
    cov = _load_coverage()
    assert set(cov) >= {"kept", "skipped_no_score", "skipped_no_price"}
