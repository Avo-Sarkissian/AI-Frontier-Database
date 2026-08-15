"""The deployed site runs the Python inside docs/pybundle.zip, not the repo.

Editing a .py file is not shipping it. GitHub Pages serves docs/ verbatim, and
everything Pyodide imports comes out of that zip, so the repo and the artifact
are two copies of the same fact — the pattern this codebase keeps rediscovering.

They had already drifted before this test existed: the bundle carried the
pre-fix `static_api.py`, `stack_recommender.py`, `pareto.py`, `quadrant.py`,
`image_scatter.py` and `video_chart.py` while HEAD carried the fixed ones, so a
commit titled "fix" changed nothing a visitor could see. Nothing said so,
because the failing artifact is a binary blob nobody reads.

Why it can drift at all: the hourly refresh workflow runs
`build_static.py --data-only`, whose `swap_bundle_csvs()` replaces the three data
CSVs *inside* the zip and copies every .py member through untouched. Only a full
`python build_static.py` re-vendors the modules, and that is a manual step. So
the bundle goes stale by default, silently, on every code change.

If this test fails, the fix is not to edit the test: run a full build
(`python build_static.py`, needs plotly>=6.1) and commit docs/.
"""
import ast
import zipfile
from pathlib import Path

import pytest

import build_static

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "docs" / "pybundle.zip"


def _bundled_python_members() -> list[str]:
    """The project .py files the bundle claims to ship.

    Derived from build_static.build_pybundle's own include list so a newly
    bundled module is covered the day it is added, rather than whenever someone
    remembers to update this test. Vendored third-party packages (plotly,
    tenacity) are excluded — they come from site-packages, not the repo.
    """
    vendored = ("plotly/", "_plotly_utils/", "tenacity/")
    with zipfile.ZipFile(BUNDLE) as z:
        return [
            n for n in z.namelist()
            if n.endswith(".py") and not n.startswith(vendored)
        ]


@pytest.mark.skipif(not BUNDLE.exists(), reason="no pybundle.zip in this checkout")
def test_every_python_module_in_the_bundle_matches_the_repo():
    stale, missing = [], []
    with zipfile.ZipFile(BUNDLE) as z:
        for name in _bundled_python_members():
            repo_file = ROOT / name
            if not repo_file.exists():
                missing.append(name)
                continue
            if z.read(name) != repo_file.read_bytes():
                stale.append(name)

    assert not missing, (
        f"bundle ships modules that no longer exist in the repo: {missing}"
    )
    assert not stale, (
        "docs/pybundle.zip is stale — the deployed site is running different "
        f"code from this checkout: {stale}. Run a full `python build_static.py` "
        "(needs plotly>=6.1) and commit docs/."
    )


def _declared_includes() -> list[str]:
    """build_pybundle's `include` list, read out of the source.

    It is a local inside the function, so it cannot be imported. Parsing it
    keeps this test in step with the build automatically: a module added to the
    bundle is covered the day it is added, not whenever someone remembers.
    """
    tree = ast.parse((ROOT / "build_static.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef) and node.name == "build_pybundle"):
            for stmt in ast.walk(node):
                if (isinstance(stmt, ast.Assign)
                        and any(getattr(t, "id", None) == "include" for t in stmt.targets)):
                    return [ast.literal_eval(e) for e in stmt.value.elts]
    pytest.fail("could not find build_pybundle's `include` list in build_static.py")


@pytest.mark.skipif(not BUNDLE.exists(), reason="no pybundle.zip in this checkout")
def test_the_bundle_ships_every_module_the_build_promises():
    """A module dropped from the zip fails at import time in the browser only —
    the suite imports from the repo and would never notice."""
    bundled = set(_bundled_python_members())
    missing = [rel for rel in _declared_includes()
               if rel.endswith(".py") and rel not in bundled]
    assert not missing, (
        f"build_pybundle promises these modules but the zip does not carry them: "
        f"{missing} — re-run `python build_static.py`"
    )


@pytest.mark.skipif(not BUNDLE.exists(), reason="no pybundle.zip in this checkout")
def test_the_bundled_data_csvs_match_the_repo():
    """`--data-only` swaps exactly these three, hourly. If they drift, the site
    is serving a catalogue the freshness badge does not describe."""
    stale = []
    with zipfile.ZipFile(BUNDLE) as z:
        names = set(z.namelist())
        for rel in build_static.DATA_CSVS:
            if rel in names and (ROOT / rel).exists():
                if z.read(rel) != (ROOT / rel).read_bytes():
                    stale.append(rel)
    assert not stale, f"bundled data CSVs differ from the repo: {stale}"
