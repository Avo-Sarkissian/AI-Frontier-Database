"""The deployed site runs the Python inside docs/pycode.zip, not the repo.

Editing a .py file is not shipping it. GitHub Pages serves docs/ verbatim, and
everything Pyodide imports comes out of that zip, so the repo and the artifact
are two copies of the same fact — the pattern this codebase keeps rediscovering.

They had already drifted before this test existed: the bundle carried the
pre-fix `static_api.py`, `stack_recommender.py`, `pareto.py`, `quadrant.py`,
`image_scatter.py` and `video_chart.py` while HEAD carried the fixed ones, so a
commit titled "fix" changed nothing a visitor could see. Nothing said so,
because the failing artifact is a binary blob nobody reads.

Why it can drift at all: the hourly refresh workflow runs
`build_static.py --data-only`, which rebuilds docs/pydata.zip (the CSVs) and
never touches docs/pycode.zip. Only a full `python build_static.py` re-vendors
the modules. (The data-only path now escalates to a full build when it sees
stale modules, but a code change pushed by hand with no build at all is still
caught only here.)

If this test fails, the fix is not to edit the test: run a full build
(`python build_static.py`, needs plotly>=6.1) and commit docs/.
"""
import ast
import zipfile
from pathlib import Path

import pytest

import build_static

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "docs" / build_static.CODE_BUNDLE
DATA_BUNDLE = ROOT / "docs" / build_static.DATA_BUNDLE


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


@pytest.mark.skipif(not BUNDLE.exists(), reason="no pycode.zip in this checkout")
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
        "docs/pycode.zip is stale — the deployed site is running different "
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


@pytest.mark.skipif(not BUNDLE.exists(), reason="no pycode.zip in this checkout")
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


@pytest.mark.skipif(not DATA_BUNDLE.exists(), reason="no pydata.zip in this checkout")
def test_the_bundled_data_csvs_match_the_repo():
    """`--data-only` rebuilds this zip hourly. If it drifts, the site is
    serving a catalogue the freshness badge does not describe."""
    with zipfile.ZipFile(DATA_BUNDLE) as z:
        names = set(z.namelist())
        missing = [rel for rel in build_static.DATA_CSVS if rel not in names]
        stale = [rel for rel in build_static.DATA_CSVS
                 if rel in names and (ROOT / rel).exists()
                 and z.read(rel) != (ROOT / rel).read_bytes()]
    assert not missing, f"pydata.zip does not carry: {missing}"
    assert not stale, f"bundled data CSVs differ from the repo: {stale}"


@pytest.mark.skipif(not BUNDLE.exists(), reason="no pycode.zip in this checkout")
def test_the_manifest_names_the_code_zip_it_was_built_with():
    """The worker caches pycode.zip under manifest.code_version. A manifest
    naming a different version would serve a returning visitor the cached old
    code indefinitely."""
    import json
    manifest = json.loads((ROOT / "docs" / "figures" / "manifest.json").read_text())
    assert manifest.get("code_version") == build_static.code_version(), (
        "manifest.code_version does not match docs/pycode.zip — run a full build"
    )


def test_the_retired_single_bundle_is_gone():
    """Nothing fetches pybundle.zip any more; left in docs/ it is 4 MB of stale
    code and data served to nobody, and a trap for the next reader."""
    assert not (ROOT / "docs" / "pybundle.zip").exists()


@pytest.mark.skipif(not BUNDLE.exists(), reason="no pycode.zip in this checkout")
def test_every_first_party_import_inside_the_bundle_is_also_bundled():
    """A bundled module importing an unbundled one is a site that boots to
    "interactivity unavailable" — and nothing else catches it, because the test
    suite imports from the repo where the file obviously exists.

    This happened: narrowing build_pybundle's include list from the whole
    components/charts directory to an explicit list was right (it had been
    shipping 1,275 lines of dead code), but an explicit list goes stale the
    moment a new first-party import appears. data/pending_models.py was added,
    imported by data/local_models.py, and left out — Pyodide then failed to
    boot at all.
    """
    import ast

    with zipfile.ZipFile(BUNDLE) as z:
        bundled = set(z.namelist())
        members = _bundled_python_members()

        missing = {}
        for name in members:
            tree = ast.parse(z.read(name))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mod = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        mod = alias.name
                        _check(mod, bundled, missing, name)
                    continue
                else:
                    continue
                _check(mod, bundled, missing, name)

    assert not missing, (
        "bundled modules import first-party modules that are NOT in the bundle — "
        f"the browser will fail to boot: {missing}"
    )


_FIRST_PARTY_ROOTS = ("data", "components", "static_api", "static_helpers", "captions")


def _check(module: str, bundled: set, missing: dict, importer: str) -> None:
    root = module.split(".")[0]
    if root not in _FIRST_PARTY_ROOTS:
        return
    candidates = {module.replace(".", "/") + ".py", module.replace(".", "/") + "/__init__.py"}
    if not (candidates & bundled):
        missing.setdefault(importer, []).append(module)
