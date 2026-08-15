import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / ".github" / "workflows" / "refresh.yml"

# Drives a scraper module through its failure path with the network forced to
# fail, then lets its __main__ block decide the process exit status.
_FAIL_DRIVER = """
import runpy, requests
def _boom(*a, **k):
    raise requests.RequestException("forced failure for test")
requests.get = _boom
runpy.run_module({mod!r}, run_name="__main__")
"""

def test_workflow_present_and_wired():
    assert WF.exists(), "refresh.yml not created"
    txt = WF.read_text()
    for needle in [
        "schedule:",
        "workflow_dispatch:",
        "contents: write",
        "pip install --require-hashes -r requirements.lock",
        "python -m data.scraper",
        "python -m data.local_scraper",
        "python -m data.image_scraper",
        "git diff --quiet -- data/raw/",
        "git ls-files --others --exclude-standard -- data/raw/",
        "build_static.py --data-only",
        "git pull --rebase --autostash origin main",
        "git push",
    ]:
        assert needle in txt, f"workflow missing: {needle}"


def test_scrape_failures_are_not_swallowed():
    """`|| echo "::warning::"` kept the job green while the image endpoint
    400'd for 29 days and stale ELOs were republished every hour."""
    txt = WF.read_text()
    steps = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert "::warning::" not in steps, "a scraper failure is being downgraded to a warning"
    assert 'echo "failed=${failed# }" >> "$GITHUB_OUTPUT"' in txt
    assert "steps.scrape.outputs.failed != ''" in txt
    assert "exit 1" in txt, "failed scrapes must fail the run"


def test_failure_is_reported_after_publishing():
    """Publishing what did succeed must come before the run is failed."""
    txt = WF.read_text()
    assert txt.index("git push") < txt.index("Report scrape failures")


@pytest.mark.parametrize("mod", ["data.scraper", "data.local_scraper", "data.image_scraper"])
def test_scraper_exits_nonzero_when_upstream_fails(mod, tmp_path):
    """The workflow records failures with `python -m data.scraper || failed=...`.
    That guard is dead code unless the module actually exits non-zero — which is
    how the image endpoint 400'd for 29 days behind a green run."""
    # Redirect the scrape-status file: this test deliberately fails a scrape,
    # and recording that into data/raw/scrape_status.json would make the live
    # freshness badge warn because the suite had been run.
    env = {**os.environ, "AI_FRONTIER_STATUS_PATH": str(tmp_path / "status.json")}
    proc = subprocess.run(
        [sys.executable, "-c", _FAIL_DRIVER.format(mod=mod)],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    assert proc.returncode != 0, (
        f"{mod} exited 0 on a failed scrape, so refresh.yml's `||` guard never "
        f"fires and stale data is published as fresh.\nstdout:\n{proc.stdout}"
    )


def test_row_loss_guard_replaces_the_bare_floor():
    txt = WF.read_text()
    assert "python data_guard.py" in txt
    assert "allow_shrink" in txt
    # the old inline absolute-only check is gone
    assert "refusing to publish'" not in txt


def test_actions_are_pinned_to_commit_shas():
    """This job holds contents:write on the branch GitHub Pages publishes, so
    "whatever v4 points at today" is a write-access dependency on someone
    else's mutable tag."""
    import re as _re
    for ref in _re.findall(r"uses:\s*(\S+)", WF.read_text()):
        _repo, _, rev = ref.partition("@")
        assert _re.fullmatch(r"[0-9a-f]{40}", rev), f"{ref} is not pinned to a SHA"


def test_dependencies_are_installed_by_hash():
    txt = WF.read_text()
    assert "--require-hashes" in txt
    assert 'pip install "plotly==' not in txt, (
        "the ad-hoc plotly override is back — pin it in requirements.txt instead"
    )
    lock = ROOT / "requirements.lock"
    assert lock.exists(), "no lockfile for the hash-pinned install"
    body = lock.read_text()
    assert "--hash=sha256:" in body
    assert "# WARNING" not in body, "the lockfile has unpinned packages"
