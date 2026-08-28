import os
import subprocess
import sys
from pathlib import Path

import pytest
import pandas as pd
import requests
from unittest import mock

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
        "python -m data.video_scraper",
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


@pytest.mark.parametrize("mod", ["data.scraper", "data.local_scraper", "data.image_scraper",
     "data.video_scraper"])
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


def test_arch_scraper_exits_nonzero_when_upstream_fails(tmp_path):
    """arch_scraper is in refresh.yml's `|| failed=` chain too, so its exit code
    carries the same contract — but only on a run that actually looks something
    up. Point it at an empty ledger so it has work to do, then kill the network."""
    driver = """
import runpy, sys, requests
def _boom(*a, **k):
    raise requests.RequestException("forced failure for test")
requests.get = _boom
sys.argv = ["arch_scraper", "--limit", "2"]
runpy.run_module("data.arch_scraper", run_name="__main__")
"""
    env = {**os.environ,
           "AI_FRONTIER_STATUS_PATH": str(tmp_path / "status.json"),
           "AI_FRONTIER_ARCH_UNRESOLVED_PATH": str(tmp_path / "unresolved.csv")}
    proc = subprocess.run([sys.executable, "-c", driver], cwd=ROOT,
                          capture_output=True, text=True, env=env)
    assert proc.returncode != 0, (
        "arch_scraper exited 0 with the network down, so refresh.yml's `||` "
        f"guard never fires.\nstdout:\n{proc.stdout}"
    )
    assert not (tmp_path / "unresolved.csv").exists(), (
        "an outage was written into the settled-non-match ledger"
    )


def test_arch_scraper_makes_no_claim_when_it_has_nothing_to_look_up(tmp_path):
    """The other side of that contract, and it is deliberate.

    Once every resolvable name is cached and the rest are resting, a run does no
    network work — so it has no evidence about upstream and must not invent
    any. An HF outage during such an hour makes nothing staler than it already
    was; failing the run would be the false alarm this whole change removes.
    """
    from datetime import date
    from data import arch_scraper as A

    catalogue = pd.read_csv(A._LOCAL_CACHE)
    ledger = {str(r["name"]): {"name": str(r["name"]),
                               "params_b": float(r.get("params_b") or 0),
                               "attempts": 1, "last_tried": date.today().isoformat(),
                               "reason": "test"}
              for _, r in catalogue.iterrows()}

    def _boom(*a, **k):
        raise AssertionError("a rested run must not touch the network")

    with mock.patch.object(A, "_load_unresolved", lambda: ledger), \
         mock.patch.object(A, "_get", _boom), \
         mock.patch.object(A, "_save_unresolved", lambda l: None):
        assert A.scrape_and_save(sleep_s=0) is True


def test_workflow_runs_the_arch_scraper():
    """It is in the `|| failed=` chain, so it is part of the red/green contract."""
    assert "python -m data.arch_scraper" in WF.read_text()


def test_arch_scraper_is_green_when_the_api_answers_but_nothing_qualifies():
    """The inverse of the 29-silent-days bug, and just as bad.

    Forty catalogue rows are product names HuggingFace has no qualifying repo
    for; they fall back to the estimator by design. arch_scraper reported that
    steady state as `resolved 0 -> exit 1`, so refresh.yml failed EVERY hourly
    run with "the upstream endpoint is failing" while the endpoint was fine.
    An alert that fires sixty times a day is not read the sixty-first time.
    """
    from data import arch_scraper as A

    calls = {"search": 0}

    def _fake_get(url, params=None):
        calls["search"] += 1
        if params and params.get("search") == "llama":      # the control
            return [{"id": "meta-llama/Llama-3.1-8B", "safetensors": {"total": 8_030_261_248}}]
        return [{"id": "someone/unrelated-7b", "safetensors": {"total": 7_000_000_000}}]

    with mock.patch.object(A, "_get", _fake_get), \
         mock.patch.object(A, "_load_unresolved", lambda: {}), \
         mock.patch.object(A, "_save_unresolved", lambda ledger: None), \
         mock.patch.object(A, "time"):
        # 400B catalogue row vs a 7B candidate: nothing can clear the guard.
        assert A.scrape_and_save(limit=1, sleep_s=0) is True
    assert calls["search"], "the control query never ran"


def test_arch_scraper_is_red_when_the_control_query_fails():
    """...but a genuinely dead API must still turn the run red."""
    from data import arch_scraper as A

    def _fake_get(url, params=None):
        if params and params.get("search") == "llama":
            raise requests.RequestException("upstream down")
        return [{"id": "someone/unrelated-7b", "safetensors": {"total": 7_000_000_000}}]

    with mock.patch.object(A, "_get", _fake_get), \
         mock.patch.object(A, "_load_unresolved", lambda: {}), \
         mock.patch.object(A, "_save_unresolved", lambda ledger: None), \
         mock.patch.object(A, "time"):
        assert A.scrape_and_save(limit=1, sleep_s=0) is False


def test_a_transport_failure_is_never_recorded_as_a_settled_non_match():
    """A rate-limited lookup must not buy itself a week of silence.

    _RETRY_DAYS exists to stop the bot re-asking about names upstream has never
    published. If a 429 were written into that ledger it would suppress a name
    that was only ever unlucky.
    """
    from data import arch_scraper as A

    saved = {}
    def _boom(url, params=None):
        raise requests.RequestException("429")

    with mock.patch.object(A, "_get", _boom), \
         mock.patch.object(A, "_load_unresolved", lambda: {}), \
         mock.patch.object(A, "_save_unresolved", lambda ledger: saved.update(ledger)), \
         mock.patch.object(A, "time"):
        assert A.scrape_and_save(limit=2, sleep_s=0) is False
    assert saved == {}, f"a transport failure was cached as a non-match: {saved}"


def test_settled_non_matches_rest_between_retries():
    """Otherwise the hourly bot re-searches every permanent non-match forever —
    ~100 anonymous HuggingFace requests an hour to re-learn the same answer,
    which is how a client earns the rate limit that becomes a real outage."""
    from datetime import date, timedelta
    from data import arch_scraper as A

    today = date.today()
    fresh = {"params_b": 550.0, "last_tried": (today - timedelta(days=1)).isoformat()}
    stale = {"params_b": 550.0, "last_tried": (today - timedelta(days=A._RETRY_DAYS)).isoformat()}
    assert A._due(fresh, 550.0, today) is False
    assert A._due(stale, 550.0, today) is True
    assert A._due(None, 550.0, today) is True
    # the catalogue moved the parameter count, so the guard's verdict may change
    assert A._due(fresh, 551.0, today) is True
