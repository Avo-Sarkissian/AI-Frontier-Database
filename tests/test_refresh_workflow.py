from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / ".github" / "workflows" / "refresh.yml"

def test_workflow_present_and_wired():
    assert WF.exists(), "refresh.yml not created"
    txt = WF.read_text()
    for needle in [
        "schedule:",
        "workflow_dispatch:",
        "contents: write",
        "pip install -r requirements.txt",
        'pip install "plotly==6.5.2"',
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


def test_row_loss_guard_replaces_the_bare_floor():
    txt = WF.read_text()
    assert "python data_guard.py" in txt
    assert "allow_shrink" in txt
    # the old inline absolute-only check is gone
    assert "refusing to publish'" not in txt
