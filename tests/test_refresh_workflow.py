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
        "build_static.py --data-only",
        "git push",
    ]:
        assert needle in txt, f"workflow missing: {needle}"
