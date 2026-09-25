"""Seams between the five fixer groups — each fix spans two groups' files.

* The Run Local docstring quoted "±30%" for estimated KV after the hover moved
  to the measured band (median 33%, p90 272%).
* The open-weight scraper read the frozen-score sidecar but never wrote it, and
  dropped AA's flapping `deprecated` flag straight through, unlike the hosted one.
* The static site kept Value Leaders frozen on the full catalogue while Dash
  filtered it, and looked model details up by the HTML-escaped hover name.
* model_options ordered quality ties differently between Dash and Pyodide.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

from components.charts.local_compat import ESTIMATED_KV_NOTE
from data import carried
import data.local_scraper as L
from static_helpers import model_options

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "docs" / "app.js").read_text()


# ── Run Local: one error band, quoted everywhere ──────────────────────────────

def test_the_local_df_docstring_quotes_the_hover_band():
    from data.local_models import get_local_df
    median, p90 = re.search(r"median error (\d+)%, p90 (\d+)%", ESTIMATED_KV_NOTE).groups()
    doc = get_local_df.__doc__
    assert "±30%" not in doc
    assert f"median error {median}%, p90 {p90}%" in doc


# ── model_options: ties break on the name ─────────────────────────────────────

def test_model_options_breaks_quality_ties_by_name():
    df = pd.DataFrame({"model": ["Zeta", "Alpha", "Mid", "Gone"],
                       "provider": ["P", "P", "P", "P"],
                       "quality": [50.0, 50.0, 60.0, 0.0]})
    assert [o["value"] for o in model_options(df)] == ["Mid", "Alpha", "Zeta"]


# ── Static site: Value Leaders and the detail lookup ──────────────────────────

def test_the_static_rankings_refresh_redraws_value_leaders():
    block = APP_JS[APP_JS.index('rankings: makeRefresher("rankings"'):
                   APP_JS.index('budget: makeRefresher("budget"')]
    assert 'callPy("update_value_leaders", p, q, s, e)' in block
    assert 'renderJsonFig("chart-value_leaders"' in block
    assert "no callback" not in block


def test_the_pareto_click_looks_up_the_raw_model_name():
    fn = APP_JS[APP_JS.index("function attachParetoClickHandler"):]
    fn = fn[:fn.index("\n}\n")]
    assert "cd[5] ?? cd[0]" in fn and "cd[6] ?? cd[1]" in fn


def test_the_raw_names_ride_at_customdata_5_and_6():
    """The frozen hosted catalogue from the charts fixture, with one model and
    provider renamed to carry characters the hover escapes."""
    import gzip, pickle
    from components.charts.pareto import build_pareto_scatter
    with gzip.open(ROOT / "tests" / "fixtures" / "charts_annotation_inputs.pkl.gz", "rb") as f:
        hosted = pickle.load(f)["hosted"].copy()
    top = hosted["quality"].idxmax()
    hosted.loc[top, "model"] = "A&B <x>"
    hosted.loc[hosted["provider"] == hosted.loc[top, "provider"], "provider"] = "P&Q"
    fig = build_pareto_scatter(hosted)
    cds = [list(cd) for tr in fig.data if tr.customdata is not None
           for cd in tr.customdata if len(cd) > 6]
    hit = [cd for cd in cds if cd[5] == "A&B <x>"]
    assert hit and hit[0][6] == "P&Q"
    assert hit[0][0] != "A&B <x>", "cd[0] is expected to be the escaped hover text"


# ── Open-weight scrape: sidecar and deprecation hysteresis ────────────────────

def _rec(i, deprecated=False, coding=None):
    return {"name": f"Open {i}", "creator": {"id": "c", "name": "LabX"},
            "isOpenWeights": True, "deprecated": deprecated,
            "intelligenceIndex": 40.0 + i, "intelligenceIndexIsEstimated": False,
            "parameters": 20.0, "inferenceParametersActiveBillions": 20.0,
            "contextWindowTokens": 128_000, "licenseName": "MIT",
            "codingIndex": coding, "agenticIndex": None, "omniscience": -5.0}


@pytest.fixture
def local_sandbox(tmp_path, monkeypatch):
    """_scrape_and_save end to end, with the network, the cache and data/carried
    all redirected into tmp_path."""
    box = type("Box", (), {})()
    box.records = [_rec(i, coding=30.0 + i) for i in range(10)]
    monkeypatch.setattr(L, "_CACHE", tmp_path / "aa_local_models.csv")
    monkeypatch.setattr(carried, "CARRIED_DIR", tmp_path / "carried")

    class _Resp:
        text = ""
        def raise_for_status(self):
            pass

    monkeypatch.setattr(L.requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(L, "payload_from_html", lambda html: "")
    monkeypatch.setattr(L, "_entry_slug", lambda payload: "open-0")
    monkeypatch.setattr(L, "_extract_models", lambda html: box.records)

    def run(deprecated=()):
        box.records = [_rec(i, deprecated=i in deprecated,
                            coding=None if i in deprecated else 30.0 + i)
                       for i in range(10)]
        assert L._scrape_and_save() is True
        return set(pd.read_csv(L._CACHE)["name"])
    box.run = run
    box.state_path = tmp_path / "carried" / L._DEPRECATION_STATE_FILE
    return box


def test_the_local_scrape_folds_its_scores_into_the_sidecar(local_sandbox):
    local_sandbox.run()
    side = carried.load_sidecar(carried.sidecar_path("local_scraper"), "name")
    assert len(side) == 10
    assert side.set_index("name").loc["Open 3", "coding"] == pytest.approx(33.0)


def test_a_local_flap_is_held_on_its_own_streak(local_sandbox):
    local_sandbox.run()
    for _ in range(carried.DEPRECATED_DROP_AFTER_SCRAPES - 1):
        assert "Open 3" in local_sandbox.run(deprecated={3})
    assert carried.load_deprecation_state(local_sandbox.state_path)["Open 3"]["count"] \
        == carried.DEPRECATED_DROP_AFTER_SCRAPES - 1
    # The hosted scraper's state file is untouched: separate files, separate streaks.
    assert not (carried.CARRIED_DIR / carried.DEPRECATION_STATE_FILE).exists()


def test_a_clean_local_scrape_resets_the_streak(local_sandbox):
    local_sandbox.run()
    local_sandbox.run(deprecated={3})
    assert "Open 3" in local_sandbox.run()
    assert carried.load_deprecation_state(local_sandbox.state_path) == {}


def test_an_unpublished_deprecated_model_is_still_skipped(local_sandbox):
    # First scrape ever: nothing is published yet, so nothing is held.
    assert "Open 3" not in local_sandbox.run(deprecated={3})
