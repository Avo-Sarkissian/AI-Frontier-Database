"""AA's Intelligence Index v5.0 (2026-10-28), and the next one.

v5.0 re-scored the leaderboard on a new eval suite: only 77 of 273 models kept
a score, the survivors moved non-uniformly (x1.08 at the top, x0.36 at the
bottom), the hosted catalogue went 162 -> 73 and the open-weight one 173 -> 21.
Both scrapers refused the shrink for nine hours, then data_guard refused the
publish on every hourly run, so the site served v4 scores for a day and needed
a hand fix. These pin the self-healing path: the page names its version, the
scrapers record it per dataset, a change is accepted as a rescale at once, and
an unknown version is never mistaken for a change.
"""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from data import carried
from data.rsc import detect_index_version


def _page(payload: str) -> str:
    return f"<script>self.__next_f.push([1,{json.dumps(payload)}])</script>"


# The two forms the live page carried on 2026-10-29, verbatim.
_EVAL = ('{"slug":"artificial-analysis-intelligence-index",'
         '"name":"Artificial Analysis Intelligence Index v5.0","keywords":[]}')
_CHANGELOG = '{"type":"updated","description":"Updated to Intelligence Index v5"}'


# ── Detection ────────────────────────────────────────────────────────────────

def test_the_evaluation_entry_names_the_version_in_raw_html_and_in_the_payload():
    from data.rsc import payload_from_html
    html = _page("0:" + _CHANGELOG + _EVAL)
    assert '\\"slug\\"' in html, "the fixture must carry the escaped form raw HTML has"
    assert detect_index_version(html) == "5.0"
    assert detect_index_version(payload_from_html(html)) == "5.0"


def test_a_bare_major_version_is_normalised():
    assert detect_index_version(_page("0:" + _CHANGELOG)) == "5.0"


def test_the_evaluation_entry_wins_over_other_mentions():
    """A changelog can name an older version; the leaderboard's own entry is
    the one it renders."""
    text = ("Intelligence Index v6 is coming" +
            _EVAL.replace("v5.0", "v5.1"))
    assert detect_index_version(text) == "5.1"


def test_without_the_entry_the_highest_mention_wins():
    assert detect_index_version("Intelligence Index v4.2 ... Intelligence Index v10") == "10.0"


@pytest.mark.parametrize("text", ["", None, "<html>Intelligence Index</html>"])
def test_a_page_naming_no_version_is_unknown(text):
    assert detect_index_version(text) is None


# ── The per-dataset record ───────────────────────────────────────────────────

T = datetime(2026, 10, 29, 11, tzinfo=timezone.utc)


def test_the_record_moves_only_when_the_version_does(tmp_path):
    path = tmp_path / "index_version.json"
    assert carried.index_rescale("hosted", "5.0", path=path) is None, \
        "no record yet is not a change"
    assert carried.record_index_version("hosted", "5.0", now=T, path=path)
    assert not carried.record_index_version("hosted", "5.0", path=path)
    assert not carried.record_index_version("hosted", None, path=path)
    rec = json.loads(path.read_text())["hosted"]
    assert rec == {"version": "5.0", "since": T.isoformat(), "previous": None}

    assert carried.index_rescale("hosted", "6.0", path=path) == "5.0 -> 6.0"
    assert carried.index_rescale("hosted", None, path=path) is None
    assert carried.record_index_version("hosted", "6.0", path=path)
    rec = json.loads(path.read_text())["hosted"]
    assert rec["version"] == "6.0" and rec["previous"] == "5.0"


def test_the_two_datasets_are_recorded_separately(tmp_path):
    """The hosted scrape publishing first must not tell the open-weight scrape
    that nothing changed: they publish or refuse independently."""
    path = tmp_path / "index_version.json"
    carried.record_index_version("hosted", "5.0", path=path)
    carried.record_index_version("local", "5.0", path=path)
    carried.record_index_version("hosted", "6.0", path=path)
    assert carried.index_rescale("local", "6.0", path=path) == "5.0 -> 6.0"
    assert carried.index_rescale("hosted", "6.0", path=path) is None


def test_the_committed_record_says_v5():
    for ds in ("hosted", "local"):
        assert carried.recorded_index_version(ds) == "5.0"


@pytest.mark.parametrize("text", [None, "", "{", "[]", '{"hosted": {"version": 5}}'])
def test_an_unreadable_record_is_no_record(text):
    assert carried.index_version_record(text, "hosted") == {}


# ── The scrapers' own shrink confirmation ────────────────────────────────────

BEFORE = {f"m{i}" for i in range(162)}
AFTER = {f"m{i}" for i in range(73)}
REFUSED = ["row count 162 -> 73, a 55% drop (limit 20%)"]


def test_a_rescale_publishes_its_shrink_on_the_first_scrape(tmp_path):
    path = tmp_path / "shrink.json"
    assert carried.confirm_shrink("hosted", list(REFUSED), BEFORE, AFTER, now=T,
                                  path=path, rescale="5.0 -> 6.0") == []
    acc = carried.load_shrink_state(path)["hosted"]["accepted"]
    assert acc["rows"] == 73 and acc["index_version"] == "5.0 -> 6.0"
    # and data_guard's cumulative check measures against it
    assert carried.accepted_shrink("hosted", 48, now=T, path=path) == 73


def test_without_a_rescale_the_same_shrink_waits(tmp_path):
    path = tmp_path / "shrink.json"
    assert carried.confirm_shrink("hosted", list(REFUSED), BEFORE, AFTER, now=T, path=path)


def test_a_rescale_does_not_excuse_a_different_array(tmp_path):
    """A version change with strangers in the rows is still a wrong array."""
    strangers = {f"x{i}" for i in range(73)}
    assert carried.confirm_shrink("hosted", list(REFUSED), BEFORE, strangers, now=T,
                                  path=tmp_path / "s.json", rescale="5.0 -> 6.0")


# ── End to end through the hosted scraper ────────────────────────────────────

def _record(i: int) -> dict:
    return {"slug": f"m-{i}", "name": f"m{i}", "modelCreatorName": "Lab",
            "intelligenceIndex": 30.0 + i % 20, "intelligenceIndexIsEstimated": False,
            "contextWindowTokens": 128_000, "deprecated": False,
            "price1mInputTokens": 1.0, "price1mOutputTokens": 4.0,
            "medianOutputTokensPerSecond": 50.0, "medianTimeToFirstTokenSeconds": 0.5}


class _Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    """data.scraper with every write redirected and a 162-row published cache."""
    from data import scraper
    from data.ingest import load_from_raw

    published = load_from_raw(scraper._parse_api_response([_record(i) for i in range(162)]))
    saved = {}
    monkeypatch.setattr(carried, "CARRIED_DIR", tmp_path)
    monkeypatch.setattr(scraper, "load_cached", lambda: saved.get("df", published))
    monkeypatch.setattr(scraper, "save_cache", lambda df: saved.__setitem__("df", df))
    monkeypatch.setattr(scraper, "_save_coverage", lambda: None)
    for fn in ("learn_creator_colors", "remember", "learn_open_weights"):
        monkeypatch.setattr(carried, fn, lambda *a, **k: False)

    def run(version: str | None, n: int = 73):
        recs = json.dumps([_record(i) for i in range(n)])
        name = (f',"x":{{"slug":"artificial-analysis-intelligence-index",'
                f'"name":"Artificial Analysis Intelligence Index v{version}"}}'
                if version else "")
        html = _page('0:{"initialModels":' + recs + name + "}")
        monkeypatch.setattr(scraper.requests, "get", lambda *a, **k: _Resp(html))
        return scraper._scrape_and_save()

    carried.record_index_version("hosted", "5.0")
    return run, saved


def test_the_hosted_scraper_publishes_a_rescale_at_once_and_records_it(hosted):
    run, saved = hosted
    assert run("6.0") is True
    assert len(saved["df"]) == 73
    assert carried.recorded_index_version("hosted") == "6.0"


@pytest.mark.parametrize("version", ["5.0", None])
def test_the_hosted_scraper_refuses_the_same_shrink_with_no_rescale(hosted, version):
    """Same version, or a page that names none: the 12h confirmation stands."""
    run, saved = hosted
    assert run(version) is False
    assert "df" not in saved
    assert carried.recorded_index_version("hosted") == "5.0"


def test_a_refused_scrape_does_not_record_the_new_version(hosted, monkeypatch):
    """The record describes what was published. Recording a version the cache
    is not on would make the next run's rescale invisible."""
    from data import scraper
    run, saved = hosted
    monkeypatch.setattr(scraper, "column_health_violations", lambda df: ["broken"])
    assert run("6.0") is False
    assert carried.recorded_index_version("hosted") == "5.0"


def test_both_llm_scrapers_detect_and_record_against_their_own_dataset():
    """The open-weight scraper reads the same page; its wiring mirrors the
    hosted one's, against the "local" record."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for mod, ds in (("data/scraper.py", "hosted"), ("data/local_scraper.py", "local")):
        src = (root / mod).read_text()
        assert f'carried.index_rescale("{ds}", index_version)' in src, mod
        assert f'carried.record_index_version("{ds}", index_version)' in src, mod
        assert "rescale=rescale)" in src, mod
        # recorded only after the CSV is written
        assert src.index(f'carried.record_index_version("{ds}"') > \
            src.index("to_csv(_CACHE" if ds == "local" else "save_cache(df)"), mod
