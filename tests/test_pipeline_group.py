"""The scrape's memory between runs, and the CI that watches the bot.

Four defects, all of the same shape: something the pipeline needed to remember
or check lived only for one run.

* Frozen coding/agentic scores were carried from the committed CSV alone, so a
  model missing from ONE published scrape lost them forever (143 -> 100 coding
  scores in two weeks). data/carried/ now holds a sidecar that only gains rows.
* AA's `deprecated` flag toggled hourly and the scrape passed every toggle
  straight to the site. A published model is now dropped only after a streak.
* Bot commits never ran the Tests workflow (GITHUB_TOKEN pushes trigger
  nothing), and a test named a model that had just been deprecated.
* The 24h row-loss guard ran against a depth-1 checkout and skipped itself.
"""
import io
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import data_guard
from data import carried
from data import scraper as S

ROOT = Path(__file__).resolve().parent.parent
REFRESH = ROOT / ".github" / "workflows" / "refresh.yml"
TESTS = ROOT / ".github" / "workflows" / "tests.yml"
NAN = float("nan")


# ── Frozen-score sidecar ──────────────────────────────────────────────────────

def test_the_sidecar_only_gains_or_updates_rows():
    side = pd.DataFrame({"model": ["A", "B"], "coding": [70.0, 60.0],
                         "agentic": [50.0, NAN]})
    # B is missing from this scrape, A has a hole, C is new.
    scrape = pd.DataFrame({"model": ["A", "C"], "coding": [NAN, 40.0],
                           "agentic": [55.0, NAN]})
    out = carried.merge_scores(side, scrape, "model").set_index("model")

    assert set(out.index) == {"A", "B", "C"}, "a model absent from one scrape lost its row"
    assert out.loc["A", "coding"] == 70.0, "a NaN in the scrape overwrote a stored score"
    assert out.loc["A", "agentic"] == 55.0, "a newer value did not update the row"
    assert out.loc["B", "coding"] == 60.0
    assert out.loc["C", "coding"] == 40.0


def test_a_rewrite_with_nothing_new_leaves_the_file_byte_identical(tmp_path):
    """The refresh workflow commits whenever data/carried/ moves, so a no-op
    upsert must not churn the file."""
    p = tmp_path / "s.csv"
    df = pd.DataFrame({"model": ["B", "A"], "coding": [60.0, 70.123],
                       "agentic": [NAN, 50.0]})
    assert carried.save_sidecar(df, p, "model") is True
    first = p.read_text()
    assert carried.remember(df, "scraper", path=p) is False
    assert p.read_text() == first
    assert first.splitlines()[1].startswith("A,70.12"), "not sorted / rounded"


def test_a_model_that_flapped_out_gets_its_score_back_from_the_sidecar(tmp_path):
    """The actual failure: the committed CSV lost the row for one hour, so when
    the model came back there was nothing to carry."""
    side = tmp_path / "hosted.csv"
    carried.save_sidecar(pd.DataFrame({"model": ["Opus"], "coding": [77.98],
                                       "agentic": [56.22]}), side, "model")
    cached = pd.DataFrame({"model": ["Other"], "coding": [10.0], "agentic": [5.0]})
    live = pd.DataFrame({"model": ["Opus", "Other", "New"],
                         "coding": [NAN] * 3, "agentic": [NAN] * 3})

    out = S._carry_dropped_columns(live, cached, sidecar=side)

    assert out.loc[0, "coding"] == 77.98 and out.loc[0, "agentic"] == 56.22
    assert out.loc[1, "coding"] == 10.0, "the cached CSV no longer fills its own rows"
    assert pd.isna(out.loc[2, "coding"]), "a model in neither source inherited a score"


def test_live_beats_cache_beats_sidecar(tmp_path):
    side = tmp_path / "hosted.csv"
    carried.save_sidecar(pd.DataFrame({"model": ["A"], "coding": [1.0],
                                       "agentic": [1.0]}), side, "model")
    cached = pd.DataFrame({"model": ["A"], "coding": [2.0], "agentic": [NAN]})
    live = pd.DataFrame({"model": ["A"], "coding": [NAN], "agentic": [NAN]})
    out = S._carry_dropped_columns(live, cached, sidecar=side)
    assert out.loc[0, "coding"] == 2.0 and out.loc[0, "agentic"] == 1.0

    live = pd.DataFrame({"model": ["A"], "coding": [3.0], "agentic": [NAN]})
    assert S._carry_dropped_columns(live, cached, sidecar=side).loc[0, "coding"] == 3.0


def test_the_open_weight_catalogue_reads_its_own_sidecar(tmp_path, monkeypatch):
    """local_scraper calls _carry_dropped_columns(key="name", tag="local_scraper")
    and must get the local sidecar, keyed on its own name column."""
    monkeypatch.setattr(carried, "CARRIED_DIR", tmp_path)
    carried.save_sidecar(pd.DataFrame({"name": ["M"], "coding": [30.0],
                                       "agentic": [NAN]}),
                         carried.sidecar_path("local_scraper"), "name")
    live = pd.DataFrame({"name": ["M"], "coding": [NAN], "agentic": [NAN]})
    out = S._carry_dropped_columns(live, None, key="name", tag="local_scraper")
    assert out.loc[0, "coding"] == 30.0


def test_the_committed_sidecars_hold_every_score_ever_published():
    """Backfilled from git history. Every coding/agentic value in the current
    CSVs must be in the sidecar, and so must every one in b2ecf2c — the last
    commit before the flaps started erasing them (143 coding, 106 agentic)."""
    for tag, (fname, key, csv) in carried.SIDECARS.items():
        side = carried.load_sidecar(carried.CARRIED_DIR / fname, key)
        assert not side.empty, f"{fname} is missing or empty"
        idx = side.set_index(key)
        current = pd.read_csv(ROOT / csv)
        for col in carried.FROZEN_COLUMNS:
            have = current.dropna(subset=[col])
            missing = sorted(set(have[key]) - set(idx[col].dropna().index))
            assert not missing, f"{fname}: {col} missing for {missing[:5]}"

    try:
        text = subprocess.run(["git", "show", "b2ecf2c:data/raw/aa_models.csv"],
                              cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("b2ecf2c not in this checkout")
    old = pd.read_csv(io.StringIO(text))
    idx = carried.load_sidecar(carried.CARRIED_DIR / "hosted_scores.csv",
                               "model").set_index("model")
    for col in carried.FROZEN_COLUMNS:
        lost = sorted(set(old.dropna(subset=[col])["model"]) - set(idx[col].dropna().index))
        assert not lost, f"{col} scores from b2ecf2c were not backfilled: {lost[:5]}"
    assert idx["coding"].notna().sum() >= 143
    assert idx["agentic"].notna().sum() >= 106


def test_backfill_reads_history_and_never_drops_a_score(tmp_path):
    out = tmp_path / "hosted.csv"
    stats = carried.backfill_from_git("scraper", path=out, include_worktree=False)
    if stats["versions_read"] == 0:
        pytest.skip("no git history with coding/agentic in this checkout")
    committed = carried.load_sidecar(carried.CARRIED_DIR / "hosted_scores.csv", "model")
    rebuilt = carried.load_sidecar(out, "model")
    lost = sorted(set(committed["model"]) - set(rebuilt["model"]))
    extra = sorted(set(rebuilt["model"]) - set(committed["model"]))
    # History alone may lack scores first seen in an uncommitted worktree, but
    # the committed sidecar must never be missing something history has.
    assert not extra, f"history has scores the committed sidecar lost: {extra[:5]}"
    assert len(lost) <= 5, f"history rebuild is far behind the committed sidecar: {lost[:5]}"
    assert stats["coding_scored"] >= 143


# ── Deprecation hysteresis ────────────────────────────────────────────────────

T0 = datetime(2026, 9, 22, 16, 0, tzinfo=timezone.utc)


def test_a_published_model_survives_a_one_hour_deprecated_flap():
    state, pub = {}, {"Opus 5", "Other"}
    hold, state = carried.deprecation_hysteresis(state, {"Opus 5"}, pub, T0)
    assert hold == {"Opus 5"}
    # Next hour the flag is gone: the streak resets.
    hold, state = carried.deprecation_hysteresis(state, set(), pub, T0 + timedelta(hours=1))
    assert hold == set() and state == {}
    # And a new flap starts from one again.
    hold, state = carried.deprecation_hysteresis(state, {"Opus 5"}, pub, T0 + timedelta(hours=2))
    assert state["Opus 5"]["count"] == 1


def test_a_steady_deprecation_is_dropped_after_the_streak():
    state, pub = {}, {"Opus 5"}
    for i in range(carried.DEPRECATED_DROP_AFTER_SCRAPES - 1):
        hold, state = carried.deprecation_hysteresis(
            state, {"Opus 5"}, pub, T0 + timedelta(hours=2 * i))
        assert hold == {"Opus 5"}, f"dropped after only {i + 1} scrape(s)"
    hold, state = carried.deprecation_hysteresis(
        state, {"Opus 5"}, pub, T0 + timedelta(hours=24))
    assert hold == set(), "a model deprecated for a day is still being held"
    assert state == {}, "a dropped model is still being tracked"


def test_several_scrapes_inside_one_flap_do_not_count_as_a_streak():
    """Four cron entries an hour plus an external dispatch: three runs fifteen
    minutes apart must not ride out a 40-90 minute flap."""
    state, pub = {}, {"Opus 5"}
    for i in range(6):
        hold, state = carried.deprecation_hysteresis(
            state, {"Opus 5"}, pub, T0 + timedelta(minutes=15 * i))
    assert hold == {"Opus 5"}


def test_models_never_published_are_not_tracked():
    """~95 long-superseded models were never in the catalogue. Holding them
    on first deploy would be a flap of its own."""
    hold, state = carried.deprecation_hysteresis({}, {"Ancient"}, {"Current"}, T0)
    assert hold == set() and state == {}


def test_parse_counts_and_holds_deprecated_models():
    rec = lambda n, dep: {  # noqa: E731
        "name": n, "slug": n, "modelCreatorName": "Lab", "intelligenceIndex": 50.0,
        "contextWindowTokens": 128_000, "deprecated": dep,
        "price1mInputTokens": 1.0, "price1mOutputTokens": 2.0,
        "medianOutputTokensPerSecond": 100, "medianTimeToFirstTokenSeconds": 1.0,
    }
    models = [rec("Live", False), rec("Flapping", True), rec("Gone", True)]

    names = {r[0] for r in S._parse_api_response(models, hold_deprecated={"Flapping"})}
    assert names == {"Live", "Flapping"}
    assert S._last_coverage["skipped_deprecated"] == ["Gone"]
    assert S._last_coverage["held_deprecated"] == ["Flapping"]
    # The quoted denominator is unchanged in meaning: deprecated is not "failed to carry".
    assert S._last_coverage["distinct_upstream_models"] == S._last_coverage["kept"] == 2

    # The default is still the pure, pre-hysteresis behaviour.
    assert {r[0] for r in S._parse_api_response(models)} == {"Live"}


@pytest.mark.parametrize("pulled", [
    {"price1mInputTokens": None, "price1mOutputTokens": None},
    {"intelligenceIndex": None},
])
def test_a_held_model_that_loses_its_price_or_score_is_deprecated_not_uncarried(pulled):
    """AA can pull a price in the same scrape it flags the model deprecated.
    Held or not, that model is deprecated: it must not surface as "not
    carried — no published price" nor inflate the quoted denominator."""
    m = {"name": "X", "slug": "x", "modelCreatorName": "Lab", "deprecated": True,
         "intelligenceIndex": 50.0, "price1mInputTokens": 1.0,
         "price1mOutputTokens": 2.0, **pulled}
    assert S._parse_api_response([m], hold_deprecated={"X"}) == []
    cov = S._last_coverage
    assert cov["skipped_deprecated"] == ["X"]
    assert cov["skipped_no_price"] == cov["skipped_no_score"] == []
    assert cov["held_deprecated"] == []
    assert cov["distinct_upstream_models"] == 0


# ── End to end through _scrape_and_save ───────────────────────────────────────

def _record(i: int, deprecated: bool = False) -> dict:
    return {
        "name": f"Model {i}", "slug": f"m{i}", "modelCreatorName": f"Lab{i % 3}",
        "intelligenceIndex": 40.0 + i, "contextWindowTokens": 128_000,
        "deprecated": deprecated,
        "price1mInputTokens": 1.0, "price1mOutputTokens": 4.0 + i,
        "medianOutputTokensPerSecond": 50 + i, "medianTimeToFirstTokenSeconds": 0.5,
    }


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Every file _scrape_and_save touches, redirected into tmp_path."""
    from data.ingest import load_from_raw

    monkeypatch.setattr(carried, "CARRIED_DIR", tmp_path / "carried")
    monkeypatch.setattr(carried, "DEPRECATED_DROP_AFTER_HOURS", 0.0)
    monkeypatch.setattr(S, "COVERAGE_PATH", tmp_path / "coverage.json")

    # Round 0: the committed cache, every model carrying frozen scores.
    seed = load_from_raw(S._parse_api_response([_record(i) for i in range(10)]))
    num = seed["model"].str.removeprefix("Model ").astype(int)
    seed["coding"] = 60.0 + num
    seed["agentic"] = 30.0 + num
    box = SimpleNamespace(cache=seed, records=[], path=tmp_path)

    monkeypatch.setattr(S, "load_cached", lambda: box.cache.copy())
    monkeypatch.setattr(S, "save_cache", lambda df: setattr(box, "cache", df.copy()))
    monkeypatch.setattr(S, "_extract_models", lambda html: json.loads(json.dumps(box.records)))
    monkeypatch.setattr(S.requests, "get", lambda *a, **k: SimpleNamespace(
        text="", raise_for_status=lambda: None))

    def run(deprecated=()):
        box.records = [_record(i, deprecated=i in deprecated) for i in range(10)]
        assert S._scrape_and_save() is True
        return set(box.cache["model"]), json.loads(box.path.joinpath("coverage.json").read_text())
    box.run = run
    return box


def test_a_flap_no_longer_reaches_the_catalogue_or_erases_scores(sandbox):
    names, cov = sandbox.run(deprecated={3})
    assert "Model 3" in names, "a one-scrape deprecated flag removed a published model"
    assert cov["held_deprecated"] == ["Model 3"]

    names, cov = sandbox.run()
    assert "Model 3" in names and cov["held_deprecated"] == []
    state = carried.load_deprecation_state()
    assert state == {}, "a clean scrape did not reset the streak"
    sidecar = carried.load_sidecar(carried.sidecar_path("scraper"), "model")
    assert len(sidecar) == 10, "the published scrape was not folded into the sidecar"


def test_a_real_deprecation_drops_after_the_streak_and_scores_come_back(sandbox):
    for _ in range(carried.DEPRECATED_DROP_AFTER_SCRAPES - 1):
        names, _cov = sandbox.run(deprecated={3})
        assert "Model 3" in names
    names, cov = sandbox.run(deprecated={3})
    assert "Model 3" not in names
    assert cov["skipped_deprecated"] == ["Model 3"], "the drop is not counted in coverage.json"
    assert "Model 3" not in set(sandbox.cache["model"])

    # AA un-deprecates it. The committed CSV no longer has its row — before the
    # sidecar, this is where coding/agentic were lost for good.
    names, _cov = sandbox.run()
    row = sandbox.cache.set_index("model").loc["Model 3"]
    assert row["coding"] == 63.0 and row["agentic"] == 33.0


def test_a_refused_scrape_does_not_advance_the_streak(sandbox, monkeypatch):
    sandbox.run(deprecated={3})
    before = carried.load_deprecation_state()
    monkeypatch.setattr(S, "column_health_violations", lambda df: ["forced"])
    sandbox.records = [_record(i, deprecated=i == 3) for i in range(10)]
    assert S._scrape_and_save() is False
    assert carried.load_deprecation_state() == before


# ── The 24h guard needs history ───────────────────────────────────────────────

def test_the_guard_says_when_the_24h_baseline_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(data_guard, "_baseline_rev", lambda h: None)
    monkeypatch.setattr(data_guard, "_is_shallow", lambda: True)
    monkeypatch.setattr(data_guard, "check", lambda **k: [])
    msg = data_guard.baseline_problem()
    assert msg and "fetch-depth: 0" in msg

    monkeypatch.setattr("sys.argv", ["data_guard.py", "--require-baseline"])
    assert data_guard.main() == 1, "a blind cumulative check passed under --require-baseline"
    monkeypatch.setattr("sys.argv", ["data_guard.py"])
    assert data_guard.main() == 0
    assert "::warning::shallow checkout" in capsys.readouterr().out


def test_the_guard_is_quiet_when_a_baseline_exists(monkeypatch):
    monkeypatch.setattr(data_guard, "_baseline_rev", lambda h: "abc123")
    assert data_guard.baseline_problem() is None


def _step(txt: str, marker: str, width: int = 1200) -> str:
    i = txt.index(marker)
    return txt[i:i + width]


def test_refresh_checks_out_full_history():
    txt = REFRESH.read_text()
    checkout = _step(txt, "uses: actions/checkout@", 800)
    assert "fetch-depth: 0" in checkout, "refresh.yml checkout is shallow again"
    assert "--require-baseline" in txt, "the guard may silently skip its 24h check"


def test_refresh_commits_the_carried_state():
    """A runner starts from a fresh checkout: uncommitted sidecar or streak
    state is state of one run."""
    txt = REFRESH.read_text()
    assert "git add -- data/raw data/carried docs" in txt
    guard = _step(txt, "Change guard")
    assert "git diff --quiet -- data/raw/ data/carried/" in guard
    assert "--exclude-standard -- data/raw/ data/carried/" in guard


def test_bot_commits_trigger_the_test_suite():
    """GITHUB_TOKEN pushes start no workflows, so `on: push` never saw a bot
    commit. workflow_run must name the refresh workflow by its exact name."""
    refresh_name = REFRESH.read_text().splitlines()[0].split(":", 1)[1].strip()
    txt = TESTS.read_text()
    assert "workflow_run:" in txt
    assert f'workflows: ["{refresh_name}"]' in txt, (
        f"tests.yml does not listen for {refresh_name!r}"
    )
    assert "types: [completed]" in txt
    # Concurrency resolves at queue time, before the job `if`: a no-op
    # refresh must not share the push group, or it cancels a human commit's
    # in-flight run and then skips itself.
    import re
    group = re.search(r"^concurrency:\n  group: (.+)$", txt, re.M).group(1)
    noop = "github.event.workflow_run.head_sha == github.sha"
    assert noop in group and "github.run_id" in group, group
    job_if = txt.split("jobs:", 1)[1]
    assert "github.event.workflow_run.head_sha != github.sha" in job_if


def test_ci_actions_stay_pinned_to_shas():
    import re
    for wf in (REFRESH, TESTS):
        for ref in re.findall(r"uses:\s*(\S+)", wf.read_text()):
            assert re.fullmatch(r"[0-9a-f]{40}", ref.partition("@")[2]), f"{wf.name}: {ref}"
