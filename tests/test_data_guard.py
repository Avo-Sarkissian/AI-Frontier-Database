# tests/test_data_guard.py
"""The relative row-loss guard that the 2026-07-24 contraction slipped past."""
import pytest

import data_guard


def _fake_counts(monkeypatch, now: dict, before: dict):
    monkeypatch.setattr(data_guard, "current_rows", lambda p: now.get(p))
    monkeypatch.setattr(data_guard, "committed_rows", lambda p, rev="HEAD": before.get(p))


def test_the_real_2026_07_24_contraction_is_rejected(monkeypatch):
    """329 -> 155 rows is a 53% drop; the old `> 50` floor let it publish."""
    _fake_counts(monkeypatch, {"a.csv": 155}, {"a.csv": 329})
    problems = data_guard.check(["a.csv"])
    assert len(problems) == 1
    assert "53% drop" in problems[0]


def test_normal_churn_passes(monkeypatch):
    _fake_counts(monkeypatch, {"a.csv": 152}, {"a.csv": 155})
    assert data_guard.check(["a.csv"]) == []


def test_growth_is_never_a_violation(monkeypatch):
    _fake_counts(monkeypatch, {"a.csv": 400}, {"a.csv": 155})
    assert data_guard.check(["a.csv"]) == []


def test_drop_exactly_at_the_limit_passes(monkeypatch):
    _fake_counts(monkeypatch, {"a.csv": 80}, {"a.csv": 100})
    assert data_guard.check(["a.csv"], max_drop_pct=20) == []


def test_drop_just_past_the_limit_fails(monkeypatch):
    _fake_counts(monkeypatch, {"a.csv": 79}, {"a.csv": 100})
    assert data_guard.check(["a.csv"], max_drop_pct=20)


def test_absolute_floor_still_applies(monkeypatch):
    """A tiny file fails even when the baseline is tiny too (no big % drop)."""
    _fake_counts(monkeypatch, {"a.csv": 12}, {"a.csv": 13})
    problems = data_guard.check(["a.csv"])
    assert problems and "floor" in problems[0]


def test_missing_file_is_a_violation(monkeypatch):
    _fake_counts(monkeypatch, {}, {"a.csv": 155})
    problems = data_guard.check(["a.csv"])
    assert problems and "missing" in problems[0]


def test_new_dataset_without_baseline_is_allowed(monkeypatch):
    _fake_counts(monkeypatch, {"a.csv": 155}, {})
    assert data_guard.check(["a.csv"]) == []


def test_real_repo_data_passes_against_head():
    """The committed CSVs must not themselves trip the guard."""
    assert data_guard.check() == []


def test_defaults_cover_every_scraped_csv():
    assert set(data_guard.DATA_CSVS) == {
        "data/raw/aa_models.csv",
        "data/raw/aa_local_models.csv",
        "data/raw/aa_image_models.csv",
        "data/raw/aa_video_models.csv",
    }



# ── AA Intelligence Index rescales (v5.0, 2026-10-28) ────────────────────────
#
# v5.0 took the hosted catalogue 162 -> 73 and the open-weight one 173 -> 21,
# moved the median price and latency >100%, and every hourly run failed for a
# day. A version change recorded by the scrapers is a rescale, not a fault.

_HOSTED, _LOCAL = "data/raw/aa_models.csv", "data/raw/aa_local_models.csv"


def _versions(monkeypatch, then: dict, now: dict):
    """index_version(dataset, rev) -> then[dataset]; (dataset, None) -> now.

    Also sets aside the scraper's accepted-shrink marker: the committed
    data/carried/shrink_state.json would otherwise lower these baselines."""
    monkeypatch.setattr(data_guard, "index_version",
                        lambda ds, rev=None: (now if rev is None else then).get(ds))
    monkeypatch.setattr(data_guard, "_accepted_floor", lambda path, rows: rows)


def _tripwires(monkeypatch):
    """Make the relative checks fail loudly whenever they run at all."""
    monkeypatch.setattr(data_guard, "_cumulative_violations",
                        lambda path, now: [f"{path}: CUMULATIVE RAN"])
    monkeypatch.setattr(data_guard, "_value_violations",
                        lambda path, rev: [f"{path}: MEDIAN RAN"])


def test_the_v5_rescale_is_accepted_with_its_reason_printed(monkeypatch, capsys):
    """The real numbers: 162 -> 73 hosted and 173 -> 21 local under 4.0 -> 5.0.
    Every relative check is waived for the run, and the log says why."""
    _fake_counts(monkeypatch, {_HOSTED: 73, _LOCAL: 21}, {_HOSTED: 162, _LOCAL: 173})
    _versions(monkeypatch, {"hosted": "4.0", "local": "4.0"},
              {"hosted": "5.0", "local": "5.0"})
    _tripwires(monkeypatch)
    assert data_guard.check([_HOSTED, _LOCAL]) == []
    out = capsys.readouterr().out
    assert out.count("4.0 -> 5.0") == 2 and "accepted as a rescale" in out
    assert "waived this run" in out


def test_without_a_version_change_the_same_numbers_still_fail(monkeypatch):
    """The waiver is the version change, nothing else: the identical counts on
    one version are exactly the broken parse the guard exists for."""
    _fake_counts(monkeypatch, {_HOSTED: 73, _LOCAL: 21}, {_HOSTED: 162, _LOCAL: 173})
    _versions(monkeypatch, {"hosted": "5.0", "local": "5.0"},
              {"hosted": "5.0", "local": "5.0"})
    problems = data_guard.check([_HOSTED, _LOCAL])
    assert any("55% drop" in p for p in problems), problems
    assert any(_LOCAL in p and "floor 50" in p for p in problems), problems


@pytest.mark.parametrize("then, now", [(None, "5.0"), ("5.0", None), (None, None)])
def test_an_unknown_version_on_either_side_is_never_a_rescale(monkeypatch, then, now):
    """A page that stops naming its version, or a baseline from before the
    record existed, must keep the guard at full strength."""
    _fake_counts(monkeypatch, {_HOSTED: 73}, {_HOSTED: 162})
    _versions(monkeypatch, {"hosted": then}, {"hosted": now})
    _tripwires(monkeypatch)
    problems = data_guard.check([_HOSTED])
    assert any("55% drop" in p for p in problems), problems


def test_a_rescale_keeps_an_absolute_backstop(monkeypatch):
    """A rescale that leaves a handful of rows is a broken parse wearing a
    version change; RESCALE_MIN_ROWS still refuses it."""
    _fake_counts(monkeypatch, {_LOCAL: data_guard.RESCALE_MIN_ROWS - 1}, {_LOCAL: 173})
    _versions(monkeypatch, {"local": "4.0"}, {"local": "5.0"})
    _tripwires(monkeypatch)
    problems = data_guard.check([_LOCAL])
    assert problems == [f"{_LOCAL}: only {data_guard.RESCALE_MIN_ROWS - 1} rows "
                        f"(floor {data_guard.RESCALE_MIN_ROWS})"]


def test_a_refilling_catalogue_keeps_the_lower_floor_and_every_relative_check(monkeypatch):
    """The hour after the rescale, HEAD holds 21 rows under 5.0 and so does the
    scrape. MIN_ROWS would refuse every run until AA had scored 50 open-weight
    models again; the lowered floor holds instead, and the row-drop, cumulative
    and median checks are all back."""
    _versions(monkeypatch, {"local": "5.0"}, {"local": "5.0"})
    _fake_counts(monkeypatch, {_LOCAL: 22}, {_LOCAL: 21})
    monkeypatch.setattr(data_guard, "_cumulative_violations", lambda path, now: [])
    monkeypatch.setattr(data_guard, "_value_violations", lambda path, rev: [])
    assert data_guard.check([_LOCAL]) == []
    _tripwires(monkeypatch)
    assert {p.split(": ", 1)[1] for p in data_guard.check([_LOCAL])} == \
        {"CUMULATIVE RAN", "MEDIAN RAN"}
    _fake_counts(monkeypatch, {_LOCAL: 16}, {_LOCAL: 21})          # a 24% drop
    assert any("24% drop" in p for p in data_guard.check([_LOCAL]))


def test_refilling_needs_a_recorded_version_and_a_published_low_count(monkeypatch):
    """The lowered floor is not a general discount: an unrecorded dataset, or
    one whose HEAD copy is above MIN_ROWS, keeps the original floor."""
    _fake_counts(monkeypatch, {_LOCAL: 21}, {_LOCAL: 21})
    _versions(monkeypatch, {}, {})
    assert any("floor 50" in p for p in data_guard.check([_LOCAL]))
    _versions(monkeypatch, {"local": "5.0"}, {"local": "5.0"})
    _fake_counts(monkeypatch, {_LOCAL: 21}, {_LOCAL: 60})
    assert any("floor 50" in p for p in data_guard.check([_LOCAL]))


def test_only_the_aa_llm_datasets_have_an_index_version():
    assert set(data_guard._INDEX_DATASET) == {_HOSTED, _LOCAL}
    assert "data/raw/aa_image_models.csv" not in data_guard._INDEX_DATASET


def test_the_cumulative_check_skips_a_window_that_spans_a_rescale(monkeypatch, capsys):
    monkeypatch.setattr(data_guard, "_baseline_rev", lambda hours: "abc123")
    monkeypatch.setattr(data_guard, "_show", lambda path, rev: "model\n" + "x\n" * 162)
    monkeypatch.setattr(data_guard, "_accepted_floor", lambda path, rows: rows)
    _versions(monkeypatch, {"hosted": "4.0"}, {"hosted": "5.0"})
    assert data_guard._cumulative_violations(_HOSTED, 73) == []
    assert "cumulative check skipped" in capsys.readouterr().out
    _versions(monkeypatch, {"hosted": "5.0"}, {"hosted": "5.0"})
    assert data_guard._cumulative_violations(_HOSTED, 73)


def test_index_version_reads_the_working_tree_and_git(monkeypatch, tmp_path):
    rec = '{"local": {"version": "5.0", "since": "x", "previous": null}}'
    (tmp_path / "data" / "carried").mkdir(parents=True)
    (tmp_path / data_guard.INDEX_VERSION_PATH).write_text(rec)
    monkeypatch.setattr(data_guard, "ROOT", tmp_path)
    assert data_guard.index_version("local") == "5.0"
    assert data_guard.index_version("hosted") is None
    monkeypatch.setattr(data_guard, "_show", lambda path, rev: None)
    assert data_guard.index_version("local", "HEAD") is None
    monkeypatch.setattr(data_guard, "_show", lambda path, rev: "not json")
    assert data_guard.index_version("local", "HEAD") is None


# ── Economics archives (data/econ_archive.py) ────────────────────────────────

def _csv(header: str, rows: list[str]) -> str:
    return header + "\n" + "\n".join(rows) + "\n"


_SOFR = "data/carried/econ/sofr.csv"
_SOFR_H = "effective_date,percent_rate,volume_bn,revision_indicator"


def _sofr_rows(n: int, rate: str = "3.93") -> list[str]:
    return [f"2020-01-{i % 28 + 1:02d}x{i},{rate},3000," for i in range(n)]


def test_append_only_archive_losing_one_row_is_a_violation():
    before = _csv(_SOFR_H, _sofr_rows(1200))
    after = _csv(_SOFR_H, _sofr_rows(1199))
    problems = data_guard.archive_violations(_SOFR, before, after)
    assert problems and "never lose a row" in problems[0]
    assert data_guard.archive_violations(_SOFR, before, _csv(_SOFR_H, _sofr_rows(1201))) == []


def test_epoch_mirror_drop_limits(monkeypatch):
    path = "data/raw/epoch/ai_companies_revenue_reports.csv"
    h = "Company,Date"
    rows = [f"Lab{i},2026-01-01" for i in range(100)]
    before = _csv(h, rows)
    assert data_guard.archive_violations(path, before, _csv(h, rows[:85])) == []      # 15%
    assert data_guard.archive_violations(path, before, _csv(h, rows[:75]))            # 25%
    floor = data_guard.archive_violations(path, _csv(h, rows[:41]), _csv(h, rows[:39]))
    assert floor and "floor" in floor[0]                                              # < 40 rows
    # --allow-shrink waives check_archives exactly as it waives check()
    monkeypatch.setattr(data_guard, "check", lambda **k: [])
    monkeypatch.setattr(data_guard, "check_archives", lambda rev="HEAD": ["x: a 25% drop"])
    monkeypatch.setattr(data_guard, "baseline_problem", lambda: None)
    monkeypatch.setattr("sys.argv", ["data_guard.py", "--allow-shrink"])
    assert data_guard.main() == 0
    monkeypatch.setattr("sys.argv", ["data_guard.py"])
    assert data_guard.main() == 1


def test_archive_median_shift_guard():
    """SOFR in basis points instead of percent is a units change, not a market."""
    before = _csv(_SOFR_H, _sofr_rows(1200))
    after = _csv(_SOFR_H, _sofr_rows(1200, rate="393") + _sofr_rows(1, rate="393"))
    problems = data_guard.archive_violations(_SOFR, before, after)
    assert problems and "units change" in problems[0]


def test_archive_rules_cover_only_known_paths():
    assert data_guard.archive_violations("data/raw/other.csv", None, "a\n1\n") == []
    assert data_guard.archive_violations(_SOFR, None, "")


def test_real_repo_archives_pass_against_head():
    assert data_guard.check_archives() == []
