# tests/test_data_guard.py
"""The relative row-loss guard that the 2026-07-24 contraction slipped past."""
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
