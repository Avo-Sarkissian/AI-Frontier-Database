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
