"""The Artificial Analysis fields the hosted scrape used to drop (2026-10-27).

AA's leaderboard records always carried cost per Intelligence Index task, cache
prices, the open-weights and reasoning flags and a slug, and a separate releases
array carries each base model's release date. They are now appended to
data/raw/aa_models.csv at indices 13..19 so their history accrues in
data/raw/history/ from the day they shipped.

What must hold:
  * each field parses to the number / flag AA published, and "AA did not say"
    stays unknown — never $0, never False;
  * small values written in e-notation survive (the old numeric parser strips
    everything after an "E");
  * the release-date join is the four exact tiers, never a prefix match;
  * none of it can take the scrape down: the record finder still needs only
    slug + intelligenceIndex;
  * every older CSV shape (7, 9, 13 fields; the 160-odd history snapshots)
    still loads, and today's snapshot is upgraded in place without touching
    what it already holds.
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data import ingest
from data import scraper as sc
from data.ingest import (AA_LATE_FIELDS, TRI_BOOL_COLUMNS, _parse_float,
                         _parse_numeric, _with_price_sides, load_from_raw)

ROOT = Path(__file__).resolve().parent.parent

OLD_HEADER = ["model", "provider", "context", "quality", "price", "speed",
              "latency", "price_in", "price_out", "quality_estimated", "coding",
              "agentic", "omniscience"]


def _page(payload: str) -> str:
    return f"<script>self.__next_f.push([1,{json.dumps(payload)}])</script>"


def _record(i=0, **over) -> dict:
    """One metrics record in the 2026-10 /leaderboards/models shape."""
    rec = {
        "slug": f"lab-model-{i}", "name": f"Lab Model {i}", "deprecated": False,
        "isReasoning": True, "isOpenWeights": False,
        "modelCreatorName": "Lab", "contextWindowTokens": 128_000,
        "intelligenceIndex": 40.0 + i, "intelligenceIndexIsEstimated": False,
        "omniscience": -3.0,
        "intelligenceIndexCostPerTask": 1.25,
        "price1mInputTokens": 4, "price1mOutputTokens": 20,
        "cacheHitPrice": 0.4, "cacheWritePrice": 5,
        "medianOutputTokensPerSecond": 80.0, "medianTimeToFirstTokenSeconds": 1.0,
    }
    rec.update(over)
    return rec


def _row_for(rec, releases=None) -> pd.Series:
    payload = '0:{"initialModels":' + json.dumps([rec])
    if releases is not None:
        payload += ',"releases":' + json.dumps(releases)
    payload += "}"
    models = sc._extract_models(_page(payload))
    df = load_from_raw(sc._parse_api_response(models))
    assert len(df) == 1
    return df.iloc[0]


# ── Parsing ──────────────────────────────────────────────────────────────────

def test_every_late_field_is_carried_with_its_value():
    row = _row_for(_record(), releases=[
        {"slug": "lab-model-0", "name": "Lab Model 0", "releaseDate": "2026-09-22"}])
    assert row["cost_per_task"] == pytest.approx(1.25)
    assert row["cache_hit_price"] == pytest.approx(0.4)
    assert row["cache_write_price"] == pytest.approx(5.0)
    assert row["is_open_weights"] is False or row["is_open_weights"] == False  # noqa: E712
    assert bool(row["is_reasoning"]) is True
    assert row["release_date"] == "2026-09-22"
    assert row["aa_slug"] == "lab-model-0"


def test_the_raw_row_puts_the_late_fields_at_13_to_19_in_order():
    """load_from_raw reads positionally, and the scraper's index map is what
    coverage.json counts by — the two must agree with the published order."""
    rows = sc._parse_api_response([_record()])
    assert len(rows[0]) == 20
    assert list(sc._LATE_FIELD_INDEX) == list(AA_LATE_FIELDS)
    assert list(sc._LATE_FIELD_INDEX.values()) == list(range(13, 20))
    df = load_from_raw(rows)
    assert list(df.columns) == OLD_HEADER + list(AA_LATE_FIELDS)


def test_small_cache_prices_survive_e_notation():
    """str(float) writes 5e-05 for a sub-cent cache price, and _parse_numeric
    strips from the first E — it would read that as $5."""
    assert _parse_numeric("5e-05") == 5.0          # the trap, pinned
    assert _parse_float("5e-05") == pytest.approx(5e-05)
    row = _row_for(_record(cacheHitPrice=0.00005, intelligenceIndexCostPerTask=0.00012))
    assert row["cache_hit_price"] == pytest.approx(0.00005)
    assert row["cost_per_task"] == pytest.approx(0.00012)


def test_a_zero_cost_per_task_is_not_a_free_task():
    """AA publishes 0 for five models it never costed (2026-10-27). $0 would put
    them at the cheap end of any cost-per-task chart."""
    row = _row_for(_record(intelligenceIndexCostPerTask=0))
    assert pd.isna(row["cost_per_task"])


def test_a_zero_cache_price_is_a_real_price():
    row = _row_for(_record(cacheHitPrice=0))
    assert row["cache_hit_price"] == 0.0


def test_absent_fields_are_unknown_not_zero_or_false():
    """154 of 272 records carry no intelligenceIndexCostPerTask key at all.
    "AA did not say" must not become $0 or closed-weights."""
    rec = _record()
    for k in ("intelligenceIndexCostPerTask", "cacheHitPrice", "cacheWritePrice",
              "isOpenWeights", "isReasoning"):
        rec.pop(k)
    row = _row_for(rec)
    for col in ("cost_per_task", "cache_hit_price", "cache_write_price",
                "release_date"):
        assert pd.isna(row[col]), f"{col} invented a value"
    assert row["is_open_weights"] is pd.NA
    assert row["is_reasoning"] is pd.NA


def test_null_and_undefined_upstream_values_are_unknown():
    row = _row_for(_record(cacheHitPrice=None, cacheWritePrice="$undefined",
                           isOpenWeights=None, intelligenceIndexCostPerTask=-1))
    assert pd.isna(row["cache_hit_price"]) and pd.isna(row["cache_write_price"])
    assert pd.isna(row["cost_per_task"])
    assert row["is_open_weights"] is pd.NA


def test_the_booleans_are_nullable_not_float():
    df = load_from_raw(sc._parse_api_response([
        _record(0, isOpenWeights=True), _record(1, isOpenWeights=False),
        {k: v for k, v in _record(2).items() if k != "isOpenWeights"}]))
    for col in TRI_BOOL_COLUMNS:
        assert str(df[col].dtype) == "boolean"
    assert int(df["is_open_weights"].eq(True).sum()) == 1
    assert int(df["is_open_weights"].eq(False).sum()) == 1
    assert int(df["is_open_weights"].isna().sum()) == 1


def test_the_record_finder_still_needs_only_slug_and_intelligence_index():
    """A late field AA drops must blank one column, not take the scrape down."""
    bare = {"slug": "s", "name": "Bare", "intelligenceIndex": 30.0,
            "modelCreatorName": "Lab", "price1mInputTokens": 1,
            "price1mOutputTokens": 2}
    models = sc._extract_models(_page('0:{"initialModels":' + json.dumps([bare]) + "}"))
    df = load_from_raw(sc._parse_api_response(models))
    assert len(df) == 1
    assert df.iloc[0]["aa_slug"] == "s"
    assert pd.isna(df.iloc[0]["release_date"])


# ── The release-date join ────────────────────────────────────────────────────

_RELEASES = [
    {"slug": "claude-sonnet-5-5", "name": "Claude Sonnet 5.5", "releaseDate": "2026-09-28"},
    {"slug": "step-5-preview-x", "name": "Step 5 Preview", "releaseDate": "2026-09-18"},
    {"slug": "nova-2-0-pro-preview", "name": "Nova 2.0 Pro Preview", "releaseDate": "2025-11-27"},
    {"slug": "gpt-6", "name": "GPT-6", "releaseDate": "2026-03-01"},
    {"slug": "gpt-6-sol-a", "name": "GPT-6 Sol", "releaseDate": "2026-05-01"},
    {"slug": "gpt-6-sol-b", "name": "GPT-6 Sol", "releaseDate": "2026-06-01"},
]


@pytest.mark.parametrize("slug,name,want,tier", [
    ("claude-sonnet-5-5", "Claude Sonnet 5.5 (Max, Default Fallback)", "2026-09-28", "slug"),
    ("step-5", "Step 5 Preview", "2026-09-18", "name"),
    ("claude-sonnet-5-5-medium", "Claude Sonnet 5.5 (Medium)", "2026-09-28", "effort_slug"),
    ("claude-sonnet-5-5-non-reasoning", "X", "2026-09-28", "effort_slug"),
    ("nova-2-0-pro-reasoning-low", "Nova 2.0 Pro Preview (Low)", "2025-11-27", "paren_name"),
])
def test_release_dates_join_through_the_four_tiers(slug, name, want, tier):
    models = [_record(slug=slug, name=name)]
    sc._join_release_dates('0:{"r":' + json.dumps(_RELEASES) + "}", models)
    assert models[0]["releaseDate"] == want
    assert models[0]["_releaseJoin"] == tier


def test_the_release_join_never_matches_by_prefix():
    """gpt-6-1-sol is a different, later model than gpt-6. A longest-prefix join
    would date it by its predecessor."""
    models = [_record(slug="gpt-6-1-sol-xhigh", name="GPT-6.1 Sol (Xhigh)")]
    sc._join_release_dates('0:{"r":' + json.dumps(_RELEASES) + "}", models)
    assert models[0]["releaseDate"] is None
    assert models[0]["_releaseJoin"] == ""


def test_a_name_two_releases_share_is_left_blank():
    models = [_record(slug="gpt-6-sol-max", name="GPT-6 Sol (Max)")]
    sc._join_release_dates('0:{"r":' + json.dumps(_RELEASES) + "}", models)
    assert models[0]["releaseDate"] is None


def test_a_record_that_carries_its_own_date_keeps_it():
    models = [_record(slug="claude-sonnet-5-5", releaseDate="2026-01-02")]
    sc._join_release_dates('0:{"r":' + json.dumps(_RELEASES) + "}", models)
    assert models[0]["releaseDate"] == "2026-01-02"
    assert models[0]["_releaseJoin"] == "record"


def test_a_missing_releases_array_blanks_dates_and_publishes_the_rest():
    row = _row_for(_record())
    assert pd.isna(row["release_date"])
    assert row["cost_per_task"] == pytest.approx(1.25)


def test_coverage_counts_the_late_fields_and_the_join():
    recs = [_record(0), _record(1, intelligenceIndexCostPerTask=None,
                                cacheWritePrice=None)]
    models = sc._extract_models(_page(
        '0:{"initialModels":' + json.dumps(recs) + ',"releases":'
        + json.dumps([{"slug": "lab-model-0", "name": "Lab Model 0",
                       "releaseDate": "2026-01-01"}]) + "}"))
    sc._parse_api_response(models)
    cov = sc._last_coverage
    assert cov["field_coverage"] == {
        "rows": 2, "cost_per_task": 1, "cache_hit_price": 2,
        "cache_write_price": 1, "is_open_weights": 2, "is_reasoning": 2,
        "release_date": 1, "aa_slug": 2}
    assert cov["release_date_join"]["slug"] == 1
    assert cov["release_date_join"]["unmatched"] == ["Lab Model 1"]


# ── Disappearance is a warning, never a refusal ──────────────────────────────

def test_late_fields_are_not_density_checked():
    for col in AA_LATE_FIELDS:
        assert col not in sc._COLUMN_HEALTH, (
            f"{col} is sparse or boolean upstream; a health floor would refuse "
            f"healthy scrapes")


def test_a_collapsed_late_field_is_reported():
    cached = load_from_raw(sc._parse_api_response([_record(i) for i in range(20)]))
    now = load_from_raw(sc._parse_api_response(
        [_record(i, intelligenceIndexCostPerTask=None, isOpenWeights=None)
         for i in range(20)]))
    warns = sc.late_field_collapse(now, cached)
    assert any("cost_per_task" in w for w in warns)
    assert any("is_open_weights" in w for w in warns)
    assert not any("cache_hit_price" in w for w in warns)
    assert sc.late_field_collapse(cached, cached) == []


def test_a_cache_from_before_the_fields_raises_no_warning():
    """The first scrape after deploy compares against a 13-column cache."""
    old = _with_price_sides(pd.DataFrame([{"model": "a", "provider": "p"}]))
    new = load_from_raw(sc._parse_api_response([_record()]))
    assert sc.late_field_collapse(new, old) == []


def test_cost_per_task_is_never_carried_forward():
    """It is a live measurement of today's prices and token counts; an old one
    filling a hole would be a stale price presented as current."""
    assert "cost_per_task" not in sc._CARRIED_COLUMNS
    cached = load_from_raw(sc._parse_api_response([_record()]))
    now = load_from_raw(sc._parse_api_response(
        [_record(intelligenceIndexCostPerTask=None)]))
    out = sc._carry_dropped_columns(now, cached, sidecar=None)
    assert pd.isna(out.iloc[0]["cost_per_task"])


# ── Backwards-compatible loading ─────────────────────────────────────────────

@pytest.mark.parametrize("width", [7, 9, 13])
def test_every_older_row_shape_still_loads(width):
    full = ["Old", "128k", "Lab", "40.0", "$2.0", "100", "1.0", "1.0", "4.0",
            "0", "", "", "-5.0"]
    df = load_from_raw([full[:width]])
    assert len(df) == 1
    for col in AA_LATE_FIELDS:
        assert col in df.columns
    assert df.iloc[0]["is_open_weights"] is pd.NA
    assert pd.isna(df.iloc[0]["cost_per_task"])


def test_a_13_column_csv_loads_with_unknown_late_fields(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text(",".join(OLD_HEADER) + "\n"
                    '"M",Lab,128k,40.0,2.0,100,1.0,1.0,4.0,False,,,-5.0\n')
    df = _with_price_sides(pd.read_csv(path))
    assert str(df["is_open_weights"].dtype) == "boolean"
    assert df["is_open_weights"].isna().all()
    assert not df["is_open_weights"].eq(False).any(), "unknown became closed"
    assert df["cost_per_task"].isna().all()


@pytest.fixture
def raw_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    monkeypatch.setattr(ingest, "CACHE_PATH", tmp_path / "aa_models.csv")
    monkeypatch.setattr(ingest, "HIST_DIR", tmp_path / "history")
    return tmp_path


def test_the_cache_round_trips_with_one_zero_blank_booleans(raw_dirs):
    df = load_from_raw(sc._parse_api_response([
        _record(0, isOpenWeights=True, cacheHitPrice=0.00005),
        {k: v for k, v in _record(1).items() if k != "isOpenWeights"}]))
    ingest.save_cache(df)
    text = (raw_dirs / "aa_models.csv").read_text()
    header = text.splitlines()[0].split(",")
    assert header == OLD_HEADER + list(AA_LATE_FIELDS)
    back = ingest.load_cached()
    by = back.set_index("model")
    assert by.loc["Lab Model 0", "is_open_weights"] == True  # noqa: E712
    assert by.loc["Lab Model 1", "is_open_weights"] is pd.NA
    assert by.loc["Lab Model 0", "cache_hit_price"] == pytest.approx(0.00005)
    raw = pd.read_csv(raw_dirs / "aa_models.csv", dtype=str, keep_default_na=False)
    assert set(raw["is_open_weights"]) == {"1", ""}
    assert set(raw["is_reasoning"]) == {"1"}


def test_todays_old_shape_snapshot_is_upgraded_without_touching_it(raw_dirs):
    """The first scrape on deploy day ran the old code. Its snapshot gains the
    new columns from the current scrape; every existing cell stays as written."""
    hist = raw_dirs / "history"
    hist.mkdir()
    snap = hist / f"aa_models_{date.today().isoformat()}.csv"
    old_text = (",".join(OLD_HEADER + ["scraped_at"]) + "\n"
                '"Lab Model 0",Lab,128k,40.0,8.25,80.0,1.0,4,20,False,,,-3.0,'
                f"{date.today().isoformat()}\n"
                '"Gone Model",Lab,128k,30.0,1.0,50.0,1.0,1,1,True,,,-9.0,'
                f"{date.today().isoformat()}\n")
    snap.write_text(old_text)
    now = load_from_raw(sc._parse_api_response([_record(0)]))
    ingest.save_cache(now)

    up = pd.read_csv(snap, dtype=str, keep_default_na=False)
    assert list(up.columns) == OLD_HEADER + list(AA_LATE_FIELDS) + ["scraped_at"]
    old = pd.read_csv(pd.io.common.StringIO(old_text), dtype=str,
                      keep_default_na=False)
    assert up[old.columns].equals(old), "an existing snapshot cell was rewritten"
    by = up.set_index("model")
    assert by.loc["Lab Model 0", "cost_per_task"] == "1.25"
    assert by.loc["Lab Model 0", "is_open_weights"] == "0"
    assert by.loc["Gone Model", "cost_per_task"] == ""
    assert by.loc["Gone Model", "is_open_weights"] == ""

    before = snap.read_text()
    ingest.save_cache(now)
    assert snap.read_text() == before, "the upgrade is not idempotent"


def test_history_mixes_old_and_new_snapshots(raw_dirs):
    hist = raw_dirs / "history"
    hist.mkdir()
    (hist / "aa_models_2026-01-01.csv").write_text(
        "model,provider,context,quality,price,speed,latency\n"
        "A,Lab,128k,40.0,2.0,100,1.0\n")
    new = load_from_raw(sc._parse_api_response([_record(0, isOpenWeights=True)]))
    ingest.save_cache(new)
    h = ingest.load_history()
    assert str(h["is_open_weights"].dtype) == "boolean"
    assert h["is_open_weights"].isna().sum() == 1
    assert h["is_open_weights"].eq(True).sum() == 1
    assert h["cost_per_task"].notna().sum() == 1


def test_the_committed_csv_header_is_old_or_complete():
    """The bot publishes the 20-column shape; until its first run after the
    deploy, the committed file may still be 13 columns. Nothing in between."""
    header = (ROOT / "data/raw/aa_models.csv").read_text().splitlines()[0].split(",")
    assert header[:13] == OLD_HEADER
    assert header[13:] in ([], list(AA_LATE_FIELDS))
