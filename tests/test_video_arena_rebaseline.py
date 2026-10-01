"""The text-to-video arena was re-anchored upstream, and the guard read it as a collapse.

On 2026-09-30 Artificial Analysis replaced its textToVideo leaderboard with a
new rating pool: 86 models became 30 (all `isDefault`), Kling 3.0 Pro sits at
exactly 1000.00, and every one of the 21 models present in both pools moved
by -124 to -315 Elo (median -239) while their order largely held (Spearman
0.82). imageToVideo did not change at all — same 83 models, same Elos.

The scraper gated `elo_t2v` and `price_per_min_t2v` on their share of the
*union* of both arenas. With 30 of 93 rows that share is 32%/31%, below the
60%/50% floors, so every hourly run refused to publish and the Video tab
served the previous day's cache under a failing badge — the guard measuring
the relative size of two leaderboards, not the health of either.

These tests pin the replacement: coverage is checked per arena, against the
records that arena actually published; an arena that shrinks past the shrink
limit is accepted only when its overlap carries the signature of a re-scaled
pool, never when it looks truncated; and an old-scale score is never carried
into the new scale.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from data import scrape_status
from data import video_scraper as v


# ── Synthetic arenas shaped like the 2026-09-30 change ───────────────────────

def _rec(slug: str, elo: float, price: float | None = 6.0, **extra) -> dict:
    rec = {"slug": slug, "name": slug.replace("-", " ").title(), "elo": elo,
           "creator": {"name": "Lab"}, "isDefault": True, "price": price}
    rec.update(extra)
    return rec


def _old_cache() -> pd.DataFrame:
    """86 text-to-video and 83 image-to-video rows on the old scale, 107 distinct."""
    rows = []
    for i in range(107):
        rows.append({
            "model": f"M{i}", "slug": f"m{i}", "provider": "Lab", "family": "",
            "release_date": "2026-06-01", "is_current": True,
            "open_weights": False,
            # t2v on m0..m85, i2v on m24..m106
            "elo_t2v": 1330.0 - 5 * i if i < 86 else None,
            "price_per_min_t2v": 6.0 if i < 86 else None,
            "elo_i2v": 1370.0 - 5 * (i - 24) if i >= 24 else None,
            "price_per_min_i2v": 7.0 if i >= 24 else None,
            "audio": False, "price_per_min_audio": None,
            "gen_time_s": None, "gen_time_host": None,
        })
    return pd.DataFrame(rows)


def _i2v_unchanged() -> list[dict]:
    return [_rec(f"m{i}", 1370.0 - 5 * (i - 24), 7.0) for i in range(24, 107)]


def _t2v_rebaselined() -> list[dict]:
    """21 survivors re-scored ~240 lower in the same order, plus 9 new models."""
    survivors = [_rec(f"m{i}", 1330.0 - 5 * i - 240.0) for i in range(0, 42, 2)]
    new = [_rec(f"new{i}", 1100.0 - 20 * i) for i in range(9)]
    return survivors + new


def _t2v_truncated() -> list[dict]:
    """The top 30 of the OLD pool, scores untouched — a cut-off page."""
    return [_rec(f"m{i}", 1330.0 - 5 * i) for i in range(30)]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the cache and the status file at tmp so nothing committed moves."""
    cache = tmp_path / "aa_video_models.csv"
    _old_cache().to_csv(cache, index=False)
    monkeypatch.setattr(v, "_CACHE", cache)
    monkeypatch.setattr(scrape_status, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(v, "_fetch", lambda url: "")
    monkeypatch.setattr(v, "_generation_times", lambda html: {})

    def run(t2v, i2v=None):
        arenas = {"textToVideo": t2v,
                  "imageToVideo": _i2v_unchanged() if i2v is None else i2v,
                  "textToVideoAudio": [], "imageToVideoAudio": []}
        monkeypatch.setattr(v, "_extract_arenas", lambda html: arenas)
        before = cache.read_bytes()
        ok = v.scrape_and_save()
        return ok, before != cache.read_bytes(), pd.read_csv(cache)

    run.status = lambda: json.loads((tmp_path / "status.json").read_text())
    return run


# ── The change that actually happened must publish ───────────────────────────

def test_a_rebaselined_text_to_video_arena_publishes(sandbox):
    ok, wrote, df = sandbox(_t2v_rebaselined())
    assert ok and wrote, "a genuinely re-scaled arena was refused"
    assert int(df["elo_t2v"].notna().sum()) == 30
    assert int(df["elo_i2v"].notna().sum()) == 83
    status = sandbox.status()["video"]
    assert status["ok"] is True


def test_old_scale_scores_are_never_carried_into_the_new_pool(sandbox):
    """m25 was ranked in the old t2v pool and is absent from the new one. Its old
    1205 is on a scale the new pool does not use — publishing it beside
    re-anchored scores would rank a dropped model above the new leader."""
    _, _, df = sandbox(_t2v_rebaselined())
    by_slug = df.set_index("slug")
    assert pd.isna(by_slug.loc["m25", "elo_t2v"]), "an old-scale t2v Elo survived"
    assert pd.isna(by_slug.loc["m25", "price_per_min_t2v"])
    # …and it is still listed, because it is ranked in the image arena.
    assert by_slug.loc["m30", "elo_i2v"] == pytest.approx(1370.0 - 5 * 6)
    # A model dropped from BOTH arenas leaves the catalogue.
    assert "m23" not in by_slug.index
    assert by_slug["elo_t2v"].max() <= 1100.0, "the new pool's ceiling was exceeded"


def test_the_rebaseline_is_recorded_as_provenance(sandbox):
    sandbox(_t2v_rebaselined())
    status = sandbox.status()["video"]
    assert status["arenas"] == {"t2v": 30, "i2v": 83}
    notes = " ".join(status.get("notes") or [])
    assert "t2v" in notes and "re-scaled" in notes, status


# ── The guard must still bite ────────────────────────────────────────────────

def test_a_truncated_arena_is_still_refused(sandbox):
    """Same 65% drop, but the surviving scores did not move: that is a page cut
    off at 30 rows, not a new pool, and must not publish."""
    ok, wrote, _ = sandbox(_t2v_truncated())
    assert not ok and not wrote
    assert sandbox.status()["video"]["ok"] is False


def test_an_arena_below_its_floor_is_refused_even_when_rescaled(sandbox):
    ok, wrote, _ = sandbox(_t2v_rebaselined()[:12])
    assert not ok and not wrote


def test_an_arena_whose_elo_key_was_renamed_is_refused(sandbox):
    """Half the records carry the score under a new key. Before, they were
    dropped silently and the survivors judged against the union."""
    t2v = _t2v_rebaselined()
    for rec in t2v[::2]:
        rec["score"] = rec.pop("elo")
    ok, wrote, _ = sandbox(t2v)
    assert not ok and not wrote


def test_an_arena_whose_price_key_was_renamed_is_refused(sandbox):
    t2v = [dict(r, price=None, cost=r["price"]) for r in _t2v_rebaselined()]
    ok, wrote, _ = sandbox(t2v)
    assert not ok and not wrote


def test_an_unchanged_image_arena_collapsing_is_refused(sandbox):
    """The image arena did NOT re-scale. Losing most of it is a fault."""
    ok, wrote, _ = sandbox(_t2v_rebaselined(), i2v=_i2v_unchanged()[:30])
    assert not ok and not wrote


# ── Coverage is per arena, not per union ─────────────────────────────────────

def test_column_coverage_is_judged_per_arena_not_against_the_union():
    """30 fully-priced t2v rows beside 83 i2v rows is a healthy catalogue."""
    rows = [{"model": f"T{i}", "slug": f"t{i}", "provider": "Lab",
             "elo_t2v": 1000.0 + i, "price_per_min_t2v": 6.0,
             "elo_i2v": None, "price_per_min_i2v": None} for i in range(30)]
    rows += [{"model": f"I{i}", "slug": f"i{i}", "provider": "Lab",
              "elo_t2v": None, "price_per_min_t2v": None,
              "elo_i2v": 1200.0 + i, "price_per_min_i2v": 7.0} for i in range(63)]
    df = pd.DataFrame(rows)
    assert v._column_violations(df) == []

    unpriced = df.copy()
    unpriced.loc[unpriced["elo_t2v"].notna(), "price_per_min_t2v"] = None
    assert v._column_violations(unpriced), "a t2v price collapse went unnoticed"

    gone = df.copy()
    gone["elo_t2v"] = None
    assert v._column_violations(gone), "an emptied arena went unnoticed"


def test_a_refusal_says_why_and_recovery_clears_it(sandbox):
    """The badge could say a dataset was failing but never why; the reason now
    rides in scrape_status and must not outlive the failure."""
    sandbox(_t2v_truncated())
    failed = sandbox.status()["video"]
    assert failed["ok"] is False
    assert "textToVideo" in failed["error"] and "truncated" in failed["error"]

    sandbox(_t2v_rebaselined())
    healed = sandbox.status()["video"]
    assert healed["ok"] is True and "error" not in healed


def test_the_badge_tooltip_carries_the_failure_reason():
    from pathlib import Path

    js = (Path(__file__).resolve().parent.parent / "docs" / "app.js").read_text()
    body = js.split("function renderFreshness")[1].split("\nfunction ")[0]
    assert "e.error" in body, "a failing dataset's reason never reaches the reader"


# ── Record completeness: a renamed key, not the odd unrated entry ────────────

def test_a_couple_of_unrated_entries_in_a_small_arena_are_not_a_schema_change():
    """30 rated + 2 unrated is 94% — a pure share floor refused it, and with a
    30-model pool every hourly run would have gone amber over two new entries."""
    recs = [_rec(f"s{i}", 1000.0 - i) for i in range(30)]
    recs += [_rec("new-a", None), _rec("new-b", None)]
    assert v._record_violations({"textToVideo": recs}) == []


def test_a_renamed_elo_key_is_caught_by_the_completeness_guard_itself():
    """Half of 60 records renamed still leaves 30 scored rows, above the
    20-row floor, so only this guard stands between it and a publish."""
    recs = [_rec(f"s{i}", 1000.0 - i) for i in range(60)]
    for rec in recs[::2]:
        rec["score"] = rec.pop("elo")
    out = v._record_violations({"textToVideo": recs})
    assert out and "30 of 60" in out[0]
