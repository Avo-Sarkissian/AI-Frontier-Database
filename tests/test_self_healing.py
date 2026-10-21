"""The three things that used to need a person every time AA changed something.

1. A renamed array. ~2026-10-14 AA renamed /leaderboards/models' metrics array
   "models" -> "initialModels" with the records unchanged, and the hosted and
   open-weight catalogues stopped for six days.
2. A real upstream prune. The text-to-image arena went 169 -> 105 the same week
   and the 20% shrink guard refused it hourly until someone overrode it.
3. A new company. Every one AA added rendered grey and failed a test until a
   hex was picked by hand.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from components.charts import auto_palette as ap
from data import carried
from data.rsc import find_records


def _page(payload: str) -> str:
    """A minimal Next.js page whose flight payload is `payload`."""
    return f"<script>self.__next_f.push([1,{json.dumps(payload)}])</script>"


def _rec(i: int, **extra) -> dict:
    return {"slug": f"m-{i}", "name": f"Model {i}", "intelligenceIndex": 40.0 + i,
            "isOpenWeights": i % 2 == 0, **extra}


# ── 1. Records are found by their fields, not their key ───────────────────────

@pytest.mark.parametrize("key", ["models", "initialModels", "leaderboardRows"])
def test_records_are_found_under_any_key(key):
    payload = '0:{"' + key + '":' + json.dumps([_rec(i) for i in range(5)]) + "}"
    found = find_records(payload, ("slug", "intelligenceIndex"))
    assert found is not None and len(found) == 5


def test_the_catalogue_beats_a_short_quote_of_it():
    """A page can quote a few records elsewhere (a headline strip); the
    catalogue is the long array."""
    short = json.dumps([_rec(i) for i in range(3)])
    long_ = json.dumps([_rec(i) for i in range(40)])
    payload = '0:{"headline":' + short + ',"initialModels":' + long_ + "}"
    assert len(find_records(payload, ("slug", "intelligenceIndex"))) == 40


def test_a_renamed_field_still_fails_loudly():
    """Finding by fields must not turn a field rename into a silent empty
    column — no array carrying the field means no records at all."""
    recs = [{"slug": "a", "intelligenceScore": 50}]
    payload = '0:{"initialModels":' + json.dumps(recs) + "}"
    assert find_records(payload, ("slug", "intelligenceIndex")) is None


def test_hosted_scraper_reads_the_2026_10_page_shape():
    from data.scraper import _extract_models
    recs = [_rec(i) for i in range(4)]
    html = _page('0:{"initialModels":' + json.dumps(recs) + "}")
    models = _extract_models(html)
    assert [m["name"] for m in models] == [f"Model {i}" for i in range(4)]


def test_local_scraper_reads_the_2026_10_page_shape():
    from data.local_scraper import _leaderboard
    payload = '0:{"initialModels":' + json.dumps([_rec(i) for i in range(4)]) + "}"
    assert len(_leaderboard(payload)) == 4


# ── 2. A shrink that holds publishes itself ───────────────────────────────────

T0 = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
BEFORE = {f"m{i}" for i in range(169)}
AFTER = {f"m{i}" for i in range(105)}
REFUSED = ["row count 169 -> 105, a 38% drop (limit 20%)"]


def _run(path, hours, violations=REFUSED, now_keys=AFTER, before=BEFORE):
    return carried.confirm_shrink("image", list(violations), before, now_keys,
                                  now=T0 + timedelta(hours=hours), path=path)


def test_a_shrink_is_refused_until_it_has_held(tmp_path):
    path = tmp_path / "shrink.json"
    for h in (0, 4, 8):
        assert _run(path, h), f"published after only {h}h"
    assert _run(path, 12) == [], "a shrink that held 4 scrapes / 12h should publish"
    assert carried.accepted_shrink("image", 48, now=T0 + timedelta(hours=13),
                                   path=path) == 105


def test_hours_alone_or_scrapes_alone_are_not_enough(tmp_path):
    path = tmp_path / "shrink.json"
    assert _run(path, 0) and _run(path, 20)          # 2 scrapes over 20h
    assert _run(path, 20.1)                          # 3rd scrape
    assert _run(path, 20.2) == []                    # 4th, and > 12h since first
    path2 = tmp_path / "shrink2.json"
    for h in (0, 0.25, 0.5, 0.75, 1.0, 1.25):        # many scrapes, one hour
        assert _run(path2, h)


def test_one_full_size_scrape_resets_the_streak(tmp_path):
    """The 2026-09-22 deprecation flaps lasted 40-90 minutes each."""
    path = tmp_path / "shrink.json"
    _run(path, 0), _run(path, 4), _run(path, 8)
    assert _run(path, 9, violations=[]) == []
    assert _run(path, 12), "the streak should have restarted at 9h"
    assert "streak" in carried.load_shrink_state(path)["image"]


def test_a_shrink_that_shares_few_rows_never_publishes_itself(tmp_path):
    """A wrong array or a renamed catalogue is a shrink with strangers in it."""
    path = tmp_path / "shrink.json"
    strangers = {f"x{i}" for i in range(105)}
    for h in range(0, 48, 4):
        assert _run(path, h, now_keys=strangers)


def test_data_guard_measures_against_an_accepted_shrink(tmp_path, monkeypatch):
    import data_guard
    monkeypatch.setattr(carried, "CARRIED_DIR", tmp_path)
    assert data_guard._accepted_floor("data/raw/aa_image_models.csv", 169) == 169
    for h in (0, 4, 8, 12):
        carried.confirm_shrink("image", list(REFUSED), BEFORE, AFTER,
                               now=datetime.now(timezone.utc) - timedelta(hours=12 - h))
    assert data_guard._accepted_floor("data/raw/aa_image_models.csv", 169) == 105
    assert data_guard._accepted_floor("data/raw/aa_models.csv", 167) == 167


# ── 3. New companies get a colour without anyone picking one ──────────────────

def test_an_automatic_colour_is_legible_and_keeps_aas_hue():
    for aa in ("#195428", "#1521a9", "#e1002a", "#00b8ea", "#795bcf"):
        c = ap.resolve("Lab", aa, taken=[])
        assert ap.contrast(c, ap.SURFACE) >= ap.MIN_CONTRAST
        assert ap.delta_e(c, ap.OTHER_GREY) >= ap.MIN_DELTA_E
        dh = abs(ap._oklch(c)[2] - ap._oklch(aa)[2]) % 360
        assert min(dh, 360 - dh) <= 1.0, f"{aa} -> {c} moved hue with nothing to avoid"


def test_an_automatic_colour_steps_aside_but_stays_near_aas_hue():
    aa = "#3c8bff"
    c = ap.resolve("Lab", aa, taken=[aa])
    assert c != aa and ap.delta_e(c, aa) >= ap.MIN_DELTA_E
    dh = abs(ap._oklch(c)[2] - ap._oklch(aa)[2]) % 360
    assert min(dh, 360 - dh) <= ap._MAX_HUE_SHIFT + 1


def test_a_company_aa_paints_black_still_gets_a_stable_hue():
    a = ap.resolve("Liquid AI", "#000000", taken=[])
    assert a == ap.resolve("Liquid AI", "#000000", taken=[])
    assert ap._oklch(a)[1] >= ap._ACHROMATIC
    assert ap.contrast(a, ap.SURFACE) >= ap.MIN_CONTRAST


def test_assign_never_overrides_a_hand_picked_colour_and_never_drifts():
    curated = {"Anthropic": "#cc785c"}
    out = ap.assign({"Anthropic": "#000000", "NewLab": "#195428"}, curated, {})
    assert "Anthropic" not in out and "NewLab" in out
    again = ap.assign({"NewLab": "#195428", "Another": "#195429"}, curated, out)
    assert again["NewLab"] == out["NewLab"], "a newcomer moved an existing colour"
    moved = ap.assign({"NewLab": "#e1002a"}, curated, out)
    assert moved["NewLab"][0] == "#e1002a", "AA's new colour was not picked up"


def test_learning_round_trips_through_the_stored_file(tmp_path):
    path = tmp_path / ap.FILE_NAME
    assert carried.learn_creator_colors({"Brand New Lab": "#195428"}, "llm", path=path)
    assert not carried.learn_creator_colors({"Brand New Lab": "#195428"}, "llm", path=path), \
        "an unchanged set must not rewrite the file, or the bot commits hourly"
    assert "Brand New Lab" in ap.stored_colors(path)


def test_every_stored_colour_is_legible_and_distinct_from_hand_picked():
    from components.charts.constants import CURATED_PROVIDER_COLORS
    from data.image_models import _IMAGE_ONLY_COLORS
    from data.local_models import _LOCAL_ONLY_COLORS
    from data.video_models import _VIDEO_ONLY_COLORS
    curated = {**_IMAGE_ONLY_COLORS, **_VIDEO_ONLY_COLORS, **_LOCAL_ONLY_COLORS,
               **CURATED_PROVIDER_COLORS}
    for name, colour in ap.stored_colors().items():
        assert name not in curated, f"{name} is hand-picked and stored automatically"
        assert ap.contrast(colour, ap.SURFACE) >= ap.MIN_CONTRAST, name
        assert colour not in curated.values(), f"{name} duplicates a hand-picked hex"
