"""The numbers have to mean what the labels say, and a schema change has to be loud.

Theme 1 of audit/2026-08-12. Every upstream field was read as `.get(x) or <default>`,
so a rename degraded into a constant rather than raising, and the only thing standing
between that and the public site was a row count.

Three distinct failures, one root cause each:

1. `price` was computed by a fallback that had become the only path, while three
   surfaces credited Artificial Analysis for it. AA publishes the opposite weighting.
2. `context` came from a host record, so it was whichever host AA listed first — and
   it contradicted the local catalogue, which reads the model's own figure.
3. Renaming any key outside the three that drop a row published an all-zero column,
   and the site still looked ~90% healthy.
"""
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import data_guard
from data.ingest import get_models
from data.scraper import (
    _parse_api_response, load_from_raw, column_health_violations,
    _shrink_violations, MAX_SHRINK_PCT,
)

ROOT = Path(__file__).resolve().parent.parent
DF = get_models()


# ── 1.1 — the blend is ours, and nothing may say otherwise ───────────────────

def test_price_is_the_output_weighted_blend():
    """Pinned deliberately. Output-weighted is the honest basis for agentic and
    RAG workloads; switching to AA's input-weighted blend would re-rank Value
    and break continuity with 90 blended-only history snapshots."""
    rows = DF[DF["price_in"].notna() & DF["price_out"].notna()]
    assert len(rows) > 50, "not enough priced rows to test"
    expected = (3 * rows["price_out"] + rows["price_in"]) / 4
    assert (rows["price"] - expected).abs().max() < 1e-6


def test_price_is_not_attributed_to_artificial_analysis():
    """The basis is a product decision; the attribution was a false statement.
    AA publishes `price_1m_blended_0_3_1` = (3*in + out)/4 — a median 1.91x
    lower — and no output-weighted blend at all."""
    surfaces = {
        "docs/app.js": 'title="Artificial Analysis blended price',
        "captions.py": "Price uses Artificial Analysis's blended rate",
        "README.md": "| Price | Artificial Analysis |",
    }
    for rel, claim in surfaces.items():
        text = (ROOT / rel).read_text()
        assert claim not in text, (
            f"{rel} still credits Artificial Analysis for a blend they do not publish"
        )


def test_the_price_label_says_whose_blend_it_is():
    """Removing the false claim is not enough — a bare '3:1' still reads as
    upstream. Each surface must say the weighting is ours."""
    # captions.py is now the single source for the Budget tab's prose, and it
    # rides the manifest to the browser — so checking it covers both renderings.
    for rel in ("docs/app.js", "captions.py", "README.md"):
        text = (ROOT / rel).read_text().lower()
        assert "our blend" in text or "our own blend" in text or "not aa's" in text, (
            f"{rel} does not disclose that the 3:1 blend is ours"
        )


def test_the_scraper_does_not_read_a_key_that_does_not_exist():
    """`price_1m_blended_3_to_1` is absent from all 443 live records, so reading
    it first made the arithmetic look like a fallback when it was the only path.
    Dead code that misrepresents provenance is worse than no code — the name may
    survive in a comment explaining the history, but never in a lookup."""
    src = (ROOT / "data" / "scraper.py").read_text()
    code = re.sub(r"^\s*#.*$", "", src, flags=re.M)
    assert "price_1m_blended_3_to_1" not in code, (
        "scraper still reads the non-existent blended key"
    )


# ── 1.2 — context is the model's, not an arbitrary host's ────────────────────

def test_context_matches_the_local_catalogue_where_they_overlap():
    """The two catalogues describe the same models. They disagreed on 40 of 91
    because this one took a host's cap and local_scraper took the model's."""
    local_csv = ROOT / "data" / "raw" / "aa_local_models.csv"
    if not local_csv.exists():
        pytest.skip("no local catalogue in this checkout")
    local = pd.read_csv(local_csv)

    def _to_k(v):
        s = str(v).strip().lower().replace(",", "")
        try:
            if s.endswith("m"):
                return float(s[:-1]) * 1000
            if s.endswith("k"):
                return float(s[:-1])
            return float(s) / 1000 if float(s) > 10000 else float(s)
        except ValueError:
            return None

    hosted = {r["model"]: _to_k(r["context"]) for _, r in DF.iterrows()}
    disagree = []
    for _, r in local.iterrows():
        name = r["name"]
        if name in hosted and hosted[name] and pd.notna(r.get("context_k")):
            a, b = hosted[name], float(r["context_k"])
            if a and b and abs(a - b) / max(a, b) > 0.05:
                disagree.append((name, a, b))
    assert len(disagree) <= 2, (
        f"{len(disagree)} models have contradictory context windows between the "
        f"hosted and local catalogues: {disagree[:5]}"
    )


def test_the_context_window_published_is_the_models_own():
    """Nemotron 3.5 Lightning once published 29k against a real 1,000,000.

    The old endpoint returned host x model rows carrying BOTH the model's window
    and the host's smaller cap, and this module preferred the host's — 34.5x
    understated, and disagreeing with the open-weight catalogue on 40 of 91
    shared models. The leaderboard publishes only the model's own window, so
    that class of mismatch cannot recur; what remains testable is that a large
    window survives the formatting intact rather than being truncated or
    rendered in the wrong unit.
    """
    from static_helpers import ctx_to_k

    rows = _parse_api_response([{
        "name": "Ctx Test", "slug": "ctx-test",
        "modelCreatorName": "TestLab",
        "intelligenceIndex": 50.0,
        "contextWindowTokens": 1_000_000,
        "deprecated": False,
        "price1mInputTokens": 1.0,
        "price1mOutputTokens": 2.0,
        "medianOutputTokensPerSecond": 100,
        "medianTimeToFirstTokenSeconds": 1.0,
    }])
    assert rows, "the synthetic record did not parse"
    got = str(load_from_raw(rows).iloc[0]["context"])
    assert ctx_to_k(got) == 1000, f"1,000,000 tokens parsed back as {got!r}"
    assert got == "1m", f"1,000,000 tokens should render as '1m', not {got!r}"


# ── 1.3 — a rename must be loud ──────────────────────────────────────────────

def _synthetic_payload(n: int = 60) -> list[dict]:
    """A leaderboard-shaped records array.

    Rewritten on 2026-08-20 with the source. This used to build the old
    /api/data/website/host-models/performance shape — nested hostModels with
    snake_case keys — which AA retired that day with a 404. A rename test
    against a schema that no longer exists proves nothing, so the mutations
    below now rename the fields the scrape actually reads.
    """
    return [{
        "name": f"Model {i}", "slug": f"m{i}",
        "modelCreatorName": f"Lab{i % 5}",
        "intelligenceIndex": 40.0 + i % 20,
        "contextWindowTokens": 128_000,
        "deprecated": False,
        "price1mInputTokens": 1.0 + i % 3,
        "price1mOutputTokens": 4.0 + i % 5,
        "medianOutputTokensPerSecond": 50 + i,
        "medianTimeToFirstTokenSeconds": 0.5 + (i % 4),
    } for i in range(n)]


def _mutate(payload: list[dict], mutation: str) -> list[dict]:
    p = json.loads(json.dumps(payload))
    for rec in p:
        if mutation == "medianOutputTokensPerSecond":
            rec["outputTokensPerSecondMedian"] = rec.pop("medianOutputTokensPerSecond")
        elif mutation == "medianTimeToFirstTokenSeconds":
            rec["timeToFirstTokenSecondsMedian"] = rec.pop("medianTimeToFirstTokenSeconds")
        elif mutation == "modelCreatorName":
            rec["creatorName"] = rec.pop("modelCreatorName")
        elif mutation == "contextWindowTokens":
            rec.pop("contextWindowTokens", None)
        elif mutation == "undefined_speed":
            # Not a rename: RSC encodes JS undefined as the STRING "$undefined",
            # so a field that stops being published arrives truthy. Reading it
            # without _real() would publish "$undefined" as a speed.
            rec["medianOutputTokensPerSecond"] = "$undefined"
            rec["medianTimeToFirstTokenSeconds"] = "$undefined"
    return p


@pytest.mark.parametrize("mutation", [
    "medianOutputTokensPerSecond",    # -> speed all zero
    "medianTimeToFirstTokenSeconds",  # -> latency all zero
    "modelCreatorName",               # -> every provider blank
    "contextWindowTokens",            # -> every context '0'
    "undefined_speed",                # -> speed and latency all zero
])
def test_an_upstream_rename_is_caught_before_publishing(mutation):
    """Each of these renames used to produce 60 rows and a silent all-zero
    column. The audit's own table; now a test."""
    rows = _parse_api_response(_mutate(_synthetic_payload(), mutation))
    if not rows:
        return  # dropping every row is already loud — exit 1 downstream
    df = load_from_raw(rows)
    assert column_health_violations(df), (
        f"renaming {mutation} published {len(df)} rows with no complaint"
    )


def test_the_two_arrays_under_models_are_told_apart():
    """/leaderboards/models carries the lightweight picker index AND the metrics
    records under the same "models" key. Taking the first gives a full row count
    with every metric missing — the silent-degradation shape this file exists
    for."""
    from data.rsc import find_array
    import json as _json

    picker = [{"name": "Model 0", "slug": "m0", "creator": {}, "deprecated": False}]
    metrics = _synthetic_payload(3)
    payload = f'{{"models":{_json.dumps(picker)},"x":1,"models":{_json.dumps(metrics)}}}'
    got = find_array(payload, "models",
                     where=lambda a: bool(a) and isinstance(a[0], dict)
                     and "intelligenceIndex" in a[0])
    assert got is not None and len(got) == 3, "picked the picker index, not the metrics"


def test_healthy_data_raises_no_column_complaints():
    """The guard must not cry wolf on the real catalogue — speed and latency
    are genuinely absent for a handful of models."""
    assert column_health_violations(DF) == []


# ── 1.4 — the guard must see values and slopes, not just this hour's count ───

def test_the_guard_notices_a_units_change_that_leaves_row_counts_intact():
    """Across the 2026-05-06 price flip every price doubled while rows moved
    +11% — in the safe direction — and the guard said OK."""
    before = (ROOT / "data" / "raw" / "aa_models.csv").read_text()
    df = pd.read_csv(io.StringIO(before))
    df["price"] = df["price"] * 2
    doubled = df.to_csv(index=False)

    was = data_guard._medians(before, ["price"])
    now = data_guard._medians(doubled, ["price"])
    shift = abs(now["price"] - was["price"]) / was["price"] * 100
    assert shift > data_guard._NUMERIC_MEDIAN_MAX_SHIFT_PCT, (
        "a doubling of every price does not exceed the median-shift limit"
    )


def test_a_slow_drain_fails_before_it_becomes_a_collapse():
    """148 -> 119 -> 96 -> 77 -> 62 -> 50 passed every per-run check while
    losing 66% of the catalogue. The cumulative budget must fail by step 2."""
    steps = [148, 119, 96, 77, 62, 50]
    first_failure = None
    for i, now in enumerate(steps[1:], start=1):
        drop = (steps[0] - now) / steps[0] * 100
        if drop > data_guard.CUMULATIVE_MAX_DROP_PCT:
            first_failure = i
            break
    assert first_failure is not None and first_failure <= 2, (
        f"cumulative guard first fires at step {first_failure}; the audit "
        f"requires it by step 2"
    )


def test_the_scrape_path_refuses_a_shrinking_catalogue():
    """data_guard runs only in CI against committed files. app.py starts the
    scraper on every Dash boot and ingest freezes a history snapshot, so that
    path wrote whatever it got."""
    small = DF.head(max(1, int(len(DF) * (1 - MAX_SHRINK_PCT / 100)) - 5))
    assert _shrink_violations(small), (
        f"a {MAX_SHRINK_PCT:.0f}%+ shrink was accepted on the scrape path"
    )
    assert _shrink_violations(DF) == [], "the current catalogue trips its own guard"


def test_data_guard_still_passes_on_the_real_tree():
    """Whatever we added must not make the live pipeline red."""
    proc = subprocess.run([sys.executable, "data_guard.py"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"data_guard failed on real data:\n{proc.stdout}\n{proc.stderr}"


# ── 1.6 — the image arena's flat schema (2026-09-07) ─────────────────────────
#
# AA rebuilt /text-to-image between 00:00Z and 04:52Z on 2026-09-07. Each
# textToImage record lost its nested `elos[]` array — the `tag: null` entry the
# parser read the global ELO from, plus ~34 tagged per-category entries — and
# gained flat `elo` / `lower95ci` / `upper95ci` fields. `pricePer1kImages`
# became `price` (same unit); `openWeightsUrl` disappeared with no replacement.
#
# The parser looked for `elos[]`, found none, skipped every model, and returned
# no rows. That is the loud failure the workflow is built to surface, and it
# worked: 7 consecutive runs went red instead of republishing frozen numbers.
# These tests pin the shape so the next rebuild is caught by a test rather than
# by a day of red cron mail.

def _flat_image_record(name="GPT Image 2 (high)", elo=1178.11, price=211):
    """One record in the schema AA serves today."""
    return {"id": "9570e1d0", "slug": "gpt-image-2", "name": name,
            "url": "/image/model-families/openai-gpt", "elo": elo,
            "lower95ci": elo - 10, "upper95ci": elo + 10,
            "creator": {"name": "OpenAI", "logo": "/img/logos/openai_small.svg"},
            "isDefault": True, "price": price}


def _nested_image_record(name="FLUX.1 [dev]", elo=842.77, price=24):
    """One record in the pre-2026-09-07 schema, kept for an upstream revert."""
    return {"name": name, "creator": {"name": "Black Forest Labs"},
            "pricePer1kImages": price, "openWeightsUrl": "https://hf.co/x",
            "elos": [{"tag": None, "elo": elo},
                     {"tag": {"displayName": "Anime"}, "elo": elo + 5}]}


def test_the_image_parser_reads_the_flat_arena_schema():
    """The regression itself: today's payload must produce rows."""
    from data.image_scraper import _parse

    df = _parse([_flat_image_record()])
    assert df is not None and len(df) == 1, "today's AA schema still parses to nothing"
    row = df.iloc[0]
    assert row["model"] == "GPT Image 2 (high)"
    assert row["provider"] == "OpenAI"
    assert row["elo"] == 1178.11
    assert row["price_per_1k"] == 211, "price/pricePer1kImages share a unit"


def test_the_image_parser_still_reads_the_nested_arena_schema():
    """AA has reverted a redesign before. The old shape must not bit-rot."""
    from data.image_scraper import _parse

    df = _parse([_nested_image_record()])
    assert df is not None and len(df) == 1
    row = df.iloc[0]
    assert row["elo"] == 842.77
    assert row["price_per_1k"] == 24
    assert bool(row["open_weights"]) is True
    assert row["elo_anime"] == 847.77, "tagged per-category ELOs were dropped"


def test_image_open_weights_is_not_asserted_false_when_upstream_stops_saying():
    """51 of 158 image models are open-weights. The flat schema carries no
    openWeightsUrl, and defaulting the column to False would publish a false
    commercial fact for all 51 and silently empty the Open Weights filter —
    the same error _price_label() exists to prevent for price."""
    from data.image_scraper import _parse

    df = _parse([_flat_image_record()])
    assert df["open_weights"].isna().all(), (
        "an absent openWeightsUrl was recorded as 'not open weights'"
    )


def test_image_columns_upstream_dropped_are_carried_forward_not_erased():
    """open_weights and the per-category ELOs are still only published behind
    AA's key-gated arena API. Writing the flat payload straight out would empty
    34 populated columns while the row count stayed at 158 — a full-length CSV
    with dead columns, which is the exact shape this file exists to catch."""
    from data.image_scraper import _merge_cached_columns

    live = pd.DataFrame([{"model": "GPT Image 2 (high)", "provider": "OpenAI",
                          "elo": 1178.11, "price_per_1k": 211.0,
                          "open_weights": None},
                         {"model": "Brand New Model", "provider": "OpenAI",
                          "elo": 900.0, "price_per_1k": 5.0,
                          "open_weights": None}])
    cached = pd.DataFrame([{"model": "GPT Image 2 (high)", "provider": "OpenAI",
                            "elo": 1170.0, "price_per_1k": 200.0,
                            "open_weights": True, "elo_anime": 1201.38}])

    out = _merge_cached_columns(live, cached)
    assert len(out) == 2, "the merge changed the row count"
    known = out[out["model"] == "GPT Image 2 (high)"].iloc[0]
    fresh = out[out["model"] == "Brand New Model"].iloc[0]

    assert known["elo"] == 1178.11, "a carried column overwrote a live one"
    assert known["price_per_1k"] == 211.0, "a carried column overwrote a live one"
    assert bool(known["open_weights"]) is True, "open_weights was not carried forward"
    assert known["elo_anime"] == 1201.38, "the category ELO was not carried forward"
    assert pd.isna(fresh["elo_anime"]), "a model AA never scored got invented data"


def test_the_live_image_columns_are_still_health_checked():
    """Carrying columns forward must not make the guard blind to the columns
    that ARE live: a provider rename still has to stop the publish."""
    from data.image_scraper import _column_violations

    broken = pd.DataFrame([{"model": f"m{i}", "provider": "", "elo": 1000.0}
                           for i in range(20)])
    assert _column_violations(broken), "an all-blank provider column published"


# ── 1.7 — AA's intelligence overhaul (2026-09) ───────────────────────────────
#
# AA rebuilt the Intelligence Index onto a new eval suite (GDPval-AA v2,
# tau3-Banking, Terminal-Bench v2.1, SciCode, AA-Omniscience, HLE, GPQA Diamond,
# CritPt, IFBench, MMMU-Pro, ITBench-SRE) and shipped three changes that matter
# here. Measured against the live payload on 2026-09-08:
#
#   * The SCALE did not move. Max intelligenceIndex is 53.37 upstream and 53.37
#     in the committed cache, and all 198 published rows match live to the
#     hundredth — so QUALITY_INDEX_MAX, the MIN SCORE ladder and the tier
#     quality floors stay calibrated. Pinned below so a future rescale is loud.
#   * `intelligenceIndexIsEstimated` is new and true for 503 of 644 upstream
#     records — 105 of our 198 hosted rows and 147 of our 190 open-weight rows.
#     AA sets it when it has not run the full suite: those records average 6.0
#     of 12 sub-evals against 10.3 for measured ones, and carry codingIndex on
#     25% of rows against 91%. Publishing an extrapolation as a measurement is
#     the same class of error as printing "free" for an unpublished price.
#   * codingIndex / agenticIndex / omniscience are new sibling scores.

def _overhaul_record(i=0, estimated=False, coding=70.0, agentic=27.0, omni=-2.0):
    rec = {
        "name": f"Model {i}", "slug": f"m{i}", "modelCreatorName": "Lab",
        "intelligenceIndex": 32.98, "intelligenceIndexIsEstimated": estimated,
        "contextWindowTokens": 128_000, "deprecated": False,
        "price1mInputTokens": 1.0, "price1mOutputTokens": 4.0,
        "medianOutputTokensPerSecond": 50, "medianTimeToFirstTokenSeconds": 0.5,
        "codingIndex": coding, "agenticIndex": agentic, "omniscience": omni,
    }
    return rec


def test_an_estimated_intelligence_score_is_recorded_as_estimated():
    """The flag must survive the scrape. Without it every consumer — the axis,
    the leaderboard, the detail panel, the CSV a reader opens — presents an
    extrapolation and a measurement as the same kind of number."""
    df = load_from_raw(_parse_api_response([_overhaul_record(0, estimated=True)]))
    assert not df.empty, "the record did not survive the parse at all"
    assert "quality_estimated" in df.columns, "the scrape dropped AA's estimated flag"
    assert bool(df.iloc[0]["quality_estimated"]) is True


def test_a_measured_intelligence_score_is_not_marked_estimated():
    df = load_from_raw(_parse_api_response([_overhaul_record(0, estimated=False)]))
    assert bool(df.iloc[0]["quality_estimated"]) is False


def test_the_new_sibling_indices_are_carried():
    """Coding, Agentic and Omniscience are separate AA scores, not derivable
    from the Intelligence Index."""
    df = load_from_raw(_parse_api_response([_overhaul_record(0)]))
    for col, want in (("coding", 70.0), ("agentic", 27.0), ("omniscience", -2.0)):
        assert col in df.columns, f"{col} was not carried through the scrape"
        assert df.iloc[0][col] == pytest.approx(want)


def test_a_missing_sibling_index_is_empty_not_zero():
    """AA scores Coding on 144 of our 198 models and Agentic on 107. Zero is a
    real score on these scales — omniscience is genuinely negative for most
    models — so an unscored model must be blank, never 0."""
    rec = _overhaul_record(0)
    rec["codingIndex"] = None
    rec["agenticIndex"] = None
    df = load_from_raw(_parse_api_response([rec]))
    assert pd.isna(df.iloc[0]["coding"]), "an unscored model was given coding 0"
    assert pd.isna(df.iloc[0]["agentic"]), "an unscored model was given agentic 0"


def test_omniscience_survives_being_negative():
    """AA-Omniscience is hallucination-penalised and runs from -88.6 to +43.7 in
    the live payload; a >0 guard copied from the price/quality columns would
    silently drop the majority of the catalogue."""
    df = load_from_raw(_parse_api_response([_overhaul_record(0, omni=-55.58)]))
    assert df.iloc[0]["omniscience"] == pytest.approx(-55.58)


def test_the_estimated_flag_is_not_health_checked_like_a_dense_column():
    """quality_estimated is a boolean and coding/agentic are legitimately sparse.
    Registering them in _COLUMN_HEALTH would make a healthy scrape fail."""
    from data.scraper import _COLUMN_HEALTH
    for col in ("quality_estimated", "coding", "agentic", "omniscience"):
        assert col not in _COLUMN_HEALTH, (
            f"{col} is sparse or boolean upstream; a density floor would cry wolf"
        )


def test_the_intelligence_scale_has_not_been_rescaled():
    """AA rebuilt the index's eval suite in 2026-09 WITHOUT moving the scale.
    QUALITY_INDEX_MAX, the MIN SCORE ladder (up to 50) and the Agent Stack
    quality floors (28.0 / 30.0) are all absolute numbers on this scale, so a
    silent rescale would miscalibrate every one of them at once."""
    from components.charts.constants import QUALITY_INDEX_MAX

    peak = DF["quality"].max()
    assert 45.0 <= peak <= QUALITY_INDEX_MAX, (
        f"peak intelligence is {peak:.2f}. Below 45 or above {QUALITY_INDEX_MAX} "
        f"means AA moved the scale: recheck the MIN SCORE options in app.py and "
        f"docs/app.js, _API_TIERS min_quality, and static_helpers' >=40/>=50 rules"
    )


def _local_record(estimated=False, coding=44.0, agentic=None, omni=-13.1):
    """One open-weight record in the shape /models/<slug> ships.

    Re-keyed on 2026-09-10 with the source: totalParameters -> parameters,
    activeParameters -> inferenceParametersActiveBillions, modelCreatorName ->
    a nested creator object. codingIndex and agenticIndex are kept here on
    purpose even though AA has stopped sending them — the scrape must still
    prefer a live score to the carried-forward one if they ever come back.
    """
    return {
        "name": "Qwen3.5 122B",
        "creator": {"id": "alibaba", "name": "Alibaba", "logo": "a.svg"},
        "isOpenWeights": True, "deprecated": False,
        "intelligenceIndex": 20.38, "intelligenceIndexIsEstimated": estimated,
        "parameters": 122.0, "inferenceParametersActiveBillions": 10.0,
        "contextWindowTokens": 256_000, "licenseName": "Apache 2.0",
        "codingIndex": coding, "agenticIndex": agentic, "omniscience": omni,
    }


def test_the_open_weight_catalogue_records_the_estimated_flag_too():
    """77% of the open-weight catalogue (147 of 190) carries an estimated
    index — a higher share than the hosted side, because AA re-runs the full
    suite on frontier API models first. The tab that says "what can I run"
    must not present those as measured."""
    from data.local_scraper import _parse as _local_parse

    df = _local_parse([_local_record(estimated=True)])
    assert df is not None and not df.empty
    assert "quality_estimated" in df.columns, "the open-weight scrape drops the flag"
    assert bool(df.iloc[0]["quality_estimated"]) is True

    measured = _local_parse([_local_record(estimated=False)])
    assert bool(measured.iloc[0]["quality_estimated"]) is False


def test_the_open_weight_catalogue_carries_the_sibling_indices():
    from data.local_scraper import _parse as _local_parse

    df = _local_parse([_local_record()])
    assert df.iloc[0]["coding"] == pytest.approx(44.0)
    assert df.iloc[0]["omniscience"] == pytest.approx(-13.1)
    assert pd.isna(df.iloc[0]["agentic"]), "an unscored agentic index became a number"


def test_the_local_export_carries_the_overhaul_columns_too():
    """`export_frame_for_tab` comments that the export "is where someone checks
    the dashboard's work, so it must carry more than the screen does, not less".
    data/local_models._load_models_raw rebuilds each row as an explicit dict, so
    any column not named there is silently dropped on the way to the Run Local
    tab and its ↓CSV — which is how the open-weight export ended up without the
    estimated flag while the hosted one had it."""
    from data.local_models import get_local_df

    df = get_local_df()
    for col in ("quality_estimated", "coding", "agentic", "omniscience"):
        assert col in df.columns, f"the open-weight frame drops {col}"
    scored = df["quality_estimated"].eq(True)
    assert scored.any(), "no open-weight model is flagged estimated (147 of 190 are)"
    assert not scored.all(), "every open-weight model is flagged estimated"


def test_a_curated_model_aa_has_not_benchmarked_is_not_marked_estimated():
    """The curated pending list carries no AA score at all, so it must not be
    labelled with AA's estimation flag either — 'AA estimated this' and 'AA has
    never seen this' are different claims."""
    from data.local_models import get_local_df

    df = get_local_df()
    if "pending" not in df.columns or not df["pending"].eq(True).any():
        pytest.skip("no pending curated models in this catalogue")
    pend = df[df["pending"].eq(True)]
    assert not pend["quality_estimated"].eq(True).any(), (
        "a model AA has never benchmarked is flagged as AA-estimated"
    )


# ── 1.6 — AA's 2026-09-10 leaderboard restructure ────────────────────────────
#
# The fourth AA schema break this project has outlived. /leaderboards/models
# split its single 94-field records array in two: a picker index that now owns
# `name`, and a slimmed 50-field metrics array where `name` became `shortName`
# and the parameter, modality and coding/agentic keys were dropped outright.
#
# Both scrapers read `m.get("name")`, so every one of the 646 rows failed the
# empty-name check and the hourly refresh exited 1 for two days while serving a
# frozen cache. The row-dropping WAS loud — the failure mode this codebase
# actually fears is the silent one — but a catalogue that cannot parse at all is
# still a catalogue nobody can refresh.
#
# `shortName` is not a substitute for `name`: it matched only 144 of the 198
# cached models, so taking it would have renamed 54 models, orphaned their
# history snapshots and broken the spotlight colour mapping.

def _rsc_html(payload: str) -> str:
    """One AA page's flight payload, wrapped the way Next.js ships it."""
    return (
        "<html><body><script>self.__next_f.push([1,"
        + json.dumps(payload)
        + "])</script></body></html>"
    )


def _restructured_pair(n: int = 4):
    """The two arrays /leaderboards/models has served since 2026-09-10."""
    picker = [{
        "slug": f"m{i}",
        "name": f"Model {i} (Adaptive Reasoning, High Effort)",
        "deprecated": False,
        "isReasoning": True,
        "releaseDate": "2026-09-01",
        "creator": {"id": f"c{i}", "name": f"Lab{i % 5}", "logo": "x.svg"},
    } for i in range(n)]
    metrics = [{
        "slug": f"m{i}",
        "shortName": f"Model {i} (high)",          # NOT the name we publish
        "modelCreatorName": f"Lab{i % 5}",
        "intelligenceIndex": 40.0 + i % 20,
        "intelligenceIndexIsEstimated": bool(i % 2),
        "omniscience": -10.0 - i,
        "contextWindowTokens": 128_000,
        "deprecated": False,
        "isOpenWeights": bool(i % 2),
        "price1mInputTokens": 1.0 + i % 3,
        "price1mOutputTokens": 4.0 + i % 5,
        "medianOutputTokensPerSecond": 50 + i,
        "medianTimeToFirstTokenSeconds": 0.5 + (i % 4),
    } for i in range(n)]
    return picker, metrics


def _restructured_html(n: int = 4) -> str:
    picker, metrics = _restructured_pair(n)
    return _rsc_html(
        '{"models":' + json.dumps(picker)
        + ',"label":"Models","models":' + json.dumps(metrics) + "}"
    )


def test_the_restructured_leaderboard_still_yields_rows():
    """The regression itself: 646 upstream rows parsed to zero."""
    from data.scraper import _extract_models

    rows = _parse_api_response(_extract_models(_restructured_html(4)))
    assert len(rows) == 4, (
        f"the 2026-09-10 payload parsed to {len(rows)} rows — the scrape is dead"
    )


def test_the_published_name_is_the_pickers_not_shortname():
    """shortName matched only 144 of 198 cached models. Publishing it would
    rename 54 of them and orphan their history."""
    from data.scraper import _extract_models

    df = load_from_raw(_parse_api_response(_extract_models(_restructured_html(4))))
    names = set(df["model"])
    assert "Model 0 (Adaptive Reasoning, High Effort)" in names, (
        f"the picker's name was not used; got {sorted(names)[:3]}"
    )
    assert not any(n.endswith("(high)") for n in names), (
        "shortName leaked into the published catalogue"
    )


def test_a_metrics_row_missing_from_the_picker_is_not_silently_dropped():
    """A slug the picker does not carry still has a shortName. Dropping the row
    would shrink the catalogue invisibly; falling back keeps it, and the name is
    the only thing that degrades."""
    from data.scraper import _extract_models

    picker, metrics = _restructured_pair(4)
    picker = [p for p in picker if p["slug"] != "m2"]        # m2 only in metrics
    html = _rsc_html('{"models":' + json.dumps(picker)
                     + ',"models":' + json.dumps(metrics) + "}")
    names = {r[0] for r in _parse_api_response(_extract_models(html))}
    assert "Model 2 (high)" in names, "a picker-less row vanished from the scrape"


def test_the_picker_index_is_not_mistaken_for_the_metrics_array():
    """The picker now carries `name`, which is exactly what the metrics array
    lost — so a name-based predicate would select the wrong one and publish a
    full row count with every metric empty."""
    from data.scraper import _extract_models

    models = _extract_models(_restructured_html(4))
    assert all("intelligenceIndex" in m for m in models), (
        "picked the picker index — every metric column would publish empty"
    )


def test_coding_and_agentic_are_carried_forward_not_blanked():
    """The same restructure dropped codingIndex and agenticIndex outright.

    AA replaced the two composites with the raw component benchmarks
    (terminalbench*, tau2, scicode, ifbench ...) rather than renaming them, so
    there is no key to re-point at. Blanking two columns of real measurements
    because upstream stopped recomputing them is the data loss the image arena
    already taught this project to avoid — carry, and say so.
    """
    from data.scraper import _carry_dropped_columns

    live = pd.DataFrame({
        "model":   ["A", "B"],
        "coding":  [float("nan"), float("nan")],
        "agentic": [float("nan"), float("nan")],
    })
    cached = pd.DataFrame({"model": ["A"], "coding": [70.0], "agentic": [50.0]})
    out = _carry_dropped_columns(live, cached)

    assert out.loc[0, "coding"] == 70.0, "the cached coding score was dropped"
    assert out.loc[0, "agentic"] == 50.0, "the cached agentic score was dropped"
    assert pd.isna(out.loc[1, "coding"]), (
        "a model absent from the cache inherited another model's score"
    )


def test_a_live_index_always_beats_the_cached_one():
    """If AA ever publishes these again, the cache must not pin the scrape to
    its own history."""
    from data.scraper import _carry_dropped_columns

    live = pd.DataFrame({"model": ["A"], "coding": [80.0], "agentic": [float("nan")]})
    cached = pd.DataFrame({"model": ["A"], "coding": [70.0], "agentic": [50.0]})
    out = _carry_dropped_columns(live, cached)

    assert out.loc[0, "coding"] == 80.0, "a stale cached score overwrote a live one"
    assert out.loc[0, "agentic"] == 50.0, "the hole was not filled"


# ── 1.7 — the open-weight catalogue after the same restructure ───────────────
#
# The Run Local tab needs total and active parameter counts to size a model
# against a card's VRAM and bandwidth. The slimmed leaderboard dropped
# totalParameters, activeParameters and every inputModality flag — `paramClass`
# ("small"/"medium"/"large") is all that is left, which cannot drive a roofline.
#
# Those fields still exist, on the per-model pages under /models/<slug>, which
# ship the full 88-field catalogue for all 646 models rather than just the one
# in the URL. `totalParameters` is now `parameters`, `activeParameters` is
# `inferenceParametersActiveBillions`, and `modelCreatorName` became a nested
# `creator` object.

def _detail_record(i: int = 0, **over) -> dict:
    """One record as /models/<slug> ships it since 2026-09-10."""
    rec = {
        "slug": f"ow{i}",
        "name": f"Open Model {i}",
        "shortName": f"OM{i}",
        "creator": {"id": f"c{i}", "name": "LabX", "logo": "x.svg"},
        "isOpenWeights": True,
        "deprecated": False,
        "isReasoning": True,
        "intelligenceIndex": 40.0 + i,
        "intelligenceIndexIsEstimated": True,
        "omniscience": -12.0,
        "contextWindowTokens": 256_000,
        "parameters": 120.0,
        "inferenceParametersActiveBillions": 12.0,
        "licenseName": "Apache 2.0",
        "inputModalityImage": True,
        "inputModalitySpeech": False,
    }
    rec.update(over)
    return rec


def test_the_local_catalogue_reads_the_renamed_parameter_keys():
    """params_b and active_b are what the roofline is computed from. Reading
    the retired key names yields None and drops every row."""
    from data.local_scraper import _parse

    df = _parse([_detail_record(0)])
    assert df is not None and len(df) == 1, "the detail-page record did not parse"
    row = df.iloc[0]
    assert row["params_b"] == 120.0, f"params_b came through as {row['params_b']}"
    assert row["active_b"] == 12.0, f"active_b came through as {row['active_b']}"
    assert row["family"] == "LabX", (
        f"the nested creator object was not read; family={row['family']!r}"
    )
    assert row["context_k"] == 256, f"context_k came through as {row['context_k']}"
    assert row["license"] == "Apache 2.0"
    assert row["moe"], "120B total against 12B active is an MoE"


def test_a_dense_model_falls_back_to_its_total_parameters():
    """AA leaves the active count null on dense models — 98 of 193 open-weight
    rows. Dropping them would halve the tab."""
    from data.local_scraper import _parse

    df = _parse([_detail_record(0, parameters=27.0,
                                inferenceParametersActiveBillions=None)])
    assert df is not None and len(df) == 1, "a dense model was dropped"
    row = df.iloc[0]
    assert row["active_b"] == 27.0, "a dense model did not fall back to its total"
    assert not row["moe"], "a dense model was labelled MoE"


def test_the_detail_page_catalogue_is_told_apart_from_the_slim_leaderboard():
    """/models/<slug> carries more than one array under "models". Picking the
    slim one would lose the parameter counts this page was fetched for."""
    from data.local_scraper import _extract_models

    slim = [{"slug": "ow0", "shortName": "OM0", "intelligenceIndex": 40.0}]
    full = [_detail_record(0), _detail_record(1)]
    html = _rsc_html('{"models":' + json.dumps(slim)
                     + ',"models":' + json.dumps(full) + "}")
    got = _extract_models(html)
    assert all("parameters" in m for m in got), (
        "picked an array with no parameter counts"
    )
    assert len(got) == 2


def test_the_catalogue_page_slug_is_discovered_not_hardcoded():
    """Any model's page serves the whole catalogue, so the entry point is just
    a slug that exists. Pinning one means the scrape dies the day that model is
    deprecated — which is how this project lost three endpoints already."""
    from data.local_scraper import _entry_slug

    picker = [
        {"slug": "gone", "name": "Retired", "deprecated": True},
        {"slug": "alive", "name": "Current", "deprecated": False},
    ]
    payload = '{"models":' + json.dumps(picker) + "}"
    assert _entry_slug(payload) == "alive", "picked a deprecated model's page"
