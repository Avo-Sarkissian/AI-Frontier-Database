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
