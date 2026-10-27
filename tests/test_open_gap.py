# tests/test_open_gap.py
"""The open-vs-closed gap (data/open_gap.py, components/charts/open_gap.py).

Pinned here:
  * AA's index rescales are FOUND from the committed snapshots — 2026-06-16,
    2026-09-05, 2026-09-08 — and the 2026-03-07 one-day 0-100 glitch is
    excluded rather than treated as two rescales; partial re-evaluations
    (2026-08-07, 2026-09-20) are not breaks;
  * no line is drawn across a rescale, and each one has a rule;
  * hand-checked gaps (2026-05-06 6.3 via Kimi K2.6, which only the registry
    still knows is open; 2026-10-21 11.3 / 19.6%);
  * release lag right-censors an unmatched record instead of dropping it,
    takes a Kaplan-Meier median over matched AND censored records, does not
    count the oldest surviving closed model (a "record" only because AA
    dropped everything stronger), and dates a checkpoint updated in place by
    its score's first snapshot;
  * the 2026-02-25 snapshot's unsuffixed names ("GLM-5") resolve open, so its
    gap is 7.0, not 15.0; an empty history never aborts the build;
  * the iso-quality ratio points the stated way, uses the stored price at the
    default mix, moves with the mix by the hand-computed amount, uses only
    AA-measured scores (and says so), and counts what it excludes;
  * the three figures are pre-rendered, the history never ships, and the
    Economics charts never use hollow marks for open weights.
"""
import json
import math
from pathlib import Path

import pandas as pd
import pytest

import build_static
from captions import CAPTIONS
from data.ingest import load_history, get_models
from data import open_gap as og
from data.workload_mix import resolve_mix
from components.charts.open_gap import build_gap_history, build_release_lag, build_iso_price

ROOT = Path(__file__).resolve().parent.parent
DF = get_models()


@pytest.fixture(scope="module")
def hist():
    h = load_history()
    if h.empty:
        pytest.skip("no committed history snapshots")
    # Pin to the snapshots that existed when the detector was tuned; the bot
    # adds one a day and a later rescale must not break these assertions.
    return h[h["scraped_at"].astype(str).str[:10] <= "2026-10-27"]


@pytest.fixture(scope="module")
def gap(hist):
    return og.gap_history(hist)


# ── Rescale detection ─────────────────────────────────────────────────────────

def test_detects_the_three_verified_rescales_and_drops_the_glitch(gap):
    _, seg = gap
    assert seg["breaks"] == ["2026-06-16", "2026-09-05", "2026-09-08"]
    assert seg["excluded"] == ["2026-03-07"]


def test_partial_reevaluations_stay_below_the_threshold(gap):
    _, seg = gap
    steps = {s["to"]: s for s in seg["steps"]}
    for d in ("2026-08-07", "2026-09-20"):
        assert not steps[d]["break"], d
        assert steps[d]["moved"] < og.BREAK_MOVED_SHARE


def _snap(day, scores):
    return pd.DataFrame({"model": list(scores), "quality": list(scores.values()),
                         "scraped_at": day})


def test_a_uniform_rescale_is_a_break_and_a_bounce_is_a_glitch():
    base = {f"m{i}": 30.0 + i for i in range(25)}
    scaled = {k: v * 0.75 for k, v in base.items()}
    h = pd.concat([_snap("2026-01-01", base), _snap("2026-01-02", base),
                   _snap("2026-01-03", {k: v * 1.7 for k, v in base.items()}),   # glitch
                   _snap("2026-01-04", base), _snap("2026-01-05", scaled)])
    seg = og.index_segments(h)
    assert seg["excluded"] == ["2026-01-03"]
    assert seg["breaks"] == ["2026-01-05"]
    assert seg["segments"]["2026-01-04"] == 0 and seg["segments"]["2026-01-05"] == 1


def test_too_few_common_models_is_never_a_break():
    base = {f"m{i}": 30.0 + i for i in range(og.BREAK_MIN_COMMON - 1)}
    h = pd.concat([_snap("2026-01-01", base),
                   _snap("2026-01-02", {k: v * 0.5 for k, v in base.items()})])
    assert og.index_segments(h)["breaks"] == []


# ── Gap values ────────────────────────────────────────────────────────────────

def test_gap_values_match_hand_computed_snapshots(gap):
    gh, _ = gap
    by = gh.set_index("date")
    may = by.loc["2026-05-06"]
    assert may["open_model"] == "Kimi K2.6", "the registry must still know Kimi K2.6 is open"
    assert may["closed_q"] == pytest.approx(60.24) and may["gap"] == pytest.approx(6.34)
    oct21 = by.loc["2026-10-21"]
    assert oct21["gap"] == pytest.approx(11.30, abs=0.005)
    assert oct21["gap_pct"] == pytest.approx(19.6, abs=0.05)
    assert "2026-03-07" not in by.index


def test_every_snapshot_says_how_its_models_were_classified(gap):
    """Source counts are per MODEL, like n_open / n_closed beside them in the
    hover (2026-03-06 has 708 host rows for 226 models)."""
    gh, _ = gap
    total = gh["src_aa"] + gh["src_catalogue"] + gh["src_inferred"]
    assert (total > 0).all()
    assert (total == gh["n_open"] + gh["n_closed"]).all()


def test_the_first_snapshot_resolves_its_unsuffixed_open_models(gap):
    """2026-02-25 named models without AA's later variant tag: "GLM-5" is the
    registry's "GLM-5 (Reasoning)". Exact matching alone counted GLM-5, Kimi
    K2.5 and Qwen3.5 397B A17B as closed and drew 57.0 - 42.0 = 15.0."""
    gh, _ = gap
    feb = gh.set_index("date").loc["2026-02-25"]
    assert feb["open_model"] == "GLM-5" and feb["open_q"] == pytest.approx(50.0)
    assert feb["gap"] == pytest.approx(7.0)


def test_an_empty_history_is_no_snapshots_not_an_exception():
    gh, seg = og.gap_history(pd.DataFrame())
    assert gh.empty and list(gh.columns)[:3] == ["date", "segment", "closed_model"]
    assert seg == {"segments": {}, "breaks": [], "excluded": [], "steps": []}
    assert og.snapshot_dates(pd.DataFrame()) == []
    assert og.score_dates(pd.DataFrame(), DF) == {}
    assert build_gap_history(gh, seg).layout.annotations    # the empty-figure note


def test_the_build_survives_a_missing_history(monkeypatch):
    monkeypatch.setattr(build_static, "load_history", lambda: pd.DataFrame())
    figs = build_static.open_gap_figures(DF)
    assert set(figs) == {"econ_open_gap_history", "econ_open_gap_lag", "econ_open_gap_price"}


def test_no_gap_line_crosses_a_break(gap):
    gh, seg = gap
    fig = build_gap_history(gh, seg)
    line = fig.data[0]
    xs, ys = list(line.x), list(line.y)
    assert line.connectgaps is False
    for b in seg["breaks"]:
        i = xs.index(b)
        assert xs[i - 1] is None and ys[i - 1] is None, f"line joins across {b}"
    assert xs.count(None) == len(seg["breaks"])
    rules = [s for s in fig.layout.shapes if s.type == "line"]
    assert sorted(r.x0 for r in rules) == seg["breaks"]
    labels = [a.text for a in fig.layout.annotations]
    assert labels.count("AA rescale") == len(seg["breaks"])


# ── Release lag ───────────────────────────────────────────────────────────────

def _fam(rows):
    return pd.DataFrame(rows, columns=["model", "provider", "quality", "release_date",
                                       "open_weights"]).assign(weights_source="aa")


def test_release_lag_matches_open_leads_and_censors_the_unmatched():
    df = _fam([
        ("Cold", "OpenAI", 30.0, "2025-06-01", False),  # oldest listed: not counted
        ("C1", "OpenAI", 40.0, "2026-01-01", False),
        ("O1", "Meta", 41.0, "2025-12-01", True),       # open got there first: lag 0
        ("C2", "OpenAI", 50.0, "2026-02-01", False),
        ("O2", "Meta", 50.0, "2026-05-02", True),       # 90 days later
        ("C3", "Google", 60.0, "2026-03-01", False),    # nothing open reaches 60
        ("C0", "Google", 45.0, "2026-01-15", False),    # 45 > 40: also a record
    ])
    recs, s = og.release_lag(df, "2026-06-01")
    by = recs.set_index("closed_model")
    assert list(by.index) == ["C1", "C0", "C2", "C3"]
    assert by.loc["C1", "lag_months"] == 0 and by.loc["C1", "open_model"] == "O1"
    assert by.loc["C2", "lag_months"] == pytest.approx(90 / og.DAYS_PER_MONTH)
    assert bool(by.loc["C3", "censored"]) and by.loc["C3", "open_model"] is None
    assert by.loc["C3", "lag_months"] == pytest.approx(92 / og.DAYS_PER_MONTH)
    assert s["matched"] == 3 and s["censored"] == 1
    assert s["uncounted"] == ["Cold"] and "Cold" in list(s["closed_records"]["model"])
    # matched lags: C1 0, C2 90 d, C0 107 d (O2 is the first open model >= 45)
    assert by.loc["C0", "lag_months"] == pytest.approx(107 / og.DAYS_PER_MONTH)
    # Kaplan-Meier: 0 (S=3/4), 90 d (S=1/2) -> median 90 d
    assert s["median_lag_months"] == pytest.approx(by.loc["C2", "lag_months"])
    assert s["median_matched_months"] == pytest.approx(by.loc["C2", "lag_months"])


@pytest.mark.parametrize("times, censored, expected", [
    ([1, 2, 3, 4, 5], [0, 1, 1, 1, 0], 5),      # matched-only median would be 3
    ([1, 2, 3, 4, 5], [0, 0, 0, 0, 0], 3),
    ([1, 4, 4, 5], [0, 0, 1, 0], 4),            # events count before a tie's censoring
    ([1, 2, 3, 4], [0, 1, 1, 1], None),         # S never reaches 1/2: not identifiable
    ([], [], None),
])
def test_km_median_counts_censored_records(times, censored, expected):
    assert og.km_median(times, censored) == expected


def test_a_checkpoint_renamed_with_a_higher_score_is_dated_by_its_snapshot():
    base = {f"m{i}": 20.0 + i for i in range(25)}
    h = pd.concat([_snap("2026-08-01", {**base, "Q Max": 40.0}),
                   _snap("2026-08-10", {**base, "Q Max (0902)": 45.0, "R (Max)": 30.0}),
                   _snap("2026-08-02", {**base, "R": 30.0})])
    cur = pd.DataFrame({"model": ["Q Max (0902)", "R (Max)", "m0"],
                        "quality": [45.0, 30.0, 20.0],
                        "release_date": ["2026-07-01", "2026-07-01", "2026-07-01"]})
    # Q: same base, +12.5% under a new name -> redated; R: renamed, same score -> not
    assert og.score_dates(h, cur) == {"Q Max (0902)": pd.Timestamp("2026-08-10")}


def test_the_live_lag_redates_qwen_max_and_skips_the_oldest_record(hist):
    dated = og.score_dates(hist, DF)
    assert dated.get("Qwen3.8 Max (0902)") == pd.Timestamp("2026-09-16")
    recs, s = og.release_lag(DF, "2026-10-27", dated)
    assert "Nova Micro" in s["uncounted"] and "Nova Micro" not in set(recs["closed_model"])
    assert "Qwen3.8 Max (0902)" in s["redated"]
    assert (recs["closed_date"] >= pd.Timestamp("2025-01-01")).all()


def test_release_lag_on_the_live_catalogue_reports_both_counts():
    recs, s = og.release_lag(DF, "2026-10-27")
    assert s["matched"] + s["censored"] == len(recs) > 0
    fig = build_release_lag(recs, s)
    title = fig.layout.title.text
    assert f"{s['matched']} matched + {s['censored']} not yet" in title
    assert "Kaplan–Meier" in title, "the median's basis must be on the chart"
    assert "still lists" in title, "the survivorship limit must be on the chart"


def test_lag_connectors_are_accent_and_labelled_not_hidden_on_the_staircase():
    recs, s = og.release_lag(DF, "2026-10-27")
    fig = build_release_lag(recs, s)
    conns = [t for t in fig.data if t.legendgroup in (
        "Lag to first open model ≥ that score", "Not matched yet (≥ months)")]
    drawn = recs[recs["censored"] | (recs["lag_months"] > 0)]
    assert len(conns) == len(drawn) > 0
    assert all(t.line.color == "#00d4ff" for t in conns)
    assert fig.data[-1] is conns[-1], "connectors are drawn above the staircases"
    texts = [a.text for a in fig.layout.annotations]
    for _, r in drawn.iterrows():
        want = (f"≥{r['lag_months']:.1f} mo" if r["censored"] else f"{r['lag_months']:.1f} mo")
        assert want in texts


# ── Iso-quality price ─────────────────────────────────────────────────────────

def _priced(rows):
    return pd.DataFrame(rows, columns=["model", "provider", "quality", "price_in",
                                       "price_out", "open_weights"]).assign(
        price=lambda d: (d["price_in"] + 3 * d["price_out"]) / 4,
        cache_hit_price=float("nan"), weights_source="aa")


def test_iso_ratio_direction_default_price_and_exclusions():
    df = _priced([
        ("Open A", "Meta", 40.0, 1.0, 1.0, True),       # price 1.0
        ("Open B", "Meta", 55.0, 1.0, 1.0, True),       # no closed peer at >= 55
        ("Closed X", "OpenAI", 45.0, 2.0, 2.0, False),  # price 2.0, cheapest >= 40
        ("Closed Y", "OpenAI", 50.0, 9.0, 9.0, False),
    ])
    out, s = og.iso_quality_ratios(df)
    assert list(out["open_model"]) == ["Open A"]
    row = out.iloc[0]
    assert row["closed_model"] == "Closed X"
    assert row["ratio"] == pytest.approx(2.0), ">1 means the open model is cheaper"
    assert row["open_price"] == df.loc[0, "price"], "default mix must use the stored price"
    assert s["n_excluded"] == 1 and s["excluded"] == ["Open B"]
    assert s["mix_short"] == resolve_mix(None).short


def test_iso_ratio_moves_with_the_mix_by_the_hand_computed_amount():
    df = _priced([
        ("Open A", "Meta", 40.0, 1.0, 4.0, True),
        ("Closed X", "OpenAI", 45.0, 3.0, 2.0, False),
    ])
    at_default, _ = og.iso_quality_ratios(df)
    assert at_default["ratio"].iloc[0] == pytest.approx(((3 + 6) / 4) / ((1 + 12) / 4))
    chat, _ = og.iso_quality_ratios(df, "chat")         # 1:1
    assert chat["ratio"].iloc[0] == pytest.approx(((3 + 2) / 2) / ((1 + 4) / 2))


def test_iso_ratio_uses_only_measured_scores_and_says_so():
    df = _priced([
        ("Open A", "Meta", 40.0, 1.0, 1.0, True),
        ("Closed Est", "Sapiens AI", 45.0, 0.1, 0.1, False),   # cheap, AA estimate
        ("Closed X", "OpenAI", 45.0, 2.0, 2.0, False),
    ]).assign(quality_estimated=[False, True, False])
    out, s = og.iso_quality_ratios(df)
    assert out.iloc[0]["closed_model"] == "Closed X"
    assert s["measured_only"] and s["n_estimated_rows"] == 1
    every, s2 = og.iso_quality_ratios(df, measured_only=False)
    assert every.iloc[0]["closed_model"] == "Closed Est" and s2["n_estimated_rows"] == 0
    assert "AA-measured scores only (1 estimated rows left out)" in \
        build_iso_price(out, s).layout.title.text


def test_the_live_iso_price_figure_asserts_no_headline():
    out, s = og.iso_quality_ratios(DF)
    fig = build_iso_price(out, s)
    title = fig.layout.title.text
    assert f"{s['n']} open families, AA-measured scores only" in title and "priced at" in title
    est = set(DF.loc[DF["quality_estimated"].eq(True), "model"])
    assert not (set(out["closed_model"]) | set(out["open_model"])) & est
    assert s["share_below_1"] is not None and 0 < s["share_below_1"] < 1, \
        "the distribution runs both ways on 2026-10-27; the chart must show it"
    assert any(sh.x0 == 1 for sh in fig.layout.shapes), "missing the 1x rule"


# ── Figures, bundle, captions ─────────────────────────────────────────────────

def test_economics_figures_never_use_hollow_for_open_weights(gap):
    figs = [build_gap_history(*gap), build_release_lag(*og.release_lag(DF, "2026-10-27")),
            build_iso_price(*og.iso_quality_ratios(DF))]
    for fig in figs:
        for t in fig.data:
            sym = getattr(getattr(t, "marker", None), "symbol", None)
            assert sym is None or "-open" not in str(sym)


def test_open_gap_figures_are_prerendered_and_history_never_ships():
    figs = build_static.open_gap_figures(DF)
    assert set(figs) == {"econ_open_gap_history", "econ_open_gap_lag", "econ_open_gap_price"}
    assert not any("history" in p for p in build_static.DATA_CSVS)
    assert "data/carried/open_weight_names.csv" in build_static.DATA_CSVS


def test_the_bundle_ships_the_new_modules():
    src = (ROOT / "build_static.py").read_text()
    for mod in ("data/open_weights.py", "data/open_gap.py", "components/charts/open_gap.py"):
        assert f'"{mod}"' in src, mod


def test_the_economics_caption_cites_its_sources_and_limits():
    c = CAPTIONS["economics_open_gap"]
    for needle in ("artificialanalysis.ai", "epoch.ai/data-insights/open-closed-eci-gap",
                   "CC BY 4.0", "openrouter.ai/data/newsletter/the-token-state-of-play",
                   "quoted, not scraped", "nousresearch.com", "August 2025",
                   "Kaplan–Meier",
                   "2026-06-16", "2026-09-05", "2026-09-08", "2026-03-07", "survivor"):
        assert needle in c, needle


def test_the_hosted_captions_explain_hollow():
    assert "Hollow = open weights" in CAPTIONS["overview_price"]
    assert "Hollow = open weights" in CAPTIONS["overview_speed"]
    assert "Outlined bars = open weights" in CAPTIONS["rankings_intelligence"]


def test_the_readme_no_longer_claims_the_index_scale_never_moved():
    readme = (ROOT / "README.md").read_text()
    assert "without changing the scale" not in readme
    for d in ("2026-09-05", "2026-09-08"):
        assert d in readme
