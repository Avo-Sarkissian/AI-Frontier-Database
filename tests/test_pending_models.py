"""Curated open-weight entries may state facts, never invent a benchmark.

The Run Local tab is captioned "open-weight models you can run on your own
hardware" but was built only from Artificial Analysis's endpoint, so it actually
showed "models AA has benchmarked". Qwen3.8-27B — 28B dense, Apache 2.0, aimed
at 24 GB cards, exactly this tab's subject — was invisible for the days AA took
to score it.

The overlay closes that gap. These tests hold the line on what it may claim.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from data.local_models import get_local_df
from data.pending_models import (
    PENDING_MODELS, merge_pending, redundant_entries, _norm,
)

ROOT = Path(__file__).resolve().parent.parent


def _pending_rows(df):
    return df[df["pending"].fillna(False)] if "pending" in df.columns else df.iloc[0:0]


# ── An entry may only carry facts ────────────────────────────────────────────

@pytest.mark.parametrize("entry", PENDING_MODELS, ids=lambda e: e["name"])
def test_no_entry_invents_a_benchmark_score(entry):
    """`quality` is a benchmark result, not an architectural fact. A plausible
    number here would propagate into the value rankings and the recommender —
    the hand-set-constant pattern this codebase keeps paying for."""
    assert entry["quality"] is None, (
        f"{entry['name']} carries a quality score that nobody measured"
    )


@pytest.mark.parametrize("entry", PENDING_MODELS, ids=lambda e: e["name"])
def test_every_entry_is_checkable(entry):
    """A curated claim without a source is folklore."""
    assert entry.get("source", "").startswith("http"), "no source URL"
    assert entry.get("announced"), "no announcement date"
    for field in ("name", "family", "params_b", "active_b", "context_k", "license"):
        assert entry.get(field) not in (None, ""), f"missing {field}"
    assert entry["active_b"] <= entry["params_b"]
    assert entry["moe"] == (entry["active_b"] < entry["params_b"]), (
        "moe flag disagrees with the active/total parameter split"
    )


# ── The overlay yields to real data ──────────────────────────────────────────

def test_a_scored_model_wins_over_its_curated_entry():
    """The moment AA publishes a score the real record must take over, or the
    catalogue carries two competing claims about one model."""
    scraped = [{"name": "Qwen3.8 27B (Reasoning)", "quality": 41.0, "params_b": 28.0,
                "active_b": 28.0, "context_k": 262, "license": "Apache 2.0",
                "tags": [], "moe": False, "family": "Alibaba"}]
    merged = merge_pending(scraped)
    assert len(merged) == 1, "the curated duplicate was not suppressed"
    assert merged[0]["quality"] == 41.0
    assert merged[0]["pending"] is False


def test_matching_ignores_effort_suffixes_and_punctuation():
    assert _norm("Qwen3.8 27B") == _norm("Qwen3.8-27B (Reasoning)")
    assert _norm("Qwen3.8 27B") != _norm("Qwen3.6 27B")


def test_entries_that_are_now_redundant_are_reported():
    """Fails once an entry can be deleted, so the list cannot quietly rot."""
    live = get_local_df(include_pending=False)
    stale = redundant_entries(live["name"])
    assert not stale, (
        f"Artificial Analysis now scores these — delete them from "
        f"data/pending_models.py: {stale}"
    )


# ── It behaves like a real row where it can, and nowhere else ────────────────

def test_a_pending_model_gets_real_vram_and_speed_numbers():
    """VRAM fit and speed are COMPUTED from published architecture, so they are
    as sound for a curated entry as for a scraped one — that is the whole point
    of carrying it."""
    df = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=1792)
    rows = _pending_rows(df)
    if rows.empty:
        pytest.skip("no pending entries in the catalogue")
    for _, r in rows.iterrows():
        assert r["vram_req_gb"] > 0
        assert r["speed_tps"] > 0
        assert r["fits"] in ("yes", "tight", "no")
        assert pd.isna(r["quality"])


def test_quantisation_moves_a_pending_model_in_and_out_of_range():
    """It is subject to exactly the same hardware arithmetic as everything else."""
    small = get_local_df(quant="FP16", vram_gb=8, bandwidth_gbps=288)
    large = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=1792)
    s, l = _pending_rows(small), _pending_rows(large)
    if s.empty or l.empty:
        pytest.skip("no pending entries")
    assert set(s["fits"]) == {"no"}, "a 28B model claims to fit 8 GB at FP16"
    assert "yes" in set(l["fits"])


def test_the_recommender_never_recommends_an_unscored_model():
    """You cannot rank on a number nobody has. NaN fails the quality floor, and
    this test pins that rather than trusting the comparison semantics."""
    from data.ingest import get_models
    from components.stack_recommender import select_stack

    local = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=1792)
    unscored = set(_pending_rows(local)["name"])
    if not unscored:
        pytest.skip("no pending entries")
    result = select_stack(get_models(), None, "local",
                          local_df=local, full_local_df=local)
    for tier in result["tiers"]:
        picks = tier["picks"]
        if picks is None or picks.empty:
            continue
        leaked = unscored & set(picks["name"])
        assert not leaked, f"{tier['key']} recommends an unscored model: {leaked}"


# ── The charts must not let an absent score read as a low one ────────────────

def test_the_compat_chart_draws_no_length_for_an_unscored_model():
    """A zero-length bar would say "scored zero"; a filled one would invent a
    score. An outline says "this exists and fits" and encodes no magnitude."""
    from components.charts.local_compat import build_local_compat

    df = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=1792)
    if _pending_rows(df).empty:
        pytest.skip("no pending entries")
    fig = json.loads(build_local_compat(df, quant="Q4", vram_gb=32).to_json())

    outlines = [t for t in fig["data"]
                if (t.get("marker") or {}).get("color") == "rgba(0,0,0,0)"]
    assert outlines, "unscored models are not drawn as outlines"
    assert "Not yet scored" in (outlines[0].get("hovertemplate") or "")

    # and the quality bars must contain no fabricated value for them
    names = set(_pending_rows(df)["name"])
    for trace in fig["data"]:
        if (trace.get("marker") or {}).get("color") == "rgba(0,0,0,0)":
            continue
        for y, x in zip(trace.get("y") or [], trace.get("x") or []):
            if y in names and isinstance(x, (int, float)):
                assert x == 0 or trace.get("hoverinfo") == "skip", (
                    f"{y} is drawn with a quality bar of {x}"
                )


def test_both_local_charts_disclose_what_they_leave_out():
    from components.charts.local_compat import build_local_compat
    from components.charts.local_scatter import build_local_scatter

    df = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=1792)
    if _pending_rows(df).empty:
        pytest.skip("no pending entries")

    compat = json.loads(build_local_compat(df, quant="Q4", vram_gb=32).to_json())
    assert "not yet scored" in compat["layout"]["title"]["text"]

    scatter = json.loads(build_local_scatter(df, vram_gb=32, quant="Q4").to_json())
    assert "not yet scored" in scatter["layout"]["title"]["text"], (
        "the scatter drops unscored models from a quality axis without saying so"
    )


def test_the_tab_caption_says_the_list_is_partly_curated():
    from captions import CAPTIONS

    caption = CAPTIONS["local"].lower()
    assert "curated" in caption and "not benchmarked" in caption.replace("-", " ")


def test_the_overlay_can_be_switched_off():
    """Callers that need only what AA vouches for must be able to say so."""
    with_pending = get_local_df(include_pending=True)
    without = get_local_df(include_pending=False)
    assert len(with_pending) >= len(without)
    assert not without["pending"].any()
