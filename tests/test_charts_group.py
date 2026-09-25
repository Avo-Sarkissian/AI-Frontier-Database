"""Chart builders: batched annotations, Run Local labels, stable ordering.

The golden fixture was captured from the builders BEFORE the per-row
fig.add_annotation loops were replaced with one batched assignment, on inputs
frozen alongside it (tests/fixtures/charts_annotation_inputs.pkl.gz: the hosted,
image and video catalogues plus get_local_df at 8x B200 SXM, Q4, 128k). The
refactor had to be invisible in the output, so every builder's annotations are
compared against that capture. The only differences allowed are the intended
ones, each asserted explicitly: tied bars in the Run Local ranking now order by
name, and the Run Local scatter keeps its "N GB" VRAM label.

One edit to the capture since: the Institute of Foundation Models labels in
local_compat / local_compat_thr were recoloured #6b7280 -> #4a6ae4. The
capture was taken while _LOCAL_ONLY_COLORS still keyed that family under its
retired "MBZUAI " name, so the grey was the palette miss, not the output.
"""
import gzip
import json
import pickle
import re
import time
from pathlib import Path

import pandas as pd
import pytest

from components.charts.cost_calc import build_cost_calc
from components.charts.image_scatter import build_image_faceted, build_image_rankings
from components.charts.local_compat import (
    ESTIMATED_KV_NOTE, build_local_compat, ctx_labels, speed_notes,
)
from components.charts.local_scatter import build_local_scatter
from components.charts.provider_leaderboard import build_provider_leaderboard
from components.charts.video_chart import build_video_rankings
from data.local_models import DEFAULT_SLO, SLO_FLOORS_TPS

FIXTURES = Path(__file__).parent / "fixtures"
B200X8_VRAM = 192 * 8
CTX = 131072


@pytest.fixture(scope="module")
def inputs():
    with gzip.open(FIXTURES / "charts_annotation_inputs.pkl.gz", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def golden():
    with gzip.open(FIXTURES / "charts_annotations_golden.json.gz", "rt") as f:
        return json.load(f)


def _anns(fig) -> list[dict]:
    return json.loads(fig.to_json())["layout"].get("annotations", [])


def _key(d) -> str:
    return json.dumps(d, sort_keys=True)


# ── #1: batched annotations are the same annotations ────────────────────────

@pytest.mark.parametrize("name, build", [
    ("cost_calc", lambda i: build_cost_calc(i["hosted"], monthly_tokens_m=1.0)),
    ("provider_leaderboard", lambda i: build_provider_leaderboard(i["hosted"])),
    ("video_rankings", lambda i: build_video_rankings(i["video"], full_df=i["video"])),
    ("image_faceted", lambda i: build_image_faceted(i["image"])),
    ("image_rankings", lambda i: build_image_rankings(i["image"])),
])
def test_batched_annotations_match_the_per_row_capture_exactly(inputs, golden, name, build):
    """Same annotations, same order, same properties as the add_annotation loop."""
    assert _anns(build(inputs)) == golden[name]


@pytest.mark.parametrize("mode, gold", [("single", "local_compat"),
                                        ("throughput", "local_compat_thr")])
def test_local_compat_annotations_match_capture_up_to_tie_order(inputs, golden, mode, gold):
    """The row set is identical; only tied scores may move, and they now sort by name."""
    fig = build_local_compat(inputs["local"], quant="Q4", vram_gb=B200X8_VRAM,
                             ctx_tokens=CTX, speed_mode=mode)
    after = _anns(fig)
    assert sorted(map(_key, after)) == sorted(map(_key, golden[gold]))
    # Every annotation still sits on the bar it labels, in the bar order.
    assert [a["y"] for a in after] == list(fig.data[-1].y)


def test_local_scatter_keeps_its_legend_key_and_gains_the_vram_label(inputs, golden):
    fig = build_local_scatter(inputs["local"], vram_gb=B200X8_VRAM, quant="Q4",
                              ctx_tokens=CTX)
    after = _anns(fig)
    assert golden["local_scatter"][0] in after, "legend key changed"
    vram = [a for a in after if a.get("xref") == "x" and a.get("x") == B200X8_VRAM]
    assert len(vram) == 1 and vram[0]["text"].strip() == f"{B200X8_VRAM} GB"


def test_local_compat_is_fast_at_the_largest_realistic_hardware(inputs):
    """8x B200 makes nearly the whole catalogue runnable, the worst case for the
    old quadratic loop (2.5 s natively; several times that in Pyodide)."""
    df = inputs["local"]
    assert int(df["fits"].isin(["yes", "tight"]).sum()) >= 150
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        build_local_compat(df, quant="Q4", vram_gb=B200X8_VRAM, ctx_tokens=CTX)
        best = min(best, time.perf_counter() - t0)
    assert best < 0.5, f"build_local_compat took {best:.3f}s"


# ── #25: deterministic tie order ────────────────────────────────────────────

def test_row_order_does_not_depend_on_input_order(inputs):
    """The CI build and the Pyodide render must emit the same figure."""
    df = inputs["local"]
    a = build_local_compat(df, quant="Q4", vram_gb=B200X8_VRAM, ctx_tokens=CTX)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    b = build_local_compat(shuffled, quant="Q4", vram_gb=B200X8_VRAM, ctx_tokens=CTX)
    assert a.to_json() == b.to_json()


# ── Synthetic rows for the label fixes ──────────────────────────────────────

def _row(name, **kw):
    base = dict(name=name, family="Meta", quality=40.0, fits="yes", pending=False,
                vram_req_gb=10.0, weights_gb=8.0, kv_gb=0.5, overhead_gb=1.5,
                kv_source="config", ctx_used=CTX, context_k=128, speed_tps=50.0,
                total_tps=400.0, sessions=8, per_session_tps=50.0,
                concurrency_bound="kv_vram", license="open", tags_str="general",
                moe=False)
    base.update(kw)
    return base


def _hover(fig, name) -> list:
    cd = fig.data[-1].customdata
    return next(r for r in cd if r[0] == name)


# ── #5: the estimator's documented error band, not "±30%" ───────────────────

def test_estimated_rows_quote_the_documented_error_band():
    df = pd.DataFrame([_row("Guess", kv_source="estimated"), _row("Known")])
    compat = build_local_compat(df, "Q4", vram_gb=32, ctx_tokens=CTX)
    scatter = build_local_scatter(df, vram_gb=32, quant="Q4", ctx_tokens=CTX)
    assert "±30%" not in compat.to_json() and "±30%" not in scatter.to_json()
    assert _hover(compat, "Guess")[10] == ESTIMATED_KV_NOTE
    assert _hover(compat, "Known")[10] == "published architecture"
    # The band is the one documented beside the estimator; they cannot drift.
    src = (Path(__file__).parents[1] / "data" / "local_models.py").read_text()
    median, p90 = re.search(r"median error (\d+)%, p90 (\d+)%", ESTIMATED_KV_NOTE).groups()
    assert re.search(rf"median[\s#]+{median}%, p90 {p90}%", src)


# ── #11: context labelled per row ───────────────────────────────────────────

def test_context_label_names_the_models_own_max_when_capped():
    df = pd.DataFrame([
        _row("Short", ctx_used=4000, context_k=4, quality=30.0),
        # 128,000 on the 131,072 choice: a units gap, not a cap.
        _row("Full", ctx_used=128000, context_k=128, quality=50.0),
    ])
    labels, capped = ctx_labels(df, CTX)
    assert labels == ["4k (model max)", "128k"] and capped == 1
    compat = build_local_compat(df, "Q4", vram_gb=32, ctx_tokens=CTX)
    assert _hover(compat, "Short")[11] == "4k (model max)"
    assert "(1 only up to their own shorter max)" in compat.layout.title.text
    scatter = build_local_scatter(df, vram_gb=32, quant="Q4", ctx_tokens=CTX)
    short_cd = [r for tr in scatter.data if tr.customdata is not None
                for r in tr.customdata if r[0] == "Short"]
    assert short_cd and short_cd[0][8] == "4k (model max)"
    assert "(1 priced at their own shorter max)" in scatter.layout.title.text


def test_nothing_capped_leaves_the_title_alone():
    df = pd.DataFrame([_row("Full", ctx_used=8192, context_k=128)])
    assert ctx_labels(df, 8192) == (["8k"], 0)
    assert "shorter max" not in build_local_compat(df, "Q4", 32, 8192).layout.title.text


# ── #24: latency-bound rows are not blamed on memory ────────────────────────

def test_latency_bound_single_session_names_the_floor_not_the_memory():
    df = pd.DataFrame([
        _row("Slow405", sessions=1, speed_tps=8.6, total_tps=8.6,
             concurrency_bound="latency"),
        _row("Full", sessions=1, concurrency_bound="kv_vram"),
    ])
    floor = SLO_FLOORS_TPS[DEFAULT_SLO]
    for mode in ("single", "throughput"):
        slow, full = speed_notes(df, mode)
        assert "one session is all this fits" not in slow
        assert f"below the {floor:g} tok/s per-session floor" in slow
        assert full.endswith("one session is all this fits")


def test_latency_bound_single_session_above_the_floor_is_not_below_it():
    # B=1 passes the floor and B=2 misses: the scan ends "latency" with a
    # single stream that is FAST enough. The hover must not call 12 tok/s
    # "below the 10 tok/s floor even alone", nor blame memory.
    floor = SLO_FLOORS_TPS[DEFAULT_SLO]
    df = pd.DataFrame([
        _row("K2", sessions=1, speed_tps=floor + 2, total_tps=floor + 2,
             concurrency_bound="latency"),
    ])
    for mode in ("single", "throughput"):
        (note,) = speed_notes(df, mode)
        assert "even alone" not in note
        assert "one session is all this fits" not in note
        assert f"a second session would drop each below the {floor:g} tok/s floor" in note


# ── #14 / #27: the scatter's VRAM label and axis ticks ──────────────────────

@pytest.mark.parametrize("vram", [32, 640, 1536, 2304])
def test_scatter_ticks_reach_the_end_of_the_axis(inputs, vram):
    fig = build_local_scatter(inputs["local"], vram_gb=vram, quant="Q4", ctx_tokens=CTX)
    ticks = list(fig.layout.xaxis.tickvals)
    axis_end = 10 ** fig.layout.xaxis.range[1]
    # The last labelled tick is within one doubling of the axis end, so no
    # stretch of the axis is wider than the gap between two ticks.
    assert ticks[-1] * 2 >= axis_end, (ticks[-1], axis_end)
    assert list(fig.layout.xaxis.ticktext) == [f"{v:g}" for v in ticks]
    assert any(a.text.strip() == f"{vram} GB" for a in fig.layout.annotations)
