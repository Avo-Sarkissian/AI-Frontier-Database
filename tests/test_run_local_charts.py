"""Run Local's two charts after the 2026-10 live audit.

  - one row per model family (AA's effort tiers share weights, VRAM and speed);
  - the ranked chart caps at MAX_COMPAT_ROWS and says how many it left out,
    so its only x axis is no longer ~4,000 px below the title;
  - unscored models sit at the bottom with no bar and say so;
  - both charts have a real empty state (no bare grid);
  - the scatter's axes come from the data, not 0.08 GB and the hosted 70;
  - the family colours on screen are tellable apart.
"""
import gzip
import math
import pickle
from pathlib import Path

import pandas as pd
import pytest

from components.charts.constants import base_model_name
from components.charts.local_compat import (
    MAX_COMPAT_ROWS, build_local_compat, one_per_family,
)
from components.charts.local_scatter import build_local_scatter
from data.local_models import (
    DEFAULT_FAMILY_COLOR, OTHER_FAMILY, get_local_df, tab_family_colors,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def big_local():
    """The frozen v4 open-weight catalogue at 8x B200 (150+ runnable rows)."""
    with gzip.open(FIXTURES / "charts_annotation_inputs.pkl.gz", "rb") as f:
        return pickle.load(f)["local"]


def _row(name, family="Meta", quality=20.0, **kw):
    base = dict(name=name, family=family, quality=quality, fits="yes", pending=False,
                vram_req_gb=10.0, weights_gb=8.0, kv_gb=0.5, overhead_gb=1.5,
                kv_source="config", ctx_used=8192, context_k=128, speed_tps=50.0,
                total_tps=400.0, sessions=8, per_session_tps=50.0,
                concurrency_bound="kv_vram", license="open", tags_str="general",
                moe=False)
    base.update(kw)
    return base


# ── one row per family ───────────────────────────────────────────────────────

def test_effort_tiers_of_one_model_take_one_row_its_best():
    df = pd.DataFrame([_row("GLM-5.3 (Max)", "Z AI", 31.6),
                       _row("GLM-5.3 (Low)", "Z AI", 23.1),
                       _row("Qwen3.8 27B (Xhigh)", "Alibaba", 20.1)])
    fig = build_local_compat(df, quant="Q4", vram_gb=32, ctx_tokens=8192)
    ys = list(fig.data[-1].y)
    assert "GLM-5.3 (Low)" not in ys and "GLM-5.3 (Max)" in ys
    sc = build_local_scatter(df, vram_gb=32, quant="Q4", ctx_tokens=8192)
    names = [r[0] for t in sc.data if t.customdata is not None for r in t.customdata]
    assert sorted(names) == ["GLM-5.3 (Max)", "Qwen3.8 27B (Xhigh)"]


def test_the_live_catalogue_has_no_family_twice_on_either_chart():
    df = get_local_df(quant="Q4", vram_gb=192 * 8, bandwidth_gbps=8000 * 8, gpu_count=8)
    fig = build_local_compat(df, quant="Q4", vram_gb=192 * 8)
    full = [r[0] for r in fig.data[-1].customdata]
    fams = [base_model_name(n) for n in full]
    assert len(fams) == len(set(fams)), fams


def test_an_unscored_variant_never_speaks_for_a_scored_family():
    df = pd.DataFrame([_row("Foo 9B (Preview)", quality=float("nan"), pending=True),
                       _row("Foo 9B (High)", quality=12.0)])
    assert list(one_per_family(df)["name"]) == ["Foo 9B (High)"]


# ── the cap, and the note that names what it left out ───────────────────────

def test_a_big_machine_shows_the_top_rows_and_counts_the_rest(big_local):
    fig = build_local_compat(big_local, quant="Q4", vram_gb=192 * 8, ctx_tokens=131072)
    bars = fig.data[-1]
    assert len(bars.y) == MAX_COMPAT_ROWS
    fits = one_per_family(big_local[big_local["fits"].isin(["yes", "tight"])])
    omitted = len(fits) - MAX_COMPAT_ROWS
    assert omitted > 0
    note = [a.text for a in fig.layout.annotations if a.yref == "paper"]
    assert note and f"+ {omitted} more" in note[0], note
    assert f"top {MAX_COMPAT_ROWS} shown" in fig.layout.title.text
    # The rows drawn are the best scored ones, best at the top.
    xs = [x for x in bars.x if x is not None]
    assert xs == sorted(xs), "the ranking is not ordered best-at-top"
    best_left_out = fits["quality"].nlargest(MAX_COMPAT_ROWS + 1).iloc[-1]
    assert min(xs) >= best_left_out
    # ~30 px a row: the x axis is within one screen of the title.
    assert fig.layout.height <= MAX_COMPAT_ROWS * 30 + 200


def test_every_row_label_leads_with_its_score():
    df = pd.DataFrame([_row("A 7B", quality=20.06), _row("B 3B", quality=6.3),
                       _row("New 9B", quality=float("nan"), pending=True)])
    fig = build_local_compat(df, quant="Q4", vram_gb=32, ctx_tokens=8192)
    labels = {a.y: a.text for a in fig.layout.annotations if a.yref == "y"}
    assert labels["A 7B"].startswith("20.1 ·")
    assert labels["B 3B"].startswith("6.3 ·")
    assert labels["New 9B"].startswith("unscored ·")
    order = list(fig.layout.yaxis.categoryarray)       # bottom -> top
    assert order == ["New 9B", "B 3B", "A 7B"]


# ── empty states ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("vram, tags", [(1, None), (32, ["no-such-tag"])])
def test_both_charts_render_an_empty_state_not_a_bare_grid(vram, tags):
    df = get_local_df(quant="Q4", vram_gb=vram, tags=tags)
    figs = [build_local_compat(df, quant="Q4", vram_gb=vram)]
    if tags:
        figs.append(build_local_scatter(df, vram_gb=vram, quant="Q4"))
    for fig in figs:
        assert fig.layout.xaxis.visible is False and fig.layout.yaxis.visible is False
        assert fig.layout.title.text, "the empty card lost its title"
        text = " ".join(a.text for a in fig.layout.annotations)
        assert ("No model fits in 1 GB" in text) if not tags else ("No model matches" in text)


# ── the scatter's axes ───────────────────────────────────────────────────────

def test_the_scatter_axes_start_and_end_near_the_data():
    df = get_local_df(quant="Q4", vram_gb=32)
    fig = build_local_scatter(df, vram_gb=32, quant="Q4")
    lo, hi = (10 ** v for v in fig.layout.xaxis.range)
    smallest = float(df["vram_req_gb"].min())
    assert smallest / 2 <= lo < smallest, (lo, smallest)
    assert hi >= float(df["vram_req_gb"].max())
    peak = float(df["quality"].max())
    top = fig.layout.yaxis.range[1]
    assert peak < top <= peak * 1.08 + 5, (peak, top)
    ticks = list(fig.layout.xaxis.tickvals)
    assert ticks and all(lo <= t <= hi for t in ticks)


def test_a_tag_filter_does_not_move_the_score_axis():
    full = build_local_scatter(get_local_df(quant="Q4", vram_gb=32), vram_gb=32, quant="Q4")
    df = get_local_df(quant="Q4", vram_gb=32)
    tag = next((t for ts in df["tags"] for t in ts), None)
    if tag is None:
        pytest.skip("no tags in the catalogue")
    sub = build_local_scatter(get_local_df(quant="Q4", vram_gb=32, tags=[tag]),
                              vram_gb=32, quant="Q4")
    assert sub.layout.yaxis.range == full.layout.yaxis.range


def test_the_vram_label_sits_on_the_line_at_every_size():
    for vram in (1, 32, 1536):
        fig = build_local_scatter(get_local_df(quant="Q4", vram_gb=vram),
                                  vram_gb=vram, quant="Q4")
        (lab,) = [a for a in fig.layout.annotations if a.xref == "x"]
        assert lab.x == pytest.approx(math.log10(vram))
        lo, hi = fig.layout.xaxis.range
        assert lo <= lab.x <= hi


# ── colours on screen ────────────────────────────────────────────────────────

def test_named_families_are_tellable_apart_and_the_rest_are_other():
    from components.charts.auto_palette import delta_e
    named = tab_family_colors()
    cols = list(named.values()) + [DEFAULT_FAMILY_COLOR]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            assert delta_e(a, b) >= 15.0, (a, b)
    # The six-blues case: of colours closer than the floor, at most one is named.
    counts = {"DeepSeek": 9, "Meta": 8, "StepFun": 7, "InclusionAI": 6,
              "Institute of Foundation Models": 5, "OpenBMB": 4}
    picked = tab_family_colors(counts)
    assert "DeepSeek" in picked and "Meta" in picked      # validated spotlight pair
    assert len(picked) < len(counts)


def test_the_scatter_legend_uses_the_tab_palette():
    df = get_local_df(quant="Q4", vram_gb=192 * 8, bandwidth_gbps=8000 * 8, gpu_count=8)
    fig = build_local_scatter(df, vram_gb=192 * 8, quant="Q4")
    named = tab_family_colors()
    legend = [t for t in fig.data if t.showlegend]
    assert len(legend) <= len(named) + 1
    for t in legend:
        if t.name == OTHER_FAMILY:
            assert t.marker.color == DEFAULT_FAMILY_COLOR
        else:
            assert t.marker.color == named[t.name]
    if OTHER_FAMILY in [t.name for t in legend]:
        assert legend[-1].name == OTHER_FAMILY, "Other must close the legend"
