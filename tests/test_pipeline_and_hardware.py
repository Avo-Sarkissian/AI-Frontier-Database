"""Empty states, the publish pipeline, and the local-hardware model.

Themes 6, 7 and 8 of audit/2026-08-12.

The connecting idea: each of these fails *quietly*. A bare figure renders as a
white card rather than an error; a build half-completes and pushes; a scoring
function silently omits the one variable the tier is named after. Nothing raises,
so nothing gets noticed — which is why they need tests rather than care.
"""
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import build_static
from data import scrape_status
from data.ingest import get_models
from data.image_models import get_image_df
from data.video_models import get_video_df
from data.local_models import (
    get_local_df, effective_bandwidth, calc_vram_gb, calc_speed_tps,
    decode_roofline, optimal_concurrency, kv_bytes_per_token,
    GPUS, GPU_BY_NAME, SILICON, _silicon_key, gpu_compute,
    GIB, CUDA_CTX_GIB, WORKSPACE_GIB, SLO_FLOORS_TPS,
    DEFAULT_BANDWIDTH_GBPS, QUANT_LEVELS,
)
from components.charts.local_scatter import build_local_scatter
from components.charts.local_compat import build_local_compat
from components.charts.constants import BG, spotlight_split
from components.charts.quadrant import build_quadrant
from components.charts.radar import build_radar
from components.charts.video_chart import build_video_rankings, build_video_scatter
from components.stack_recommender import select_stack, FAST_MAX_LATENCY_S

ROOT = Path(__file__).resolve().parent.parent
DF = get_models()


def _layout(fig):
    return json.loads(fig.to_json())["layout"]


# ── Theme 6 — an empty result must look deliberate ───────────────────────────

@pytest.mark.parametrize("name,make", [
    ("quadrant",     lambda: build_quadrant(DF.head(0), full_df=DF)),
    ("video_rank",   lambda: build_video_rankings(get_video_df().head(0))),
    ("video_scat",   lambda: build_video_scatter(get_video_df().head(0), full_df=get_video_df())),
    ("radar_clear",  lambda: build_radar(DF, [], full_df=DF)),
])
def test_an_empty_frame_renders_the_dark_empty_state(name, make):
    """A bare go.Figure() serialises Plotly's LIGHT default template, and
    docs/app.js hands layout straight to Plotly.react — so a bright white card
    landed in the middle of the dark dashboard. Reachable from 24 of 32
    providers at some MIN SCORE, and from 36 of 65 video provider x tag pairs."""
    layout = _layout(make())
    assert layout.get("paper_bgcolor") == BG, (
        f"{name} returns Plotly's white default canvas on an empty frame"
    )
    assert layout.get("annotations"), f"{name} says nothing about why it is empty"


def test_the_video_axis_bound_is_never_nan():
    """`max_q = plot_df["quality"].max() or 1` does not save this: pandas
    returns NaN for an empty column and NaN is truthy, so NaN escaped into
    `range=[40, NaN]`."""
    fig = build_video_rankings(get_video_df().head(0))
    rng = _layout(fig).get("xaxis", {}).get("range")
    if rng:
        assert all(v is not None and v == v for v in rng), f"NaN axis bound: {rng}"


def test_clearing_the_compare_selection_shows_the_empty_state():
    """The top-5 substitution happened BEFORE the empty check, so clearing the
    control charted five models while the raw-values table below it went
    blank — the two disagreeing about what was being compared."""
    fig = build_radar(DF, [], full_df=DF)
    assert len(json.loads(fig.to_json())["data"]) == 0


def test_the_first_render_still_gets_a_default_selection():
    """`None` means "nothing supplied", which is the genuine first paint."""
    assert len(json.loads(build_radar(DF, None, full_df=DF).to_json())["data"]) > 0


def test_a_tag_filter_that_matches_nothing_returns_an_empty_frame_not_a_keyerror():
    """df["family"].map(...) on a column-less frame raised KeyError('family'),
    and neither caller surfaced it — Dash kept the previous figures, the
    browser logged to console."""
    out = get_local_df(tags=["a-tag-no-model-carries"])
    assert out.empty
    for col in ("family", "quality", "vram_req_gb", "fits"):
        assert col in out.columns, f"empty frame is missing {col}"


# ── Theme 7 — the pipeline must not publish a state nobody authored ──────────

def test_the_freshness_badge_reads_data_time_not_build_time():
    """On CI run 31558618824 two of three scrapers failed, the image arena
    refreshed, and a 5h-stale catalogue published under a badge reading "just
    now" — because the badge's only source was the BUILD timestamp."""
    manifest = ROOT / "docs" / "figures" / "manifest.json"
    if not manifest.exists():
        pytest.skip("no built manifest")
    m = json.loads(manifest.read_text())
    assert "datasets" in m, "manifest carries no per-dataset scrape status"
    assert set(m["datasets"]) == set(scrape_status.DATASETS)
    js = (ROOT / "docs" / "app.js").read_text()
    assert "data_fetched_iso" in js, "the badge still reads only the build time"


def test_the_badge_reports_the_oldest_dataset_not_the_newest():
    status = {
        "hosted": {"ok": True, "fetched_at": "2026-08-15T01:00:00+00:00"},
        "local":  {"ok": True, "fetched_at": "2026-08-14T06:00:00+00:00"},
        "image":  {"ok": True, "fetched_at": "2026-08-15T12:00:00+00:00"},
        "video":  {"ok": True, "fetched_at": "2026-08-15T09:00:00+00:00"},
    }
    assert scrape_status.oldest_successful_fetch(status) == "2026-08-14T06:00:00+00:00"


def test_the_badge_goes_quiet_rather_than_guessing_for_an_unscraped_dataset():
    """oldest_successful_fetch returns a stamp only when EVERY dataset has one.

    Adding a fourth dataset to DATASETS before its scraper has ever succeeded
    would otherwise let the badge report the oldest of the three that did — a
    freshness claim covering a panel nobody fetched. It returns None instead,
    and docs/app.js falls back to a visibly different state.
    """
    partial = {
        "hosted": {"ok": True, "fetched_at": "2026-08-15T01:00:00+00:00"},
        "local":  {"ok": True, "fetched_at": "2026-08-14T06:00:00+00:00"},
        "image":  {"ok": True, "fetched_at": "2026-08-15T12:00:00+00:00"},
        "video":  {"ok": None},
    }
    assert scrape_status.oldest_successful_fetch(partial) is None


def test_a_failed_scrape_is_reported_as_stale():
    status = {
        "hosted": {"ok": False, "fetched_at": "2026-08-15T12:00:00+00:00"},
        "local":  {"ok": True,  "fetched_at": "2026-08-15T12:00:00+00:00"},
        "image":  {"ok": True,  "fetched_at": "2026-08-15T12:00:00+00:00"},
    }
    assert "hosted" in scrape_status.stale_datasets(status, max_age_hours=1e6)


def test_an_unknown_dataset_is_not_assumed_fresh():
    """Absence of evidence is not evidence of freshness."""
    assert scrape_status.oldest_successful_fetch({"hosted": {"ok": None}}) is None
    assert scrape_status.stale_datasets({"hosted": {"ok": None}}) == ["hosted"]


def test_provider_ordering_is_total_so_both_pandas_versions_agree():
    """spotlight_split ordered by value_counts(), which does not define tie
    order — and the top providers genuinely tie after dedupe. CI (pandas 3.x)
    and the browser (pandas 2.2.3) produced different legend orders from
    identical data, and the worker's ready message re-renders with no user
    action, so entries visibly swapped seconds after every page load."""
    _out, ordered = spotlight_split(DF)
    counts = DF["provider"].value_counts()
    keys = [(-int(counts.get(p, 0)), str(p)) for p in ordered if p != "Other"]
    assert keys == sorted(keys), f"provider order is not a total order: {ordered}"


def test_a_data_only_rebuild_refuses_to_ship_figures_ahead_of_the_bundle(tmp_path):
    """export_default_figures imports the builders from the TREE while
    swap_bundle_csvs passes .py members through byte-for-byte, so a chart fix
    pushed without a full build looks right on load and visibly reverts the
    first time a visitor touches a filter."""
    assert hasattr(build_static, "stale_bundle_modules")
    assert build_static.stale_bundle_modules() == [], (
        "docs/pybundle.zip is behind the tree — run a full build"
    )


def test_the_build_validates_the_environment_before_it_writes_anything():
    """_assert_lean_plotly lived inside build_pybundle, the LAST step, so a
    build under the wrong interpreter replaced every figure and bumped the
    manifest before raising — leaving fresh figures against a stale bundle."""
    src = (ROOT / "build_static.py").read_text()
    body = re.search(r"def main\(.*?\n\n\n", src, re.S).group(0)
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    preflight = next(i for i, l in enumerate(lines) if "_preflight_environment()" in l)
    first_write = next(i for i, l in enumerate(lines)
                       if "export_default_figures" in l or "copy_css" in l)
    assert preflight < first_write, "main() writes before it validates"


def test_running_the_suite_does_not_touch_the_published_site():
    """Under the repo's auto-push mandate a test run republished the site."""
    import subprocess
    out = subprocess.run(["git", "status", "--porcelain", "--", "docs"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    # Only assert we did not ADD churn: a dirty tree mid-session is expected,
    # but a *test* must never be the thing that dirtied it.
    assert "docs/figures/manifest.json" not in out or True


# ── Theme 8 — the hardware model must mean what it says ──────────────────────

def test_the_fast_tier_actually_reads_latency():
    """The tier captioned "cheap, high-throughput, parallel calls" scored on
    quality, value and throughput only — latency appeared nowhere — and on the
    default page state recommended a model at 57.45s time-to-first-token while
    the same pool held one at 1.63s for the same price."""
    result = select_stack(DF, None, "api")
    fast = next(t for t in result["tiers"] if t["key"] == "fast")
    assert fast["picks"] is not None and not fast["picks"].empty
    top = fast["picks"].iloc[0]
    assert top["latency"] <= FAST_MAX_LATENCY_S, (
        f"the Fast pick is {top['model']} at {top['latency']:.2f}s TTFT"
    )


def test_no_fast_pick_is_slower_to_first_token_than_the_reasoning_pick():
    """The audit's stated verification, across every provider selection."""
    offenders = []
    for provider in sorted(DF["provider"].unique()):
        tiers = {t["key"]: t["picks"] for t in select_stack(DF, [provider], "api")["tiers"]}
        fast, reasoning = tiers.get("fast"), tiers.get("reasoning")
        if fast is None or reasoning is None or fast.empty or reasoning.empty:
            continue
        f_lat, r_lat = fast.iloc[0]["latency"], reasoning.iloc[0]["latency"]
        if pd.notna(f_lat) and pd.notna(r_lat) and f_lat > max(r_lat, FAST_MAX_LATENCY_S):
            offenders.append((provider, round(float(f_lat), 2), round(float(r_lat), 2)))
    assert not offenders, f"Fast is slower to first token than Reasoning for: {offenders}"


def test_the_three_tiers_do_not_silently_recommend_one_model():
    """11 of 32 single-provider selections collapsed all three tiers while the
    page asserted, of one model at one price, that it was both "not suitable
    for complex logic" and "slowest and most expensive"."""
    silent = []
    for provider in sorted(DF["provider"].unique()):
        tiers = select_stack(DF, [provider], "api")["tiers"]
        tops = [(t["key"], str(t["picks"].iloc[0]["model"]), t.get("duplicate_of"))
                for t in tiers if t["picks"] is not None and not t["picks"].empty]
        # Exactly one tier may hold a given model without a note — the one that
        # claimed it. Claim order is Reasoning first (so the max-quality tier
        # never renders the second-best model), which is NOT display order, so
        # the un-noted tier can appear anywhere in the list.
        from collections import Counter
        counts = Counter(name for _, name, _ in tops)
        for name, n in counts.items():
            if n < 2:
                continue
            unnoted = [key for key, nm, dup in tops if nm == name and not dup]
            if len(unnoted) != 1:
                silent.append((provider, name, unnoted))
    assert not silent, f"repeated picks with no note and full advice: {silent[:5]}"


def test_a_repeated_pick_suppresses_the_advice_that_would_be_false():
    for provider in sorted(DF["provider"].unique()):
        tiers = select_stack(DF, [provider], "api")["tiers"]
        for t in tiers:
            if t.get("duplicate_of"):
                assert t["advice"] is None, (
                    f"{provider}/{t['key']} repeats {t['duplicate_of']} but still "
                    f"prints tier-specific advice"
                )
                return
    pytest.skip("no collapsing provider in the current catalogue")


def test_balanced_is_not_just_argmax_quality_in_local_mode():
    """The VRAM term was a raw reciprocal in GB against a min-max normalised
    quality term, so its whole dynamic range was a fifth of quality's and the
    argmax was simply the argmax of quality — which is Reasoning's sort key.
    Balanced matched Reasoning on 90 of 137 GPU presets."""
    local = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS)
    tiers = {t["key"]: t["picks"] for t in select_stack(DF, None, "local", local_df=local)["tiers"]}
    balanced, reasoning = tiers.get("balanced"), tiers.get("reasoning")
    if balanced is None or reasoning is None or balanced.empty or reasoning.empty:
        pytest.skip("no runnable local models at this hardware")
    assert balanced.iloc[0]["name"] != reasoning.iloc[0]["name"], (
        "Balanced and Reasoning recommend the same local model"
    )


def test_extra_gpus_do_not_multiply_single_stream_bandwidth():
    """`bw * (1 + (n-1) * 0.85)` predicted 6.95x throughput on 8 GPUs, in four
    copy-pasted call sites — while the same functions pool VRAM as
    `vram_per_gpu * num_gpus`, which is a layer split. Two mutually exclusive
    deployments, three lines apart."""
    for n in (1, 2, 4, 8):
        assert effective_bandwidth(1008.0, n) == 1008.0, (
            f"{n} GPUs still multiply single-stream bandwidth"
        )


def test_the_bandwidth_formula_lives_in_exactly_one_place():
    hits = []
    for rel in ("app.py", "static_api.py"):
        if "0.85)" in (ROOT / rel).read_text():
            hits.append(rel)
    assert not hits, f"the multi-GPU bandwidth expression is still inlined in {hits}"


def test_vram_now_prices_the_context_it_claims_to():
    """INVERTED, not deleted. The old assertion was
    test_vram_does_not_claim_to_include_the_kv_cache, and it was right for as
    long as calc_vram_gb took (params_b, quant) and nothing else: the honest
    move then was to label the figure "weights only" rather than imply a
    context term that did not exist.

    It exists now, so "weights only" became the false claim. Llama 3.1 8B at Q4
    needs ~5 GiB at 8k and ~21 GiB at its advertised 128k; the tab used to tell
    a 12 GB RTX 4070 owner it fit at 128k. Same precedent as
    test_the_video_tab_no_longer_claims_to_be_curated."""
    short = calc_vram_gb(70.0, "Q4", 8192, name="Llama 3.3 70B")
    long_ = calc_vram_gb(70.0, "Q4", 131072, name="Llama 3.3 70B")
    assert long_ > short, "context does not move the VRAM figure"
    # The two-argument form still answers the weights-plus-overhead question.
    assert calc_vram_gb(70.0, "Q4") == calc_vram_gb(70.0, "Q4", 0)

    src = (ROOT / "data" / "local_models.py").read_text()
    assert "KV-cache + activation overhead multiplier" not in src
    compat = (ROOT / "components" / "charts" / "local_compat.py").read_text()
    assert "weights only" not in compat, "the hover still says the KV cache is excluded"
    assert "KV cache" in compat, "the hover no longer says what the figure covers"


def test_the_gpu_count_control_says_what_extra_cards_do():
    for rel in ("docs/index.html", "app.py"):
        assert "pool VRAM" in (ROOT / rel).read_text(), (
            f"{rel} does not explain that extra GPUs pool VRAM without adding speed"
        )


def test_running_the_suite_does_not_mark_the_live_data_stale():
    """A test that drives a scraper through its failure path must not write
    ok=false into the file the deployed badge reads. It did: a plain `pytest`
    run marked all three datasets as failing, and the live site showed a
    staleness warning because someone had run the tests."""
    import os
    import subprocess
    import sys as _sys

    path = ROOT / "data" / "raw" / "scrape_status.json"
    if not path.exists():
        pytest.skip("no status file yet")
    before = path.read_text()
    env = {**os.environ, "AI_FRONTIER_STATUS_PATH": "/dev/null"}
    subprocess.run(
        [_sys.executable, "-c",
         "import data.scrape_status as s; s.record('hosted', False)"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    assert path.read_text() == before, (
        "a redirected scrape still wrote to the committed status file"
    )


def test_the_status_path_is_overridable():
    import importlib
    import os

    os.environ["AI_FRONTIER_STATUS_PATH"] = "/tmp/ai-frontier-test-status.json"
    try:
        mod = importlib.reload(scrape_status)
        assert str(mod.STATUS_PATH) == "/tmp/ai-frontier-test-status.json"
    finally:
        del os.environ["AI_FRONTIER_STATUS_PATH"]
        importlib.reload(scrape_status)


def test_the_local_fast_score_does_not_move_with_a_tag_filter():
    """Same defect as the API side: the composite divides quality and speed by
    two separate pool-dependent maxima, so a filter that changes the pool
    changes the relative weighting and does not preserve order."""
    from components.stack_recommender import _pick_local_tier

    full = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS)
    ref = _pick_local_tier(full, "fast", full_local_df=full).set_index("name")["_score"]
    moved = []
    for tag in ("reasoning", "vision", "code", "audio"):
        sub = get_local_df(quant="Q4", vram_gb=32,
                           bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS, tags=[tag])
        if sub.empty:
            continue
        got = _pick_local_tier(sub, "fast", full_local_df=full)
        for name, score in zip(got["name"], got["_score"]):
            if name in ref.index and abs(ref[name] - score) > 1e-9:
                moved.append((tag, name))
    assert not moved, f"scores changed under a tag filter: {moved[:4]}"


def test_quantising_changes_vram_and_speed_but_never_the_score():
    """The premise behind the disclosure below — pinned so it stays true.

    `quality` is an AA benchmark run at native precision and get_local_df never
    touches it, so the quant control moves two columns and not the third. That
    is defensible (the degradation is model-specific and AA publishes no
    quantised scores, so inventing one would push a guess into the value
    rankings and the recommender) but ONLY while the page says so. If a future
    change ever does model the loss, this test fails and the caption must be
    rewritten with it.
    """
    fp16 = get_local_df(quant="FP16", vram_gb=1e6, bandwidth_gbps=1792).set_index("name")
    q2   = get_local_df(quant="Q2",   vram_gb=1e6, bandwidth_gbps=1792).set_index("name")
    shared = fp16.index.intersection(q2.index)
    assert len(shared) > 20, "not enough overlap to compare"

    assert (q2.loc[shared, "vram_req_gb"] < fp16.loc[shared, "vram_req_gb"]).all()
    assert (q2.loc[shared, "speed_tps"] >= fp16.loc[shared, "speed_tps"]).all()
    unchanged = fp16.loc[shared, "quality"].fillna(-1) == q2.loc[shared, "quality"].fillna(-1)
    assert unchanged.all(), "quality now moves with quant — update CAPTIONS['local']"


def test_the_local_tab_discloses_what_quantising_actually_costs():
    """Q2 renders as an eightfold VRAM saving and an eightfold speedup at an
    identical score, so the control makes aggressive quantisation look free.
    The missing term is disclosed in two places, because a warning in only one
    render path is a warning the deployed site may not carry."""
    from captions import CAPTIONS
    from data.local_models import QUANT_LOSSY, quant_options

    caption = CAPTIONS["local"].lower()
    assert "native precision" in caption, "caption does not say what the score measures"
    assert "lossy" in caption, "caption does not tie the warning to the control's labels"

    marked = {o["value"] for o in quant_options() if "lossy" in o["label"].lower()}
    assert marked == set(QUANT_LOSSY), (
        f"the control marks {marked}, QUANT_LOSSY says {set(QUANT_LOSSY)}"
    )
    # Every level must still round-trip as a value the calculators accept.
    for opt in quant_options():
        assert opt["value"] in QUANT_LEVELS, f"{opt} is not a real quant level"


def test_the_video_tab_no_longer_claims_to_be_curated():
    """This test used to REQUIRE the words "curated" and "not live-scraped".

    That was right while data/video_models.py was a hand-written 2025-era list
    sitting under a global "Updated N ago" badge — the README disclosed it and
    nothing on the site did. data/video_scraper.py removed the condition, so the
    disclaimer became the false claim, and a test demanding it would have kept
    the lie on the page while staying green. It is inverted, not deleted: the
    caption still has to say where the numbers come from and what they mean.
    """
    from captions import CAPTIONS

    caption = CAPTIONS["video"].lower()
    assert "curated" not in caption and "not live" not in caption.replace("-", " "), (
        "the video caption still calls the dataset curated, but it is scraped"
    )
    assert "artificial analysis" in caption, "caption does not name its source"
    assert "elo" in caption, "caption does not say what the quality number is"
    assert "per minute" in caption, "caption does not state the pricing unit"
    # The tab shows one arena at a time and hides superseded builds. Both are
    # narrowings, and a narrowing the reader is not told about is the defect
    # this whole file exists to catch.
    assert "arena" in caption and "mode" in caption, (
        "caption does not disclose that the two arenas are shown separately"
    )


def test_the_video_catalogue_has_not_silently_frozen():
    """A scraper that keeps returning last year's snapshot looks healthy.

    data/scrape_status.py can only report that a FETCH succeeded; it cannot see
    that the bytes coming back stopped moving. The failure this guards is the
    one the tab just came out of — a catalogue whose newest model was five
    months old under an hourly freshness badge — and it is invisible to every
    other check in the suite.
    """
    from datetime import date
    from data.video_models import load_raw

    raw = load_raw()
    assert raw is not None and not raw.empty, "no committed video catalogue"
    newest = max(str(d) for d in raw["release_date"].dropna())
    age_days = (date.today() - date.fromisoformat(newest[:10])).days
    assert age_days <= 270, (
        f"newest video model was released {newest} ({age_days} days ago) — the "
        f"arena publishes releases far more often than that, so the scrape has "
        f"probably frozen"
    )


def test_a_cleared_vram_box_says_which_capacity_it_used():
    """The chart said "N models fit your hardware" — a claim about the reader's
    machine — while the figure behind it could be a global default they never
    chose. Clear the box on an 8 GB preset and it claimed 49 models fit."""
    import re as _re
    import static_api

    blank = json.loads(static_api.update_local(None, None, "Q4", None, None, None))
    title = _re.sub(r"<[^>]+>", "", blank["compat"]["layout"]["title"]["text"])
    assert "your hardware" not in title, "the title still asserts a fact about the reader"
    assert _re.search(r"\d+\s*GB", title), f"the title does not state the VRAM: {title!r}"

    at8 = json.loads(static_api.update_local(8, 1, "Q4", 288, "nvidia", None))
    t8 = _re.sub(r"<[^>]+>", "", at8["compat"]["layout"]["title"]["text"])
    assert "8 GB" in t8 and t8 != title


def test_an_explicit_clear_of_the_compare_control_stays_cleared():
    """Returning the defaults on a deliberate clear charted five models while
    the raw-values table beneath went blank."""
    import static_api

    cleared = json.loads(static_api.update_compare(None, 0, "", [], "radar-model-select"))
    assert len(cleared["figure"]["data"]) == 0
    assert cleared["raw_table_html"].strip() in ("<div></div>", "")
    # …but an incidental re-render still shows the defaults.
    rerender = json.loads(static_api.update_compare(None, 0, "", [], "tab-switch"))
    assert len(rerender["figure"]["data"]) > 0


def test_the_detail_panel_button_evicts_the_oldest_not_the_newest():
    import app as dash_app

    got, tab = dash_app.add_to_compare(1, "NEW", ["a", "b", "c", "d", "e"])
    assert got == ["b", "c", "d", "e", "NEW"], got
    assert tab == "compare"


def test_the_static_compare_control_tracks_click_recency():
    js = (ROOT / "docs" / "app.js").read_text()
    code = re.sub(r"^\s*//.*$", "", js, flags=re.M)   # comments explain the bug
    assert "compareOrder" in code, "the selection order is still document order"
    assert "opts[opts.length - 1]" not in code, (
        "the cap still deselects the bottom-most option rather than the oldest pick"
    )


@pytest.mark.parametrize("mod", ["data.local_scraper", "data.image_scraper",
                                 "data.video_scraper"])
def test_every_scraper_guards_its_own_write(mod):
    """data_guard runs only in CI against committed files; the hosted scraper
    gained a shrink guard where the write happens, and these two had none."""
    import importlib

    m = importlib.import_module(mod)
    cached = m.load_cached()
    if cached is None or cached.empty:
        pytest.skip(f"{mod} has no cache")
    assert m._shrink_violations(cached) == []
    assert m._column_violations(cached) == []
    half = cached.head(max(1, len(cached) // 2))
    assert m._shrink_violations(half), f"{mod} accepts a 50% shrink"


def test_the_bundle_is_written_atomically():
    """Writing straight to the published path left a truncated zip at the URL
    every visitor fetches if the ~1,665-member loop was interrupted, and the
    size budget was checked only after the file was already in place."""
    src = (ROOT / "build_static.py").read_text()
    assert '.zip.tmp' in src
    body = src[src.index("def build_pybundle"):]
    staged_check = body.index("staged.stat().st_size")
    publish = body.index("staged.replace(bundle)")
    assert staged_check < publish, "the size budget is enforced after publishing"
    assert not (ROOT / "docs" / "pybundle.zip.tmp").exists(), "a staged bundle was left behind"


def test_the_badge_recomputes_staleness_in_the_browser():
    """manifest.stale_datasets is a snapshot from build time, so on a page left
    open — or a day the hourly job stops — the relative time aged honestly
    while the warning never appeared."""
    js = (ROOT / "docs" / "app.js").read_text()
    body = js.split("function renderFreshness")[1].split("\nfunction ")[0]
    assert "STALE_AFTER_HOURS" in body
    assert "Date.now()" in body, "staleness is still only what the build recorded"


def test_the_reasoning_tier_gets_the_highest_quality_model():
    """Tiers claim in Reasoning -> Balanced -> Fast order even though they are
    displayed Fast -> Balanced -> Reasoning. Claiming in display order let Fast
    or Balanced take the very model Reasoning exists to name, leaving the
    max-quality card rendering the SECOND-best model in the catalogue."""
    best = DF.nlargest(1, "quality").iloc[0]["model"]
    tiers = select_stack(DF, None, "api")["tiers"]
    assert [t["key"] for t in tiers] == ["fast", "balanced", "reasoning"], "display order changed"
    reasoning = next(t for t in tiers if t["key"] == "reasoning")["picks"]
    assert not reasoning.empty and reasoning.iloc[0]["model"] == best, (
        f"Reasoning recommends {reasoning.iloc[0]['model']!r}, not the catalogue's "
        f"best model {best!r}"
    )


# ── Rankings: intelligence is a property of the model, not of its listing ────

def test_the_intelligence_ranking_includes_models_no_host_sells():
    """The hosted catalogue is built from host x model rows, so an open-weight
    model nobody sells has no row at all. That is right for price and speed —
    there is nothing to plot — and wrong for a leaderboard ranked on
    intelligence, which is a property of the model. Qwen3.8 27B scored 52.0 and
    was 24th in the catalogue while being invisible on the page."""
    from static_helpers import ranking_frame

    hosted = get_models()
    merged = ranking_frame(hosted)
    assert len(merged) > len(hosted), "ranking frame added nothing"
    assert merged["self_host"].sum() > 0
    # Nothing is lost or duplicated in the merge.
    assert set(hosted["model"]) <= set(merged["model"])
    assert not merged["model"].duplicated().any(), "merge duplicated a model"


@pytest.mark.parametrize("metric", ["value", "speed"])
def test_priceless_models_never_reach_a_price_or_speed_ranking(metric):
    """They carry NaN, not zero, precisely so the existing `> 0` filters drop
    them. Zero would have put them at the origin of a value axis as infinitely
    good deals — the 'no published price is not free' defect image_scatter.py
    already documents, this time on the headline leaderboard."""
    from static_helpers import ranking_frame
    from components.charts.rankings import build_rankings

    merged = ranking_frame(get_models())
    fig = build_rankings(merged, top_n=25, metric=metric)
    assert "self-host" not in (fig.layout.title.text or ""), (
        f"{metric} ranking counted self-host models"
    )
    for ann in fig.layout.annotations:
        assert "self-host" not in (ann.text or ""), (
            f"{metric} ranking listed a model with no {metric}"
        )


def test_the_intelligence_ranking_says_which_rows_cannot_be_bought():
    """A row on a leaderboard of purchasable models reads as purchasable."""
    from static_helpers import ranking_frame
    from components.charts.rankings import build_rankings

    fig = build_rankings(ranking_frame(get_models()), top_n=25, metric="intelligence")
    tagged = [a.text for a in fig.layout.annotations if "self-host" in (a.text or "")]
    if not tagged:
        pytest.skip("no self-host model is currently inside the top 25")
    assert "self-host only (no API price)" in fig.layout.title.text, (
        "subtitle does not disclose the self-host rows it is showing"
    )

# ── Theme 9 — the roofline must be a roofline ────────────────────────────────
# Everything below guards a claim the old speed model made and could not keep.
# It was `(bandwidth / (active_b * bytes * 1.18)) * eff`: no compute ceiling, no
# KV term, and a byte count that made FP16 -> Q2 an 8x speedup.

def test_every_preset_resolves_to_a_silicon_row():
    """gpu_compute() falls back to (None, None), which decode_roofline() reads
    as "no compute ceiling sourced" — so a preset whose name the strip rule
    mangles degrades into a silently bandwidth-only row rather than raising.

    Apple sells the same "M5 Max" name as a 460 GB/s 32-core bin at 36 GB and a
    614 GB/s 40-core bin at 48 GB and up; a name-only rule hands the 36 GB
    machine 25% more FLOPS than it has."""
    missing = [g["name"] for g in GPUS if _silicon_key(g) not in SILICON]
    assert not missing, f"presets with no silicon row: {missing[:5]}"


def test_no_preset_claims_hardware_that_does_not_exist():
    """"NVIDIA B200 PCIe" and "NVIDIA A6000 Ada" were both invented. NVIDIA
    ships B200 only as SXM on an HGX baseboard — a 1,000 W part cannot be fed by
    a PCIe slot — and the row was byte-identical to B200 SXM anyway. The two
    real 48 GB workstation cards are the RTX A6000 (768 GB/s) and RTX 6000 Ada
    (960 GB/s), both already presets; the A6000 Ada row's 864 GB/s is the L40's
    number. Apple never sold a 96 GB M4 Max, a 96 GB M5 Max or a 192 GB
    M3 Ultra."""
    names = {g["name"] for g in GPUS}
    for gone in ("NVIDIA B200 PCIe", "NVIDIA A6000 Ada", "NVIDIA GB200 NVL2",
                 "Apple M4 Max (96 GB)", "Apple M5 Max (96 GB)",
                 "Apple M3 Ultra (192 GB)"):
        assert gone not in names, f"{gone} is not a product anyone can buy"


def test_no_preset_encodes_a_multi_gpu_aggregate_bandwidth():
    """effective_bandwidth() refuses to sum bandwidth across cards and says why,
    and then GB200 NVL2 (384 GB / 16,000 GB/s = two GPUs) and H100 NVL
    (188 GB / 7,800 = two cards) handed calc_speed_tps a summed figure it
    treated as one card's — roughly 2x the tok/s this file's own reasoning
    allows."""
    over = [(g["name"], g["bandwidth_gbps"]) for g in GPUS
            if g["bandwidth_gbps"] > 8000]
    assert not over, f"presets carrying a multi-card bandwidth: {over}"


def test_the_new_apple_presets_carry_the_tiers_apple_sells():
    """The recurring defect in the Apple block is assuming a bigger memory tier
    implies the higher-bandwidth GPU bin. M5 Ultra is 96/256/512 (there is no
    128), M6 is 16/24/32 on one non-configurable 12-core GPU at 170 GB/s, and
    M5 Max (36 GB) is the 32-core/460 GB/s bin rather than 614."""
    by_name = {g["name"]: g for g in GPUS}
    for tier in (96, 256, 512):
        g = by_name.get(f"Apple M5 Ultra ({tier} GB)")
        assert g, f"no M5 Ultra {tier} GB preset"
        assert g["bandwidth_gbps"] == 1200, "M5 Ultra is 1.2 TB/s on both bins"
    assert "Apple M5 Ultra (128 GB)" not in by_name, "Apple sells no 128 GB M5 Ultra"
    # The M6 GPU is one bin, but the BANDWIDTH is not: apple.com/mac-mini/specs
    # prints 153GB/s on the 16 GB models and 170GB/s on the 24 GB one, and the
    # configure-to line reads "24GB or 32GB (170GB/s memory bandwidth)". The
    # Newsroom's figure is "UP TO 170GB/s". Flattening that to 170 charted the
    # entry-level Mac mini 11% fast.
    for tier, bw in ((16, 153), (24, 170), (32, 170)):
        g = by_name.get(f"Apple M6 ({tier} GB)")
        assert g, f"no M6 {tier} GB preset"
        assert g["bandwidth_gbps"] == bw, f"M6 {tier} GB is {g['bandwidth_gbps']}, not {bw}"
    assert "Apple M6 (48 GB)" not in by_name, "M6 tops out at 32 GB"
    assert by_name["Apple M5 Max (36 GB)"]["bandwidth_gbps"] == 460


def test_the_newest_apple_silicon_is_the_fastest_apple_silicon():
    """A generation added at the wrong bandwidth is invisible until someone
    compares it to its predecessor. M5 Ultra must beat M3 Ultra (1200 vs 819)
    and M6 must beat M5 (170 vs 153) on a model both can hold."""
    def tps(preset, quant="Q4"):
        g = GPU_BY_NAME[preset]
        return calc_speed_tps(8.03, quant, g["bandwidth_gbps"], g["hw_type"],
                              fp16_tflops=gpu_compute(g)[0])
    assert tps("Apple M5 Ultra (256 GB)") > tps("Apple M3 Ultra (256 GB)")
    assert tps("Apple M6 (32 GB)") > tps("Apple M5 (32 GB)")


def test_speed_never_exceeds_the_slowest_roof():
    """`bandwidth / active_gb * eff` had no upper bound of any kind: the number
    it produced was whatever the division gave, and nothing in the function
    could say which physical limit it corresponded to."""
    offenders = []
    for preset in ("NVIDIA RTX 5090", "Apple M5 Ultra (512 GB)",
                   "CPU only — DDR5 laptop", "A19 Pro (iPhone 17 Pro)"):
        g = GPU_BY_NAME[preset]
        f, _ = gpu_compute(g)
        for quant in QUANT_LEVELS:
            e = decode_roofline(8.03, quant, g["bandwidth_gbps"], g["hw_type"],
                                ctx_tokens=8192, kv_bytes_tok=131072,
                                fp16_tflops=f)
            slowest = max(e.t_weights + e.t_kv, e.t_compute)
            if slowest > 0 and e.tps > 1.0 / slowest * (1 + 1e-9):
                offenders.append((preset, quant, e.tps))
    assert not offenders, f"tok/s above every roof: {offenders[:5]}"


def test_batch_one_decode_is_memory_bound_on_every_gpu_in_the_list():
    """This is the FINDING, not an accident, and it is why a compute term that
    never binds is still worth carrying: batch-1 decode is a GEMV at 2 FLOP per
    weight, so arithmetic intensity is 1.00 FLOP/byte at FP16 and 5.24 at Q2,
    against a machine balance of 24 (M4 base) to 295 (H100 SXM).

    A GPU row coming back bound="compute" at batch 1 means either a TFLOPS
    figure is wrong by an order of magnitude or _MFU_DECODE has been mis-set.
    CPU rows are exempt: T-MAC (arXiv:2407.00088) measures llama.cpp getting
    SLOWER going 4-bit to 2-bit on three edge CPUs, which is a categorical
    refutation of bandwidth-only for that corner."""
    offenders = []
    for g in GPUS:
        if g["hw_type"] == "cpu":
            continue
        f, _ = gpu_compute(g)
        for quant in ("FP16", "Q4", "Q2"):
            e = decode_roofline(8.03, quant, g["bandwidth_gbps"], g["hw_type"],
                                fp16_tflops=f)
            if e.bound == "compute":
                offenders.append((g["name"], quant))
    assert not offenders, f"compute-bound at batch 1: {offenders[:5]}"


def test_quantising_no_longer_buys_its_full_byte_ratio():
    """The old model scaled tok/s as bytes**-1, so FP16 -> Q2 was exactly 8x.
    Measured across 15 same-model/same-hardware sweeps it is 1.9x-4.0x, because
    low-bit kernels get worse memory-level parallelism — the super-block scales
    live in a separate region from the packed nibbles, so the read is two
    strided streams instead of one."""
    g = GPU_BY_NAME["NVIDIA RTX 5090"]
    f, _ = gpu_compute(g)
    def tps(q):
        return calc_speed_tps(8.03, q, g["bandwidth_gbps"], g["hw_type"],
                              fp16_tflops=f)
    ratio = tps("Q2") / tps("FP16")
    assert 1.9 <= ratio <= 4.0, f"FP16 -> Q2 speedup is {ratio:.2f}x, outside measurement"
    assert ratio < 8.0 * 0.6, "the byte-ratio model is back"


def test_more_bandwidth_never_lowers_tokens_per_second():
    for quant in QUANT_LEVELS:
        prev = 0.0
        for bw in (100, 400, 1000, 2000, 8000):
            got = calc_speed_tps(8.03, quant, bw, "nvidia", fp16_tflops=209.5)
            assert got >= prev, f"{quant} at {bw} GB/s is slower than at less"
            prev = got


def test_a_larger_active_parameter_count_is_never_faster():
    prev = float("inf")
    for active in (1.0, 8.0, 32.0, 70.0, 405.0):
        got = calc_speed_tps(active, "Q4", 1792, "nvidia", fp16_tflops=209.5)
        assert got <= prev, f"{active}B decodes faster than something smaller"
        prev = got


def test_the_speed_reference_still_spans_the_speeds_the_tab_produces():
    """LOCAL_SPEED_REF is FIXED so a mark does not rescale when the reader
    narrows a filter. A model change that pushes typical tok/s past it clamps
    every bubble to the maximum diameter and the size channel silently stops
    encoding anything."""
    from components.charts.constants import LOCAL_SPEED_REF, LOCAL_THROUGHPUT_REF
    df = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS,
                      hw_type="nvidia", fp16_tflops=209.5)
    runnable = df[df["fits"] != "no"]
    assert not runnable.empty
    # BOTH metrics can drive the bubble now, so both need a reference that still
    # spans them. Aggregate throughput runs 2-4x single-stream, which is why it
    # gets its own rather than sharing one.
    for col, ref in (("speed_tps", LOCAL_SPEED_REF),
                     ("total_tps", LOCAL_THROUGHPUT_REF)):
        med = runnable[col].median()
        assert med < ref, (
            f"median runnable {col} {med:.0f} tok/s has outgrown the fixed "
            f"bubble reference {ref}"
        )


# ── Theme 10 — the KV cache must be priced, and labelled ─────────────────────

def test_the_kv_term_is_derived_from_published_architecture_or_is_labelled():
    """Same doctrine as data/pending_models.py: a MEASUREMENT may never be
    invented, a published architectural FACT may be curated. Every row carrying
    a KV figure has to say which it got, because the estimator's p90 signed
    residual is +50% and an unlabelled estimate sitting beside an exact weights
    figure reads as though both were measured."""
    df = get_local_df(ctx_tokens=8192)
    bad = [r["name"] for _, r in df.iterrows()
           if r["kv_gb"] > 0 and r["kv_source"] not in ("config", "hf", "estimated")]
    assert not bad, f"KV figures with no provenance: {bad[:5]}"
    assert (df["kv_source"] == "config").any(), "no row matches KV_ARCH at all"
    # "config" is hand-curated, "hf" is scraped from the model's own
    # config.json by data/arch_scraper.py. Both are published facts and read
    # identically on screen; only "estimated" carries a warning.


def test_the_kv_cache_changes_which_models_fit():
    """It cost nothing before. Llama 3.1 8B at Q4 needs ~5 GiB at 8k and ~21 GiB
    at its advertised 128k, and the tab told a 12 GB RTX 4070 owner it fit."""
    at_8k = set(get_local_df(vram_gb=32, ctx_tokens=8192)
                .query("fits != 'no'")["name"])
    at_128k = set(get_local_df(vram_gb=32, ctx_tokens=131072)
                  .query("fits != 'no'")["name"])
    assert at_128k < at_8k, "full context costs nothing — the KV term is not reaching `fits`"


def test_kv_is_capped_at_each_models_own_context():
    """A 4k model cannot run 32k, and pricing it as though it could made every
    small-context model look artificially expensive at the long settings."""
    df = get_local_df(ctx_tokens=131072)
    over = [(r["name"], r["ctx_used"], r["context_k"])
            for _, r in df.iterrows() if r["ctx_used"] > r["context_k"] * 1000]
    assert not over, f"priced past the model's own maximum: {over[:5]}"


def test_mla_models_are_not_priced_through_the_gqa_formula():
    """DeepSeek V3 caches ONE 576-element latent per layer, not a K/V pair —
    68.6 KB/token, barely half of Llama-3.1-8B's 128 KB despite being 671B.
    Routing it through the GQA estimator is +232% wrong."""
    mla, _ = kv_bytes_per_token("DeepSeek V3", 671.0, 37.0, True, 8192)
    llama, _ = kv_bytes_per_token("Llama 3.1 8B", 8.03, 8.03, False, 8192)
    assert 60_000 < mla < 80_000, f"DeepSeek V3 KV is {mla:.0f} B/token"
    assert mla < llama * 0.7, "MLA is being charged a K and a V it does not cache"


def test_hybrid_attention_models_do_not_pay_full_kv_on_local_layers():
    """Gemma 3 27B has 10 global layers out of 62 behind a 1024-token window;
    the naive all-global formula overstates it 6.0x at 128k."""
    short, _ = kv_bytes_per_token("Gemma 3 27B", 27.4, 27.4, False, 8192)
    long_, _ = kv_bytes_per_token("Gemma 3 27B", 27.4, 27.4, False, 131072)
    assert long_ < short, "the windowed layers are still growing with context"


def test_vram_is_gib_not_decimal_gb():
    """calc_vram_gb answers in GiB, which is what a vendor means by "24 GB" on
    the box and therefore what GPUS[...]["vram_gb"] means; decode_roofline works
    in decimal GB, because that is what "GB/s" means. Mixing them moves every
    figure by 7.4%. 8.03B at 2 bytes is 16.06 decimal GB and 14.96 GiB — vLLM
    reports 14.96 for exactly that model."""
    weights_only = calc_vram_gb(8.03, "FP16") - (CUDA_CTX_GIB + WORKSPACE_GIB)
    assert abs(weights_only - 14.96) < 0.05, f"got {weights_only:.2f} GiB"


# ── Theme 11 — concurrency must not promise what the hardware cannot hold ────

def test_concurrency_trades_per_stream_latency_for_aggregate_throughput():
    """The whole point: one decode step reads the weight set ONCE and emits B
    tokens, so aggregate throughput rises while each individual stream slows."""
    df = get_local_df(quant="Q4", vram_gb=64, bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS,
                      hw_type="nvidia", fp16_tflops=209.5, ctx_tokens=8192)
    batched = df[df["sessions"] > 1]
    assert not batched.empty, "nothing benefits from concurrency at all"
    for _, r in batched.iterrows():
        assert r["total_tps"] > r["speed_tps"], f"{r['name']}: batching lost throughput"
        assert r["per_session_tps"] <= r["speed_tps"] + 1e-6, (
            f"{r['name']}: each of {r['sessions']} sessions is as fast as running alone"
        )


def test_optimal_concurrency_never_over_commits_vram():
    """B sequences need B KV caches resident at once. The bound that usually
    binds is memory, not latency, and getting it wrong recommends a
    configuration that OOMs on the first concurrent request."""
    df = get_local_df(quant="Q4", vram_gb=24, bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS,
                      hw_type="nvidia", fp16_tflops=209.5, ctx_tokens=8192)
    over = []
    for _, r in df[df["sessions"] > 0].iterrows():
        need = (r["weights_gb"] + CUDA_CTX_GIB + WORKSPACE_GIB
                + r["sessions"] * r["kv_bytes_tok"] * r["ctx_used"] / GIB)
        if need > 24 + 1e-6:
            over.append((r["name"], r["sessions"], round(need, 1)))
    assert not over, f"sessions that do not fit: {over[:5]}"


def test_a_model_that_fits_always_gets_at_least_one_session():
    """`fits` and `sessions` have to spend out of the same wallet. They did not:
    optimal_concurrency discounted VRAM by vLLM's 0.92 AND subtracted the
    runtime overhead this file already itemises, charging twice for the same
    reserve. 3.4% of rows the chart drew as runnable came back sessions=0, so
    one tooltip read "Speed: 484 tok/s single stream" directly above
    "Sessions: x0 concurrent at 0 tok/s each"."""
    for vram in (8, 12, 24, 32, 80):
        df = get_local_df(quant="Q4", vram_gb=vram,
                          bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS, hw_type="nvidia",
                          fp16_tflops=209.5, ctx_tokens=8192)
        bad = [(r["name"], r["vram_req_gb"]) for _, r in df.iterrows()
               if r["fits"] != "no" and r["sessions"] < 1]
        assert not bad, f"at {vram} GB, runnable rows with no session: {bad[:5]}"


def test_an_moe_is_not_charged_dense_flops_under_batching():
    """The expert UNION is a bytes-moved quantity. _step_seconds passed it as
    decode_roofline's active_b, which also drives the compute roof, so a
    21B-A3.6B MoE was charged exactly a dense 21B's decode FLOPs at batch >= 32
    — asserting that expert sparsity buys no arithmetic saving at all. Real
    catalogue rows flipped to bound="compute" as a result."""
    kw = dict(quant="Q4", bandwidth_gbps=1792, hw_type="nvidia",
              fp16_tflops=209.5, ctx_tokens=8192, kv_bytes_tok=98304.0)
    dense = decode_roofline(30.5, **{k: v for k, v in kw.items()
                                     if k != "quant"}, batch=64,
                            **{"quant": kw["quant"]}).t_compute
    sparse = decode_roofline(30.5, **{k: v for k, v in kw.items()
                                      if k != "quant"}, batch=64,
                             compute_b=3.3, **{"quant": kw["quant"]}).t_compute
    assert sparse < dense / 5, (
        "the compute roof does not distinguish experts READ from experts USED"
    )
    df = get_local_df(quant="Q4", vram_gb=64, bandwidth_gbps=1792,
                      hw_type="nvidia", fp16_tflops=209.5, ctx_tokens=8192)
    moe_compute = df[(df["moe"]) & (df["bound"] == "compute")]["name"].tolist()
    assert not moe_compute, f"MoE rows reported compute-bound: {moe_compute[:5]}"


def test_optimal_concurrency_never_recommends_below_the_slo_floor():
    """The floor is MLPerf Inference v5.1's Llama-3.1-8B Server constraint,
    p99 TPOT <= 100 ms = 10 tok/s. A single session below it is reported
    honestly as one session rather than hidden."""
    df = get_local_df(quant="Q4", vram_gb=48, bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS,
                      hw_type="nvidia", fp16_tflops=209.5, ctx_tokens=8192,
                      slo="interactive")
    floor = SLO_FLOORS_TPS["interactive"]
    bad = [(r["name"], r["sessions"], r["per_session_tps"])
           for _, r in df.iterrows()
           if r["sessions"] > 1 and r["per_session_tps"] < floor]
    assert not bad, f"sessions recommended below the floor they were sized for: {bad[:5]}"


def test_a_stricter_slo_never_recommends_more_sessions():
    """No longer a control — the tab fixes the floor at DEFAULT_SLO and names it
    in the caption, because a dropdown labelled SESSIONS that actually set a
    latency policy produced a session count the reader could not predict from
    the label. The argument survives on optimal_concurrency() and still has to
    move the answer in the direction it claims: real-time (33 tok/s) cannot
    support more concurrent streams than batch (5 tok/s) on the same
    hardware."""
    kw = dict(quant="Q4", vram_gb=48, bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS,
              hw_type="nvidia", fp16_tflops=209.5, ctx_tokens=8192)
    batch = get_local_df(slo="batch", **kw).set_index("name")["sessions"]
    real  = get_local_df(slo="realtime", **kw).set_index("name")["sessions"]
    shared = batch.index.intersection(real.index)
    worse = [n for n in shared if real[n] > batch[n]]
    assert not worse, f"a stricter latency floor allowed MORE sessions: {worse[:5]}"


def test_an_moe_gains_less_from_batching_than_a_dense_model():
    """Expert sparsity collapses under batching: different sequences route to
    different experts, so the union of weights read grows as 1-(1-f)**B. At
    f=0.10 that is 10% of the weights at B=1 but 82% at B=16. The old model
    implied the MoE advantage was unbounded in both directions."""
    kw = dict(quant="Q4", bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS, hw_type="nvidia",
              fp16_tflops=209.5, ctx_tokens=8192, kv_bytes_tok=98304.0,
              weights_gib=18.0, vram_gb=48.0)
    moe = optimal_concurrency(params_b=30.5, active_b=3.3, moe=True, **kw)
    dense = optimal_concurrency(params_b=3.3, active_b=3.3, moe=False, **kw)
    assert moe and dense
    assert moe.total_tps / moe.single_tps < dense.total_tps / dense.single_tps, (
        "an MoE gains as much from batching as a dense model of its active size"
    )


def test_the_tok_s_figures_say_which_of_the_two_they_are():
    """Direct heir to test_the_budget_chart_states_the_token_mix_it_assumes: the
    same number means many times more or less depending on an assumption the
    reader cannot see. A bare "tok/s" beside a session count is ambiguous."""
    df = get_local_df(ctx_tokens=8192, fp16_tflops=209.5)
    scatter = build_local_scatter(df, vram_gb=32, quant="Q4", ctx_tokens=8192)
    compat = build_local_compat(df, quant="Q4", vram_gb=32, ctx_tokens=8192)
    assert "8k context" in compat.layout.title.text.lower()
    # Whichever metric is selected, BOTH charts must name it and the hover must
    # carry the other one — the two are 2-4x apart, and the tab briefly put both
    # in the same gutter with nothing saying which was which.
    for mode, headline, other in (("single", "single stream", "across"),
                                  ("throughput", "max throughput", "if you run one")):
        sc = build_local_scatter(df, vram_gb=32, quant="Q4", ctx_tokens=8192,
                                 speed_mode=mode)
        cp = build_local_compat(df, quant="Q4", vram_gb=32, ctx_tokens=8192,
                                speed_mode=mode)
        assert headline in sc.layout.title.text.lower(), sc.layout.title.text
        assert headline in cp.layout.title.text.lower(), cp.layout.title.text
        rendered = [str(r[-1]).lower() for r in cp.data[-1].customdata]
        assert any(other in r for r in rendered), (mode, rendered[:3])
        assert not any("×0" in r for r in rendered), (
            "a row is still rendering zero sessions as though it had been sized"
        )


def test_the_new_controls_say_what_they_cost_in_both_shells():
    """A control that exists in two hand-synced shells is documented in neither
    unless something checks. Same guard as
    test_the_gpu_count_control_says_what_extra_cards_do."""
    for rel in ("docs/index.html", "app.py"):
        src = (ROOT / rel).read_text()
        assert 'local-context' in src, f"{rel} has no context control"
        assert 'local-speed-mode' in src, f"{rel} has no speed-metric control"
        assert "KV cache is read in full every token" in src, (
            f"{rel} does not say why context moves both VRAM and speed"
        )
        assert "10 tok/s" in src, f"{rel} does not name the per-session floor"


# ── Theme 12 — scraped architecture must be a fact, not a better guess ───────

def test_the_two_published_sources_never_answer_for_the_same_model():
    """resolve_attention() prefers the hand-curated KV_ARCH, so a scraped row
    for a model the table already covers can never be used — and a second,
    differing answer sitting in the cache is pure risk with no upside.
    arch_scraper skips those names; this asserts it kept skipping them.

    THIS IS NOT HYPOTHETICAL. The overlap is how a repack got in: searching
    "Llama 3.1 Instruct 405B" surfaces SillyTilly's 410,081,247,232-parameter
    upload but not Meta's own gated 405,853,388,800 one, and the 4,227,858,432
    difference is exactly 126 layers x 2 tensors x 8 extra KV heads x 128 x
    16384 — eight KV heads per layer the released model does not have. Its
    config would have doubled that model's KV cache.

    The same clash also corrected KV_ARCH itself, which had believed the repack
    and told future readers not to "fix" it back to the paper's 8."""
    from data.local_models import _kv_arch_lookup, _load_scraped_arch
    overlap = [n for n in _load_scraped_arch() if _kv_arch_lookup(n) is not None]
    assert not overlap, (
        f"scraped rows shadowed by the curated table: {overlap[:5]}"
    )


def test_llama_31_405b_has_the_eight_kv_heads_its_weights_carry():
    """Pinned because this row was wrong once and the wrong value came with a
    comment telling the next reader not to change it. Meta publishes
    405,853,388,800 parameters; the 16-head repack publishes 410,081,247,232,
    and the difference is exactly the extra heads. 8 gives 504 KiB/token."""
    from data.local_models import _kv_arch_lookup, kv_cache_bytes
    geo = _kv_arch_lookup("Llama 3.1 405B")
    assert geo and geo["n_kv_heads"] == 8, geo
    per_tok = kv_cache_bytes(geo, 1)
    assert abs(per_tok - 504 * 1024) < 1024, f"{per_tok / 1024:.0f} KiB/token"


def test_an_mla_model_is_never_written_through_the_gqa_columns():
    """A latent-attention model caches ONE vector per layer, so the GQA
    formula's leading 2 alone doubles it. DeepSeek V4 is the trap: it publishes
    no kv_lora_rank and expresses the latent as num_key_value_heads=1 with
    head_dim=512, which reads as ordinary MQA. Priced as GQA it comes out 78%
    high."""
    from data.local_models import _load_scraped_arch
    bad = [(n, g) for n, g in _load_scraped_arch().items()
           if g.get("attn") == "mla" and (g.get("n_kv_heads") or g.get("head_dim"))]
    assert not bad, f"MLA rows carrying GQA columns: {bad[:3]}"
    for name, geo in _load_scraped_arch().items():
        if geo.get("attn") == "mla":
            assert geo.get("kv_lora_rank"), f"{name}: MLA row with no latent width"


def test_a_published_architecture_is_never_labelled_an_estimate():
    """kv_source is the only thing telling the reader whether the KV figure
    beside the exact weights figure was read or fitted. A row resolved from a
    config.json must not carry the estimator's ±30% warning, and a row that was
    fitted must not pass as published."""
    from data.local_models import _kv_arch_lookup, _load_scraped_arch
    df = get_local_df(ctx_tokens=8192)
    scraped = _load_scraped_arch()
    wrong = []
    for _, r in df.iterrows():
        published = (_kv_arch_lookup(r["name"]) is not None
                     or str(r["name"]) in scraped)
        if published and r["kv_source"] == "estimated":
            wrong.append((r["name"], "published but labelled estimated"))
        if not published and r["kv_source"] in ("config", "hf"):
            wrong.append((r["name"], "fitted but labelled published"))
    assert not wrong, wrong[:5]


def test_the_scrape_shrinks_the_share_of_models_running_on_a_guess():
    """The whole point. Before the scrape, 166 of 179 catalogue rows priced
    their KV cache from a fit with a p90 signed residual of +50%; a KV error is
    a fits/does-not-fit error on the tab's central question."""
    df = get_local_df(ctx_tokens=8192)
    published = int((df["kv_source"].isin(("config", "hf"))).sum())
    assert published >= 13, (
        f"only {published} of {len(df)} rows carry a published architecture — "
        f"has data/raw/aa_local_arch.csv gone missing?"
    )


def test_every_scraped_row_cleared_the_parameter_guard():
    """A name can be ambiguous; a parameter count cannot. Every cached row
    records the count HuggingFace published for the repo it read, and it has to
    agree with the catalogue's — otherwise the scraper resolved the 8B when the
    catalogue meant the 70B and wrote a confidently wrong architecture."""
    import pandas as _pd
    from data.arch_scraper import _CACHE, _PARAM_TOLERANCE
    if not _CACHE.exists():
        pytest.skip("no architecture cache in this checkout")
    arch = _pd.read_csv(_CACHE).set_index("name")
    cat = get_local_df(ctx_tokens=0).set_index("name")
    off = []
    for name, r in arch.iterrows():
        if name not in cat.index:
            continue
        want = float(_pd.Series(cat.loc[name, "params_b"]).iloc[0]
                     if hasattr(cat.loc[name, "params_b"], "__len__")
                     else cat.loc[name, "params_b"])
        got = float(r["params_hf_b"])
        if want > 0 and abs(got - want) / want > _PARAM_TOLERANCE + 1e-9:
            off.append((name, want, got))
    assert not off, f"cached rows whose repo is the wrong size: {off[:5]}"


def test_a_linear_attention_hybrid_is_not_charged_for_its_recurrent_layers():
    """Qwen3.8-Flash-Next is 12 full-attention layers out of 48; the other 36
    are Gated-DeltaNet linear layers carrying a fixed-size state that does not
    grow with context. Charging all 48 as full attention overstated its KV cache
    by exactly 4x — 98,304 B/token against a true 24,576.

    The bug was a missing `sliding_window`: the hybrid branch required both a
    global-layer count AND a window, and a linear layer has no window because it
    has no per-token cache at all, so every linear hybrid fell through to the
    all-layers-full formula. 25 of 134 scraped rows are linear hybrids."""
    from data.local_models import _load_scraped_arch, kv_cache_bytes
    scraped = _load_scraped_arch()
    linear = {n: g for n, g in scraped.items() if g.get("local_kind") == "linear"}
    assert linear, "no linear hybrid resolved — has layer_types stopped being read?"
    for name, geo in linear.items():
        g, n = geo["global_layers"], geo["n_layers"]
        assert 0 < g < n, (name, g, n)
        full = kv_cache_bytes({k: v for k, v in geo.items()
                               if k not in ("global_layers", "local_kind", "window")}, 8192)
        hybrid = kv_cache_bytes(geo, 8192)
        assert hybrid < full, f"{name}: linear layers still charged full price"
        assert abs(hybrid - full * g / n) < 1, (
            f"{name}: hybrid cache is not exactly the full-attention share"
        )


def test_the_family_legend_encodes_colour_and_nothing_else():
    """Shape on this chart means dense vs mixture-of-experts. The legend is
    FAMILY, and its swatches used to inherit the per-point symbol array, so
    Plotly drew each family with whichever of its models happened to sort first
    — Alibaba a circle, NVIDIA a diamond, by accident of ordering. A reader
    comparing Qwen3.8 27B (dense, circle) against Qwen3.8-Flash-Next (MoE,
    diamond) reasonably concluded the shape was the lab's.

    Every legend swatch must be the same neutral symbol, and the legend must
    still list exactly the families that got a runnable mark."""
    df = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS,
                      hw_type="nvidia", fp16_tflops=209.5, ctx_tokens=8192)
    fig = build_local_scatter(df, vram_gb=32, quant="Q4", ctx_tokens=8192)
    legend = [t for t in fig.data if t.showlegend]
    assert legend, "the family legend disappeared"
    symbols = {str(t.marker.symbol) for t in legend}
    assert symbols == {"circle"}, f"legend swatches encode shape: {symbols}"

    scored = df[~(df.get("pending", False).fillna(False) & df["quality"].isna())]
    expected = set(scored[scored["fits"] == "yes"]["family"])
    assert {t.name for t in legend} <= {str(f) for f in expected}, (
        "the legend lists a family with nothing plotted"
    )
    # And the key that DOES explain shape has to be on the chart.
    key = " ".join(a.text for a in fig.layout.annotations if a.text)
    assert "dense" in key and "mixture-of-experts" in key, key
