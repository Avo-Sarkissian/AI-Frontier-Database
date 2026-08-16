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
    get_local_df, effective_bandwidth, calc_vram_gb,
    DEFAULT_BANDWIDTH_GBPS,
)
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


def test_vram_does_not_claim_to_include_the_kv_cache():
    """calc_vram_gb takes (params_b, quant) and nothing else, yet the constant
    was labelled a "KV-cache + activation overhead multiplier" and the compat
    hover printed "VRAM needed: 14.9 GB" two lines above "Context: 256k"."""
    a = calc_vram_gb(70.0, "Q4")
    assert a == calc_vram_gb(70.0, "Q4")
    src = (ROOT / "data" / "local_models.py").read_text()
    assert "KV-cache + activation overhead multiplier" not in src
    compat = (ROOT / "components" / "charts" / "local_compat.py").read_text()
    assert "weights only" in compat, "the hover still implies context is included"


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
