"""Filter controls must do what the user clicked.

Two failure modes, one root cause each:

1. Python truthiness conflates "the user chose none" with "the user chose
   nothing yet". `if providers:` is False for BOTH `None` and `[]`, so
   unchecking every provider fell through to the "no filter" branch and showed
   *all* providers -- the exact opposite of the request.

2. Filter options outlive the data they were written against. A hardcoded option
   whose value no longer matches any row silently selects nothing, and the UI
   blames the user with "No models match these filters". app.py carries a
   comment about 'multilingual' being removed for exactly this reason -- the
   option was fixed, the class of bug was not.
"""
import json
import re
from pathlib import Path

import pytest

from data.ingest import get_models
from data.image_models import get_image_df, get_image_tags
from data.video_models import get_video_df, get_video_tags
from components.charts.constants import PROVIDER_ALIASES
from components.stack_recommender import select_stack
from static_helpers import apply_filters

ROOT = Path(__file__).resolve().parent.parent
DF = get_models()
INDEX_HTML = (ROOT / "docs" / "index.html").read_text()
APP_PY = (ROOT / "app.py").read_text()


def _picked(result) -> list[str]:
    out = []
    for tier in result["tiers"]:
        picks = tier["picks"]
        out.extend([] if picks is None or picks.empty else list(picks["model"]))
    return out


# ── "none selected" must mean none ────────────────────────────────────────────

def test_selecting_no_providers_returns_no_recommendations():
    """`[]` is a deliberate empty selection; `None` means "no filter applied"."""
    assert _picked(select_stack(DF, [], "api")) == []


def test_selecting_no_providers_is_not_the_same_as_selecting_all():
    none_selected = _picked(select_stack(DF, [], "api"))
    all_providers = _picked(select_stack(DF, None, "api"))
    assert all_providers, "the unfiltered call should recommend something"
    assert none_selected != all_providers


def test_selecting_all_still_recommends_across_providers():
    assert len(_picked(select_stack(DF, None, "api"))) > 0


def test_a_named_provider_only_recommends_that_provider():
    result = select_stack(DF, ["Anthropic"], "api")
    for tier in result["tiers"]:
        picks = tier["picks"]
        if picks is not None and not picks.empty:
            assert set(picks["provider"]) == {"Anthropic"}


# ── Provider aliases must be applied where users type them ────────────────────

@pytest.mark.parametrize("retired,current", sorted(PROVIDER_ALIASES.items()))
def test_recommender_resolves_retired_provider_names(retired, current):
    """constants.py has carried this alias map since the Microsoft rename, and
    data/local_models.py applies it. The recommender did a raw .isin(), so the
    UI's own "xAI" checkbox matched zero rows against data that says
    "SpaceXAI"."""
    if DF[DF["provider"] == current].empty:
        pytest.skip(f"no {current} models in the current catalogue")
    assert _picked(select_stack(DF, [retired], "api")) == \
           _picked(select_stack(DF, [current], "api"))


# ── Every offered option must match at least one row ──────────────────────────

def _html_select_values(select_id: str) -> list[str]:
    block = re.search(rf'<select[^>]*id="{select_id}".*?</select>', INDEX_HTML, re.S)
    return re.findall(r'<option value="([^"]+)"', block.group(0)) if block else []


def _html_checkbox_values(name: str) -> list[str]:
    return re.findall(rf'<input[^>]*name="{name}"[^>]*value="([^"]+)"', INDEX_HTML)


def _app_checklist_values(component_id: str) -> list[str]:
    block = re.search(rf'id="{component_id}".*?options=\[(.*?)\]', APP_PY, re.S)
    return re.findall(r'"value":\s*"([^"]+)"', block.group(1)) if block else []


_SENTINELS = {"__all__"}


def test_static_site_offers_a_provider_checkbox_for_every_named_provider():
    offered = [v for v in _html_checkbox_values("recommend-providers") if v not in _SENTINELS]
    assert offered, "no provider checkboxes found — selector drifted"
    dead = [p for p in offered if _picked(select_stack(DF, [p], "api")) == []]
    assert not dead, f"provider options matching zero models: {dead}"


def test_dash_offers_a_provider_checkbox_for_every_named_provider():
    offered = [v for v in _app_checklist_values("recommend-providers") if v not in _SENTINELS]
    assert offered, "no provider checkboxes found — selector drifted"
    dead = [p for p in offered if _picked(select_stack(DF, [p], "api")) == []]
    assert not dead, f"provider options matching zero models: {dead}"


def _image_tag_matches(tag: str) -> int:
    d = get_image_df()
    if tag == "open_weights":
        return int((d["open_weights"] == True).sum())  # noqa: E712
    return int(d["tags"].apply(lambda t: tag in t).sum())


def _video_tag_matches(tag: str) -> int:
    d = get_video_df()
    # Capability tags are columns, not derived category standings.
    if tag in ("open-weights", "audio"):
        col = "open_weights" if tag == "open-weights" else "audio"
        return int((d[col] == True).sum())  # noqa: E712
    return int(d["tags"].apply(lambda t: tag in t).sum())


def test_derived_image_tags_cover_every_tag_that_can_narrow_the_view():
    """The complement of the dead-option bug: a tag the data carries but the UI
    never offers is an unselectable group. The one exclusion is a tag on *every*
    row, which cannot narrow anything and so is not a filter."""
    tag_lists = [t for t in get_image_df()["tags"] if isinstance(t, list)]
    live = {t for tags in tag_lists for t in tags}
    universal = {t for t in live if sum(t in tags for tags in tag_lists) == len(get_image_df())}
    assert {t["value"] for t in get_image_tags()} == live - universal


@pytest.mark.parametrize("derive,frame,matches", [
    (get_image_tags, get_image_df, _image_tag_matches),
    (get_video_tags, get_video_df, _video_tag_matches),
])
def test_no_offered_tag_selects_the_entire_catalogue(derive, frame, matches):
    """A control that always returns everything is as useless as one that always
    returns nothing, and looks broken in the same way."""
    total = len(frame())
    useless = [t["value"] for t in derive() if matches(t["value"]) >= total]
    assert not useless, f"tag options that match every model: {useless}"


# The tests below take `offered` from the ARTEFACTS A USER ACTUALLY MEETS — the
# published manifest and app.py's own options — and only the expected value from
# get_image_tags(). An earlier draft read both sides from get_image_tags(), which
# is `Counter(...)` filtered to count >= 1: the assertion could not fail, and
# re-hardcoding the dead 'fast' option in app.py left it green.

def _manifest_tags(key: str) -> list[dict] | None:
    path = ROOT / "docs" / "figures" / "manifest.json"
    if not path.exists():
        pytest.skip("no built manifest in this checkout")
    return json.loads(path.read_text()).get(key)


@pytest.mark.parametrize("key,matches", [
    ("image_tags", _image_tag_matches),
    ("video_tags", _video_tag_matches),
])
def test_every_tag_the_published_manifest_offers_matches_a_model(key, matches):
    """The manifest is what fills the control on the deployed site. A tag in
    here that no row carries is a filter that always returns nothing."""
    offered = _manifest_tags(key)
    assert offered, (
        f"manifest has no {key}; both TAGS controls ship as empty <select>s now, "
        f"so a missing key is a dead control — re-run build_static.py"
    )
    dead = [t["value"] for t in offered if matches(t["value"]) == 0]
    assert not dead, f"{key} options in the published manifest matching zero models: {dead}"


@pytest.mark.parametrize("key,derive", [
    ("image_tags", get_image_tags),
    ("video_tags", get_video_tags),
])
def test_published_manifest_tags_match_the_live_vocabulary(key, derive):
    """Pins the producer, not just the consumer. Without this the manifest can
    go stale against the data and every other tag test still passes."""
    offered = _manifest_tags(key)
    assert offered is not None, f"manifest has no {key} — re-run build_static.py"
    assert [t["value"] for t in offered] == [t["value"] for t in derive()], (
        f"docs/figures/manifest.json {key} has drifted from the data"
    )


@pytest.mark.parametrize("select_id,key", [
    ("image-tag-filter", "image_tags"),
    ("video-tag-filter", "video_tags"),
])
def test_tag_selects_are_populated_from_the_manifest_not_hardcoded(select_id, key):
    """Both renderings must read one source. Hardcoded <option>s in index.html
    are how the two sides drifted apart in the first place."""
    assert _html_select_values(select_id) == [], (
        f"{select_id} still ships hardcoded options; they will outlive the data"
    )
    # Strip comments first — an earlier version of this assertion was satisfied
    # by the explanatory comment sitting above the <select>.
    js = re.sub(r"/\*.*?\*/", "", (ROOT / "docs" / "app.js").read_text(), flags=re.S)
    js = re.sub(r"^\s*//.*$", "", js, flags=re.M)
    assert f'fillTagSelect("{select_id}", m.{key})' in js, (
        f"{select_id} is empty but nothing populates it from manifest.{key} — "
        f"the control would be permanently blank"
    )


@pytest.mark.parametrize("component_id,fn", [
    ("image-tag-filter", "get_image_tags"),
    ("video-tag-filter", "get_video_tags"),
])
def test_dash_tag_options_are_derived_not_literal(component_id, fn):
    block = re.search(rf'id="{component_id}",(.*?)multi=True', APP_PY, re.S)
    assert block, f"could not find the {component_id} component in app.py"
    body = re.sub(r"^\s*#.*$", "", block.group(1), flags=re.M)
    # Not `'"value":' not in body` — that only catches double-quoted dict keys,
    # so the identical list written with single quotes slipped straight through.
    assert re.search(r"""["']value["']\s*:""", body) is None, (
        f"{component_id} still has a hardcoded options list in app.py"
    )
    assert f"options={fn}()" in body.replace(" ", "").replace("\n", ""), (
        f"{component_id} does not derive its options from {fn}()"
    )


# ── `0` is a number the user typed, not a missing value ───────────────────────
# `float(x or DEFAULT)` treats 0 as unset, so typing 0 silently produced the
# default. Three controls did this, and the two VRAM sites disagreed on what the
# default even was (8 in Dash, 32 on the public site) for the same user action.

def test_zero_token_volume_is_not_treated_as_one_million():
    import json as _json
    import static_api

    zero = _json.loads(static_api.update_cost_calc(0, None, 0, ""))
    one = _json.loads(static_api.update_cost_calc(1, None, 0, ""))
    assert zero["figure"] != one["figure"], (
        "a monthly volume of 0 rendered the same chart as 1M tokens"
    )


def test_a_degenerate_token_volume_still_names_the_cheapest_model_cheapest():
    """The clamp alone was not enough. At 0 tokens every model costs $0.000, so
    an ascending sort on cost is a no-op and the callout — whose literal text is
    CHEAPEST MODEL SCORING N+ — named Claude Opus 5 at $20/M instead of DeepSeek
    V4 Flash at $0.1575/M: a 127x error. Rank has to be total, not clamped."""
    import static_api

    truth = json.loads(static_api.update_cost_calc(1, None, 0, "", 40))["best"]
    for tokens in (0, -5, 1e-9):
        best = json.loads(static_api.update_cost_calc(tokens, None, 0, "", 40))["best"]
        assert best["model"] == truth["model"], (
            f"at {tokens}M tokens the CHEAPEST callout named {best['model']} "
            f"(${best['price']}/M); the cheapest qualifying model is "
            f"{truth['model']} (${truth['price']}/M)"
        )


def test_negative_token_volume_never_plots_a_negative_cost():
    """`min="0.1"` in the HTML is a hint, not a guard: pressing '-' passes
    straight through to Python."""
    import static_api

    fig = json.loads(static_api.update_cost_calc(-5, None, 0, ""))["figure"]
    xs = [x for tr in fig["data"] for x in (tr.get("x") or []) if isinstance(x, (int, float))]
    assert all(x >= 0 for x in xs), f"negative monthly costs plotted: {[x for x in xs if x < 0]}"


def test_zero_vram_fits_no_models_at_all():
    """Asserts what 0 GB *means*, not merely that it differs from one arbitrary
    default. The earlier form compared 0 against 32 and passed against the very
    bug it named, because the pre-fix code turned 0 into 8 — and 8 != 32."""
    from data.local_models import get_local_df

    fitted = get_local_df(quant="Q4", vram_gb=0, bandwidth_gbps=1792, hw_type="nvidia")
    runnable = fitted[fitted["vram_req_gb"] <= 0]
    assert runnable.empty, (
        f"{len(runnable)} models reported as fitting in 0 GB of VRAM — a typed 0 "
        f"is still being read as a default"
    )
    assert not get_local_df(quant="Q4", vram_gb=8, bandwidth_gbps=1792).empty


@pytest.mark.parametrize("swallowed_default", [8, 24, 32])
def test_a_blank_hardware_box_means_exactly_the_shared_default(swallowed_default):
    """Behaviour, not grep. The previous form asserted the constant NAMES appear
    in each file — which they do from the import line alone, so reverting every
    call site to `float(x or 8)` / `or 32` left it green and the 8-vs-32
    disagreement undetected."""
    import static_api
    from data.local_models import DEFAULT_VRAM_GB, DEFAULT_GPU_COUNT, DEFAULT_BANDWIDTH_GBPS

    blank = static_api.update_local(None, None, "Q4", None, None, None)
    explicit = static_api.update_local(
        DEFAULT_VRAM_GB, DEFAULT_GPU_COUNT, "Q4", DEFAULT_BANDWIDTH_GBPS, "nvidia", None
    )
    assert blank == explicit, "a blank box does not resolve to the shared default"

    if swallowed_default != DEFAULT_VRAM_GB:
        other = static_api.update_local(
            swallowed_default, DEFAULT_GPU_COUNT, "Q4", DEFAULT_BANDWIDTH_GBPS, "nvidia", None
        )
        assert blank != other, (
            f"a blank box renders the same chart as an explicit {swallowed_default} GB — "
            f"the default in use is not DEFAULT_VRAM_GB ({DEFAULT_VRAM_GB})"
        )


def test_neither_rendering_declares_its_own_hardware_default():
    """One default, in Python. Both entry points must reach it by import, and
    the browser must send `null` rather than inventing a number of its own."""
    static_api_src = (ROOT / "static_api.py").read_text()
    js = (ROOT / "docs" / "app.js").read_text()

    for name in ("DEFAULT_VRAM_GB", "DEFAULT_GPU_COUNT", "DEFAULT_BANDWIDTH_GBPS"):
        assert name in static_api_src, f"static_api hardcodes a {name} default"
        assert name in APP_PY, f"app.py hardcodes a {name} default"

    for box in ("local-vram", "local-num-gpus", "recommend-vram", "recommend-num-gpus"):
        assert f'numOrNull("{box}")' in js, (
            f"{box} is not read through numOrNull; a `|| N` fallback re-declares a "
            f"default that lives in data/local_models.py"
        )

    stray = re.findall(r'getElementById\("(?:local|recommend)-[\w-]+"\)[^;\n]*\|\|\s*\d', js)
    assert not stray, f"hardware inputs still carry literal numeric fallbacks: {stray}"

    from data.local_models import DEFAULT_BANDWIDTH_GBPS
    assert re.search(rf"\b{int(DEFAULT_BANDWIDTH_GBPS)}\b", js) is None, (
        f"docs/app.js still hardcodes {int(DEFAULT_BANDWIDTH_GBPS)}; bandwidth defaults "
        f"belong to data/local_models.DEFAULT_BANDWIDTH_GBPS alone"
    )


# ── One spelling in, one spelling out ─────────────────────────────────────────
# constants.py has carried PROVIDER_ALIASES since Microsoft's rename, but the
# map was applied in only some of the places a provider name arrives. Each site
# below silently mis-answered instead of erroring.

def test_a_retired_spelling_selects_the_same_rows_as_the_current_one():
    """The global PROVIDER filter, shared by every tab in both renderings."""
    for retired, current in PROVIDER_ALIASES.items():
        if DF[DF["provider"] == current].empty:
            continue
        assert len(apply_filters(DF, [retired], 0, "")) == \
               len(apply_filters(DF, [current], 0, "")), \
               f"?p={retired} does not resolve to {current}"


def test_an_old_share_link_does_not_silently_show_everything():
    """An unresolved name matches no row; an empty selection means "all". So a
    ?p=xAI link shared before the rename rendered the entire catalogue under a
    filter the user believed was applied — wrong, and wrong silently."""
    for retired in PROVIDER_ALIASES:
        got = apply_filters(DF, [retired], 0, "")
        if not got.empty:
            assert len(got) < len(DF), (
                f"?p={retired} returned the whole catalogue instead of one provider"
            )


def test_the_static_site_resolves_aliases_from_the_manifest_not_a_second_copy():
    """docs/app.js needs the alias map to fix a ?p= link before matching. It
    must ride the manifest — a hand-copied literal in JS is precisely how the
    palettes and labels drifted apart before."""
    js = (ROOT / "docs" / "app.js").read_text()
    assert "provider_aliases" in js, "docs/app.js does not resolve retired ?p= names"
    code = re.sub(r"^\s*//.*$", "", js, flags=re.M)
    for retired, current in PROVIDER_ALIASES.items():
        assert not re.search(rf'["\']{re.escape(retired)}["\']\s*:', code), (
            f"docs/app.js hardcodes the {retired}->{current} alias; derive it "
            f"from the manifest instead"
        )
    manifest = ROOT / "docs" / "figures" / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()).get("provider_aliases") == \
               dict(PROVIDER_ALIASES), "manifest alias map has drifted from constants.py"


def test_the_dash_url_handler_canonicalises_before_selecting():
    import app as dash_app

    for retired, current in PROVIDER_ALIASES.items():
        _tab, providers, _q = dash_app.init_from_url(f"?p={retired}")
        assert providers == [current], (
            f"init_from_url left ?p={retired} unresolved; the dropdown has no such "
            f"option, so the filter reads as empty and shows every model"
        )


def test_the_recommender_labels_and_colours_use_the_current_spelling():
    """Filtering on the canonical name while rendering the raw one meant ticking
    SpaceXAI returned the right models and then captioned them "xAI" in fallback
    grey, because PROVIDER_COLORS is keyed by the canonical spelling only."""
    from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR
    from components.stack_recommender import build_stack_cards_html

    for retired, current in PROVIDER_ALIASES.items():
        if current not in PROVIDER_COLORS:
            continue
        frame = DF.copy()
        frame["provider"] = frame["provider"].replace({current: retired})
        if (frame["provider"] == retired).sum() == 0:
            frame.loc[frame.index[:5], "provider"] = retired
        html_out = build_stack_cards_html(frame, [current], mode="api")
        assert f">{retired}<" not in html_out, (
            f"a card still prints the retired spelling {retired!r}"
        )
        assert DEFAULT_COLOR not in html_out or PROVIDER_COLORS[current] in html_out, (
            f"{current} rows rendered in the grey fallback instead of "
            f"{PROVIDER_COLORS[current]}"
        )


# ── The two provider controls use opposite empty-state rules, on purpose ──────

def test_the_global_filter_treats_an_empty_selection_as_all():
    """A multi-select whose placeholder reads "All providers" is unfiltered when
    empty. Pinned so the asymmetry with select_stack stays deliberate."""
    assert len(apply_filters(DF, [], 0, "")) == len(DF)
    assert len(apply_filters(DF, None, 0, "")) == len(DF)


def test_the_recommender_treats_an_empty_selection_as_none():
    """Checkboxes are the opposite idiom: every box unticked is a choice."""
    assert _picked(select_stack(DF, [], "api")) == []
    assert _picked(select_stack(DF, None, "api")) != []
