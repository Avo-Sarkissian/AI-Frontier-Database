"""Controls must do what they say, and both renderings must say the same thing.

Themes 4 and 5 of audit/2026-08-12. Two patterns:

* A control that silently does something other than what it is labelled — the
  Compare selection being discarded rather than pruned, a quality preset also
  clearing the provider filter, a filter bar live over content it does not
  touch, ↓CSV exporting a different dataset from the one on screen.
* app.py and docs/app.js drifting, always in the same direction: the fix lands
  on the Dash side, which nobody deploys, and the public site keeps the bug.
"""
import json
import re
from pathlib import Path

import pytest

from captions import CAPTIONS
from data.ingest import get_models
from static_helpers import (
    apply_filters, cap_compare_selection, compute_diverse5, quality_options,
    export_frame_for_tab, TABS_WITHOUT_GLOBAL_FILTERS, COMPARE_MAX,
)

ROOT = Path(__file__).resolve().parent.parent
DF = get_models()
APP_JS = (ROOT / "docs" / "app.js").read_text()
APP_PY = (ROOT / "app.py").read_text()
INDEX = (ROOT / "docs" / "index.html").read_text()


def _manifest():
    p = ROOT / "docs" / "figures" / "manifest.json"
    if not p.exists():
        pytest.skip("no built manifest")
    return json.loads(p.read_text())


# ── 4.4 — a filter must prune the Compare selection, not replace it ──────────

def test_a_selection_that_still_passes_the_filter_survives_it():
    """A curated three-model comparison vanished on one click of "Top 25%"
    while every one of the three still cleared the new floor."""
    strong = DF.nlargest(3, "quality")["model"].tolist()
    floor = float(DF["quality"].quantile(0.75))
    filtered = apply_filters(DF, None, floor, "")
    kept = cap_compare_selection(strong, filtered, "filter-quality")
    assert set(kept) == set(strong), (
        f"models that still pass a >= {floor} floor were discarded: "
        f"{set(strong) - set(kept)}"
    )


def test_a_selection_excluded_by_the_filter_is_dropped_not_kept():
    """Pruning, not ignoring — a model the filter excludes must go."""
    weakest = DF.nsmallest(1, "quality")["model"].tolist()
    filtered = apply_filters(DF, None, float(DF["quality"].quantile(0.9)), "")
    kept = cap_compare_selection(weakest, filtered, "filter-quality")
    assert weakest[0] not in kept


def test_a_tab_switch_does_not_touch_the_selection():
    """The static site routed every tab change through the filter path, so
    going to Table and back replaced the user's picks with the defaults. Dash
    has no tabs input on this callback and never did it."""
    picked = DF.nlargest(3, "quality")["model"].tolist()
    assert cap_compare_selection(picked, DF, "tab-switch") == picked


def test_an_empty_selection_still_falls_back_to_the_defaults():
    assert cap_compare_selection([], DF, "tab-switch") == compute_diverse5(DF)
    assert cap_compare_selection(None, DF, None) == compute_diverse5(DF)


def test_the_static_site_passes_a_tab_switch_trigger():
    assert 'refreshCompare("filter-provider")' not in APP_JS, (
        "a tab switch still claims to be a provider-filter change"
    )
    assert "tab-switch" in APP_JS


# ── 4.8 — the cap evicts the oldest pick, not the newest ─────────────────────

def test_the_sixth_pick_is_kept_and_the_oldest_is_evicted():
    """`[:5]` threw away the model the user had just clicked, which reads as
    the click doing nothing at all."""
    six = DF.nlargest(6, "quality")["model"].tolist()
    kept = cap_compare_selection(six, DF, None)
    assert len(kept) == COMPARE_MAX
    assert six[-1] in kept, "the newest pick was discarded"
    assert six[0] not in kept, "the oldest pick was not evicted"


def test_the_static_site_trims_an_over_selection_back_into_the_control():
    """Shift-selecting 9 left 8 highlighted while 5 were charted."""
    assert "selected.slice(-COMPARE_MAX)" in APP_JS


def test_the_add_to_compare_button_is_not_a_no_op_when_full():
    """Both renderings default to exactly 5 models, so the detail panel's only
    call to action did nothing on a fresh page load — and switched tabs
    anyway."""
    assert "chosen.length < 5" not in APP_JS, (
        "the button still refuses to act when the selection is already full"
    )
    assert "COMPARE_MAX - 1" in APP_JS


# ── 4.7 — a quality preset touches quality ───────────────────────────────────

def test_quality_presets_leave_the_provider_selection_alone():
    block = re.search(r"def apply_preset\(.*?\n\n\n", APP_PY, re.S)
    assert block, "could not find apply_preset"
    body = block.group(0)
    assert 'return _P75, no_update, no_update' in body
    assert 'return _P90, no_update, no_update' in body


def test_the_static_presets_leave_the_provider_selection_alone():
    assert "setPreset(window.AF.manifest.p75, null, false)" in APP_JS
    assert "setPreset(window.AF.manifest.p90, null, false)" in APP_JS


def test_the_button_that_clears_everything_says_so():
    """"All" cleared SEARCH and PROVIDER too — three buttons, three scopes,
    one label that named none of it."""
    assert "Reset filters" in APP_PY and "Reset filters" in INDEX


# ── 4.9 — the filter bar and ↓CSV must match what is on screen ───────────────

@pytest.mark.parametrize("tab", TABS_WITHOUT_GLOBAL_FILTERS)
def test_the_global_filter_bar_is_hidden_where_it_does_nothing(tab):
    assert "global-filters" in APP_PY, "the Dash filter bar has no id to hide by"
    assert "TABS_WITHOUT_GLOBAL_FILTERS" in APP_JS
    assert f'"{tab}"' in APP_JS


@pytest.mark.parametrize("tab,expect_col", [
    ("image", "elo"),
    ("video", "quality"),
    ("local", "params_b"),
])
def test_csv_export_returns_the_dataset_on_screen(tab, expect_col):
    """From Image Gen, export_csv(['Anthropic'], 50, '') returned the LLM
    header and seven text models."""
    frame, name = export_frame_for_tab(tab, DF, ["Anthropic"], 50, "")
    assert expect_col in frame.columns, f"{tab} exported the wrong dataset: {list(frame.columns)[:5]}"
    assert tab in name, f"{tab} export is named {name!r}"


def test_csv_export_still_honours_the_filters_where_the_bar_is_live():
    frame, name = export_frame_for_tab("overview", DF, ["Anthropic"], 0, "")
    assert set(frame["provider"]) == {"Anthropic"}
    assert name == "ai_frontier_export.csv"


# ── 5.1 — an unknown ?tab= must not blank the page ───────────────────────────

def test_an_unknown_tab_id_falls_back_instead_of_hiding_every_panel():
    """`insights`, `performance` and `embeddings` were all live tab values this
    app once emitted into share URLs. Dash has guarded this since 82effa0; the
    static site never got the same fix."""
    assert "VALID_TABS" in APP_JS
    assert "if (!VALID_TABS.includes(id)) id = VALID_TABS[0];" in APP_JS


def test_dash_still_validates_the_tab_too():
    assert "_VALID_TABS" in APP_PY


# ── 5.3 — one quality value, one representation ──────────────────────────────

def test_the_quality_control_can_hold_the_value_its_presets_set():
    """42.1 into a round-ladder dropdown left the Dash control visually blank
    and made the static site snap 52.7 down to 50 — so "Top 10%" returned
    15.5% of the catalogue under a label that is a precise numeric claim."""
    p75 = round(float(DF["quality"].quantile(0.75)), 1)
    p90 = round(float(DF["quality"].quantile(0.90)), 1)
    values = [o["value"] for o in quality_options(p75, p90)]
    assert p75 in values and p90 in values
    assert values == sorted(values)


def test_the_preset_percentiles_are_labelled_not_bare_numbers():
    p90 = round(float(DF["quality"].quantile(0.90)), 1)
    labels = {o["value"]: o["label"] for o in quality_options(None, p90)}
    assert "top 10%" in labels[p90]


def test_a_fractional_q_param_is_not_silently_dropped():
    """`int("42.1")` raised, the error was swallowed, and the filter became 0 —
    a shared "Top 25%" link opened as no filter at all."""
    import app as dash_app

    p75 = round(float(DF["quality"].quantile(0.75)), 1)
    _tab, _providers, quality = dash_app.init_from_url(f"?q={p75}")
    assert quality == pytest.approx(p75), (
        f"?q={p75} parsed as {quality}; the filter was silently dropped"
    )


def test_the_static_site_inserts_an_option_rather_than_snapping():
    assert "ensureQualityOption" in APP_JS
    assert "snapped" not in APP_JS, "the nearest-option snap is still there"


def test_the_manifest_ships_the_quality_options():
    opts = _manifest().get("quality_options")
    assert opts, "manifest has no quality_options — re-run build_static.py"
    p90 = round(float(DF["quality"].quantile(0.90)), 1)
    assert p90 in [o["value"] for o in opts]


# ── 5.4 — the deployed site must explain itself too ──────────────────────────

def test_every_caption_has_one_source():
    """app.py rendered 14 captions; docs/app.js rendered 1, so nine tabs of the
    site that actually deploys explained nothing."""
    assert len(CAPTIONS) >= 12
    assert "from captions import CAPTIONS" in APP_PY
    literal_descs = re.findall(r'_desc\(\s*\n?\s*"', APP_PY)
    assert not literal_descs, (
        f"{len(literal_descs)} captions are still inline string literals in app.py"
    )


def test_the_manifest_ships_the_captions():
    caps = _manifest().get("captions")
    assert caps, "manifest has no captions — re-run build_static.py"
    assert caps == dict(CAPTIONS), "manifest captions have drifted from captions.py"


def test_the_static_site_renders_a_caption_on_every_tab():
    assert "renderCaptions" in APP_JS
    block = re.search(r"const TAB_CAPTIONS = \{(.*?)\n\};", APP_JS, re.S)
    assert block, "TAB_CAPTIONS not found"
    covered = set(re.findall(r"^\s*(\w+):", block.group(1), re.M))
    tabs = set(re.findall(r'\{ id: "(\w+)"', APP_JS))
    assert tabs <= covered, f"tabs with no caption: {sorted(tabs - covered)}"


def test_every_caption_key_the_site_asks_for_exists():
    block = re.search(r"const TAB_CAPTIONS = \{(.*?)\n\};", APP_JS, re.S)
    wanted = set(re.findall(r'"(\w+)"', block.group(1)))
    missing = wanted - set(CAPTIONS)
    assert not missing, f"docs/app.js asks for captions that do not exist: {missing}"
