# tests/test_open_weights.py
"""Open vs closed weights on every hosted surface.

Pinned here:
  * classification order: the row's AA flag, the registry's AA flag, the
    open-weights catalogue (registry or current file), else "inferred" closed;
    an unknown flag is never read as closed before that last step;
  * the registry is append-only, upgrades a catalogue row to AA's flag, and
    the committed file is canonical (a no-news rewrite is byte-identical);
  * WEIGHTS partitions the catalogue and composes with the other filters;
  * hollow = open weights on Overview (price, task, speed) and Rankings, with
    the untimed encoding moved to a centre dot;
  * WEIGHTS is the LAST argument everywhere the browser calls, rides the URL
    as ?w=, and Dash restores it from a shared link.
"""
import inspect
import io
import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

import static_api as api
from data import carried
from data import open_weights as ow
from data.ingest import get_models
from static_helpers import apply_filters, ranking_frame, weights_options
from components.charts.constants import (
    weights_symbol, row_is_open, with_alpha, PROVIDER_COLORS, PROVIDER_SHAPES,
    SPOTLIGHT_PROVIDERS, DEFAULT_SHAPE, UNTIMED_RIM,
)
from components.charts.pareto import build_pareto_scatter, _pareto_frontier, _plottable
from components.charts.quadrant import build_quadrant
from components.charts.rankings import build_rankings
from components.charts.bump_chart import build_value_leaders

ROOT = Path(__file__).resolve().parent.parent
DF = get_models()
APP_JS = (ROOT / "docs" / "app.js").read_text()
INDEX = (ROOT / "docs" / "index.html").read_text()


def _marker_traces(fig):
    return [t for t in fig.data if getattr(t, "mode", "") == "markers"]


# ── Classification ────────────────────────────────────────────────────────────

def _reg(rows):
    return pd.DataFrame(rows, columns=list(ow.REGISTRY_COLUMNS))


def test_annotate_prefers_aa_flag_then_registry_then_catalogue():
    reg = _reg([
        ("RegOpen", "1", ow.SOURCE_AA, "2026-10-27"),
        ("RegClosed", "0", ow.SOURCE_AA, "2026-10-27"),
        ("Listed", "1", ow.SOURCE_CATALOGUE, "2026-05-06"),
        # AA's flag on the row beats a registry entry that disagrees.
        ("Own", "0", ow.SOURCE_AA, "2026-10-27"),
    ])
    df = pd.DataFrame({
        "model": ["Own", "RegOpen", "RegClosed", "Listed", "InCatalogueFile", "Nobody"],
        "is_open_weights": pd.array([True, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA], dtype="boolean"),
    })
    out = ow.annotate_open_weights(df, registry=reg, catalogue={"InCatalogueFile"})
    got = dict(zip(out["model"], zip(out["open_weights"], out["weights_source"])))
    assert got == {
        "Own": (True, "aa"), "RegOpen": (True, "aa"), "RegClosed": (False, "aa"),
        "Listed": (True, "catalogue"), "InCatalogueFile": (True, "catalogue"),
        "Nobody": (False, "inferred"),
    }
    assert out["open_weights"].dtype == bool
    assert "open_weights" not in df.columns, "annotate must not modify its input"


def test_an_unknown_flag_is_never_read_as_closed_before_resolution():
    """Every pre-2026-10-27 snapshot has is_open_weights NA; NA must fall
    through to the registry, not count as False."""
    df = pd.DataFrame({"model": ["Kimi K2.6"],
                       "is_open_weights": pd.array([pd.NA], dtype="boolean")})
    reg = _reg([("Kimi K2.6", "1", ow.SOURCE_CATALOGUE, "2026-05-06")])
    out = ow.annotate_open_weights(df, registry=reg, catalogue=set())
    assert bool(out["open_weights"].iloc[0]) and out["weights_source"].iloc[0] == "catalogue"


def test_a_frame_without_names_or_flags_is_inferred_closed_not_an_error():
    out = ow.annotate_open_weights(pd.DataFrame({"x": [1, 2]}), registry=_reg([]),
                                   catalogue=set())
    assert list(out["open_weights"]) == [False, False]
    assert set(out["weights_source"]) == {"inferred"}


def test_the_live_catalogue_is_annotated_on_load():
    assert {"open_weights", "weights_source"} <= set(DF.columns)
    assert DF["open_weights"].dtype == bool
    assert set(DF["weights_source"]) <= {"aa", "catalogue", "inferred"}


def test_the_aa_flag_and_the_catalogue_name_join_never_disagree():
    """Snapshot-independent invariant: where AA flags a hosted row, an exact
    name in AA's open-weights catalogue never contradicts it (0 of 162 did on
    2026-10-27)."""
    names = ow.catalogue_names()
    flagged = DF[DF["is_open_weights"].notna()]
    contradict = flagged[flagged["model"].isin(names) & flagged["is_open_weights"].eq(False)]
    assert contradict.empty, list(contradict["model"])


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_is_append_only_and_upgrades_the_source():
    reg = _reg([("Old Open", "1", ow.SOURCE_CATALOGUE, "2026-05-06"),
                ("Gone", "1", ow.SOURCE_CATALOGUE, "2026-05-07"),
                ("Flip", "0", ow.SOURCE_AA, "2026-10-27")])
    scrape = pd.DataFrame({
        "model": ["Old Open", "New Closed", "Flip", "Silent"],
        "is_open_weights": pd.array([True, False, True, pd.NA], dtype="boolean"),
    })
    out = carried.merge_open_weights(reg, scrape, today="2026-10-28").set_index("name")
    assert set(out.index) == {"Old Open", "Gone", "New Closed", "Flip"}, \
        "a row was dropped, or an NA flag was recorded as a fact"
    assert out.loc["Old Open", "source"] == ow.SOURCE_AA
    assert out.loc["Old Open", "first_seen"] == "2026-05-06", "first_seen moved"
    assert out.loc["Gone", "source"] == ow.SOURCE_CATALOGUE
    assert out.loc["New Closed"].tolist() == ["0", ow.SOURCE_AA, "2026-10-28"]
    assert out.loc["Flip", "open_weights"] == "1", "AA's latest flag must win"


def test_learn_open_weights_writes_only_on_change(tmp_path):
    path = tmp_path / "open_weight_names.csv"
    scrape = pd.DataFrame({"model": ["A", "B"],
                           "is_open_weights": pd.array([True, False], dtype="boolean")})
    assert carried.learn_open_weights(scrape, path) is True
    assert carried.learn_open_weights(scrape, path) is False
    reg = ow.load_registry(path)
    assert sorted(reg["name"]) == ["A", "B"]


def test_the_committed_registry_is_canonical_and_covers_the_catalogue():
    path = ROOT / "data" / "carried" / ow.REGISTRY_FILE
    text = path.read_text()
    reg = ow.load_registry(path)
    assert carried._registry_text(reg) == text, "a no-news rewrite would change the file"
    assert set(reg["source"]) <= {ow.SOURCE_AA, ow.SOURCE_CATALOGUE}
    assert reg["first_seen"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert reg["open_weights"].isin(["0", "1"]).all()
    assert ow.catalogue_names() <= set(reg["name"]), "the current catalogue is not seeded"


def test_the_scrape_feeds_the_registry():
    src = (ROOT / "data" / "scraper.py").read_text()
    body = src.split("def _scrape_and_save")[1].split("\ndef ")[0]
    assert "carried.learn_open_weights(df)" in body
    assert body.index("save_cache(df)") < body.index("carried.learn_open_weights(df)"), \
        "a refused scrape must not reach the registry"


# ── Filter ────────────────────────────────────────────────────────────────────

def test_weights_options_are_all_open_closed_with_all_first():
    assert [o["value"] for o in weights_options()] == ["", "open", "closed"]
    assert ow.valid_weights("OPEN") == "" and ow.valid_weights("closed") == "closed"


@pytest.mark.parametrize("effort", [None, "best", "low"])
@pytest.mark.parametrize("providers", [None, ["Alibaba", "OpenAI"]])
def test_weights_partitions_the_catalogue_and_composes(effort, providers):
    every = apply_filters(DF, providers, 20, "", effort)
    op = apply_filters(DF, providers, 20, "", effort, "open")
    cl = apply_filters(DF, providers, 20, "", effort, "closed")
    assert set(op.index).isdisjoint(cl.index)
    assert set(op.index) | set(cl.index) == set(every.index)
    assert op["open_weights"].all() and not cl["open_weights"].any()
    assert apply_filters(DF, providers, 20, "", effort, "bogus").equals(every)


def test_rankings_tag_the_unhosted_rows_as_open():
    rf = ranking_frame(DF)
    extra = rf[rf.get("self_host", pd.Series(False, index=rf.index)).eq(True)]
    if extra.empty:
        pytest.skip("every open-weights model is hosted today")
    assert extra["open_weights"].all() and set(extra["weights_source"]) == {"aa"}
    assert rf["open_weights"].dtype == bool


def test_the_frontier_never_grows_under_weights():
    """WEIGHTS hides points; the frontier is still decided market-wide."""
    front = _pareto_frontier(_plottable(DF))
    market = {(round(p, 9), round(q, 9)) for p, q in zip(front["price"], front["quality"])}
    for w in ("open", "closed"):
        fig = build_pareto_scatter(apply_filters(DF, None, 0, "", None, w), full_df=DF)
        for t in fig.data:
            if t.name == "Pareto Frontier":
                pts = {(round(p, 9), round(q, 9)) for p, q in zip(t.x, t.y)}
                assert pts <= market, w


# ── Marks ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("base,is_open,timed,want", [
    ("square", False, True, "square"),
    ("square", True, True, "square-open"),
    ("square", False, False, "square-dot"),
    ("square", True, False, "square-open-dot"),
    ("square-open", False, True, "square"),          # hollow can only mean open
    ("hourglass", False, False, "hourglass"),        # Plotly has no hourglass-dot
    ("bowtie", True, False, "bowtie-open"),
])
def test_weights_symbol(base, is_open, timed, want):
    assert weights_symbol(base, is_open, timed) == want


@pytest.mark.parametrize("provider", list(SPOTLIGHT_PROVIDERS) + ["Other"])
def test_every_spotlight_symbol_variant_is_a_real_plotly_symbol(provider):
    base = PROVIDER_SHAPES.get(provider, DEFAULT_SHAPE)
    for o in (False, True):
        for t in (False, True):
            go.scatter.Marker(symbol=weights_symbol(base, o, t))   # raises if invalid


def _mixed_frame():
    return pd.DataFrame([
        dict(model="Shut", provider="OpenAI", price=2.0, quality=50.0, speed=200.0,
             latency=1.0, context="1m", open_weights=False, weights_source="aa"),
        dict(model="Free", provider="OpenAI", price=1.0, quality=45.0, speed=150.0,
             latency=1.0, context="1m", open_weights=True, weights_source="aa"),
        dict(model="FreeSlow", provider="Google", price=3.0, quality=40.0, speed=0.0,
             latency=0.0, context="1m", open_weights=True, weights_source="catalogue"),
        dict(model="ShutSlow", provider="Google", price=4.0, quality=41.0, speed=0.0,
             latency=0.0, context="1m", open_weights=False, weights_source="inferred"),
    ]).assign(cost_per_task=[1.0, 0.5, 0.7, 0.9])


def _symbols(fig):
    return {cd[5]: str(t.marker.symbol) for t in _marker_traces(fig) for cd in t.customdata}


@pytest.mark.parametrize("x", ["price", "task"])
def test_overview_draws_open_weights_hollow_on_both_axes(x):
    fig = build_pareto_scatter(_mixed_frame(), x=x)
    sym = _symbols(fig)
    assert sym["Shut"] == PROVIDER_SHAPES["OpenAI"]
    assert sym["Free"] == PROVIDER_SHAPES["OpenAI"] + "-open"
    assert sym["FreeSlow"].endswith("-open-dot")
    assert sym["ShutSlow"].endswith("-dot") and "-open" not in sym["ShutSlow"]
    title = str(fig.layout.title.text)
    assert "hollow = open weights" in title and "centre dot = speed not yet measured" in title
    hover = {cd[5]: cd[10] for t in _marker_traces(fig) for cd in t.customdata}
    assert hover == {"Shut": "Closed · AA flag", "Free": "Open · AA flag",
                     "FreeSlow": "Open · name match", "ShutSlow": "Closed · inferred"}
    assert all("Weights: %{customdata[10]}" in t.hovertemplate for t in _marker_traces(fig))


def test_untimed_closed_marks_carry_a_light_rim_so_the_dot_shows():
    fig = build_pareto_scatter(_mixed_frame())
    for t in _marker_traces(fig):
        if str(t.marker.symbol).endswith("-dot") and "-open" not in str(t.marker.symbol):
            assert t.marker.line.color == UNTIMED_RIM


def test_no_hollow_key_when_nothing_open_is_drawn():
    df = _mixed_frame()
    fig = build_pareto_scatter(df[~df["open_weights"]])
    assert "hollow" not in str(fig.layout.title.text)


def test_each_provider_keeps_one_legend_entry_and_a_filled_swatch():
    fig = build_pareto_scatter(_mixed_frame())
    shown = [t for t in _marker_traces(fig) if t.showlegend is not False]
    assert sorted(t.name for t in shown) == sorted({t.name for t in _marker_traces(fig)})
    assert all("-open" not in str(t.marker.symbol) for t in shown)
    for t in _marker_traces(fig):
        assert t.legendgroup == t.name


def test_speed_view_draws_open_weights_hollow():
    fig = build_quadrant(_mixed_frame())
    sym = _symbols(fig)
    assert sym["Free"].endswith("-open") and not sym["Shut"].endswith("-open")
    assert "hollow = open weights" in str(fig.layout.title.text)
    assert all(t.legendgroup == t.name for t in _marker_traces(fig))


def _bar(fig):
    return [t for t in fig.data if isinstance(t, go.Bar) and t.customdata is not None][0]


def test_rankings_open_bars_are_outlined_and_closed_bars_unchanged():
    df = _mixed_frame().assign(self_host=[False, False, False, True])
    fig = build_rankings(df, top_n=4)
    bar = _bar(fig)
    names = [cd[0] for cd in bar.customdata]
    for i, n in enumerate(names):
        c = PROVIDER_COLORS[df.set_index("model").loc[n, "provider"]]
        if n in ("Free", "FreeSlow", "ShutSlow"):        # ShutSlow is self-host
            assert bar.marker.color[i] == with_alpha(c, 0.14)
            assert bar.marker.line.color[i] == c and bar.marker.line.width[i] > 0
        else:
            assert bar.marker.color[i] == c and bar.marker.line.width[i] == 0
    assert "hollow = open weights" in str(fig.layout.title.text)
    assert "Weights: %{customdata[5]}" in bar.hovertemplate


def test_value_leaders_open_bars_are_outlined():
    fig = build_value_leaders(_mixed_frame())
    bar = fig.data[0]
    by = {cd[0]: i for i, cd in enumerate(bar.customdata)}
    assert bar.marker.line.width[by["Free"]] > 0 and bar.marker.line.width[by["Shut"]] == 0
    assert bar.customdata[by["Free"]][6] == "Open · AA flag"


def test_row_is_open_reads_self_host_and_unannotated_frames():
    df = pd.DataFrame({"model": ["a", "b"], "self_host": [True, False]})
    assert list(row_is_open(df)) == [True, False]


# ── Static API and browser wiring ─────────────────────────────────────────────

_GLOBAL_FNS = (api.update_overview, api.update_treemap, api.update_provider_leaderboard,
               api.update_rankings, api.update_value_leaders, api.update_compare,
               api.update_cost_calc, api.update_table, api.export_csv)


@pytest.mark.parametrize("fn", _GLOBAL_FNS, ids=lambda f: f.__name__)
def test_weights_is_the_last_parameter_of_every_global_filter_function(fn):
    params = list(inspect.signature(fn).parameters.values())
    assert params[-1].name == "weights" and params[-1].default is None
    assert params[-2].name == "mix"


def test_static_overview_at_open_draws_only_open_models():
    fig = json.loads(api.update_overview(None, 0, "", "price", None, None, "open"))
    drawn = {cd[5] for t in fig["data"] if t.get("mode") == "markers"
             for cd in t["customdata"]}
    assert drawn and drawn <= set(DF[DF["open_weights"]]["model"])
    default = api.update_overview(None, 0, "", "price", None, None)
    assert api.update_overview(None, 0, "", "price", None, None, None) == default


def test_export_honours_weights_and_says_where_each_flag_came_from():
    closed = pd.read_csv(io.StringIO(api.export_csv(None, 0, "", "table", None, None, None,
                                                    "closed")))
    assert len(closed) == int((~DF["open_weights"]).sum())
    assert {"open_weights", "weights_source"} <= set(closed.columns)
    assert not closed["open_weights"].any()


def test_detail_panel_names_the_weights_and_their_source():
    row = DF[DF["open_weights"]].iloc[0]
    html = api.model_detail(row["model"], row["provider"])
    assert ">Weights<" in html and "Open · AA flag" in html


def test_the_browser_sends_weights_last_and_shares_it_in_the_url():
    assert 'getElementById("filter-weights")' in APP_JS
    assert "return [providers, minQuality, search, effort, mix, weights];" in APP_JS
    assert 'params.set("w", weights)' in APP_JS and 'u.has("w")' in APP_JS
    assert 'onFilter("filter-weights")' in APP_JS
    assert "m.weights_options" in APP_JS
    # Reset filters clears it.
    reset = APP_JS.split('getElementById("preset-all").onclick')[1][:600]
    assert 'getElementById("filter-weights")' in reset


def test_the_filter_bar_orders_weights_between_effort_and_mix():
    i_e, i_w, i_m = (INDEX.index(f'id="filter-{k}"') for k in ("effort", "weights", "mix"))
    assert i_e < i_w < i_m


# ── Dash parity ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dash_app():
    import app as module
    return module


def test_the_weights_setting_survives_a_shared_link(dash_app):
    assert dash_app.init_weights_from_url("?w=open") == "open"
    assert dash_app.init_weights_from_url("?w=closed&tab=rankings") == "closed"
    assert dash_app.init_weights_from_url("?w=everything") == ""
    assert dash_app.init_weights_from_url("") == ""
    assert len(dash_app.init_from_url("?w=open")) == 4


@pytest.mark.parametrize("output", [
    "pareto-chart.figure", "treemap-chart.figure", "rankings-chart.figure",
    "value-leaders-chart.figure", "provider-leaderboard-chart.figure",
    "radar-chart.figure", "cost-calc-chart.figure", "model-table.data",
])
def test_every_dash_hosted_chart_listens_to_weights(dash_app, output):
    from dash._callback import GLOBAL_CALLBACK_MAP
    specs = [s for k, s in GLOBAL_CALLBACK_MAP.items() if output in k]
    assert specs, output
    assert "filter-weights" in {d["id"] for d in specs[0]["inputs"]}, output


def test_dash_and_static_filter_weights_identically(dash_app):
    dash_fig = json.loads(dash_app.update_value_leaders(None, 0, None, "", 0, None, None,
                                                        None, "open").to_json())
    static_fig = json.loads(api.update_value_leaders(None, 0, "", None, None, "open"))
    assert dash_fig["data"][0]["customdata"] == static_fig["data"][0]["customdata"]


def test_dash_csv_export_honours_weights(dash_app):
    out = dash_app.export_csv(1, "table", None, 0, "", None, None, None, None, None,
                              None, None, None, None, None, "open")
    exported = pd.read_csv(io.StringIO(out["content"]))
    assert len(exported) == int(DF["open_weights"].sum()) and exported["open_weights"].all()
