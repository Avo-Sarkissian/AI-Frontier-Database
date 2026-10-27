# tests/test_task_cost.py
"""Overview's "Cost / task" x axis and the detail panel's token burn.

The axis is Artificial Analysis's measured cost per Intelligence Index task
(`cost_per_task`, scraped since 2026-10-27). Pinned here:
  * a model AA has not costed is never drawn and never imputed, and the title
    says how many families lack one;
  * the frontier is computed on cost per task, not on price;
  * MIX does not move the axis, the frontier or the counts, only the hover's
    price line;
  * token burn divides by the LIST blend and names it, whatever the MIX;
  * the static site and the Dash app say the same thing.
"""
import base64
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import static_api as api
from components.charts.pareto import build_pareto_scatter, _pareto_frontier
from data import task_cost as tc
from data.workload_mix import apply_mix, resolve_mix
from data.ingest import get_models

ROOT = Path(__file__).resolve().parent.parent
DF = get_models()
HAS_REAL = bool((pd.to_numeric(DF.get("cost_per_task"), errors="coerce") > 0).any()) \
    if "cost_per_task" in DF.columns else False
real = pytest.mark.skipif(not HAS_REAL, reason="no committed row has cost_per_task > 0")


def _decoded(json_str):
    spec = json.loads(json_str)
    for trace in spec.get("data", []):
        for key, val in list(trace.items()):
            if isinstance(val, dict) and "bdata" in val and "dtype" in val:
                trace[key] = np.frombuffer(base64.b64decode(val["bdata"]),
                                           dtype=np.dtype(val["dtype"])).tolist()
    return spec


def _row(model, provider, quality, price, cpt, speed=100.0):
    return {
        "model": model, "provider": provider, "context": "128k",
        "quality": quality, "price": price, "speed": speed, "latency": 0.5,
        "price_in": price / 2, "price_out": price * 7 / 6,
        "cache_hit_price": price / 20, "cost_per_task": cpt,
    }


@pytest.fixture()
def frame():
    """Five families. "Burner" is the cheapest per token but the dearest per
    task, so the price frontier and the task frontier disagree on it. "Ghost"
    has no AA cost per task; "Zero" has AA's $0 (no published cost)."""
    return pd.DataFrame([
        _row("Burner", "Anthropic", 60.0, 1.0, 2.00),
        _row("Lean", "OpenAI", 55.0, 4.0, 0.10),
        _row("Mid", "Google", 50.0, 2.0, 0.50),
        _row("Ghost", "Meta", 70.0, 3.0, float("nan")),
        _row("Zero", "Mistral", 40.0, 0.5, 0.0),
    ])


def _xs(fig, mode="markers"):
    return sorted(x for t in fig.data if t.mode == mode for x in t.x)


def _marker_models(fig):
    return {cd[5] for t in fig.data if t.mode == "markers" for cd in t.customdata}


# ── data/task_cost ───────────────────────────────────────────────────────────

def test_task_costs_blank_missing_zero_and_negative():
    df = pd.DataFrame({"cost_per_task": [1.5, 0.0, -2.0, None, "x"]})
    out = tc.task_costs(df)
    assert out.iloc[0] == 1.5
    assert out.iloc[1:].isna().all()
    assert tc.task_costs(pd.DataFrame({"price": [1.0]})).isna().all()


def test_token_burn_is_cost_over_list_price_in_tokens(frame):
    burn = tc.token_burn(frame)
    assert burn.iloc[0] == pytest.approx(2.00 / 1.0 * 1e6)
    assert burn.iloc[1] == pytest.approx(0.10 / 4.0 * 1e6)
    assert math.isnan(burn.iloc[3]) and math.isnan(burn.iloc[4])


def test_token_burn_divides_by_the_list_blend_not_the_mixed_price(frame):
    mixed = apply_mix(frame, resolve_mix("agentic"))
    assert "price_list_blend" in mixed.columns
    assert not np.allclose(mixed["price"], frame["price"])
    pd.testing.assert_series_equal(tc.token_burn(mixed), tc.token_burn(frame),
                                   check_names=False)


def test_median_token_burn_skips_rows_without_one(frame):
    burns = [2e6, 0.025e6, 0.25e6]
    assert tc.median_token_burn(frame) == pytest.approx(float(np.median(burns)))
    assert tc.median_token_burn(frame.assign(cost_per_task=float("nan"))) is None


@pytest.mark.parametrize("v,text", [
    (374_000, "374K"), (1_984_953, "1.98M"), (11_291.5, "11.3K"), (950, "950"),
    (None, "—"), (float("nan"), "—"), (0, "—"),
])
def test_fmt_tokens(v, text):
    assert tc.fmt_tokens(v) == text


def test_task_cost_keeps_sub_cent_precision():
    assert tc.fmt_task_cost(0.0045166) == "$0.004517"
    assert tc.fmt_task_cost(5.982) == "$5.98"
    assert tc.fmt_task_cost(None) == "—"


def test_burn_multiple_text():
    assert tc.burn_multiple_text(2.3, 1.0) == "2.3× median"
    assert tc.burn_multiple_text(0.07, 1.0) == "0.07× median"
    assert tc.burn_multiple_text(12.4, 1.0) == "12× median"
    assert tc.burn_multiple_text(None, 1.0) == ""
    assert tc.burn_multiple_text(1.0, None) == ""


def test_rank_flip_share_counts_cheaper_per_token_dearer_per_task(frame):
    # Burner/Lean/Mid with both values: prices 1 < 2 < 4, costs 2.0 > 0.5 > 0.1.
    # Every one of the 3 pairs is reversed.
    share, n = tc.rank_flip_share(frame)
    assert (share, n) == (1.0, 3)
    agree = frame.assign(cost_per_task=frame["price"] / 10)
    assert tc.rank_flip_share(agree)[0] == 0.0
    assert tc.rank_flip_share(frame.head(1)) is None


def test_detail_rows_and_note_name_the_list_blend(frame):
    rows = dict(tc.detail_rows(frame.iloc[0], 1e6))
    assert rows["Cost / task"] == "$2.00  ·  AA-measured"
    assert rows["Token burn"] == "2M blend-eq. tok/task  ·  2× median"
    assert resolve_mix(None).short in tc.detail_note(frame.iloc[0])
    missing = dict(tc.detail_rows(frame.iloc[3], 1e6))
    assert missing == {"Cost / task": "—  ·  AA publishes none", "Token burn": "—"}
    assert tc.detail_note(frame.iloc[3]) == ""


# ── The chart ────────────────────────────────────────────────────────────────

def test_task_axis_draws_only_costed_families_and_counts_the_rest(frame):
    fig = build_pareto_scatter(frame, full_df=frame, x="task")
    assert _marker_models(fig) == {"Burner", "Lean", "Mid"}
    assert _xs(fig) == [0.10, 0.50, 2.00]
    title = fig.layout.title.text
    assert "3 of 5 model families have an AA cost per task (2 lack one; not imputed)" in title


def test_task_frontier_is_computed_on_cost_per_task(frame):
    task = build_pareto_scatter(frame, full_df=frame, x="task")
    price = build_pareto_scatter(frame, full_df=frame, x="price")
    line = [t for t in task.data if t.mode == "lines"][0]
    # Per task: Lean (0.10, 55) then Burner (2.00, 60); Mid is dominated by Lean.
    assert list(line.x) == [0.10, 2.00] and list(line.y) == [55.0, 60.0]
    assert set(_pareto_frontier(frame, "cost_per_task")["model"]) == {"Lean", "Burner"}
    # Per token, Ghost (no cost per task) is on the frontier; it must not leak in.
    assert "Ghost" in set(_pareto_frontier(frame)["model"])
    assert "Ghost" not in _marker_models(task)
    assert list([t for t in price.data if t.mode == "lines"][0].x) != list(line.x)


def test_task_axis_is_labelled_as_aa_measured_and_mix_free(frame):
    fig = build_pareto_scatter(frame, full_df=frame, x="task")
    assert fig.layout.xaxis.title.text == (
        "Cost per Intelligence Index task  (USD, log, AA-measured)  ·  MIX does not apply")
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.title.text.startswith("Cost per Task vs. Intelligence")
    ann = fig.layout.annotations[0].text
    assert "log cost/task vs quality" in ann and "cheaper per token, dearer per task" in ann


@pytest.mark.parametrize("mix", ["agentic", "chat", ["custom", 20, 50]])
def test_mix_moves_only_the_hover_price_on_the_task_axis(frame, mix):
    basis = resolve_mix(mix)
    plain = build_pareto_scatter(frame, full_df=frame, x="task")
    mixed_df = apply_mix(frame, basis)
    mixed = build_pareto_scatter(mixed_df, full_df=mixed_df, basis=basis, x="task")
    assert [list(t.x) for t in plain.data] == [list(t.x) for t in mixed.data]
    assert [list(t.y) for t in plain.data] == [list(t.y) for t in mixed.data]
    assert plain.layout.title.text == mixed.layout.title.text
    assert plain.layout.xaxis.title.text == mixed.layout.xaxis.title.text
    hp = [t.hovertemplate for t in plain.data if t.mode == "markers"]
    hm = [t.hovertemplate for t in mixed.data if t.mode == "markers"]
    assert all(basis.short in h for h in hm) and hp != hm
    # The burn ([9]) does not move with MIX; the price string ([7]) does.
    cp = {cd[5]: cd for t in plain.data if t.mode == "markers" for cd in t.customdata}
    cm = {cd[5]: cd for t in mixed.data if t.mode == "markers" for cd in t.customdata}
    assert all(cp[m][8] == cm[m][8] and cp[m][9] == cm[m][9] for m in cp)
    assert any(cp[m][7] != cm[m][7] for m in cp)


@pytest.mark.parametrize("x", ["price", "task"])
def test_customdata_layout_is_the_same_on_both_axes(frame, x):
    """[5]/[6] raw names for the click handler, [7] price, [8] cost per task,
    [9] burn: ten fields in both modes, so a later field has one index."""
    fig = build_pareto_scatter(frame, full_df=frame, x=x)
    rows = [cd for t in fig.data if t.mode == "markers" for cd in t.customdata]
    assert rows and all(len(cd) == 10 for cd in rows)
    by = {cd[5]: cd for cd in rows}
    assert by["Burner"][8] == "$2.00" and by["Burner"][9] == "2M blend-eq. tok/task"
    if x == "price":
        assert by["Ghost"][8] == "—" and by["Ghost"][9] == "—"


def test_task_hover_names_the_list_blend_for_the_burn(frame):
    fig = build_pareto_scatter(frame, full_df=frame, x="task")
    h = [t.hovertemplate for t in fig.data if t.mode == "markers"][0]
    assert "Cost / task: %{customdata[8]}  ·  AA-measured" in h
    assert f"Token burn: %{{customdata[9]}}  ·  at {resolve_mix(None).short}" in h
    price_h = [t.hovertemplate for t in build_pareto_scatter(frame, full_df=frame).data
               if t.mode == "markers"][0]
    assert "customdata[8]" not in price_h


def test_task_frontier_is_market_wide_not_filter_local(frame):
    """Filtering out Lean must not promote Mid onto the frontier."""
    shown = frame[frame["model"] != "Lean"]
    fig = build_pareto_scatter(shown, full_df=frame, x="task")
    line = [t for t in fig.data if t.mode == "lines"][0]
    assert list(line.x) == [2.00]


def test_frame_without_the_column_draws_an_empty_task_chart(frame):
    fig = build_pareto_scatter(frame.drop(columns="cost_per_task"), x="task")
    assert not _marker_models(fig)
    assert "0 of 5 model families" in fig.layout.title.text


def test_empty_frame_does_not_raise():
    build_pareto_scatter(DF.head(0), full_df=DF, x="task")


# ── static_api ───────────────────────────────────────────────────────────────

def test_update_overview_task_mode():
    fig = _decoded(api.update_overview(None, 0, "", "task", None, None))
    assert fig["layout"]["title"]["text"].startswith("Cost per Task vs. Intelligence")
    assert "MIX does not apply" in fig["layout"]["xaxis"]["title"]["text"]
    assert api.update_overview(None, 0, "", "task") != api.update_overview(None, 0, "", "price")


@real
def test_real_task_axis_counts_and_never_imputes():
    fig = _decoded(api.update_overview(None, 0, "", "task", None, None))
    drawn = {cd[5] for t in fig["data"] if t.get("mode") == "markers" for cd in t["customdata"]}
    costed = DF[pd.to_numeric(DF["cost_per_task"], errors="coerce") > 0]
    assert drawn and drawn <= set(costed["model"])
    from components.charts.constants import base_model_name
    priced = DF[(DF["price"] > 0) & (DF["quality"] > 0)]
    n_fam = priced["model"].map(base_model_name).nunique()
    n_cost = priced[priced["model"].isin(costed["model"])]["model"].map(base_model_name).nunique()
    assert len(drawn) == n_cost
    assert (f"{n_cost} of {n_fam} model families have an AA cost per task "
            f"({n_fam - n_cost} lack one; not imputed)") in fig["layout"]["title"]["text"]
    xs = [x for t in fig["data"] if t.get("mode") == "markers" for x in t["x"]]
    by_model = dict(zip(DF["model"], DF["cost_per_task"]))
    for t in fig["data"]:
        if t.get("mode") == "markers":
            for x, cd in zip(t["x"], t["customdata"]):
                assert x == pytest.approx(by_model[cd[5]])
    assert min(xs) > 0


@real
def test_real_task_frontier_is_monotone():
    fig = _decoded(api.update_overview(None, 0, "", "task", None, None))
    line = [t for t in fig["data"] if t.get("mode") == "lines"][0]
    assert line["x"] == sorted(line["x"])
    assert all(b > a for a, b in zip(line["y"], line["y"][1:]))


@real
def test_real_mix_does_not_move_the_task_axis():
    a = _decoded(api.update_overview(None, 0, "", "task", None, None))
    b = _decoded(api.update_overview(None, 0, "", "task", None, ["agentic", None, None]))
    assert [t.get("x") for t in a["data"]] == [t.get("x") for t in b["data"]]
    assert a["layout"]["title"] == b["layout"]["title"]


def _panel_text(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " | ", html))


@real
def test_detail_panel_shows_cost_per_task_and_burn_at_any_mix():
    row = DF[pd.to_numeric(DF["cost_per_task"], errors="coerce") > 0].iloc[0]
    default = _panel_text(api.model_detail(row["model"], row["provider"]))
    agentic = _panel_text(api.model_detail(row["model"], row["provider"],
                                           ["agentic", None, None]))
    burn = row["cost_per_task"] / row["price"] * 1e6
    want = f"{tc.fmt_tokens(burn)} blend-eq. tok/task"
    assert "Cost / task" in default and want in default and want in agentic
    assert tc.fmt_task_cost(row["cost_per_task"]) in default
    assert "ours 1:3 in:out (not AA&#x27;s)" in default


@real
def test_detail_panel_says_when_aa_published_no_cost():
    missing = DF[DF["cost_per_task"].isna()]
    if missing.empty:
        pytest.skip("every row has a cost per task")
    row = missing.iloc[0]
    text = _panel_text(api.model_detail(row["model"], row["provider"]))
    assert "AA publishes none" in text and "blend-eq." not in text


# ── Static site wiring ───────────────────────────────────────────────────────

def test_the_radio_offers_cost_per_task_and_the_prerender_exists():
    html = (ROOT / "docs" / "index.html").read_text()
    assert re.search(r'name="overview-xaxis" value="task" />\s*Cost / task', html)
    js = (ROOT / "docs" / "app.js").read_text()
    assert 'task: "pareto_task"' in js
    assert "captions?.overview_task" in js
    import build_static
    assert "data/task_cost.py" in Path(build_static.__file__).read_text()


def test_prerendered_task_figure_equals_the_live_default(tmp_path):
    import build_static
    build_static.export_default_figures(tmp_path)
    pre = _decoded((tmp_path / "pareto_task.json").read_text())
    now = _decoded(api.update_overview(None, 0, "", "task", None, None))
    assert pre["layout"] == now["layout"]
    assert pre["data"] == now["data"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["captions"]["overview_task"]


def test_caption_says_no_imputation_and_credits_aa():
    from captions import CAPTIONS
    text = CAPTIONS["overview_task"]
    assert "never estimated" in text and "MIX does not move it" in text
    assert "artificialanalysis.ai" in text
    assert "3:1" not in text


# ── Dash parity ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dash_app():
    import app as module
    return module


def test_dash_offers_the_task_axis_and_matches_the_static_figure(dash_app):
    src = (ROOT / "app.py").read_text()
    assert '{"label": "Cost / task", "value": "task"}' in src
    fig, desc = dash_app.update_overview(None, 0, None, "", "task", None)
    static = _decoded(api.update_overview(None, 0, "", "task", None, None))
    dash = _decoded(fig.to_json())
    assert dash["layout"]["title"] == static["layout"]["title"]
    assert [t.get("x") for t in dash["data"]] == [t.get("x") for t in static["data"]]
    from captions import CAPTIONS
    assert CAPTIONS["overview_task"] in str(desc)


@real
def test_dash_detail_panel_shows_the_same_task_rows(dash_app, monkeypatch):
    class _Ctx:
        triggered_id = "filter-mix"
    monkeypatch.setattr(dash_app, "ctx", _Ctx)
    row = DF[pd.to_numeric(DF["cost_per_task"], errors="coerce") > 0].iloc[0]
    cls, body, _ = dash_app.toggle_detail_panel(None, None, "agentic", None, None,
                                                row["model"])
    text = str(body)
    for lbl, val in tc.detail_rows(row, tc.median_token_burn(DF)):
        assert lbl in text and val in text
    assert tc.detail_note(row) in text
