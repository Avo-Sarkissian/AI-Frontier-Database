# tests/test_static_api.py
import json
import pandas as pd
import pytest
import static_api as api


def test_overview_returns_figure_json():
    fig = json.loads(api.update_overview([], 0, "", "price"))
    assert "data" in fig and "layout" in fig


def test_overview_speed_mode_uses_quadrant():
    # quadrant + pareto differ; just assert valid figure for speed axis.
    fig = json.loads(api.update_overview([], 0, "", "speed"))
    assert "data" in fig
    speed = api.update_overview([], 0, "", "speed")
    price = api.update_overview([], 0, "", "price")
    assert speed != price


def test_filters_reduce_rows_via_table():
    full = json.loads(api.update_table([], 0, "", "quality", "desc"))
    anthropic = json.loads(api.update_table(["Anthropic"], 0, "", "quality", "desc"))
    assert 0 < len(anthropic) < len(full)
    assert all(r["provider"] == "Anthropic" for r in anthropic)


def test_compare_caps_at_five_and_returns_parts():
    out = json.loads(api.update_compare([], 0, "", [], "filter-provider"))
    assert len(out["value"]) <= 5
    assert "figure" in out and "raw_table_html" in out and out["options"]


def _row(model, provider, context):
    """One scraped-shaped row, as get_models() would produce it."""
    return {
        "model": model, "provider": provider, "context": context,
        "quality": 50.0, "price": 1.5, "speed": 120.0, "latency": 0.42,
    }


def test_raw_table_escapes_scraped_names():
    """Scraped model/provider/context text must never reach app.js as live markup."""
    model = "Foo<img src=x onerror=alert(1)>"
    provider = 'Evil"><script>alert(2)</script>'
    df = pd.DataFrame([_row(model, provider, "<b>128k</b>")])
    out = api._build_raw_table_html(df, [model])
    for tag in ("<img", "<script", "</script>", "<b>"):
        assert tag not in out
    assert "&lt;img src=x onerror=alert(1)&gt;" in out
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in out
    assert "&quot;&gt;" in out           # the attribute-breaking quote is escaped too
    assert "&lt;b&gt;128k&lt;/b&gt;" in out


def test_raw_table_preserves_plain_names():
    """Escaping must not change how ordinary values render."""
    df = pd.DataFrame([_row("Claude Opus 4.1", "Anthropic", "200k")])
    out = api._build_raw_table_html(df, ["Claude Opus 4.1"])
    assert ">Claude Opus 4.1</td>" in out
    assert ">Anthropic</td>" in out
    assert ">200k</td>" in out


def test_detail_panel_escapes_scraped_names():
    """The detail panel is the same sink class and must escape too."""
    model = "Bar<svg onload=alert(1)>"
    provider = "Acme<script>alert(2)</script>"
    out = api._detail_html(pd.Series(_row(model, provider, "<i>1M</i>")), provider)
    for tag in ("<svg", "<script", "</script>", "<i>"):
        assert tag not in out
    assert "&lt;svg onload=alert(1)&gt;" in out
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in out
    assert "&lt;i&gt;1M&lt;/i&gt;" in out


def test_detail_panel_preserves_plain_names():
    out = api._detail_html(pd.Series(_row("Gemini 3 Pro", "Google", "1M")), "Google")
    assert ">Gemini 3 Pro</div>" in out
    assert ">Google</div>" in out
    assert ">1M</span>" in out


def test_export_csv_is_text():
    csv = api.export_csv(["Anthropic"], 0, "")
    assert "model" in csv.splitlines()[0]


# ── Task 5 tests ──────────────────────────────────────────────────────────────

def test_local_returns_two_figs():
    out = json.loads(api.update_local(32, 1, "Q4", 1792, "nvidia", None))
    assert "scatter" in out and "compat" in out


def test_recommend_modes_toggle_rows():
    api_only = json.loads(api.update_recommend(["Anthropic"], "api", "NVIDIA RTX 5090", 32, 1, "Q4"))
    local = json.loads(api.update_recommend([], "local", "NVIDIA RTX 5090", 32, 1, "Q4"))
    assert api_only["show_hw"] is False and api_only["show_providers"] is True
    assert local["show_hw"] is True and local["show_providers"] is False
    assert "<" in api_only["cards_html"]


def test_video_and_image():
    assert "data" in json.loads(api.update_image(None, None))
    assert "rankings" in json.loads(api.update_video(None, None))


# ── Budget tab: cheapest model above a minimum intelligence ──────────────────

def _budget_df():
    """Three priceable models: cheap+dumb, mid, expensive+smart."""
    return pd.DataFrame([
        {"model": "Cheap One", "provider": "Alibaba", "quality": 12.0,
         "price": 0.10, "speed": 100.0, "latency": 1.0, "context": "128k"},
        {"model": "Middle One", "provider": "Google", "quality": 34.0,
         "price": 2.00, "speed": 100.0, "latency": 1.0, "context": "128k"},
        {"model": "Smart One", "provider": "Anthropic", "quality": 55.0,
         "price": 9.00, "speed": 100.0, "latency": 1.0, "context": "1m"},
    ])


def test_cheapest_above_picks_the_cheapest_qualifying_model():
    from components.charts.cost_calc import cheapest_above
    df = _budget_df()
    # With no floor the answer is simply the cheapest model.
    assert cheapest_above(df, 0)["model"] == "Cheap One"
    # A floor of 20 rules out the cheap one, so the mid model wins.
    best = cheapest_above(df, 20)
    assert best["model"] == "Middle One"
    assert best["n_qualifying"] == 2
    # A floor above the mid model leaves only the expensive one.
    assert cheapest_above(df, 40)["model"] == "Smart One"


def test_cheapest_above_scales_cost_with_token_volume():
    from components.charts.cost_calc import cheapest_above
    best = cheapest_above(_budget_df(), 20, monthly_tokens_m=10.0)
    assert best["monthly_cost"] == pytest.approx(20.0)   # 10M tokens x $2.00/M


def test_cheapest_above_returns_none_when_nothing_qualifies():
    """The empty case must be representable, not silently the whole list."""
    from components.charts.cost_calc import cheapest_above
    assert cheapest_above(_budget_df(), 99) is None


def test_cost_calc_chart_honours_the_floor():
    from components.charts.cost_calc import build_cost_calc
    df = _budget_df()
    unfiltered = build_cost_calc(df, monthly_tokens_m=1.0)
    filtered = build_cost_calc(df, monthly_tokens_m=1.0, min_quality=40)
    # The cost-bar trace is the second one (index 0 is the background track).
    assert len(unfiltered.data[1].x) == 3
    assert len(filtered.data[1].x) == 1
    assert filtered.data[1].x[0] == pytest.approx(9.0)


def test_cost_calc_chart_says_so_when_nothing_qualifies():
    """A blank chart reads as broken; the empty state has to be explicit."""
    from components.charts.cost_calc import build_cost_calc
    fig = build_cost_calc(_budget_df(), min_quality=99)
    text = " ".join(a.text for a in fig.layout.annotations)
    assert "No model scores 99 or higher" in text


def test_update_cost_calc_returns_figure_and_answer():
    out = json.loads(api.update_cost_calc(1, [], 0, "", 40))
    assert "data" in out["figure"] and "layout" in out["figure"]
    assert out["floor"] == 40
    assert out["best"]["quality"] >= 40


def test_update_cost_calc_floor_composes_with_the_global_filter():
    """The budget floor intersects the global MIN SCORE rather than replacing it."""
    loose = json.loads(api.update_cost_calc(1, [], 0, "", 0))["best"]
    strict = json.loads(api.update_cost_calc(1, [], 0, "", 45))["best"]
    assert strict["quality"] >= 45
    assert strict["monthly_cost"] >= loose["monthly_cost"]


def test_update_cost_calc_tolerates_a_junk_floor():
    """The slider is a DOM value; a non-numeric one must not blow up the tab."""
    out = json.loads(api.update_cost_calc(1, [], 0, "", "not-a-number"))
    assert out["floor"] == 0
    assert out["best"] is not None
