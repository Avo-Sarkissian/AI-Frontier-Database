# tests/test_static_api.py
import json
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


def test_export_csv_is_text():
    csv = api.export_csv(["Anthropic"], 0, "")
    assert "model" in csv.splitlines()[0]
