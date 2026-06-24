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
