# tests/test_static_api.py
import json
import pandas as pd
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
