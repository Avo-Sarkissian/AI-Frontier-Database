"""
Tests for the HTML-string renderer in stack_recommender.py.
TDD: write test first (RED), then implement (GREEN).
"""
import pandas as pd
from data.ingest import get_models
from components.stack_recommender import build_stack_cards, build_stack_cards_html


def test_html_renderer_returns_string_with_same_tiers():
    df = get_models()
    html = build_stack_cards_html(df, ["Anthropic", "Google", "OpenAI"], mode="api")
    assert isinstance(html, str) and len(html) > 200
    # Tier labels present in both renderers.
    for tier in ["Fast", "Balanced", "Reasoning"]:
        assert tier.lower() in html.lower()


def test_dash_renderer_still_works():
    df = get_models()
    comp = build_stack_cards(df, ["Anthropic"], mode="api")
    assert hasattr(comp, "children")
    # three tier cards (Fast / Balanced / Reasoning)
    assert len(comp.children) == 3


def test_html_renderer_contains_model_names():
    """HTML output should include actual model name text (not just tier labels)."""
    df = get_models()
    html = build_stack_cards_html(df, ["Anthropic", "Google", "OpenAI"], mode="api")
    # "tok/s" only appears when speed chip rows render — proves model rows are present
    assert "tok/s" in html


def test_html_renderer_contains_advice_sections():
    """HTML output should contain advisor sections (best_for, tradeoff, avoid_if)."""
    df = get_models()
    html = build_stack_cards_html(df, ["Anthropic", "Google", "OpenAI"], mode="api")
    # check key advice labels appear
    assert "BEST FOR" in html
    assert "TRADEOFF" in html
    assert "AVOID IF" in html


def test_html_renderer_local_mode():
    """HTML renderer should handle local mode gracefully (empty picks → no crash)."""
    df = get_models()
    html = build_stack_cards_html(df, [], mode="local", local_df=None)
    assert isinstance(html, str)
    for tier in ["Fast", "Balanced", "Reasoning"]:
        assert tier.lower() in html.lower()


def test_row_shell_escapes_scraped_text():
    """build_stack_cards_html output is assigned to innerHTML at docs/app.js:494,
    so the scraped model name and provider/family label must not reach it as
    live markup."""
    from components.stack_recommender import _row_shell_html
    out = _row_shell_html(
        "Foo<img src=x onerror=alert(1)>", 61.2,
        'Evil"><script>alert(2)</script>', "#cc4104", "", is_top=True,
    )
    for tag in ("<img", "<script", "</script>"):
        assert tag not in out
    assert "&lt;img src=x onerror=alert(1)&gt;" in out
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in out
    assert "&quot;&gt;" in out           # the attribute-breaking quote is escaped too


def test_row_shell_preserves_plain_text():
    """Escaping must not change how ordinary values render."""
    from components.stack_recommender import _row_shell_html
    out = _row_shell_html("Claude Opus 4.1", 61.2, "Anthropic", "#cc4104", "", is_top=False)
    assert ">Claude Opus 4.1</span>" in out
    assert ">Anthropic</div>" in out


def test_real_cards_carry_no_unescaped_angle_brackets_from_data():
    """End-to-end over the committed catalog: every model name rendered into the
    cards survives as text, not markup."""
    df = get_models()
    poisoned = df.copy()
    poisoned.loc[poisoned.index[0], "model"] = "Pwn<svg onload=alert(1)>"
    html = build_stack_cards_html(poisoned, list(poisoned["provider"].unique()[:5]), mode="api")
    assert "<svg" not in html
    assert "&lt;svg onload=alert(1)&gt;" in html


def test_select_stack_returns_dict():
    """select_stack returns a dict with a 'tiers' key containing 3 entries."""
    from components.stack_recommender import select_stack
    df = get_models()
    data = select_stack(df, ["Anthropic", "Google", "OpenAI"], mode="api")
    assert isinstance(data, dict)
    assert "tiers" in data
    assert len(data["tiers"]) == 3
    for t in data["tiers"]:
        assert "name" in t
        assert "picks" in t
        assert "source" in t
        assert isinstance(t["picks"], pd.DataFrame)
