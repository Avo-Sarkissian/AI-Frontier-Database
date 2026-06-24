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
