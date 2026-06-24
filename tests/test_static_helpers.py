# tests/test_static_helpers.py
from data.ingest import get_models
import static_helpers as h

def test_apply_filters_provider_and_quality():
    df = get_models()
    out = h.apply_filters(df, ["Anthropic"], 40, "")
    assert len(out) > 0 and (out["provider"] == "Anthropic").all() and (out["quality"] >= 40).all()

def test_compute_diverse5_returns_up_to_five():
    df = get_models()
    picks = h.compute_diverse5(df)
    assert 0 < len(picks) <= 5 and len(set(picks)) == len(picks)

def test_ctx_to_k_and_quality_label():
    assert h.ctx_to_k("1m") == 1000 and h.ctx_to_k("128k") == 128
    assert h.quality_label(95) == "Exceptional" and h.quality_label(10) == "Limited"

def test_options_shape():
    df = get_models()
    assert all(set(o) == {"label", "value"} for o in h.provider_options(df))
    assert all(set(o) == {"label", "value"} for o in h.model_options(df))
