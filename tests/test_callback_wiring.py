"""Every callback dependency must point at a component that exists.

Dash resolves callback ids at *runtime*, and `suppress_callback_exceptions=True`
(app.py:107) silences the startup warning — so a callback wired to a deleted or
renamed component fails only when a user interacts with it, in the browser, as a
console ReferenceError nobody sees. That is how the Overview detail panel stayed
dead: it listened to `quadrant-chart`, an id that exists nowhere in the layout,
so clicking a bubble did nothing while the static site's equivalent worked fine.

This walks the real layout and the real callback registry. It is the cheapest
guard in the suite against a whole class of silent breakage.
"""
import os
import re

import pytest

# Importing app.py starts three background scrapers (network) unless DEBUG is
# set and this process is not a Werkzeug worker child — see app.py:51-56.
os.environ.setdefault("DEBUG", "true")
os.environ.pop("WERKZEUG_RUN_MAIN", None)

import app as app_module  # noqa: E402
from dash._callback import GLOBAL_CALLBACK_MAP  # noqa: E402


def _layout_ids(node, found=None):
    """Every string `id=` in the rendered layout tree."""
    if found is None:
        found = set()
    if isinstance(node, (list, tuple)):
        for child in node:
            _layout_ids(child, found)
        return found
    cid = getattr(node, "id", None)
    if isinstance(cid, str):
        found.add(cid)
    children = getattr(node, "children", None)
    if children is not None:
        _layout_ids(children, found)
    return found


def _output_ids(key: str, spec: dict) -> set:
    """Ids on the output side.

    `spec["output"]` is a list of Output objects for multi-output callbacks and a
    single Output for one — fall back to parsing the registry key
    (`..a.prop...b.prop..`) only if neither exposes component_id.
    """
    raw = spec.get("output")
    outputs = raw if isinstance(raw, (list, tuple)) else [raw]
    ids = set()
    for out in outputs:
        cid = getattr(out, "component_id", None)
        if isinstance(cid, str):
            ids.add(cid)
    if ids:
        return ids

    for piece in str(key).strip(".").split("..."):
        piece = piece.strip(".")
        if not piece or "." not in piece:
            continue
        cid = piece.rsplit(".", 1)[0]
        # Pattern-matching ids are JSON objects; this test only covers string ids.
        if not cid.startswith("{"):
            # Dash appends @<hash> to ids in some multi-output keys.
            ids.add(cid.split("@", 1)[0])
    return ids


def _dependency_ids(spec: dict) -> set:
    ids = set()
    for dep in list(spec.get("inputs") or []) + list(spec.get("state") or []):
        cid = dep.get("id") if isinstance(dep, dict) else getattr(dep, "component_id", None)
        if isinstance(cid, str):
            ids.add(cid)
    return ids


def test_layout_exposes_ids():
    """Guard the guard: if the walker returns nothing, the real test is vacuous."""
    assert len(_layout_ids(app_module.app.layout)) > 20


def test_every_callback_dependency_resolves_to_a_real_component():
    layout_ids = _layout_ids(app_module.app.layout)
    assert GLOBAL_CALLBACK_MAP, "no callbacks registered — import order changed?"

    dangling = {}
    for key, spec in GLOBAL_CALLBACK_MAP.items():
        missing = (_dependency_ids(spec) | _output_ids(key, spec)) - layout_ids
        if missing:
            dangling[key] = sorted(missing)

    assert not dangling, (
        "callback(s) reference component ids that are not in app.layout — Dash "
        "raises a runtime ReferenceError and the feature silently does nothing:\n"
        + "\n".join(f"  {k}\n      missing: {v}" for k, v in dangling.items())
    )


# ── The callback BODY, not just its wiring ───────────────────────────────────

def test_the_detail_panel_callback_body_actually_runs(monkeypatch):
    """This file proves every callback points at a component that exists. It
    does not prove the callback RUNS — and on 2026-09-08 the Overview detail
    panel was wired correctly while its body called `_index_label`, a name that
    existed only under a different spelling after the helper moved to
    static_helpers. Dash resolves that at click time, in the browser, as a
    console error nobody sees: the same silence this file was written for, one
    layer in.

    So: drive the real callback with a real row and let a NameError be a test
    failure rather than a dead panel."""
    class _Ctx:
        triggered_id = "pareto-chart"

    monkeypatch.setattr(app_module, "ctx", _Ctx)
    row = app_module.df.iloc[0]
    click = {"points": [{"customdata": [row["model"], row["provider"]]}]}

    class_name, body, name = app_module.toggle_detail_panel(click, None)

    assert class_name == "detail-panel open"
    assert name == row["model"]
    rendered = str(body)
    for label in ("INTELLIGENCE", "Price", "Coding", "Agentic", "Omniscience"):
        assert label in rendered, f"the panel body never rendered {label}"


def test_the_dash_and_static_panels_agree_about_an_estimated_score(monkeypatch):
    """app.py renders the Dash panel and static_api renders the published site's.
    Two implementations of one claim drift; this pins that they still agree on
    the claim that matters most — whether AA measured this score or guessed it."""
    import static_api as api

    est = app_module.df[app_module.df["quality_estimated"].eq(True)]
    if est.empty:
        pytest.skip("no AA-estimated scores in the catalogue")
    row = est.iloc[0]

    class _Ctx:
        triggered_id = "pareto-chart"

    monkeypatch.setattr(app_module, "ctx", _Ctx)
    click = {"points": [{"customdata": [row["model"], row["provider"]]}]}
    _, dash_body, _ = app_module.toggle_detail_panel(click, None)

    static_html = api._detail_html(row, str(row["provider"]))
    assert "est" in str(dash_body).lower(), "the Dash panel does not mark the estimate"
    assert "estimated" in static_html.lower(), "the static panel does not mark the estimate"
