"""2026-10-29 audit: captions had grown to 2-4k characters (the Serve caption
stood ~1,000px tall on a 390px phone before its chart). Each is now one or two
sentences on how to read the chart; method, sources and caveats moved into a
closed "Method & sources" disclosure. These pin the split, not the wording."""

import re
from pathlib import Path

import pytest

from captions import CAPTIONS, CAPTION_TEXT, METHODS, link_segments, method_payload

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "docs" / "app.js").read_text()
APP_PY = (ROOT / "app.py").read_text()
CSS = (ROOT / "assets" / "style.css").read_text()


def _sentences(text: str) -> int:
    # A sentence ends at ". " / "." at the end; decimals and "e.g." never occur.
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


@pytest.mark.parametrize("key", sorted(CAPTIONS))
def test_every_caption_is_one_or_two_sentences(key):
    cap = CAPTIONS[key]
    assert _sentences(cap) <= 2, f"{key}: {_sentences(cap)} sentences"
    assert len(cap) <= 300, f"{key}: {len(cap)} characters — move detail to METHODS"


def test_long_captions_kept_their_detail_in_the_disclosure():
    for key in ("economics_serve", "economics_compute", "economics_labs",
                "economics_open_gap", "local", "overview_task"):
        assert sum(map(len, METHODS[key])) > 3 * len(CAPTIONS[key]), key


@pytest.mark.parametrize("credit, key", [
    ("Artificial Analysis", "economics_serve"),
    ("SemiAnalysis InferenceX", "economics_serve"),
    ("SemiAnalysis InferenceX", "economics_compute"),
    ("GetDeploying (CC BY 4.0", "economics_compute"),
    ("Epoch AI, 'Data on Machine Learning Hardware' (CC BY 4.0)", "economics_compute"),
    ("Epoch AI, \"Data on AI companies\"", "economics_labs"),
    ("CC BY 4.0", "economics_labs"),
    ("artificialanalysis.ai", "overview_task"),
    ("artificialanalysis.ai", "economics_open_gap"),
])
def test_required_attributions_are_still_published(credit, key):
    assert credit in " ".join(METHODS[key]), f"{credit!r} left the {key} disclosure"
    assert credit in CAPTION_TEXT[key]


def test_source_domains_become_links_but_templates_do_not():
    segs = link_segments("see inferencex.semianalysis.com/about: and "
                         "inferencex.semianalysis.com/chips/<chip>; openrouter.ai/state-of-ai).")
    links = [s["href"] for s in segs if isinstance(s, dict)]
    assert links == ["https://inferencex.semianalysis.com/about",
                     "https://openrouter.ai/state-of-ai"]
    assert "".join(s if isinstance(s, str) else s["text"] for s in segs).startswith("see ")
    hrefs = {s["href"] for paras in method_payload().values() for p in paras
             for s in p if isinstance(s, dict)}
    assert "https://getdeploying.com/dataset/gpu-prices" in hrefs
    assert "https://epoch.ai/data-insights/open-closed-eci-gap" in hrefs


def test_both_renderings_put_method_in_a_closed_details():
    assert 'document.createElement("details")' in APP_JS
    assert 'sum.textContent = "Method & sources"' in APP_JS
    assert ".open = true" not in APP_JS.split("function fillCaption")[1].split("\n}\n")[0]
    assert 'html.Details([' in APP_PY and 'html.Summary("Method & sources")' in APP_PY
    assert "open=True" not in APP_PY.split("def _desc(")[1].split("\ndef ")[0]
    # no hard-coded caption prose left in app.js: the Overview reads the manifest
    assert "OVERVIEW_CAPTIONS =" not in APP_JS


def test_dash_renders_the_disclosure():
    import app as dash_app
    div = dash_app._desc("economics_serve")
    lede, details = div.children
    assert lede.children == CAPTIONS["economics_serve"]
    assert details.children[0].children == "Method & sources"
    assert len(details.children) == 1 + len(METHODS["economics_serve"])
    assert getattr(details, "open", None) in (None, False)


# ── Table readability ───────────────────────────────────────────────────────

def _lum(hex_):
    h = hex_.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_table_header_contrast_and_sticky_cells():
    block = CSS[CSS.index(".model-table th {"):]
    block = block[:block.index("}")]
    color = re.search(r"color:\s*(#[0-9a-fA-F]{6})", block).group(1)
    assert _contrast(color, "#111111") >= 4.5
    assert "position: sticky" in block and "top: 0" in block
    first = CSS[CSS.index(".model-table td:first-child {"):]
    assert "left: 0" in first[:first.index("}")]
    assert "font-size:9px" not in APP_JS.split("function renderTableRows")[1][:3000]


def test_provider_colour_is_a_dot_not_the_text():
    rows = APP_JS.split("function renderTableRows")[1][:4000]
    assert 'class="prov-dot"' in rows
    assert "color:${pcolor}" not in rows
    import app as dash_app
    md = dash_app._provider_dot_md("Z AI")
    assert 'class="prov-dot"' in md and md.endswith("Z AI")
    assert "<script" not in dash_app._provider_dot_md("<script>x</script>")
