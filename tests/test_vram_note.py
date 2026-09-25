"""The "usable of N GB" note beside the VRAM boxes.

Unified-memory presets put less than their RAM in the box (usable_memory_gb),
so "Apple M4 Max (128 GB)" fills in 96. The note says why. It is shown only
while the box holds the preset's own figure, in both shells, from one source
(data/local_models.memory_note).
"""
import re
from pathlib import Path

from data.local_models import GPUS, GPU_BY_NAME, memory_note
from static_helpers import gpu_preset_options

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "docs" / "app.js").read_text()
HTML = (ROOT / "docs" / "index.html").read_text()
APP_PY = (ROOT / "app.py").read_text()

MAC = "Apple M4 Max (128 GB)"
IPHONE = "A19 Pro (iPhone 17 Pro)"
SNAPDRAGON = "Snapdragon X Elite (32 GB)"


def test_a_mac_preset_names_its_total_and_the_wired_limit():
    note = memory_note(MAC)
    assert note["text"] == "usable of 128 GB"
    assert "3/4" in note["title"] and "iogpu.wired_limit_mb" in note["title"]
    # <=36 GB Macs wire 2/3, and the tooltip must say the fraction it applied
    assert "2/3" in memory_note("Apple M4 Max (36 GB)")["title"]


def test_phone_and_desktop_presets_name_their_reserve():
    assert memory_note(IPHONE)["text"] == "usable of 12 GB"
    assert "2 GB" in memory_note(IPHONE)["title"]
    assert memory_note(SNAPDRAGON)["text"] == "usable of 32 GB"
    assert "4 GB" in memory_note(SNAPDRAGON)["title"]


def test_a_discrete_card_has_no_note():
    assert memory_note("NVIDIA RTX 5090") is None
    assert memory_note("no such preset") is None
    assert memory_note(None) is None


def test_the_note_applies_only_while_the_box_holds_the_preset_figure():
    usable = GPU_BY_NAME[MAC]["vram_gb"]
    assert memory_note(MAC, usable) is not None
    assert memory_note(MAC, str(usable)) is not None      # Dash may hand a string
    assert memory_note(MAC, usable + 4) is None           # a typed figure
    assert memory_note(MAC, "") is None                   # a cleared box
    assert memory_note(MAC, "abc") is None


def test_every_unified_memory_preset_ships_its_note_and_no_other_does():
    by_value = {o["value"]: o for o in gpu_preset_options()}
    for g in GPUS:
        shipped = by_value[g["name"]]["memory_note"]
        if g.get("ram_gb") is None:
            assert shipped is None, g["name"]
        else:
            assert shipped == memory_note(g["name"]), g["name"]
            assert shipped["text"].startswith("usable of "), g["name"]


def test_both_shells_render_the_note_beside_both_vram_boxes():
    for box in ("local-vram", "recommend-vram"):
        assert f'<span id="{box}-note" class="filter-note"></span>' in HTML
        assert f'html.Span(id="{box}-note", className="filter-note")' in APP_PY
        assert f'Output("{box}-note", "children")' in APP_PY


def test_the_browser_resyncs_the_note_on_preset_and_on_typing():
    assert "syncVramNote(inputId, hw);" in APP_JS            # inside setVramFromPreset
    assert re.search(r'localVram\.oninput = \(\) => \{[^}]*syncVramNote\("local-vram"', APP_JS)
    assert re.search(r'recVram\.oninput = \(\) => \{[^}]*syncVramNote\("recommend-vram"', APP_JS)
    assert "memory_note: o.memory_note" in APP_JS
