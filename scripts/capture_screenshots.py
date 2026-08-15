"""Regenerate the README screenshots from the built static site.

The six images in screenshots/ went stale and stayed stale: the hero shot read
"324 MODELS TRACKED / $0.010 FLOOR / 59.9 PEAK" against a catalogue of 155, and
its table showed models (Claude Opus 4.8, GPT-5.5 xhigh) that no longer exist.
That figure is almost certainly where the README's "300+ models" claim came
from — a screenshot outlived the data and then the prose copied it.

Two rules, both encoded below:

1. **The header stat bar is hidden before every capture.** Those numbers change
   hourly, so any image containing them is wrong within the hour. The charts
   themselves age gracefully; the counters do not.
2. **This is a script, not a manual pass.** Regenerating had to be cheap or it
   would not happen again.

Usage:
    .venv/bin/python build_static.py            # capture what actually ships
    .venv/bin/python scripts/capture_screenshots.py
"""
from __future__ import annotations

import http.server
import socketserver
import sys
import threading
from functools import partial
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "screenshots"
PORT = 8799
VIEWPORT = {"width": 1600, "height": 1000}

# tab id -> (filename, extra settle time in ms, full page?)
SHOTS = [
    ("overview",  "overview.png",    1200, False),
    ("recommend", "agent-stack.png", 2500, False),
    ("landscape", "landscape.png",   1500, False),
    ("rankings",  "rankings.png",    1500, False),
    ("compare",   "compare.png",     2000, False),
    ("table",     "table.png",       2000, False),
]

# Hide anything whose value changes hourly, plus the transient boot banner.
HIDE_CSS = """
  .stat-bar, #py-status { display: none !important; }
  #data-freshness { visibility: hidden !important; }
"""


def _serve(directory: Path, port: int) -> socketserver.TCPServer:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> int:
    if not (DOCS / "index.html").exists():
        print("docs/index.html missing — run build_static.py first", file=sys.stderr)
        return 1
    OUT.mkdir(exist_ok=True)

    httpd = _serve(DOCS, PORT)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
            page.goto(f"http://127.0.0.1:{PORT}/index.html", wait_until="load")

            # Wait for the in-browser Python runtime: several tabs render nothing
            # meaningful until it is up, and a screenshot taken early captures
            # the pre-rendered snapshot rather than the live product.
            page.wait_for_function("() => window.AF && window.AF.pyReady", timeout=180_000)
            page.add_style_tag(content=HIDE_CSS)

            for tab, filename, settle, full in SHOTS:
                page.evaluate("id => switchTab(id)", tab)
                page.wait_for_timeout(settle)
                page.screenshot(path=str(OUT / filename), full_page=full)
                print(f"  captured {filename}")

            browser.close()
    finally:
        httpd.shutdown()

    print(f"\n{len(SHOTS)} screenshots written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
