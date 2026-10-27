"""Shared plumbing for the Economics tab's data, Pyodide-safe (stdlib + pandas).

The economics archives (data/carried/econ/, data/raw/epoch/) are too large to
ship to the browser: the InferenceX points archive alone is ~700 KB against a
pydata.zip budget of 150 KB. Each Economics view therefore reads a small
DERIVED slice of them, written into pydata.zip at build time by
build_static.DERIVED_DATA (never committed: data/derived/ is git-ignored).

`read_derived(rel_path, compute)` is the one reader. In the browser the slice
exists at /bundle/<rel_path> and is read from there; locally, in Dash and in
the tests it does not exist, and `compute()` builds the same CSV text from the
archives. Both paths parse the SAME CSV text, so the frame a test checks is the
frame a visitor gets.

Provenance vocabulary for every hand-curated economics CSV (data/curated/) and
for every hard-coded figure in an economics module lives here too, so a status
word means one thing everywhere.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# published   a primary document, filing, vendor document or official API
# reported    press, or a secondary source citing documents
# derived     our arithmetic on cited inputs (the note says how)
# estimated   an analyst figure or our own assumption
# unavailable looked for and not found
STATUS_VALUES: tuple[str, ...] = ("published", "reported", "derived", "estimated",
                                  "unavailable")


# The exact texts a build is about to ship (build_static.shipped_derived sets
# them before pre-rendering and clears them after). read_derived prefers them
# over everything, so a pre-rendered figure is drawn from the bytes the browser
# will read: a slice shipped EMPTY (failed or over budget) used to pre-render
# from the full archive and then lose its points on the first control change.
_OVERRIDES: dict[str, str] = {}


def set_derived_override(rel_path: str, text: str) -> None:
    _OVERRIDES[rel_path] = text or ""


def clear_derived_overrides() -> None:
    _OVERRIDES.clear()


def derived_path(rel_path: str) -> Path:
    return ROOT / rel_path


def read_derived(rel_path: str, compute: Callable[[], str]) -> pd.DataFrame:
    """The derived slice at `rel_path`, or the same CSV computed from archives.

    `compute` returns CSV TEXT (the same callable build_static.DERIVED_DATA
    writes into pydata.zip), so both branches go through one pd.read_csv and
    cannot disagree about dtypes. A compute that fails or returns nothing gives
    an empty frame rather than raising: an Economics chart with no data says so
    on screen, and a raise here would take every other tab's import down with it.
    """
    p = derived_path(rel_path)
    if rel_path in _OVERRIDES:
        text = _OVERRIDES[rel_path]
    elif p.exists():
        text = p.read_text(encoding="utf-8")
    else:
        try:
            text = compute() or ""
        except Exception as exc:          # archives absent or malformed
            print(f"[econ] could not derive {rel_path}: {exc}")
            text = ""
    if not text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))
