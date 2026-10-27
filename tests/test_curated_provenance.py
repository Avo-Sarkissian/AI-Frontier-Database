"""Every hand-curated economics figure carries its source, its date and its
status (data/curated/*.csv; economics plan 0.2/0.3). Shared by Compute (F)
and Labs (G): a new curated file is checked by adding nothing here.

Provenance columns: source_url, as_of (ISO date), status in
econ_common.STATUS_VALUES. A row whose status is `unavailable` was looked for
and not found, so it carries no value; a `derived` row says how in its note.
"""
from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path

import pytest

import build_static
from data.econ_common import STATUS_VALUES

ROOT = Path(__file__).resolve().parent.parent
CURATED = sorted((ROOT / "data" / "curated").glob("*.csv"))
# Sources whose terms forbid what a public dashboard does with their data, in
# any form: plotted, captioned or quoted from a scrape (economics plan 0.2).
BANNED = ("ornn", "silicondata", "silicon data", "vast.ai", "runpod",
          "openrouter.ai/rankings")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return list(rd.fieldnames or []), list(rd)


def test_there_are_curated_files():
    assert CURATED, "data/curated/ is empty"


@pytest.mark.parametrize("path", CURATED, ids=lambda p: p.name)
def test_every_curated_file_has_the_provenance_columns(path):
    header, rows = _rows(path)
    for col in ("source_url", "as_of", "status"):
        assert col in header, f"{path.name} lacks {col}"
    assert rows, f"{path.name} has no rows"


@pytest.mark.parametrize("path", CURATED, ids=lambda p: p.name)
def test_every_curated_row_is_sourced_dated_and_labelled(path):
    header, rows = _rows(path)
    today = date.today()
    for i, r in enumerate(rows, start=2):
        where = f"{path.name}:{i}"
        status = r["status"].strip()
        assert status in STATUS_VALUES, f"{where}: status {status!r}"
        assert _ISO.match(r["as_of"].strip()), f"{where}: as_of {r['as_of']!r}"
        assert date.fromisoformat(r["as_of"].strip()) <= today, f"{where}: as_of in the future"
        value = r.get("value", "").strip()
        if status == "unavailable":
            assert value == "", f"{where}: an unavailable figure has a value"
            continue
        assert r["source_url"].startswith("https://"), f"{where}: source_url {r['source_url']!r}"
        if value:
            float(value)                          # numeric, or this raises
        if status == "derived":
            assert len(r.get("note", "").strip()) > 20, f"{where}: a derived row must say how"


def test_no_banned_sources_in_curated_data():
    paths = CURATED + [ROOT / "data" / "raw" / "econ" / "hyperscaler_gpu_prices.csv"]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for host in BANNED:
            assert host not in text, f"{path.name} references {host}"


def test_every_curated_file_ships_to_the_browser():
    shipped = set(build_static.DATA_CSVS)
    for path in CURATED:
        rel = path.relative_to(ROOT).as_posix()
        assert rel in shipped, f"{rel} is not in build_static.DATA_CSVS"
