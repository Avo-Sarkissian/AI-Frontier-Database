"""Reading model records out of Artificial Analysis's React Server Components pages.

WHY THIS IS A SHARED MODULE
---------------------------
Two catalogues are scraped this way. The image arena moved here first — its
``/api/text-to-image/arena/preferences`` endpoint became key-gated and answered
every request with ``400 {"error":"User key is required"}``, so the parser was
rewritten to read the same records out of the page's own RSC payload (see
data/image_scraper.py). The video arena has no JSON endpoint at all: every
``/api/data/website/video-*`` path 404s, and ``/video/models`` renders its six
leaderboards server-side exactly the way ``/image/models`` does.

Both therefore need the same three steps, and the bracket-matching in
``slice_json_array`` is subtle enough that a second hand-written copy would be a
place for the two catalogues to drift apart. It lives here once.

WHAT NEXT.JS ACTUALLY SHIPS
---------------------------
The page streams its flight payload as a series of ``self.__next_f.push([1,"…"])``
calls, each carrying one JS string literal. Concatenating the decoded literals
yields a single string that holds the model records — but it is *one long string*,
not a JSON document, so the tail cannot be handed to ``json.loads``. Finding
where an array ends means matching brackets while respecting string escapes.

This module is deliberately NOT bundled for Pyodide: it is scraper-side only,
like data/scraper.py and data/image_scraper.py. The browser reads the CSVs those
scrapers commit.
"""
from __future__ import annotations

import json
import re

# Each push carries one JS string literal; `("(?:[^"\\]|\\.)*")` keeps escaped
# quotes inside the literal from ending the match early.
_CHUNK_RE = re.compile(r'self\.__next_f\.push\(\[1,\s*("(?:[^"\\]|\\.)*")\]\)')


def payload_from_html(html: str) -> str:
    """The concatenated flight payload for one AA page.

    Raises ValueError when the page carries no RSC chunks at all, which is what
    a redesign — or a bot wall serving a different document — looks like from
    here. Callers treat that as a failed scrape and keep the previous cache.
    """
    chunks = _CHUNK_RE.findall(html)
    if not chunks:
        raise ValueError("no RSC payload found — page structure changed")
    return "".join(json.loads(c) for c in chunks)


def slice_json_array(text: str, start: int) -> str:
    """The complete JSON array beginning at ``text[start]``.

    Bracket-matched rather than parsed, because ``text`` is the whole flight
    payload and the array is embedded in it. Brackets inside string literals are
    skipped, and a backslash escape inside a string never ends it.
    """
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unterminated JSON array in RSC payload")


def find_array(payload: str, key: str, where=None) -> list | None:
    """The first array-valued occurrence of ``"<key>":`` that ``where`` accepts.

    Two different hazards make "just take the first hit" wrong:

    * The key is routinely also a UI label — ``"textToImage":"Text to Image"``,
      ``"textToVideo":"Text to Video"`` — so the first occurrence is often a
      string rather than the records. Non-array values are skipped.
    * The same key can carry two DIFFERENT arrays. On /leaderboards/models
      ``"models"`` appears twice, both 609 long: first the lightweight picker
      index (6 fields — name, slug, creator, releaseDate, isReasoning,
      deprecated), then the metrics records (94 fields). Taking the first gave a
      full-length array with every metric missing, which is the silent-degradation
      shape this project keeps paying for — a healthy row count with empty columns.

    ``where`` receives the decoded array and returns True if it is the one
    wanted; pass it whenever a page might carry more than one array under the
    key. ``None`` means no matching occurrence exists.
    """
    needle = f'"{key}":'
    at = payload.find(needle)
    while at != -1:
        cursor = at + len(needle)
        while cursor < len(payload) and payload[cursor].isspace():
            cursor += 1
        if cursor < len(payload) and payload[cursor] == "[":
            try:
                arr = json.loads(slice_json_array(payload, cursor))
            except ValueError:
                arr = None
            if isinstance(arr, list) and (where is None or where(arr)):
                return arr
        at = payload.find(needle, at + 1)
    return None


# Any `"<key>":[{` — the start of an array of records under whatever name.
_RECORD_ARRAY_RE = re.compile(r'"[A-Za-z0-9_$]+":\s*\[\s*\{')


def find_records(payload: str, fields: tuple[str, ...]) -> list[dict] | None:
    """The largest array whose first record carries every one of ``fields``,
    under ANY key.

    WHY NOT A KEY NAME
    ------------------
    The key is the part AA changes. ``/leaderboards/models`` called its metrics
    array ``"models"`` until ~2026-10-14 and ``"initialModels"`` after, with the
    records themselves unchanged — and that one rename took the hosted and
    open-weight catalogues down for six days. The fields a scraper reads are
    what it actually depends on, so they are what it should search by: a rename
    of the array is then a non-event, and a rename of a FIELD still fails loudly
    because nothing matches.

    Largest wins because a page can quote a handful of the same records
    elsewhere (a "headline models" strip, a comparison widget); the catalogue is
    the long one. Arrays are pre-screened on a window after the opening brace so
    the payload's thousands of unrelated arrays are never decoded.
    """
    needles = [f'"{f}":' for f in fields]
    best: list[dict] | None = None
    for m in _RECORD_ARRAY_RE.finditer(payload):
        start = payload.index("[", m.start() + 1)
        head = payload[start:start + 6000]
        if not all(n in head for n in needles):
            continue
        try:
            arr = json.loads(slice_json_array(payload, start))
        except ValueError:
            continue
        if not arr or not isinstance(arr[0], dict) \
                or not all(f in arr[0] for f in fields):
            continue
        if best is None or len(arr) > len(best):
            best = arr
    return best


# ── Intelligence Index version ────────────────────────────────────────────────
#
# AA versions its Intelligence Index and rescales every score when it moves to
# a new one. v5.0 (2026-10-28) cut the scored catalogue from 272 models to 77
# overnight and re-rated the survivors non-uniformly (x1.08 at the top, x0.36
# at the bottom), which every row-count and median guard in this project read
# as a broken parse. The page names the version it renders, so the scrapers
# record it (data/carried.py, "Index version") and the guards can tell a
# rescale from a fault.
#
# The leaderboard's own evaluation entry is preferred:
#   "slug":"artificial-analysis-intelligence-index",
#   "name":"Artificial Analysis Intelligence Index v5.0"
# Raw HTML carries that with escaped quotes and the flight payload without, so
# the pattern allows both. Failing that, the highest version named anywhere on
# the page (a changelog line, "Updated to Intelligence Index v5").
_INDEX_NAMED_RE = re.compile(
    r'artificial-analysis-intelligence-index\\?"\s*,\s*\\?"name\\?"\s*:\s*\\?"'
    r'[^"\\]*?Intelligence Index v(\d+)(?:\.(\d+))?')
_INDEX_ANY_RE = re.compile(r"Intelligence Index v(\d+)(?:\.(\d+))?")


def _version(m: re.Match) -> str:
    return f"{int(m.group(1))}.{int(m.group(2) or 0)}"


def detect_index_version(text: str) -> str | None:
    """'5.0' for a page rendering Intelligence Index v5.0 (or "v5"); None when
    the page names no version, which callers treat as unknown, never as a
    change."""
    if not text:
        return None
    m = _INDEX_NAMED_RE.search(text)
    if m:
        return _version(m)
    found = {_version(m) for m in _INDEX_ANY_RE.finditer(text)}
    if not found:
        return None
    return max(found, key=lambda v: tuple(int(x) for x in v.split(".")))
