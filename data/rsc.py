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
