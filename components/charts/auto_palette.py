"""Colours for the companies nobody hand-picked one for.

WHY THIS EXISTS
---------------
The palettes in constants.py, data/image_models.py and data/video_models.py
are chosen by hand against a validator, and every company AA added since
rendered in the grey "Other" colour and failed a test until someone picked a
hex: Apodex (2026-08-31), the Institute of Foundation Models (2026-09-22),
Reflection (2026-10-16). That is a chore that recurs forever, on a schedule AA
sets.

AA already answers the question. Every record it publishes carries the
company's brand colour, and the hand-picked palette exists to mirror it
(see AA_CREATOR_COLORS). So the scrapers record AA's colour for each company
they see, and a company with no hand-picked colour gets one derived from it
by the same three moves a person made by hand:

1. Keep AA's hue and chroma. A colour AA paints black or grey has no hue to
   keep, so it gets a stable hue from its name instead.
2. Lift lightness until it clears 3:1 against the #111111 chart surface,
   measured on the final quantised hex ("AA #2436d8, lifted to 3.00:1").
3. If that lands within OKLab dE 15 of a colour already on the same tab — a
   hand-picked one, the grey "Other" bucket, or an earlier automatic pick —
   try nearby colours, lightness first and hue at most 30 degrees off AA's,
   and keep the cheapest that clears (or the furthest if none does). The
   collision rule, applied mechanically: whoever was there first keeps the
   colour, and the newcomer still reads as its own brand.

RESOLVED ONCE, THEN STORED
--------------------------
The result is written to data/carried/aa_creator_colors.csv beside AA's own
value and never recomputed unless AA changes that value. Recomputing at
import would let one new company shift the colours of others whose candidates
it now competes with, and a company's colour is part of how a returning
reader finds it.

Pure Python on purpose: constants.py reads the stored file in the browser
under Pyodide, where this module ships in pycode.zip.
"""
from __future__ import annotations

import csv
import hashlib
import math
import os
from pathlib import Path

SURFACE = "#111111"
OTHER_GREY = "#6b7280"          # constants.DEFAULT_COLOR; a mark, not background
MIN_CONTRAST = 3.0
MIN_DELTA_E = 15.0              # OKLab dE x100 — the validator's normal-vision floor
_HUE_STEP = 10
_MAX_HUE_SHIFT = 30          # degrees; past this it stops reading as AA's colour
_MIN_CHROMA = 0.09              # below this a colour starts reading as grey
_ACHROMATIC = 0.04              # AA colours under this chroma have no usable hue

FILE_NAME = "aa_creator_colors.csv"
FIELDS = ("creator", "aa_color", "color")


def stored_path() -> Path:
    """Where the resolved colours live. Same override as data/carried.py."""
    root = os.environ.get("AI_FRONTIER_CARRIED_DIR") \
        or Path(__file__).resolve().parents[2] / "data" / "carried"
    return Path(root) / FILE_NAME


def load_stored(path: Path | None = None) -> dict[str, tuple[str, str]]:
    """{creator: (aa_color, color)}; {} when the file is absent or unreadable."""
    path = stored_path() if path is None else Path(path)
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return {}
    out = {}
    for r in rows:
        name, aa, colour = (r.get(k, "").strip() for k in FIELDS)
        if name and _is_hex(colour):
            out[name] = (aa.lower(), colour.lower())
    return out


def save_stored(entries: dict[str, tuple[str, str]], path: Path | None = None) -> bool:
    """Sorted and timestamp-free so an unchanged set is byte-identical."""
    path = stored_path() if path is None else Path(path)
    lines = [",".join(FIELDS)]
    lines += [f"{_csv_field(n)},{aa},{c}" for n, (aa, c) in sorted(entries.items())]
    text = "\n".join(lines) + "\n"
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def _csv_field(s: str) -> str:
    return f'"{s.replace(chr(34), chr(34) * 2)}"' if any(c in s for c in ',"\n') else s


# ── Colour math ───────────────────────────────────────────────────────────────

def _is_hex(h) -> bool:
    return isinstance(h, str) and len(h) == 7 and h[0] == "#" \
        and all(c in "0123456789abcdefABCDEF" for c in h[1:])


def _rgb(h: str) -> tuple[float, float, float]:
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gamma(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def contrast(a: str, b: str) -> float:
    la, lb = (0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(bl)
              for r, g, bl in (_rgb(a), _rgb(b)))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _oklab(h: str) -> tuple[float, float, float]:
    r, g, b = (_lin(c) for c in _rgb(h))
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def delta_e(a: str, b: str) -> float:
    return 100 * math.dist(_oklab(a), _oklab(b))


def _oklch(h: str) -> tuple[float, float, float]:
    L, a, b = _oklab(h)
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def _from_oklch(L: float, C: float, H: float) -> str | None:
    """The sRGB hex for an OKLCH colour, or None when it is out of gamut."""
    a, b = C * math.cos(math.radians(H)), C * math.sin(math.radians(H))
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (+4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
           -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
           -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_)
    if any(c < -1e-4 or c > 1 + 1e-4 for c in rgb):
        return None
    return "#" + "".join(f"{round(_gamma(min(max(c, 0.0), 1.0)) * 255):02x}" for c in rgb)


def _in_gamut(L: float, C: float, H: float) -> str:
    """Nearest in-gamut colour at this lightness and hue: chroma gives way first."""
    while C > 0:
        hexv = _from_oklch(L, C, H)
        if hexv:
            return hexv
        C -= 0.01
    return _from_oklch(min(max(L, 0.0), 1.0), 0.0, H) or "#808080"


def _legible(L: float, C: float, H: float) -> str | None:
    """The darkest colour at this hue and chroma that clears MIN_CONTRAST."""
    while L <= 0.97:
        hexv = _in_gamut(L, C, H)
        if contrast(hexv, SURFACE) >= MIN_CONTRAST:
            return hexv
        L += 0.01
    return None


def _name_hue(name: str) -> float:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 360


def _candidates(L: float, C: float, H: float):
    """Nearby colours, cheapest change first: lightness before hue, and hue
    never more than _MAX_HUE_SHIFT from AA's, so the result still reads as
    the company's colour."""
    found = []
    for dh in range(-_MAX_HUE_SHIFT, _MAX_HUE_SHIFT + 1, _HUE_STEP):
        base = _legible(L, C, (H + dh) % 360)
        if base is None:
            continue
        bL = _oklch(base)[0]
        for k in range(0, 6):
            for dl in ((0.0,) if k == 0 else (0.04 * k, -0.04 * k)):
                hexv = _in_gamut(bL + dl, C, (H + dh) % 360)
                if contrast(hexv, SURFACE) < MIN_CONTRAST:
                    continue
                found.append((abs(dh) / _MAX_HUE_SHIFT + abs(dl) / 0.2, hexv))
    found.sort()
    return [h for _, h in found]


def resolve(name: str, aa_color: str | None, taken: list[str]) -> str:
    """A legible colour for `name` that keeps AA's hue where it can."""
    if _is_hex(aa_color):
        L, C, H = _oklch(aa_color)
    else:
        L, C, H = 0.0, 0.0, 0.0
    if C < _ACHROMATIC:
        # Black, white or grey upstream: nothing to mirror. A hue from the
        # name is stable across runs and machines, unlike hash().
        L, C, H = 0.72, 0.13, _name_hue(name)
    C = max(C, _MIN_CHROMA)

    avoid = list(taken) + [OTHER_GREY]
    best: tuple[float, str] | None = None
    for hexv in _candidates(L, C, H):
        if hexv in taken:
            continue
        gap = min(delta_e(hexv, t) for t in avoid)
        if gap >= MIN_DELTA_E:
            return hexv
        if best is None or gap > best[0]:
            best = (gap, hexv)
    return best[1] if best else _legible(0.72, 0.13, _name_hue(name)) or "#a3a3a3"


def assign(seen: dict[str, str], curated: dict[str, str],
           stored: dict[str, tuple[str, str]],
           neighbours: dict[str, str] | None = None) -> dict[str, tuple[str, str]]:
    """Fold newly seen creators into the stored assignments.

    `seen` is creator -> AA colour from this scrape. `curated` is every
    hand-picked colour on any tab: a creator in it never gets an automatic
    colour. `neighbours` is the hand-picked colours that share a tab with
    these creators — the ones a new colour must stand apart from (defaults to
    all of `curated`). A stored creator keeps its colour unless AA has changed
    the colour it was derived from. Gains only — a creator absent from this
    scrape keeps its line.
    """
    neighbours = curated if neighbours is None else neighbours
    out = dict(stored)
    for name in sorted(seen):
        aa = (seen[name] or "").strip().lower()
        if not name or name in curated or not _is_hex(aa):
            continue
        if name in out and out[name][0] == aa:
            continue
        taken = list(neighbours.values()) + [c for n, (_, c) in out.items() if n != name]
        out[name] = (aa, resolve(name, aa, taken))
    return out


def stored_colors(path: Path | None = None) -> dict[str, str]:
    """{creator: colour} for every automatic assignment."""
    return {name: colour for name, (_, colour) in load_stored(path).items()}
