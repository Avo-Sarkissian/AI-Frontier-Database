"""Open or closed weights, for every hosted row — including archived ones.

Artificial Analysis publishes `isOpenWeights` on every leaderboard record, and
the scrape has kept it as `is_open_weights` since 2026-10-27 (data/ingest.py).
Every history snapshot written before that day has the column blank, and AA can
drop a field without notice (it has restructured its pages three times this
year), so a row's own flag is not always there to read.

`annotate_open_weights` therefore resolves each row from the first source that
answers, and says which one it used:

  1. the row's own AA flag (`is_open_weights`)              -> "aa"
  2. the registry's AA flag for that exact model name       -> "aa"
  3. the name appears in AA's open-weights catalogue —
     any committed revision of data/raw/aa_local_models.csv
     (the registry's catalogue rows) or the current file     -> "catalogue"
  4. only the variant tag differs: the name with one trailing
     parenthetical removed equals a known name with one removed
     ("GLM-5" <-> "GLM-5 (Reasoning)"), and every known name with
     that base agrees                                        -> "catalogue"
  5. none of the above: treated as closed                   -> "inferred"

The registry (data/carried/open_weight_names.csv, `name,open_weights,source,
first_seen`) was seeded once from the git history of the open-weights catalogue
(scripts/seed_open_weight_registry.py) and grows every hour from AA's flag
(data/carried.learn_open_weights). It is what lets the gap history classify a
model that has since left both catalogues.

Name matching is EXACT first. On 2026-10-27 the AA flag and an exact-name join
to the open-weights catalogue agreed on all 84 open and 78 closed hosted rows; a
closed model that reused an open model's exact name would be misread, which is
why step 5 is labelled "inferred" rather than claimed. Step 4 exists because
AA's earliest snapshot (2026-02-25) named models without the variant tag it
added a week later: "GLM-5", "Kimi K2.5" and "Qwen3.5 397B A17B" there are the
registry's "GLM-5 (Reasoning)" etc., and exact matching alone counted all three
as closed, doubling that day's score gap. A base shared by an open and a closed
known name matches nothing (none on 2026-10-27).

Imports only stdlib + pandas: this module runs in the browser (Pyodide) too.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

REGISTRY_FILE = "open_weight_names.csv"
REGISTRY_COLUMNS = ("name", "open_weights", "source", "first_seen")

# Registry `source` values.
SOURCE_AA = "aa_isOpenWeights"
SOURCE_CATALOGUE = "aa_open_weights_catalogue"

# `weights_source` values on an annotated frame, and how each reads on screen.
WEIGHTS_SOURCE_LABELS = {
    "aa": "AA flag",
    "catalogue": "name match",
    "inferred": "inferred",
}

# The WEIGHTS control. "" (All) is first and the default, so an untouched
# dashboard shows exactly the catalogue it always has.
WEIGHTS_OPTIONS = (
    {"label": "All", "value": ""},
    {"label": "Open", "value": "open"},
    {"label": "Closed", "value": "closed"},
)


def weights_options() -> list[dict]:
    return [dict(o) for o in WEIGHTS_OPTIONS]


def valid_weights(value) -> str:
    """'' | 'open' | 'closed'; anything else (a URL is user input) is All."""
    return value if value in ("open", "closed") else ""


def _carried_dir() -> Path:
    # Same override as data/carried.py, which is not shipped to the browser.
    return Path(os.environ.get("AI_FRONTIER_CARRIED_DIR")
                or Path(__file__).parent / "carried")


def registry_path() -> Path:
    return _carried_dir() / REGISTRY_FILE


def _flag(value):
    """'1'/'0'/True/False -> True/False; anything else -> None."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "1.0"):
        return True
    if s in ("0", "false", "0.0"):
        return False
    return None


def load_registry(path: Path | None = None) -> pd.DataFrame:
    """The registry as text columns; empty (with the columns) when absent."""
    path = Path(path) if path is not None else registry_path()
    empty = pd.DataFrame(columns=list(REGISTRY_COLUMNS))
    if not path.exists():
        return empty
    try:
        reg = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (ValueError, OSError, pd.errors.EmptyDataError):
        return empty
    if not set(REGISTRY_COLUMNS) <= set(reg.columns):
        return empty
    reg = reg[list(REGISTRY_COLUMNS)]
    return reg[reg["name"].str.strip() != ""].drop_duplicates("name", keep="last")


def catalogue_names(path: Path | None = None) -> set[str]:
    """Model names in the CURRENT open-weights catalogue (aa_local_models.csv)."""
    path = Path(path) if path is not None else \
        Path(__file__).parent / "raw" / "aa_local_models.csv"
    if not path.exists():
        return set()
    try:
        cat = pd.read_csv(path, usecols=["name"], dtype=str, keep_default_na=False)
    except (ValueError, OSError, pd.errors.EmptyDataError):
        return set()
    return {n for n in cat["name"] if n.strip()}


# One trailing parenthetical: the variant tag AA appends ("(Reasoning)",
# "(Max)", "(0902)"). Same pattern as components/charts/constants.
# base_model_name, repeated here so this module stays stdlib + pandas.
_VARIANT_TAG_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _base(name: str) -> str:
    stripped = _VARIANT_TAG_RE.sub("", str(name)).strip()
    return stripped or str(name)


class Resolver:
    """Name -> (open, source) from the registry and the catalogue, built once."""

    def __init__(self, registry: pd.DataFrame | None = None,
                 catalogue: set[str] | None = None):
        reg = load_registry() if registry is None else registry
        self.aa: dict[str, bool] = {}
        self.listed: set[str] = set(catalogue_names() if catalogue is None else catalogue)
        for name, flag, source in zip(reg.get("name", []), reg.get("open_weights", []),
                                      reg.get("source", [])):
            f = _flag(flag)
            if source == SOURCE_AA and f is not None:
                self.aa[name] = f
            elif source == SOURCE_CATALOGUE and f is not False:
                self.listed.add(name)
        # base name -> the flags of every known name with that base (step 4).
        flags: dict[str, set] = {}
        for name, f in self.aa.items():
            flags.setdefault(_base(name), set()).add(f)
        for name in self.listed:
            flags.setdefault(_base(name), set()).add(True)
        self.by_base: dict[str, bool] = {b: next(iter(v)) for b, v in flags.items()
                                         if len(v) == 1}

    def resolve(self, name, own_flag=None) -> tuple[bool, str]:
        f = _flag(own_flag) if own_flag is not None and not _isna(own_flag) else None
        if f is not None:
            return f, "aa"
        if name in self.aa:
            return self.aa[name], "aa"
        if name in self.listed:
            return True, "catalogue"
        base = _base(name)
        if base in self.by_base:
            return self.by_base[base], "catalogue"
        return False, "inferred"


def _isna(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


_DEFAULT_RESOLVER: Resolver | None = None


def default_resolver() -> Resolver:
    """The on-disk registry + catalogue, read once per process."""
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = Resolver()
    return _DEFAULT_RESOLVER


def reset_cache() -> None:
    global _DEFAULT_RESOLVER
    _DEFAULT_RESOLVER = None


def annotate_open_weights(df: pd.DataFrame, registry: pd.DataFrame | None = None,
                          catalogue: set[str] | None = None,
                          resolver: Resolver | None = None) -> pd.DataFrame:
    """`df` plus `open_weights` (bool, never NA) and `weights_source`.

    Name column: `model` (hosted) or `name` (open-weights catalogue). A frame
    with neither gets every row "inferred" closed rather than an exception.
    Returns a copy; the input is not modified.
    """
    if resolver is None:
        resolver = (default_resolver() if registry is None and catalogue is None
                    else Resolver(registry, catalogue))
    out = df.copy()
    key = "model" if "model" in out.columns else ("name" if "name" in out.columns else None)
    names = out[key].astype(str) if key else pd.Series("", index=out.index)
    own = out["is_open_weights"] if "is_open_weights" in out.columns \
        else pd.Series([None] * len(out), index=out.index, dtype=object)
    pairs = [resolver.resolve(n, f) for n, f in zip(names, own)]
    out["open_weights"] = pd.Series([p[0] for p in pairs], index=out.index, dtype=bool)
    out["weights_source"] = pd.Series([p[1] for p in pairs], index=out.index, dtype=object)
    return out


def open_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series: True where the row is open weights (annotating if needed)."""
    if "open_weights" not in df.columns:
        df = annotate_open_weights(df)
    return df["open_weights"].eq(True)


def filter_weights(df: pd.DataFrame, weights) -> pd.DataFrame:
    """The WEIGHTS filter: 'open' keeps open rows, 'closed' keeps the rest."""
    w = valid_weights(weights)
    if not w:
        return df
    mask = open_mask(df)
    return df[mask] if w == "open" else df[~mask]


def weights_label(is_open, source) -> str:
    """'Open · AA flag' / 'Open · name match' / 'Closed · inferred' etc."""
    kind = "Open" if bool(is_open) else "Closed"
    return f"{kind} · {WEIGHTS_SOURCE_LABELS.get(str(source), 'inferred')}"


def weights_labels(df: pd.DataFrame) -> list[str]:
    """weights_label per row; a frame without the columns is annotated first."""
    if "open_weights" not in df.columns or "weights_source" not in df.columns:
        df = annotate_open_weights(df)
    return [weights_label(o, s) for o, s in zip(df["open_weights"], df["weights_source"])]
