"""Open-weight models that exist but Artificial Analysis has not scored yet.

WHY THIS FILE EXISTS
--------------------
The Run Local tab is captioned "find open-weight models you can run on your own
hardware". It was built entirely from Artificial Analysis's endpoint, so what it
actually showed was "open-weight models *Artificial Analysis has benchmarked*" —
not the same set, and nothing on the page said so.

The gap is not hypothetical and it is not small. Qwen3.8-27B shipped on
2026-08-13 under Apache 2.0: 28B dense, multimodal, 262k context, and explicitly
aimed at ~24 GB consumer cards. That is exactly the model this tab exists to
surface, and for the days or weeks it takes AA to run its benchmark suite the tab
showed nothing at all. A dashboard that lags the releases it is *for* is not
doing its job.

WHAT AN ENTRY MAY AND MAY NOT CLAIM
-----------------------------------
Everything the Run Local tab needs is a published architectural fact:
parameter count, active parameters, context window, licence. VRAM fit and the
speed estimate are then COMPUTED from those, exactly as for a scraped model.

`quality` is the one thing that cannot be curated, because it is a benchmark
result. It stays None. Downstream that means:

  * the model appears in the VRAM-compatibility view, which is the question the
    tab is actually answering;
  * it is excluded from anything that ranks or recommends on intelligence —
    `_pick_local_tier` filters on `quality >= N`, and NaN fails that comparison,
    so the Agent Stack cannot recommend a model nobody has scored;
  * the UI marks it as unscored rather than implying a missing score is a zero.

Never invent a `quality`. A plausible-looking number here would propagate into
the value rankings and the recommender, which is precisely the "hand-set
constant nobody checked against the data" pattern this codebase keeps paying for.

LIFECYCLE
---------
Entries are self-expiring: `merge_pending` drops any curated row whose name
matches a scraped one, so the moment AA publishes a score the real record wins
and the curated entry becomes dead weight rather than a competing claim. Prune
this list when that happens — `tests/test_pending_models.py` fails once an entry
is redundant, so it will tell you.

Every entry carries `source` and `announced` so a reader can check the claim.
"""
from __future__ import annotations

# Fields mirror data/local_models.py's own model dicts, plus provenance.
#   params_b   - total parameters, billions
#   active_b   - parameters read per token (== params_b for dense models)
#   context_k  - native context window, thousands of tokens
#   moe        - True when active_b < params_b
#   source     - a URL a reader can check the numbers against
#   announced  - ISO date the weights were published
PENDING_MODELS: list[dict] = [
    # Ornith-1.5, released 2026-08-19 under MIT. Artificial Analysis has not
    # benchmarked the family: "ornith" appears zero times across all 610 models
    # on its leaderboard, /models/ornith-1-5 is a 404, and there is no sitemap
    # entry. So it cannot reach the Run Local tab through the scrape, and the
    # 9B is exactly what that tab exists to surface — a dense reasoning model
    # with a quantised mobile build.
    #
    # The lab's headline numbers (86.1 Terminal-Bench 2.1, 56.0 DeepSWE) are
    # VENDOR-REPORTED, from its own runs. `quality` stays None regardless: a
    # self-reported score is not the AA Intelligence Index and putting one in
    # this column would let it rank against numbers it was never measured
    # against. That is the whole point of this file.
    #
    # The blog coverage also describes a 397B MoE flagship. It is NOT here: the
    # only "397B" traceable to a primary source was Qwen3.5-397B sitting in
    # Ornith's own benchmark comparison table, so the sibling may be a
    # conflation. An unverified entry is the invention this module refuses.
    {
        "name":      "Ornith-1.5 9B",
        "family":    "Ornith AI",
        "params_b":  9.0,
        "active_b":  9.0,           # dense
        "context_k": 262,           # native; the card notes ~1M via YaRN
        "quality":   None,          # AA has not benchmarked it
        "license":   "MIT",
        "tags":      ["reasoning", "code"],
        "moe":       False,
        "source":    "https://huggingface.co/ornith-ai/Ornith-1.5-9B",
        "announced": "2026-08-19",
    },
    {
        "name":      "Ornith-1.5 35B A3B",
        "family":    "Ornith AI",
        "params_b":  36.0,          # card states "36B params"
        "active_b":  3.0,           # "activates only ~3B parameters per token"
        "context_k": 262,
        "quality":   None,
        "license":   "MIT",
        "tags":      ["vision", "reasoning", "code"],
        "moe":       True,
        "source":    "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B",
        "announced": "2026-08-19",
    },
]


def _norm(name: str) -> str:
    """Loose key for matching a curated name against a scraped one.

    Artificial Analysis writes effort variants as suffixes — "Qwen3.6 27B
    (Reasoning)" — so a curated "Qwen3.8 27B" must be recognised as already
    covered once ANY scored variant of it appears. Case, spaces and punctuation
    are all noise for that comparison.
    """
    base = name.split("(")[0]
    return "".join(ch for ch in base.lower() if ch.isalnum())


def merge_pending(scraped: list[dict]) -> list[dict]:
    """Append curated entries that the scrape does not already cover.

    Returns plain dicts in the same shape `_load_models_raw` produces, with an
    extra `pending` flag so the renderers can mark them and the recommender can
    exclude them.
    """
    covered = {_norm(m.get("name", "")) for m in scraped}
    out = [{**m, "pending": False} for m in scraped]
    for entry in PENDING_MODELS:
        if _norm(entry["name"]) in covered:
            continue        # AA scored it — the real record wins
        row = {k: v for k, v in entry.items() if k not in ("source", "announced")}
        row["pending"] = True
        out.append(row)
    return out


def redundant_entries(scraped_names) -> list[str]:
    """Curated entries the scrape now covers — i.e. safe to delete from here."""
    covered = {_norm(n) for n in scraped_names}
    return [e["name"] for e in PENDING_MODELS if _norm(e["name"]) in covered]
