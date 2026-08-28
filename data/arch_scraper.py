"""
Resolves each open-weight model to its published attention geometry, so the Run
Local tab can price the KV cache from a fact rather than a fit.

Source: each model's own ``config.json`` on HuggingFace.

WHY THIS EXISTS
---------------
data/local_models.py sizes the KV cache from n_layers x n_kv_heads x head_dim.
The AA leaderboard publishes none of those, so the catalogue shipped with a
curated table of ~35 hand-read architectures and a fitted estimator for
everything else. The estimator is honest but blunt, and it covered 166 of 179
rows. Measured against the geometry this module went and fetched, its error on
those rows is mean 89%, median 33%, p90 272%, max 885% — far worse than the
29.6% it scores in-sample, because it assumes every layer caches the full
sequence and therefore cannot see sliding-window attention (Gemma 4 E4B: +618%)
or latent attention (Sarvam 105B: +367%). A model's KV cache is the difference
between "fits your 12 GB card" and "does not", so that band is a band on the
answer the tab exists to give.

The geometry is not a measurement anybody has to estimate. It is three integers
in a file the lab published. This module fetches them.

WHY IT CANNOT JUST TRUST A NAME MATCH
-------------------------------------
The catalogue names models the way a product page does ("Qwen3.5 32B",
"gpt-oss-120b (high)"); HuggingFace names them by repo. Guessing wrongly would
write a WRONG architecture into the cache, and a confidently wrong number is
worse than a labelled estimate -- that is the whole doctrine in
data/pending_models.py.

So every match must clear a numeric guard: HuggingFace publishes the exact
parameter count in its safetensors metadata, and a candidate repo is rejected
unless that count agrees with the catalogue's params_b to within
``_PARAM_TOLERANCE``. A name can be ambiguous; 32,762,123,264 parameters cannot.
Anything that fails to resolve is simply absent from the cache and falls back to
the estimator, labelled as such.

Fields written per resolved model:
  name, repo, attn, n_layers, n_kv_heads, head_dim, kv_lora_rank,
  qk_rope_head_dim, sliding_window, global_layers, local_kind,
  params_hf_b, resolved_at

Run standalone:  python -m data.arch_scraper
"""

import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from static_helpers import csv_safe

_CACHE = Path(__file__).parent / "raw" / "aa_local_arch.csv"
_LOCAL_CACHE = Path(__file__).parent / "raw" / "aa_local_models.csv"
# Names the API answered about and produced nothing qualifying for. See
# scrape_and_save(): without this the hourly bot re-searched the same permanent
# non-matches ~100 requests a run, forever.
# Overridable for the same reason data/scrape_status.py's path is: a test has to
# drive this module through its outage path without rewriting the ledger the
# committed run depends on.
_UNRESOLVED = Path(
    os.environ.get("AI_FRONTIER_ARCH_UNRESOLVED_PATH")
    or Path(__file__).parent / "raw" / "aa_local_arch_unresolved.csv"
)

# How long a recorded non-match is left alone before being retried. Long enough
# that the hourly bot stops re-asking a question upstream has already answered,
# short enough that a name which becomes resolvable (a repo published, a
# catalogue params_b corrected) is picked up without anyone intervening.
_RETRY_DAYS = 7

_HEADERS = {"User-Agent": "AI-Frontier-Dashboard/1.0 (+github.com/Avo-Sarkissian)"}
_API = "https://huggingface.co/api/models"
_TIMEOUT = 20

# A repo is accepted only if its published parameter count agrees with the
# catalogue's to within this fraction. 8% absorbs the difference between counting
# embeddings or not and between a base and instruct checkpoint; it does not
# absorb "the 8B when you wanted the 70B", which is the failure that matters.
_PARAM_TOLERANCE = 0.08

# Organisations whose upload of a given model is the reference one. Used only to
# BREAK TIES between candidates that have already cleared the parameter guard,
# never to admit one that has not.
_PREFERRED_ORGS = (
    "meta-llama", "Qwen", "google", "mistralai", "openai", "deepseek-ai",
    "microsoft", "moonshotai", "nvidia", "allenai", "ibm-granite", "CohereLabs",
    "zai-org", "MiniMaxAI", "baidu", "tencent", "unsloth", "NousResearch",
)

# Mirrors to try when the reference repo is gated. Meta and Google gate their
# weights behind a licence click, which returns HTML rather than JSON to an
# anonymous fetch; these orgs republish the same config.
_MIRROR_ORGS = ("unsloth", "NousResearch", "nvidia")

# Product-name noise that never changes the attention config. Kept in step with
# data/local_models._KV_NAME_NOISE, which folds the same suffixes for the
# curated table's lookup.
_NOISE = re.compile(
    r"\((?:reasoning|non-reasoning|thinking|low|medium|high|xhigh|max|max effort|"
    r"high effort|standard|minimal)[^)]*\)",
    re.I,
)


def _search_terms(name: str) -> list[str]:
    """Query strings to try for a catalogue display name, best first."""
    base = _NOISE.sub(" ", str(name))
    base = re.sub(r"\s+", " ", base).strip()
    terms = [base]
    # "Qwen3.5 32B" -> "Qwen3.5-32B": HF ids hyphenate.
    terms.append(base.replace(" ", "-"))
    # Drop a leading vendor word the repo carries as the ORG instead of the id.
    parts = base.split()
    if len(parts) > 2:
        terms.append(" ".join(parts[1:]))
    out, seen = [], set()
    for t in terms:
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


class ApiError(RuntimeError):
    """HuggingFace did not answer.

    Kept strictly separate from "answered, and nothing qualified". The two are
    identical at the call site — both end in no geometry — but they mean
    opposite things: one is an outage that must turn the run red, the other is
    the documented steady state in which the estimator answers and says so.
    """


def _get(url: str, params: dict | None = None):
    r = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _candidates(name: str, limit: int = 12) -> list[dict]:
    """Search hits for a name.

    Raises ApiError if every query for this name failed at the transport level,
    so a rate-limited or unreachable API cannot be mistaken for "no such model"
    and recorded as a settled non-match.
    """
    seen, out, tried, errors = set(), [], 0, 0
    for term in _search_terms(name):
        tried += 1
        try:
            rows = _get(_API, {"search": term, "limit": limit,
                               "expand[]": "safetensors"})
        except Exception:
            errors += 1
            continue
        for m in rows or []:
            rid = m.get("id")
            if rid and rid not in seen:
                seen.add(rid)
                out.append(m)
        if out:
            break
    if tried and errors == tried:
        raise ApiError(f"every search for {name!r} failed")
    return out


def _param_match(cand: dict, params_b: float) -> float | None:
    """Published parameter count in billions, or None if it fails the guard."""
    total = ((cand.get("safetensors") or {}).get("total"))
    if not total or not params_b or params_b <= 0:
        return None
    hf_b = float(total) / 1e9
    if abs(hf_b - params_b) / params_b > _PARAM_TOLERANCE:
        return None
    return hf_b


def _score(cand: dict, name: str, params_b: float = 0.0) -> tuple:
    """Tie-break only. Every candidate scored here already cleared the guard."""
    rid = str(cand.get("id", ""))
    org, _, repo = rid.partition("/")
    org_rank = (_PREFERRED_ORGS.index(org) if org in _PREFERRED_ORGS
                else len(_PREFERRED_ORGS))
    target = re.sub(r"[^a-z0-9]", "", _NOISE.sub("", name).lower())
    got = re.sub(r"[^a-z0-9]", "", repo.lower())
    exact = 0 if got == target else 1
    # A quantised or derivative upload is not the reference architecture.
    derived = 1 if re.search(r"gguf|awq|gptq|int4|int8|fp8|nvfp4|mlx|abliterat|"
                             r"heretic|uncensor|merge|lora", repo, re.I) else 0
    # Version-token guard. The parameter check cannot separate two GENERATIONS
    # of the same size — "Hermes 4 - Llama-3.1 405B" matched
    # Hermes-3-Llama-3.1-405B, which is 405B and clears every numeric test.
    # Same-size siblings usually share an attention config, so this is a
    # provenance error rather than a wrong number, but a cache row that names
    # the wrong repo cannot be audited. Penalise a candidate whose digits are
    # not a superset of the query's.
    want_nums = set(re.findall(r"\d+(?:\.\d+)?", _NOISE.sub("", name)))
    got_nums = set(re.findall(r"\d+(?:\.\d+)?", repo))
    version_miss = 1 if (want_nums - got_nums) else 0
    # Closeness to the catalogue's own parameter count, in half-percent buckets.
    # The 8% guard admits a REPACK as readily as the model: SillyTilly's
    # Llama-3.1-405B-Instruct publishes 410.08B against Meta's 405.85B, and the
    # 4.23B difference is exactly eight extra KV heads per layer that the
    # released weights do not have. Reading its config gives a KV cache twice
    # the real one. The official upload is always the closest match, so
    # preferring closeness picks it without having to special-case anything.
    total = ((cand.get("safetensors") or {}).get("total")) or 0
    delta = (abs(total / 1e9 - params_b) / params_b) if (total and params_b) else 1.0
    return (derived, version_miss, round(delta * 200), exact, org_rank, len(repo))


def _fetch_config(repo: str) -> dict | None:
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT,
                         allow_redirects=True)
    except Exception as e:
        # Never reached the server. Gating and 404s answer with a status code;
        # this did not, so it says nothing about whether the config exists.
        raise ApiError(f"{repo}: {type(e).__name__}") from e
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _geometry(cfg: dict) -> dict | None:
    """Attention geometry from a config, as GQA or as MLA.

    Multimodal configs nest the decoder under ``text_config``.

    MLA MUST NOT GO THROUGH THE GQA COLUMNS. A latent-attention model caches ONE
    vector per layer, not a K and a V, so the GQA formula's leading 2 alone
    overstates it by double. DeepSeek V3 publishes the latent as
    ``kv_lora_rank``; DeepSeek V4 publishes no such key and expresses the same
    thing as ``num_key_value_heads: 1`` with ``head_dim: 512`` -- which reads as
    a perfectly ordinary MQA config and is not one. Its 128 attention heads
    against a hidden size of 7168 give the game away: 512 is the compressed
    latent width, not a per-head dimension.

    ``qk_rope_head_dim`` is the reliable tell, because the decoupled RoPE key is
    specific to this family. Priced as GQA, DeepSeek V4 Pro comes out at 122 KiB
    per token against a true 68.6 -- 78% high.
    """
    t = cfg.get("text_config") or cfg
    n_layers = t.get("num_hidden_layers")
    rope = t.get("qk_rope_head_dim")
    if n_layers and (t.get("kv_lora_rank") or rope):
        latent = t.get("kv_lora_rank")
        if not latent:
            # V4 form: the latent is num_key_value_heads x head_dim.
            n_kv, hd = t.get("num_key_value_heads"), t.get("head_dim")
            latent = (n_kv * hd) if (n_kv and hd) else None
        if latent:
            return {
                "attn": "mla",
                "n_layers": int(n_layers),
                "kv_lora_rank": int(latent),
                "qk_rope_head_dim": int(rope or 0),
                "n_kv_heads": "", "head_dim": "",
                "sliding_window": "", "global_layers": "",
            }
        return None
    n_kv = t.get("num_key_value_heads") or t.get("num_attention_heads")
    if not n_layers:
        return None
    head_dim = t.get("head_dim")
    if head_dim is None:
        hs, n_heads = t.get("hidden_size"), t.get("num_attention_heads")
        head_dim = (hs // n_heads) if (hs and n_heads) else None
    if not (n_layers and n_kv and head_dim):
        return None

    # Hybrid local/global attention. Gemma 3 publishes sliding_window_pattern,
    # gpt-oss and others publish layer_types; either tells us how many layers
    # actually cache the full sequence. Without one, a bare sliding_window on a
    # config whose layers are ALL windowed still means every layer is capped.
    window = t.get("sliding_window")
    layer_types = t.get("layer_types")
    global_layers, local_kind = None, ""
    if layer_types:
        n_full = sum(1 for x in layer_types if "full" in str(x).lower())
        if 0 < n_full < len(layer_types):
            global_layers = n_full
            # WHAT THE OTHER LAYERS ARE decides whether they cache anything at
            # all, and the difference is not small. A sliding-window layer
            # caches `window` tokens; a LINEAR / recurrent layer (Gated
            # DeltaNet, Mamba) carries a fixed-size state and caches nothing
            # that grows with context. Qwen3.8-Flash-Next is 12 full-attention
            # layers out of 48 with the other 36 linear, so charging all 48 as
            # full overstates its KV cache by exactly 4x.
            rest = " ".join(str(x).lower() for x in layer_types
                            if "full" not in str(x).lower())
            if any(k in rest for k in ("linear", "mamba", "recurrent", "ssm")):
                local_kind = "linear"
            elif "slid" in rest or window:
                local_kind = "sliding"
    elif window and t.get("sliding_window_pattern"):
        pat = int(t["sliding_window_pattern"])
        if pat > 1:
            global_layers, local_kind = max(1, int(n_layers) // pat), "sliding"
    return {
        "attn": "gqa",
        "n_layers": int(n_layers),
        "n_kv_heads": int(n_kv),
        "head_dim": int(head_dim),
        "kv_lora_rank": "", "qk_rope_head_dim": "",
        "sliding_window": int(window) if window and local_kind == "sliding" else "",
        "global_layers": int(global_layers) if global_layers else "",
        "local_kind": local_kind,
    }


def resolve_one(name: str, params_b: float) -> dict | None:
    """Published geometry for one catalogue row, or None if nothing qualified."""
    cands = [c for c in _candidates(name) if _param_match(c, params_b) is not None]
    if not cands:
        return None
    cands.sort(key=lambda c: _score(c, name, params_b))
    for cand in cands[:3]:
        repo = str(cand["id"])
        cfg = _fetch_config(repo)
        if cfg is None:
            # Gated. Try a mirror that republishes the same config, but only
            # under a repo id whose parameter count also cleared the guard.
            stem = repo.split("/")[-1]
            for org in _MIRROR_ORGS:
                cfg = _fetch_config(f"{org}/{stem}")
                if cfg is not None:
                    repo = f"{org}/{stem}"
                    break
        if cfg is None:
            continue
        geo = _geometry(cfg)
        if geo is None:
            continue
        return {"name": name, "repo": repo, **geo,
                "params_hf_b": round(_param_match(cand, params_b), 3),
                "resolved_at": date.today().isoformat()}
    return None


def _api_healthy() -> bool:
    """Positive control: is the search API still answering with usable rows?

    "Resolved nothing" has two causes that look identical from here. Forty of
    the catalogue's names have no qualifying repo and never will -- a product
    name the labs never uploaded under, or a marketing parameter count the
    published weights disagree with by more than the guard allows. That is the
    documented steady state: those rows fall back to the fitted estimator and
    the hover says so. A dead or rate-limited API produces the same empty
    result.

    Without a control this module had to guess which it was, and it guessed
    "failure" -- so every hourly run went red on a healthy scrape, which is the
    image endpoint's 29 silent days running in reverse: a guard that fires
    constantly stops being read exactly like one that never fires at all.

    So ask about something that cannot plausibly be missing. If "llama" comes
    back with parameter counts attached, the API is fine and an unresolved name
    is a fact about naming, not an outage.
    """
    try:
        rows = _get(_API, {"search": "llama", "limit": 5, "expand[]": "safetensors"})
    except Exception as e:
        print(f"[arch_scraper] control query failed: {type(e).__name__} — "
              f"treating this run as an upstream failure")
        return False
    if not any((r.get("safetensors") or {}).get("total") for r in rows or []):
        print("[arch_scraper] control query answered but carried no parameter "
              "counts — the search API has changed shape")
        return False
    return True


def _load_unresolved() -> dict:
    """name -> {params_b, attempts, last_tried, reason} for settled non-matches."""
    if not _UNRESOLVED.exists():
        return {}
    try:
        df = pd.read_csv(_UNRESOLVED)
    except Exception:
        return {}
    return {str(r["name"]): {k: r[k] for k in df.columns} for _, r in df.iterrows()}


def _due(rec: dict | None, params_b: float, today: date) -> bool:
    """Should this name be looked up again this run?"""
    if not rec:
        return True
    # The catalogue moved its parameter count, so the guard that rejected every
    # candidate last time is now a different guard. Ask again.
    try:
        if abs(float(rec.get("params_b") or 0) - params_b) > 1e-6:
            return True
        last = date.fromisoformat(str(rec.get("last_tried")))
    except Exception:
        return True
    return (today - last).days >= _RETRY_DAYS


def _save_unresolved(ledger: dict) -> None:
    """Write the ledger, but only when it actually changed.

    data/raw is what the workflow's change guard watches. A file rewritten every
    run with nothing but a fresh timestamp would make that guard fire hourly and
    commit nothing of substance -- the same trap scrape_status.json is excluded
    from in refresh.yml.
    """
    cols = ["name", "params_b", "attempts", "last_tried", "reason"]
    if not ledger:
        return
    out = pd.DataFrame(
        [{c: rec.get(c, "") for c in cols} for rec in ledger.values()]
    ).sort_values("name")
    body = csv_safe(out).to_csv(index=False)
    if _UNRESOLVED.exists() and _UNRESOLVED.read_text() == body:
        return
    _UNRESOLVED.parent.mkdir(parents=True, exist_ok=True)
    _UNRESOLVED.write_text(body)


def load_cached() -> pd.DataFrame | None:
    if _CACHE.exists():
        return pd.read_csv(_CACHE)
    return None


def scrape_and_save(limit: int | None = None, sleep_s: float = 0.4,
                    refresh_all: bool = False) -> bool:
    """Resolve every catalogue row not already cached, and append.

    INCREMENTAL BY DESIGN. The hourly bot must not re-fetch 179 configs every
    run: HuggingFace would rate-limit it, and an architecture does not change
    once published. Only names absent from the cache are looked up, so a steady
    state costs one search per newly scraped model and the file converges.

    A row that fails to resolve is NOT written as a blank. It stays absent and
    the estimator answers for it, labelled. It is recorded in
    aa_local_arch_unresolved.csv and retried every _RETRY_DAYS rather than every
    hour, because a name upstream has no repo for does not acquire one in the
    next sixty minutes, and re-asking 40 of them hourly is how an anonymous
    client earns a rate limit -- which would be a real outage, manufactured by
    the check for outages.

    RETURN VALUE MEANS "THE SOURCE ANSWERED", NOT "SOMETHING RESOLVED". refresh.yml
    turns a False into a red run reading "the upstream endpoint is failing", so
    False must mean exactly that. Resolving nothing is the steady state once
    every resolvable name is cached: 40 of the catalogue's rows are product
    names with no qualifying repo, and reporting that as an outage every hour
    for a healthy scrape is how the one alert that matters gets ignored.
    """
    if not _LOCAL_CACHE.exists():
        print("[arch_scraper] no aa_local_models.csv yet — run local_scraper first")
        return False
    catalogue = pd.read_csv(_LOCAL_CACHE)
    cached = load_cached()
    have = set() if (refresh_all or cached is None) else set(cached["name"].astype(str))

    # Rows the hand-curated KV_ARCH already answers are SKIPPED, not cached.
    # resolve_attention() prefers the curated value, so a scraped row for the
    # same model can never be used — and a second, possibly different answer
    # sitting in the cache is pure risk. It is how a repack got in: searching
    # "Llama 3.1 Instruct 405B" surfaces SillyTilly's 410.08B upload but not
    # Meta's own gated 405.85B one, and the extra 4.23B is eight KV heads per
    # layer the released model does not have. Curated wins; do not store a rival.
    from data.local_models import _kv_arch_lookup
    candidates = [(str(r["name"]), float(r["params_b"]))
                  for _, r in catalogue.iterrows()
                  if str(r["name"]) not in have and float(r.get("params_b") or 0) > 0
                  and _kv_arch_lookup(str(r["name"])) is None]

    today = date.today()
    ledger = {} if refresh_all else _load_unresolved()
    todo = [(n, pb) for n, pb in candidates
            if refresh_all or _due(ledger.get(n), pb, today)]
    resting = len(candidates) - len(todo)
    if limit:
        todo = todo[:limit]
    if not todo:
        print(f"[arch_scraper] nothing to resolve ({len(have)} cached, "
              f"{resting} known non-matches resting until retried)")
        return True

    rows, failed, errored = [], [], []
    for i, (name, params_b) in enumerate(todo, 1):
        try:
            got = resolve_one(name, params_b)
            reason = "no repo cleared the parameter guard"
        except ApiError as e:
            got, reason = None, str(e)
            errored.append(name)
            print(f"  ! {name}: upstream did not answer ({e})")
        except Exception as e:
            got, reason = None, f"{type(e).__name__}"
            print(f"  ! {name}: {type(e).__name__}")
        if got:
            rows.append(got)
            ledger.pop(name, None)
            shape = (f"{got['n_layers']}L/mla{got['kv_lora_rank']}+{got['qk_rope_head_dim']}"
                     if got["attn"] == "mla"
                     else f"{got['n_layers']}L/{got['n_kv_heads']}kv/{got['head_dim']}d")
            print(f"  [{i}/{len(todo)}] {name} -> {got['repo']} ({shape})")
        elif name not in errored:
            # The API answered; this name simply has no qualifying repo. Record
            # it so the next 167 hourly runs do not ask again.
            failed.append(name)
            prev = ledger.get(name) or {}
            ledger[name] = {
                "name": name, "params_b": params_b,
                "attempts": int(float(prev.get("attempts") or 0)) + 1,
                "last_tried": today.isoformat(), "reason": reason,
            }
        time.sleep(sleep_s)

    if rows:
        out = pd.DataFrame(rows)
        if cached is not None and not refresh_all:
            out = pd.concat([cached, out], ignore_index=True)
            out = out.drop_duplicates(subset=["name"], keep="last")
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        csv_safe(out).to_csv(_CACHE, index=False)
        print(f"[arch_scraper] Saved {len(out)} architectures "
              f"(+{len(rows)} this run, {len(failed)} unresolved)")
    else:
        print(f"[arch_scraper] resolved 0 of {len(todo)} — cache unchanged")
    if failed:
        print(f"  unresolved: {failed[:8]}{' …' if len(failed) > 8 else ''}")
    _save_unresolved(ledger)

    # Something resolved: the source is demonstrably up, whatever else failed.
    if rows:
        return True
    # Nothing resolved AND the API refused to talk about at least one name.
    if errored:
        print(f"[arch_scraper] {len(errored)} name(s) got no response — "
              f"reporting upstream failure")
        return False
    # Nothing resolved and nothing errored. Only the control can tell an outage
    # from a catalogue of names upstream has never published.
    return _api_healthy()


if __name__ == "__main__":
    n = None
    if "--limit" in sys.argv:
        n = int(sys.argv[sys.argv.index("--limit") + 1])
    ok = scrape_and_save(limit=n, refresh_all="--refresh-all" in sys.argv)
    sys.exit(0 if ok else 1)
