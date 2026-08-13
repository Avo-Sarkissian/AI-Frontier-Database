# Product & Quality — AI Frontier Audit

*Every fact below was re-verified against the working tree at `b9d6e19` using `./ai-frontier/bin/python`. Where a lens observation did not survive verification, I say so.*

---

## 0. Conflict resolution — the positions I took

Six lenses disagreed in five places. Rulings, with reasoning:

**a11y wants more text; CLAUDE.md wants less.** This conflict is 90% imaginary. Of the 16 accessibility findings, exactly two add visible pixels (a footer with methodology, and a `<details>` coverage disclosure). The other fourteen — contrast ratios, `role=`/`aria-*`, focus rings, live regions, `<label for>`, landmarks — are **invisible or pixel-neutral**. Raising `#666666` to `#8a8a8a` (3.29:1 → 5.33:1 on `#111111`, computed) does not add clutter; it makes the existing text legible. "Aggressively clean" is a statement about *noise*, not about *information starvation*. **Ruling: take all fourteen invisible wins unconditionally.** Of the two visible ones, take the footer (see §3 — an uncitable dashboard has no academic value) and reject the per-chart "View this data as a table" link (a11y-03 part 2) — 13 new links across 10 tabs to reach a tab that is already one click away.

**UV-03 says stop deduping (plot all 148); UV-04 says dedupe everywhere (count 95).** Both are right about the diagnosis and wrong about the cure. The real defect is that `dedupe_to_best_variant` is applied to exactly two of five surfaces — `pareto.py:52` and `quadrant.py:28`, nowhere else (verified: 95 marker points in both `pareto.json` and `quadrant.json` vs `manifest.model_count=148`). **Ruling: the collapse is a display decision that must be (a) consistent, (b) disclosed, and (c) user-controllable — in that order of urgency.** Disclosure is a one-line annotation and ships today; consistency is a vocabulary change and ships next; the variant toggle is the L-effort finish.

**ui-02 wants a JS empty-state gate; a11y-15 wants `empty_figure()` in each Python builder.** Same bug, two layers. **Ruling: fix in Python only.** A JS gate would be a *third* re-implementation of chart logic and directly feeds recurring defect pattern (c). One fix in `components/charts/*.py` serves Dash and Pages simultaneously. Verified the gap is real and asymmetric: `update_overview([],0,'zzz','price')` returns `paper_bgcolor='#111111'`, 0 traces, 0 annotations; `update_overview([],0,'zzz','speed')` returns `paper_bgcolor=None` with a 1-key layout — a bare `go.Figure()` from `quadrant.py:31`, i.e. a **white panel on a black dashboard**.

**PERF-07/PERF-12 want pandas and/or Pyodide deleted from the browser path.** Rejected — see §2.C.1. The 19.3 MB is real and the 3 s boot is real, but the cure institutionalises the codebase's single largest defect generator.

**DVC-11 (in-bar label contrast) vs DVC-12 (provider colour as the only channel on bar charts).** Both fixes touch the same lines. **Ruling: do DVC-11's per-bar luminance-aware text colour now (mechanical, 3 files); defer DVC-12's pattern-fill/palette re-solve** — the bar charts already print the provider name in the right gutter (`rankings.py:135`), so colour there is decorative redundancy, not the sole channel. The genuine sole-channel failures are `video_scatter.json` (13 traces, no `marker.symbol`) and `radar.json` (5 traces differing only in `line.color`, verified) — both get fixed inside items **1.3** and **1.4**.

---

## 1. Do next — high value, S/M, each shippable in one sitting

Ordered by (user-visible wrongness) × (cheapness). Items 1–4 are the ones where the product currently tells a user something false on arrival, with no interaction required.

### 1.1 — Agent Stack's default state recommends the wrong model by 6× on price and 104× on latency
**Files:** `docs/index.html:219-238`, `components/stack_recommender.py:30-40, 108-113, 127-129, 140`

Verified by executing `static_api.update_recommend`:

| provider selection | Fast tier pick #1 |
|---|---|
| default (`Anthropic, Google, OpenAI` — the checked boxes at `index.html:220-228`) | **GPT-5.6 Luna (max) — $0.950/M, 162 tok/s, 57.45s TTFT** |
| `["__all__"]` | DeepSeek V4 Flash 0731 — $0.158/M, 60 tok/s, **0.55s TTFT** |

The tier's own tagline is "Sub-agent workhorse — cheap, high-throughput, **parallel calls**" and its use cases are "run in parallel across many files / grep, search and classify at scale". `_score_fast_api` (`stack_recommender.py:108-113`) is `q*0.45 + v*0.30 + s*0.25` — **latency appears nowhere in the score**, and the tier dict has `min_speed: 30` but no `max_latency`. The card renders `57.45s TTFT` in the same grey as everything else, directly beneath the words "high-throughput".

Two more verified defects in the same control: the `xAI` checkbox (`index.html:232`) matches **zero rows** — `update_recommend(["xAI"], "api", …)` returns "No models match these criteria" for all three tiers, because the dataset provider is `SpaceXAI`. And unchecking everything is byte-identical to checking All: `update_recommend([], …)` returns the same DeepSeek/Ling/MiniMax picks as `["__all__"]`.

**Change:** check `__all__` and uncheck the three named boxes at `index.html:220-237`; add `max_latency: 5.0` to the fast tier dict and an `l = 1 - min(row["latency"]/5.0, 1)` term reweighted to `q*.35 + v*.25 + s*.20 + l*.20`; compute `q_max/v_max/s_max` from the unfiltered `_DF` rather than `pool` (`:127-129`); route the provider list through the existing alias map so `xAI → SpaceXAI`; treat an empty selection as empty, not as all; and call `dedupe_to_best_variant` on each tier pool before `.head(5)`.

**Outcome:** the tab's first render recommends a 0.55s model instead of a 57s one, the Balanced tier stops returning five effort-tiers of a single model, and the xAI checkbox works.

### 1.2 — Budget lands on a ranking of $0.08-vs-$0.09 and hides its own answer
**Files:** `docs/index.html:103, 108`, `docs/app.js:484`

Verified: `budget-tokens` ships `value="1"`, `budget-min-intelligence` ships `value="0"`. Decoding `docs/figures/cost_calc.json` gives `xaxis.range=[0, 0.255]` and top bars **Gemma 4 E4B (R) $0.0800 / Gemma 4 E4B (NR) $0.0800 / Granite 4.1 8B $0.0875**. On arrival, the tab that answers the single most common buying question ranks the weakest models in the catalogue by a two-cent monthly difference.

Worse, `docs/app.js:484` reads `if (!floor) { host.style.display = "none"; return; }` — the "CHEAPEST MODEL SCORING X+" callout, the only place in the entire product that names a model and a dollar figure together, **is hidden until the user guesses the unlabelled slider is the point**.

**Change:** initialise the slider to `manifest.p75` (42, verified present) and tokens to 50; delete the `!floor` hide branch and render "cheapest model overall" at floor 0 — a floor of zero is a valid query, not an absent one. Add a second line: "next cheapest above 50 pts: `<model>` at $X/mo".

**Outcome:** the landing state reads "cheapest model scoring 42+ at 50M tok/mo — `<name>` — $X/mo" instead of a two-cent bar chart.

### 1.3 — The flagship ranking labels five different models `(2)` through `(5)`
**Files:** `components/charts/constants.py:9-27, 376-394`; `components/charts/radar.py:107`; `components/stack_recommender.py:211, 235, 582, 605`

`_NAME_SUBS` (`constants.py:9-18`) has patterns for the bare forms `(Adaptive Reasoning)`, `(xhigh)`, `(high)` — verified working (`'GPT-5.6 Luna (XH)'` ships correctly). It has **no pattern for the compound form Anthropic and DeepSeek actually publish**. Measured over the live catalogue: **148 distinct model names collapse to 141 distinct labels**, with four collision groups:

```
5×  'Claude Opus 5 (Adaptive Reasoni…'
2×  'NVIDIA Nemotron 3 Nano 30B A3B …'
2×  'Llama Nemotron Super 49B v1.5 (…'
2×  'NVIDIA Nemotron Nano 12B v2 VL …'
```

`unique_labels` then stamps ordinals on top. The shipped `docs/figures/rankings.json` y-axis literally contains, in order: `'Claude Opus 5 (Adaptive Reasoni…'`, `'Claude Opus 5 (Adaptive Reas (2)'`, `'(3)'`, `'(4)'`, `'(5)'` — a reader takes those for versions or ranks. The same strings appear in `radar.json` (where `data[0].name == data[4].name`, two byte-identical legend entries on the one chart whose purpose is telling models apart) and in the Agent Stack Reasoning card.

**Change:** prepend two compound rules to `_NAME_SUBS` — `\(\s*(Adaptive\s+)?Reasoning,\s*(\w+)\s+Effort\s*\)` → `(AR·\2)` / `(R·\2)`, and the `Non-reasoning,` variant → `(NR·\1)`. That yields `Claude Opus 5 (AR·Max)` at 22 chars. Change `clean_model_name` to elide the middle rather than the tail so no future suffix can be destroyed. Route `radar.py:107` and the four `name[:35]` sites in `stack_recommender.py` through the shared helper. Add the regression test `len({clean_model_name(m) for m in df.model}) == df.model.nunique()` — it fails today, catching all four groups.

**Outcome:** the Top-25 chart names five real configurations instead of four ordinals; the Compare legend stops shipping duplicate entries.

### 1.4 — Two bar charts are drawn from zero on a truncated axis (lie factors 5.8× and 1.8×)
**Files:** `components/charts/image_scatter.py:191-195`; `components/charts/video_chart.py:105`

Verified from the shipped JSON:

- `image_faceted.json`: `xaxis.range=[1033.16, 1261.80]`, `xaxis2=[1027.70, 1220.12]`, `xaxis3=[981.37, 1244.20]`, all traces `type="bar"` with `base=None`. Bars originate at x=0 and are clipped by the axis, so visible length is `value − range[0]`. A 1.135× true Elo ratio renders as a **5.77× length ratio**.
- `video_rankings.json`: `xaxis.range=[40, 94.86]`, bar `x` spans 58…93, `base=None`, while the chart's own subtitle says "human preference score **0-100**". A 1.60× ratio renders as 2.94×.

**Change:** Elo is an interval scale, so neither zero-baseline nor truncated bar is correct — convert `build_image_faceted` to a dot plot (`go.Scatter(mode="markers")` plus a thin rule from the facet minimum), keeping `x_min`/`x_max` exactly as they are; position, not length, then carries the value, and it deletes 12 ghost-track bars per facet. For video, set `range=[0, 100]` — the metric is explicitly declared 0-100, and the fixed range also stops the axis moving when a provider filter changes `max_q`.

**Outcome:** the image tab stops implying the top model is six times better than the twelfth when it is 13% better.

### 1.5 — The treemap colour ramp gets *darker* as intelligence rises across its bottom 30%
**File:** `components/charts/treemap.py:55-60`

Computed WCAG relative luminance for the six shipped stops (`[0.0 #243056, 0.3 #16213e, 0.55 #0f3460, 0.75 #1a5276, 0.9 #00909e, 1.0 #00d4ff]`, `cmin=0, cmax=70`):

```
0.03161, 0.01606, 0.03402, 0.07562, 0.22415, 0.54307   → monotonic? False
contrast vs #111111: 1.47, 1.19, 1.51, 2.26, 4.93, 10.67
```

A sequential encoding that **inverts over scores 0–21**, which is where 13 of 32 providers sit. The comment at `treemap.py:52-54` records that the old bottom stop was raised because it sat too close to the page background — but the replacement `0.3` stop (`#16213e`, 1.19:1) is *darker* than the stop it replaced (1.47:1). The fix moved the problem.

**Change:** replace with a strictly brightening ramp whose darkest stop clears 3:1 on `#111111` — e.g. `[0.0 #1b3a5c, 0.25 #1c5178, 0.5 #1a6b91, 0.75 #0f8ba5, 1.0 #00d4ff]` (luminances 0.043 → 0.543, strictly increasing). Add a unit test asserting non-decreasing luminance across 100 samples so a future hand-edit cannot reintroduce it.

### 1.6 — Scales derived from the filtered frame: frontier, quadrant medians, tier normalisers
**Files:** `components/charts/pareto.py:106`; `components/charts/quadrant.py:33-34`; `components/stack_recommender.py:127-129`

This is recurring defect pattern (b), and **the test suite never names any of the affected helpers**. Verified:

- `pareto.py:106` calls `_pareto_frontier(plot_df)` on the *filtered* frame. Full catalogue frontier = 10 models; filtered to OpenAI = 5 models, of which `GPT-5.6 Sol (max)` and `GPT-5.6 Terra (max)` are **not on the market frontier** — the chart prints their names next to a line labelled "Pareto Frontier", asserting market-wide cost-optimality that a filter manufactured.
- `quadrant.py:33-34` computes `med_speed`/`med_quality` from the filtered frame. Full medians: speed 105.3, quality 25.7. OpenAI-only: 110.2 / 38.3. Anthropic-only: 62.8 / 55.3. `o3` moves from "Slow · Smart" to "Slow · Weak" on a filter that does not touch it. The zone captions are absolute-sounding.
- `stack_recommender.py:127-129` takes `q_max/v_max/s_max` from `pool`, so a model's printed rank changes when you tick an unrelated provider box.

Note the irony at `quadrant.py:36-39`: a comment congratulating the code for fixing the *bubble-size* version of this bug sits two lines below the unfixed median version.

**Change:** hoist to module-level constants computed once from `data.ingest.get_models()`, exactly as `QUALITY_INDEX_MAX` and `BUBBLE_SPEED_REF` already are (`constants.py:255-263`) — `_FRONTIER_MODELS: set[str]`, `SPEED_MEDIAN_REF`, `QUALITY_MEDIAN_REF` — and intersect/compare against the plotted frame. Add three regression tests asserting invariance under a provider filter.

### 1.7 — Empty results render a well-formed chart of nothing (and one of them is white)
**Files:** `components/charts/pareto.py`, `quadrant.py:31`, `treemap.py`, `cost_calc.py`, `local_scatter.py`, `local_compat.py`, `provider_leaderboard.py`, `video_chart.py`

`empty_figure()` exists at `constants.py:349-373` with a docstring stating exactly why ("every render path receives a user-filtered frame, so 'zero rows' is a reachable state"), and is called from **3 of ~20 builders**. Verified on the speed view: a bare `go.Figure()` with `paper_bgcolor=None` — a white panel in the middle of a `#111111` dashboard. On the price view: dark, but full title, full log axis 0.1→1M, zero data, zero message.

**Change:** one guard line after each frame filter, returning `empty_figure(...)` with a message that names the filter where known. Pair with a single visually-hidden `#chart-status` live region so the same string is announced.

### 1.8 — Contrast and focus sweep (18 measured failures, all mechanical)
**Files:** `docs/assets/style.css:17,72,123,142,632,662,680,712`; `docs/assets/shell.css:38,44,127,143,186`; `docs/app.js:495,526,617`; `static_api.py:54,120,155,175`; `components/stack_recommender.py:649,658,669,674,684,698,699`; eleven chart modules

Independently recomputed a sample: `#666666`/`#0a0a0a` = **3.45**, `#666666`/`#111111` = **3.29**, `#666666`/`#161616` = **3.15**, `#777777`/`#111111` = **4.22**, `#555555`/`#111111` = **2.53**, `#444444`/`#111111` = **1.94**, `#2a2a2a`/`#111111` (Plotly modebar) = **1.32**. All below the 4.5:1 floor for sub-18px text; the modebar is below the 3:1 floor for UI components. Meanwhile `#888888`/`#111111` = 5.33 and `#00d4ff`/`#0a0a0a` = **11.18**, so the fixes are free.

Focus is actively deleted: `shell.css:38` `outline: none` on every select/input, `style.css:662` `outline: none !important` on `.search-input`, plus three inline repeats (`index.html:106,147,249`). The replacement border composites to 1.91:1 and 1.64:1.

**Change:** set `--text-3: #8a8a8a`; replace `#444/#555/#666/#777` text with `#8f8f8f` on card surfaces and `#9a9a9a` inside figures; hoist the two figure greys to named constants in `constants.py` so eleven modules share one value. Delete the four `outline: none` declarations and add one rule: `select:focus-visible, input:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }`. Raise the modebar fill to `rgba(255,255,255,0.30)` and drop `"toImage"` from `modeBarButtonsToRemove` (`docs/app.js:20`) — a dashboard whose purpose is producing comparison figures currently cannot export one.

**Outcome:** zero pixels added, ~20 text elements become legible, keyboard focus becomes visible, and charts become exportable.

### 1.9 — Routing and preset correctness (three separate wrong-label bugs)
**Files:** `docs/app.js:702-710, 750-761`; `docs/index.html:48-54`

- `docs/app.js:761` is `if (u.get("tab")) switchTab(u.get("tab"))` — **no validation**. A stale `?tab=performance` link sets `display:none` on all ten panels and renders a header over a void, silently. `app.py:956` already has the `_VALID_TABS` guard; it was never ported to the JS that serves the public.
- `setPreset` (`:702-710`) snaps the manifest percentile to the nearest coarse option. Verified against the live frame: `p75=42.1 → snaps to 40 → 41 models = 27.7%` (button says "Top 25%"); `p90=52.68 → snaps to 50 → 23 models = 15.5%` (button says **"Top 10%"** — off by 55%).
- The same function does `o.selected = providers.includes(o.value)` with `providers=[]`, so clicking a preset **silently wipes the provider selection**.
- `MIN SCORE` options stop at ≥50 while peak quality is 63.05, so the frontier tier cannot be isolated at all.

**Change:** validate against `TABS` and `history.replaceState` the bad param away; populate the MIN SCORE `<select>` from the manifest so presets are exact (`≥ 42.1`, `≥ 52.7`) and extend the ladder to ≥55/≥60; drop the provider argument from `setPreset`; add `.preset-btn[aria-pressed="true"]` styling. Add a test asserting the JS `TABS` ids equal the `app.py` `dcc.Tab` values — `app.py`'s own `_VALID_TABS` is itself stale (it still lists `performance`, `imagegen`, `videogen`, `embeddings`).

### 1.10 — Performance: four independent S-effort wins in one sitting (~3 s and ~3.5 MB)
**Files:** `docs/app.js:949-950`; `docs/pyworker.js:53-71`; `build_static.py:97, 199-203`

1. **Lazy chart render.** `init()` awaits `Promise.all` over *every* chart of *every* tab before `bootPyodide()`; 11 of 12 are inside `display:none` panels. Measured by the perf lens at ~400 ms of blocked main thread (≈1.2 s on a mid-range Android) — and the work is discarded, because `switchTab` already resizes and `rerenderActiveFilterCharts` already re-renders from Python on activation.
2. **Parallel bundle fetch.** `pyworker.js:53-71` is a strict await chain; the 4.04 MB `pybundle.zip` fetch starts *after* ~15 MB of Pyodide+pandas has transferred, despite having no dependency on it until `unpackArchive`. Start the fetch as the first statement of `boot()` and await it at the unpack point.
3. **Content-hash cache keys.** `build_static.py:97` uses a wall-clock `%Y%m%dT%H%M%SZ` version appended to every figure and bundle URL. The hourly bot regenerates it whether or not anything changed, so every returning visitor re-downloads 4.09 MB every hour and a 304 is impossible. Verified the waste directly: between two consecutive hourly bundles, **1 of 1665 zip members changed** (0.36% of compressed bytes).
4. **Prune the vendored plotly.** Verified: plotly is **3,587,139 of 3,765,577 compressed bytes = 95.3%** of the bundle, and the perf lens's audit-hook trace found 80.7% of it is never opened by any of the 17 callbacks. An allowlist in the vendoring loop (`build_static.py:199-203`) takes `pybundle.zip` from 4.04 MB to roughly 1.1 MB and the unpacked MEMFS tree from 23.7 MB to ~4 MB.

### 1.11 — A test that would have caught the shipped-stale-code incident
**File:** new `tests/test_bundle_freshness.py`

`docs/pybundle.zip` is the Python the browser executes on every interactive re-render, and `--data-only` (the only mode the hourly bot runs) passes every non-CSV member through untouched. **I verified the bundle is fresh right now** (0 stale members of 31 tracked `.py` files) — so TQ-01's headline is a historical incident, not a current defect. But the incident is real and I reproduced it: at commit `1b76aec`, titled *"fix(charts): use Anthropic's exact brand orange and re-solve the palette"*, the source had `#d97757` and the shipped bundle had `#cc4104`. pytest was green; the deployed site showed the old orange on every filter interaction.

Nothing tests this: `tests/test_build_static.py:56` runs a full build *before* inspecting the bundle, so it can only ever see a fresh one, and `:69` actively asserts the stale-source-preserving behaviour is correct.

**Change:** a ~15-line test that opens `docs/pybundle.zip` *without building* and asserts byte-equality against `git ls-files components data static_api.py static_helpers.py`. Run on every push.

**Related hygiene, same sitting:** the working tree was dirty when I started this audit — `data/raw/aa_image_models.csv`, `docs/figures/manifest.json`, `docs/figures/image_faceted.json` and `docs/pybundle.zip` were all modified by an earlier lens simply running the suite. `tests/test_build_static.py:31` accepts a `tmp_path` fixture and never uses it. Give `build_static.py` an `--out DIR` argument and build into `tmp_path`; add a `conftest.py` session fixture that fails if tracked files under `docs/` or `data/raw/` changed. Under the CLAUDE.md auto-push mandate, a test run currently republishes the live site.

---

## 2. Worth doing — high value / L effort, or medium value / cheap

### A. High value, large effort

**2.A.1 — One definition of "a model", applied everywhere.** Five surfaces, five counts in the same session: stat bar 148, Overview 95, Table 148, Rankings 25 variants, treemap raw rows. Verified the consequence: `treemap.json` ranks **OpenAI 25 > Alibaba 22 > NVIDIA 13 > Google 13 > Anthropic 11**; deduped, the same counts are **Alibaba 16 > NVIDIA 9 > OpenAI 8 = Google 8 > Anthropic 5** — the headline tile inverts, purely because OpenAI publishes six effort rows per model and Alibaba publishes one. The provider leaderboard is confounded identically: Google reads avg **28.76** raw vs **34.19** deduped, penalised for benchmarking its own non-reasoning rows. **Change:** count distinct `base_model_name` as "models" everywhere; make the stat bar read "95 models · 148 priced variants"; hover reads "N models (M priced variants)"; leaderboard tick becomes the mean of per-family peaks. *(Files: `constants.py:33`, `treemap.py`, `provider_leaderboard.py`, `docs/app.js:189-193`.)*

**2.A.2 — Variants as first-class objects.** With the vocabulary fixed, add a control-row toggle `VARIANTS: peak only | all effort tiers`. Since price is identical across effort tiers (verified: all six GPT-5.6 Luna rows at $0.95, latency 57.45 → 0.75s), connecting same-family rows with a thin provider-coloured polyline turns each family into a readable "effort curve" — and surfaces the one lever a buyer actually pulls: *turn effort down, keep the model*. This is the chart's own subject on the Speed axis, currently deleted.

**2.A.3 — Effective price with an IN:OUT ratio control.** `price_in` and `price_out` are scraped for all 148 rows (verified: header `model,provider,context,quality,price,speed,latency,price_in,price_out`) and shown in hover text — but **every computation uses only the blended column** (`cost_calc.py:23`, `static_api.py:283`, `bump_chart.py:196`). Verified the ranking inverts for input-heavy work: cheapest models at quality ≥ 40 under a 20:1 blend are DeepSeek V4 Flash $0.094, Hy3 $0.151, **GPT-5.6 Luna $0.248**, then DeepSeek V4 Pro $0.456 — while the Budget tab, using the 3:1 blend, shows **DeepSeek V4 Pro ($0.761) as cheaper than GPT-5.6 Luna ($0.950)**. Anyone optimising a RAG or agent pipeline is handed the wrong answer from data already in memory. Presets: Chat 1:1, AA blend 1:3, Agent 10:1, RAG 20:1.

**2.A.4 — A "Changes" tab.** `load_history()` returns **23,742 rows across 90 daily snapshots**; `app.py:44` assigns it to `history_df`, `app.py:81` refreshes it, and **nothing reads it** (4 occurrences total, all in loading code). `build_bump_chart` is imported at `app.py:40` and never called — and it renders fine. Meanwhile the site cannot answer "what changed this week". Ship a diff strip (N added / removed / repriced), the working bump chart, and per-model sparklines in the detail panel; add a `history.json` to `build_static.py`.

**2.A.5 — Local vs API break-even.** See §3 — this is my answer to "single change that most increases the edge."

**2.A.6 — Image Gen: expose the 33 task ELOs already being scraped hourly.** Verified `data/raw/aa_image_models.csv` carries **34 `elo_*` columns**, of which **33 have ≥50 non-null values** (marketing 151, live-action film 151, consumer 151, animation/gaming 151, human anatomy 150, complex compositions 145, architecture 138, text rendering 127, UI/UX 123, fantasy 108, layout 103…). `image_scatter.py:24-30` hardcodes three facets over six of them. Replace with a TASK selector, and render the chosen task as a ranked bar with a grey overall-ELO bar behind it — "+80 ELO on text rendering relative to overall rank" is the decision; a re-sorted list is not.

**2.A.7 — Accessibility semantics pass.** Verified: the whole ten-tab dashboard contains **one heading** (`<h1>AI FRONTIER</h1>`), **zero landmarks**, `grep -c "role=" docs/index.html` → 0, `grep -c "for=" docs/index.html` → 0, 21 of 23 form controls with no accessible name, zero live regions, and 61 unnamed modebar buttons. Chart titles live inside Plotly SVG, so there is no document outline. Wrap header/nav/main, add `role="tablist"`/`tab`/`tabpanel` with roving tabindex and arrow keys, `aria-labelledby` on every control, `role="status"` on `#py-status` and `#toast`, `scope="col"` + `<caption>` on the table, and `role="img"` + a generated `aria-label` on each chart div. All pixel-neutral.

**2.A.8 — Responsive pass.** Measured by the UI lens at 390px: `#chart-pareto` renders a **124px-tall data band** (28% of the chart) after a 52px top margin, 104px bottom margin and a 124px horizontal legend, with the title clipped 158px off the right edge mid-phrase. The tab bar wraps to 2–3 stacked rows because `.tab-container` — the element `style.css:967-979` styles — **does not exist in the static DOM**; `docs/app.js:102` appends buttons straight into `#tabs`. At 2560px nothing constrains width: the scatter is 3.28:1 and captions run 215 characters on one line, because the `max-width: 720px` rule targets `.chart-desc` while `app.js:939` emits `.chart-caption`.

**2.A.9 — Custom Plotly bundle.** `docs/index.html:23` loads the full 4.65 MB / 1.41 MB-gzip CDN build render-blocking in `<head>` with no `defer`, `async` or `preconnect`. The app constructs exactly four trace types (`go.Bar` 23, `go.Scatter` 21, `go.Treemap` 1, `go.Scatterpolar` 1, zero plotly.express). A `plotly.js/lib/core` + 4 registrations build lands near 400–500 KB gzip and lets you drop `cdn.plot.ly` from the CSP.

### B. Medium value, cheap

**2.B.1 — Make the Table a hub, not a dead end.** `renderTableRows` (`docs/app.js:612-659`) emits plain `<tr>` with no handler, no data attribute, no cursor. The detail panel — the richest per-model view in the product — is bound only to `#chart-pareto` (`docs/app.js:787-806`). Combined with the 53-variant collapse, **the dropped variants are unreachable from the detail panel entirely**: they exist only in the Table, and the Table cannot open them. Traces already carry `customdata[0]/[1]` (verified in `cost_calc.json`: `['Gemma 4 E4B (Reasoning)', 'Google', 0.08, 12.18, '$0.02 in · $0.10 out']`), so one delegated handler serves rows, rankings, value leaders and cost_calc. Same sitting: sticky `<thead>` (the tab is 4,694px tall with 148 rows and the header scrolls away after ~30), `font-variant-numeric: tabular-nums` on the six numeric columns (`.stat-value` already sets it — the site is inconsistent with itself), and scaled price precision so the column stops printing `$20.0000` beside `$0.0800`.

**2.B.2 — Stat bar must respond to filters.** `docs/app.js:189-193` writes the four KPIs once from the manifest and never again. Filter to IBM + ≥50 and the largest, highest-contrast text on the page still reads **148 MODELS / 32 PROVIDERS / $0.080 / 63.0** directly above an empty chart. Return a `{n_models, n_providers, floor_price, peak_quality}` summary alongside every figure and render "37 · MODELS TRACKED · OF 148".

**2.B.3 — Provenance, methodology, and a link home.** Verified: `grep 'href=|<a |artificialanalysis' docs/index.html docs/app.js` returns nothing but the stylesheet and CDN tags. "AA Intelligence Index" is the y-axis label on five figures and the subject of the MIN SCORE filter, and **is never defined anywhere in the product**. There is no link to artificialanalysis.ai, no licence statement, no definition of the 3:1 blend beyond a `title` attribute on one `<th>`. For the student/researcher half of the audience this makes the dashboard uncitable; for the buyer it makes every number unverifiable. One footer: source + link, `manifest.generated_iso`, and a paragraph defining the Index, the blend, TTFT and the variant-collapse rule. Also port the nine explanatory captions that exist in `app.py` and not on the public site.

**2.B.4 — Value Leaders sits under the global filter bar and ignores it.** `update_value_leaders` is defined at `static_api.py:242` and **called by nothing** (verified: zero references in `docs/app.js`, `app.py` or `tests/`); `docs/app.js:681-682` states the intent in a comment. `bump_chart.py:196` hardcodes a quality ≥ 20 floor. Filter to Anthropic and you get an Anthropic ranking above a chart of Gemma/Qwen/DeepSeek/Ling/Nemotron, visually identical, with no explanation. Either wire it, or move it out of the filtered tab behind a "full catalog · ignores filters" badge. Do not leave a filter-inert chart pixel-identical to a filter-live one directly above it.

**2.B.5 — Unify the image palette (the test for this already exists, twice).** `data/image_models.py:16-46` keeps a private `PROVIDER_COLORS` that disagrees with the canonical palette for nine shared providers — Google renders **amber on Overview and blue on Image Gen**; Amazon magenta vs AWS orange; NVIDIA olive vs green. `data/video_models.py:30` fixes exactly this bug with `PROVIDER_COLORS = {**_VIDEO_ONLY_COLORS, **_SHARED_COLORS}`, and `tests/test_chart_contract.py:299` and `:213` guard the video and local tabs respectively. **The image tab is the one still broken and the one with no test.** Copy both, one file each.

**2.B.6 — Detail panel dismissal.** No Escape handler anywhere in `docs/app.js` (`grep -c "keydown\|Escape"` → 0), no scrim, no close on tab switch, no `inert` when hidden — so its two buttons are the last two Tab stops on the page, positioned at x=1718 and x=1461 in a 1440px viewport. It also covers 287px of the chart beneath it, including the right-gutter annotations that `local_compat.json` reserves `margin.r=185` for.

**2.B.7 — Compare's cap evicts the wrong model.** `docs/app.js:836-840` uses `selectedOptions` (document order, not click order), so adding a 6th model deletes whichever selected model sits lowest in the 148-item list — verified by the UI lens: the user's new pick survived and an untouched "Grok 4.3 (low)" was removed, with no message and no "max 5" hint in the DOM. Track click order; evict the oldest; say so.

**2.B.8 — Radar ceilings are set to values the data never approaches.** `constants.py:260-263` fixes `RADAR_SPEED_MAX=2500`, `RADAR_CONTEXT_K_MAX=2000`, `RADAR_PRICE_MAX=50`, `RADAR_LATENCY_MAX=30`. Observed maxima: 1560 tok/s, 1000k context, $40. Decoded `radar.json` — the five default polygons have Speed values **0.021, 0.024, 0.119, 0.064, 0.020** and Context **0.5, 0.5, 0.128, 0.5, 0.5**, with `radialaxis.range=[0,1]`, `tickformat=".0%"`. Two of five spokes carry no information, four of five traces are identical on Context, and two are pinned to Latency 0.0 (where "slowest measured" and "no data" look the same). The fixed-ceiling doctrine is correct and should stay — the ceilings just need to be anchored to observed maxima with log scaling on speed and price. Same sitting: fix `static_helpers.py:32-36` so `compute_diverse5` dedupes on family (`manifest.diverse5` currently spends two of five slots on Claude Opus 5 Max and Xhigh, whose polygons differ on one spoke), and add a `line.dash` cycle so five models are separated by more than hue.

**2.B.9 — Bubble sizing.** Four related defects, one helper: `bubble_size` (`constants.py:284-290`) interpolates in *diameter* space with a 7px additive floor, so area is not proportional to value despite the docstring's claim (a 2.46× speed advantage renders as 1.52× area). `BUBBLE_PRICE_REF` inversion is linear, so **91 of 95 quadrant bubbles fall in 19–26px with a cliff to 7px at $20**. `LOCAL_SPEED_REF=400` clamps **21 of 97** local models to the ceiling on a catalogue reaching 1113.7 tok/s. And `video_chart.py:130-132` bypasses the shared helper entirely, sizing by radius and rescaling from the *filtered* frame — filtering to Runway alone moves Gen-4 from 22.00px to 8.00px for a model whose 90s generation time never changed. Fix: interpolate in area space (`sqrt(MIN² + frac·(MAX² − MIN²))`), add a log branch for price and local speed, raise the local ref, route video through the helper.

**2.B.10 — Collapse the Landscape tab.** Both charts encode headcount. The treemap's area metric is one a buyer never optimises and inverts under deduping (§2.A.1). The provider leaderboard is **1,040px tall** for 32 providers of which 15 have exactly one model, while the top 8 hold 72% of the catalogue — and its right-gutter annotations read `"1 models · Reka Flash 3"`. Two screens of scrolling, no decision, and a pluralisation bug on visible text. Fold the useful half into Overview (click-to-filter legend, peak/median intelligence in the legend hover); if a provider view survives, cap it at the top 8 + "Other (24 providers)", switch to peak intelligence with a median tick and a $/M whisker, and cap the height at ~420px.

**2.B.11 — Video Gen: gate it or cut it.** Verified: there is **no `data/video_scraper.py`** and no `data/raw/aa_video_models.csv` — `data/` contains `scraper.py`, `image_scraper.py`, `local_scraper.py` only. `get_video_df()` returns **20 rows** whose newest entries are Veo 3, Sora HD, Kling 1.6 Pro — early-2025 models — while the LLM catalogue beside it carries GPT-5.6 and Claude Opus 5. The module's own docstring says the quality scores are "human preference ratings (0-100 scale, **approximate**)" from "public pricing pages, Artificial Analysis, EvalVid, community benchmarks", and `video_rankings.json` renders them with false precision (`$0.001/sec · 30s gen · 576p`). `docs/index.html:30` puts a live "Updated N min ago" badge above the whole app, so the tab inherits credibility it has not earned. This is the clearest violation of CLAUDE.md's "never write dummy code", and it is on a top-level tab. Minimum: a per-tab provenance banner and drop `gen_time_s` and the sub-cent annotations. Real fix: `data/video_scraper.py` mirroring `data/image_scraper.py`.

**2.B.12 — Delete the seven dead chart modules.** Verified zero references from `app.py`, `static_api.py`, `build_static.py`, `docs/app.js`, `docs/index.html` for `animated_pareto`, `context_chart`, `embedding_chart`, `price_timeline`, `trends`, `local_rankings`, `image_rankings` — ~1,275 lines, all vendored into `pybundle.zip` and downloaded by every visitor. This is where the next drift bug comes from: a maintainer editing `trends.py` to fix the trends view will see no change anywhere and no error. Add an explicit allowlist to the packing step instead of globbing `components/charts/`.

**2.B.13 — Surface failures instead of logging them.** 18 of 19 `catch` handlers in `docs/app.js` write to console and stop; `toast()` is defined at `:394` and called from exactly one place — `:415`, the *success* path. A Pyodide call that throws leaves the previous chart on screen and the user believes their filter applied. Route all render/refresh handlers through one `reportFailure(where, err)` and add the test that asserts no `refresh*`/`render*` catch body contains `console.` without `toast(` or `setStatus(`.

**2.B.14 — Reject unknown enum values instead of falling through.** `rankings.py:46-48` uses `if value / elif speed / else`, so any unrecognised metric silently renders the Intelligence ranking complete with the "AA Intelligence Index" axis title. Verified: `update_rankings([],0,'','not-a-metric')` returns **byte-identical JSON** to `update_rankings([],0,'','intelligence')`. Nothing in `tests/` calls `update_rankings` at all.

**2.B.15 — Strengthen the assertions that certify emptiness as success.** `tests/test_static_api.py:102` asserts `"<" in cards_html`, true of any HTML ever produced — `update_recommend(['NoSuchProvider'], …)` returns 9,703 bytes with **zero** `tok/s` occurrences and passes. `tests/test_chart_contract.py:117` and `tests/test_pareto_chart.py:250` assert only `is not None`. And 37 of the 43 assertions in `tests/test_static_site_wiring.py` — the **only** file that reads `docs/app.js`, which is 961 lines re-implementing `app.py`'s callbacks for the only publicly reachable build — are substring greps against raw file text (verified counts; 234 assertions suite-wide). They fail on a rename and pass on a broken implementation. Convert the six highest-value ones to executed assertions using the node-skip pattern already proven at `test_static_site_wiring.py:87`.

**2.B.16 — A parametrized contract for the 8 shipped-but-unasserted builders** (`build_treemap`, `build_rankings`, `build_radar`, `build_value_leaders`, `build_local_scatter`, `build_image_faceted`, `build_video_scatter`, `build_bump_chart`): non-zero plotted points, `paper_bgcolor == BG`, `margin.r <= MAX_RIGHT_GUTTER_PX`, no `'nan'` in annotation text. ~25 lines.

**2.B.17 — Guard the scraper against a silent field rename.** `data/scraper.py:113-115` reads `hm.get("timescaleData") or {}` then `.get("median_output_speed") or 0`. The test lens fed the parser a payload keyed `timeseriesData` and got back **speed=0, latency=0, context=0** with `coverage: {kept: 1, skipped_no_score: [], skipped_no_price: []}` and `scrape_and_save()` returning success. Nothing in the 134-test suite fails. Add a `zero_speed` counter to `_last_coverage`, refuse the scrape above a 50% threshold, and assert `(get_models()["speed"] > 0).mean() > 0.8` on the committed snapshot. AA has already renamed fields once (`tag.label` → `tag.displayName`, per `image_scraper.py:16-18`).

**2.B.18 — README and `report.tex` describe a different product.** `README.md:3`, `:36` and `:73` advertise "300+ large language models" across "30+ providers"; the dataset is **148 models across 32 providers**. `:73` even claims counts are "always visible live in the dashboard's header stats" — the header says 148, contradicting the sentence it sits in. The hero screenshot shows 324 models. `report.tex` — the source of the graded PDF — describes 255+ models, eleven tabs, a Trends tab that does not exist, and a Render deployment.

---

## 3. Deliberately not doing — and why

**3.1 — Do not delete pandas/Pyodide and reimplement the data layer in JS.** (rejects PERF-07's option (b) and PERF-12's "single biggest structural lever".) The measurement is sound: `import pandas` costs 1.26 s and +48 MB of WASM heap, and the pandas+numpy+pytz+dateutil+six wheel closure is 9.5 MB for operations over a 148-row frame. But the codebase's **single largest defect generator is that `docs/app.js` already re-implements parts of `app.py`'s callback logic by hand** — pattern (c), which produced the stale `?tab=` guard, the dropped captions, the `?q=42.1` mismatch, the Compare-selection divergence, and the preset disagreement between the two renderings. Pyodide is the mechanism that keeps `static_api.py` as *one* implementation serving both. Porting the filter/sort/groupby logic to JS would make that fork permanent, load-bearing, and untested by the Python suite. Items **1.10** and **2.A.9** recover roughly 3.5 MB and ~1.5 s without touching the architecture; take those instead, then re-measure.

**3.2 — Do not add a "View this data as a table" link under every chart.** (rejects a11y-03 part 2.) Thirteen new visible links across ten tabs to reach a tab that is one click away, against an explicit "avoid visual clutter at all costs" mandate. The accessible-alternative requirement is real, and it is satisfied more cheaply by **2.B.1** (the Table becomes reachable from any chart click) plus `role="img"` + generated `aria-label` on each chart div, which costs zero pixels.

**3.3 — Do not build a bespoke multiselect popover component now.** (defers ui-08.) The diagnosis is correct — `#filter-provider` shows **1.22 of 32 options** (clientHeight 30 vs scrollHeight 784) and renders as a text field containing the words "AI21 Labs"; `#radar-model-select` is 148 options in a 110px box. But a popover with type-ahead, select-all, and touch support is a genuine component with its own state, focus-trap and keyboard contract — days, not a sitting, and it competes with items that fix *wrong answers*. Ship the S-effort stopgap (`size="5"`, a live `3 / 32` count, moved out of the single-line bar) and revisit after §1 lands.

**3.4 — Do not revive `animated_pareto.py` or the other six dead modules.** (rejects DVC-14's "either wire it or delete it" as a genuine either/or.) An animated Pareto with autoplay frames is the chart most likely to cost more attention than it gives, it has never been subjected to a production readability judgement because it has never rendered, and it would need a reduced-motion gate on top. The history data deserves the **diff strip** in §2.A.4, which answers "what changed", not a 900ms autoplay loop that shows motion. Delete all seven.

**3.5 — Do not truncate float precision in the figure JSON.** The perf lens measured it and reported the negative result honestly: rounding every float in the 12 fetched figures to 6 significant figures saves 8,267 raw bytes but **475 gzip bytes**, because gzip already collapses the repeated `0.1111111111111111` tails. The real 27% is the byte-identical 6,621-byte Plotly template duplicated across all 13 figures (verified: 1 distinct sha256) — strip that instead, in **2.A.9**'s sitting.

**3.6 — Do not add `aria-label` to 61 Plotly modebar buttons via a post-render DOM pass.** (rejects a11y-16's naming recommendation.) It is a per-render loop over third-party DOM that will break on any Plotly upgrade, to name controls that are **not in the tab order at all** — so the naming buys nothing until keyboard reachability is also solved, which Plotly does not support. Take the contrast half of that finding (`#2a2a2a` → `rgba(255,255,255,0.30)`, 1.32:1 → ~3.4:1) in **1.8** and leave the semantics.

**3.7 — Do not keep Video Gen as a peer tab.** (rejects the implicit status quo.) 20 hand-typed rows of 2025-era models with "approximate" scores, rendered to three decimal places, beneath a global hourly-freshness badge, is not a tab that is merely thin — it is a tab that actively transfers unearned credibility from the live pipeline to stale hand-entered data. Demote or scrape; do not leave.

**3.8 — Do not run the 61%-dead-CSS purge as a standalone task.** (defers ui-04.) The measurement (83 dead rules of 148, 11,817 of 19,389 bytes) is accurate, but the *dead bytes* are not the problem — 11.8 KB uncompressed is noise next to a 20.8 MB cold load. What matters is the three specific dead blocks that cause real bugs (the phantom `.tab-container`, the never-applied `.detail-panel-*` classes, the `.chart-desc` max-width that targets the wrong class), and those are already inside **2.A.8** and **2.B.6**. Split `dash-only.css` out opportunistically when you next touch the file.

**3.9 — Do not extend the `title`-attribute pattern for disclosures.** The coverage note (`docs/app.js:79-92`) and the price-blend definition (`:628`) — the two strings that qualify every number on the page — live in `title` attributes on non-focusable elements (measured `tabIndex = -1`), reachable by mouse hover only. The fix is to *replace* them with a `<details>` disclosure and a `<caption>`, not to add more of them elsewhere.

**3.10 — Do not fix the stat-bar gutter rails as a priority.** (ui-23.) It is a real artefact (`background: var(--border)` + `gap: 1px` + `padding: 0 32px` exposes a divider-coloured strip at each end that reads as a truncated fifth stat, and there are two leftover `.stat:nth-child(5)` rules for a four-stat layout) but it is cosmetic and nobody has misread a number because of it. Fold into whichever sitting next touches `style.css`.

---

## 4. Is this useful to users today?

**Partly — and the parts that are useful are not the parts that look most finished.**

The honest test is: *on arrival, with no interaction, does each tab tell the truth?* I ran that test. **Overview** plots 95 of 148 models with a stat tile above it reading 148 and no disclosure. **Budget** ranks the thirty weakest models in the catalogue by a two-cent monthly difference, with its one real answer hidden behind a slider nobody knows to drag. **Agent Stack** recommends a 57.45s-TTFT model as the "cheap, high-throughput, parallel calls" workhorse — a claim its own on-card chip contradicts in the next line, and one that is 6× the price and 104× the latency of the answer the same function returns two lines of HTML later. **Compare** ships two byte-identical legend entries and two of five spokes that no model in the dataset can move. **Landscape**'s headline tile ranks OpenAI first on a metric that inverts the moment you count models the way every other tab counts them. Four of the six decision surfaces are currently wrong on arrival, and none of the four is wrong because the data is wrong — every one is a display, default, or scale bug over correct underlying numbers.

So: useful as a **market map** — it is fast to scan, the density is genuinely good, the provider palette work is real, and the Overview scatter with symbol-redundant encoding is a better single view of the price/quality plane than most leaderboards. Not yet reliable as a **buying tool**, because on three of the four surfaces where it commits to an answer, the answer it gives first is not the answer its own data supports.

**Where the genuine edge over artificialanalysis.ai is — and is not.**

It is not the data. The data *is* AA's, re-served with a one-hour lag, at 148 models versus AA's full catalogue, behind a 20.8 MB cold load and a ~3 s Pyodide boot. Any surface that is a re-rendered leaderboard is strictly worse than going to the source: **Rankings** is a re-sorted Table with a filter-inert chart beneath it, and **Landscape** is a headcount treemap. Those two tabs are decoration — not because they are badly made, but because they answer questions ("who publishes the most rows?") that no user has, using a metric that inverts under a definitional choice the user cannot see. **Video Gen** is worse than decoration: 20 hand-typed rows of early-2025 models with self-described "approximate" scores, rendered to three decimals under a live freshness badge, in a repo whose CLAUDE.md says "never write dummy code."

The edge is **synthesis and opinion over a dataset AA publishes but does not compose**:

1. **Run Local** — VRAM fit × quantisation × estimated tok/s across 137 GPU presets is not on AA in any form. This is the most defensible surface in the product.
2. **Agent Stack** — an opinionated three-tier recommendation is editorial. AA will never ship "here is your sub-agent model, here is your orchestrator" because AA is a benchmark, not a point of view.
3. **Budget** — "what does this cost me per month at my volume" is arithmetic AA leaves to the reader.
4. **90 days of archived daily snapshots** — 23,742 rows AA does not publish as a series.

Everything on the Do-Next list makes the dashboard *correct*. Only these four make it *necessary*.

**The single change that would most increase the edge: compute local-vs-API break-even on the Run Local tab.**

The Run Local tab currently ends its sentence halfway. It says "46 models fit your hardware" at N tok/s and stops. The decision it exists to serve — *is a $2,000 RTX 5090 cheaper than paying $0.158/M for DeepSeek V4 Flash, and above what monthly volume?* — is not computable from anything on screen, even though the API side has $/M for all 148 models and the local side has tok/s for all 97. Verified the gap is exactly two fields wide: `GPU_BY_NAME` holds 137 presets shaped `{name, vram_gb, bandwidth_gbps, hw_type, category}`, and `grep -cE 'price|cost|usd|msrp|watt|power|tdp' data/local_models.py` returns **0**. Add `msrp_usd` and `tdp_w` per preset plus three control-row inputs (amortisation, $/kWh, utilisation), plot local $/M on the same log axis the Overview already uses, draw the closest-quality API model as a horizontal reference line, and print the crossover: *"break-even vs DeepSeek V4 Flash at 41M tokens/month."*

That is the one number on this site that a user cannot get from artificialanalysis.ai, cannot get from a vendor pricing page, and would restructure a four-figure purchase around. It is one arithmetic step from data already in the repo, and it converts the tab from a compatibility checker into the reason the dashboard exists.

**One hard caveat, and I would not ship without it.** The correctness lens found that the local tab's physics are themselves wrong in three ways — VRAM has no context/KV-cache term at all despite the docstrings claiming it does, MoE throughput is computed from active params only (74–150% optimistic), and the Apple bandwidth-efficiency factor (0.82) exceeds NVIDIA's (0.55), so the tool currently claims an M3 Ultra out-generates an RTX 4090. A dollar figure built on those tok/s numbers would not merely be wrong, it would be *confidently* wrong about money, which is a strictly worse failure than being vague about speed. **Fix the local physics first, then ship the break-even.** That sequencing is the whole recommendation; inverting it would be the most damaging thing on this roadmap.

Runner-up, if the local physics turn out to be a bigger job than they look: the **Changes tab** (§2.A.4). 90 days of snapshots showing 6 models added, 18 removed and 5 repriced in the last week alone — with `Celeris-1` down $4.43 — is a second thing AA does not give you, and the chart that renders it is already written, already imported at `app.py:40`, and has never once been called.