# AI Frontier — full-scale audit, 2026-08-12

Baseline commit: `b9d6e19`. Test suite at time of audit: **134 passed**.

36 agents across 12 correctness dimensions and 6 quality lenses. Every correctness
dimension was followed by an adversarial refutation pass whose instruction was to
*kill* findings, defaulting to REFUTED when a claim could not be positively confirmed.
101 findings raised → 99 survived → 80 distinct defects after dedup.

| File | Contents |
|---|---|
| [`correctness.md`](correctness.md) | 80 defects in 10 themes, each with evidence, root cause, fix, and a verification step |
| [`roadmap.md`](roadmap.md) | Prioritized improvement roadmap (Do next / Worth doing / Deliberately not doing) + an honest answer to "is this useful today?" |
| [`findings.json`](findings.json) | Raw structured findings with per-finding verdicts |

## The six things that are wrong on arrival, no interaction required

1. **Price is the wrong blend and the UI credits AA for it.** `data/scraper.py:101` reads
   `price_1m_blended_3_to_1`, which is absent from the live feed, so 100% of rows take the
   fallback `(3·price_out + price_in)/4`. AA publishes `price_1m_blended_0_3_1` — the opposite
   weighting. Independently verified: 148/148 rows match the output-weighted blend, 8/148 match
   AA's, **median 1.914× overstatement**.
2. **The Pareto frontier admits strictly dominated models.** `pareto.py:29` sorts by price with
   pandas' unstable quicksort and no secondary key, so on a price tie the worse model can be
   appended first. Verified: at `quality ≥ 10` the code returns 12 frontier models where the true
   non-dominated set is 10.
3. **Quadrant thresholds are computed from the filtered frame.** Verified median speed moves
   103.17 → 83.24 and median quality 28.31 → 46.86 on a provider filter, so a model's
   "Fast · Smart" label is really a statement about who else is on screen.
4. **The radar's ceilings were never checked against the data.** Verified: speed ceiling 2500 vs
   population max 1560 (132/148 models render inside the innermost 10% of the spoke); latency
   ceiling 30s vs population max 101.87s, so 10 models clamp to a flat 0%. The caption claims
   normalization "relative to the best model in the full dataset".
5. **Filters silently invert.** `stack_recommender.py:425` is `if providers:` — verified that
   unchecking every provider returns byte-identical output to checking All. The `xAI` checkbox
   matches 0 rows (the data says `SpaceXAI`, and `canonical_provider()` already exists at
   `constants.py:72` to fix exactly this).
6. **"Updated just now" is a build timestamp, not a data timestamp** — proven on real CI run
   `31558618824`, where two of three scrapers failed, the site published a 5h-stale catalogue
   under a fresh badge, and the failure step ran *after* the push.

## Root causes

62 of the 80 defects trace to three patterns already known to this codebase:
silent failure, scales derived from the filtered frame, and `app.py` ↔ `docs/app.js` drift.
A fourth pattern the audit named: **hand-set encoding constants never checked against the
population** (radar ceilings, treemap ramp, bubble ramp, the `max(900, …)` ELO floor,
the `40` bar baseline) — 7 defects, all shipping to every visitor.

Three defects in this report were **already fixed once and regressed, or were fixed on one side
only**: the `?tab=` guard, the white-empty-figure, and the dead-tag filter. In all three the fix
landed as a patch at the call site rather than as an assertion. Until each has a test, expect
them back.

## The four fixes that close the most ground

1. One test that walks `app.layout`, collects every id, and asserts every callback dependency
   resolves. (Kills the dead detail panel outright and prevents its whole class.)
2. Thread a `full_frame` argument (or precomputed thresholds) into `build_quadrant`,
   `_pareto_frontier`, `_pick_elo_column`, `_add_pareto` and `_pick_api_tier` — closes all five
   filtered-frame defects at once.
3. `if x is not None` instead of `if x:` at the six numeric/list sites, plus `canonical_provider()`
   in the recommender, plus data-derived filter options.
4. Per-dataset scrape status in the manifest, driving the freshness badge from the *oldest*
   successful fetch.

## Known audit-process artifact

Running `pytest` rebuilds and dirties the published site — `tests/test_build_static.py:31`
accepts a `tmp_path` fixture and never uses it, while `build_static.py:41` hardcodes
`DOCS = ROOT/"docs"`. This is finding §7.5, and it self-demonstrated during this audit:
`docs/pybundle.zip`, `docs/figures/manifest.json`, `docs/figures/image_faceted.json` and
`data/raw/aa_image_models.csv` were all modified simply by running the suite. Under CLAUDE.md's
auto-push mandate, a test run republishes the live site.
