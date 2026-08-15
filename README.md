# AI Frontier

> An interactive dashboard mapping the entire AI model landscape — comparing 300+ large language models on cost, speed, and intelligence in one place, updated hourly.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-Plotly-informational?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-Live%20%E2%80%93%20hourly-brightgreen)

### 🔗 Live dashboard → **https://avo-sarkissian.github.io/AI-Frontier-Database/**

Hosted free on GitHub Pages. The full interactive dashboard runs entirely in your browser — the same Python chart code is executed client-side via [Pyodide](https://pyodide.org/) (WebAssembly), so there is no server. Charts paint instantly from a pre-rendered snapshot; the live filters/calculators activate after a one-time background load (a few seconds, cached thereafter).

![Overview tab — cost vs. intelligence scatter with Pareto frontier](screenshots/overview.png)

---

## Overview

Every week, new AI models ship from OpenAI, Google, Anthropic, Meta, and dozens of others. Each makes different tradeoffs — some excel at reasoning but cost more, some are fast but less capable, and some punch well above their price point.

**AI Frontier** makes those tradeoffs visible. A GitHub Actions bot scrapes live benchmark data from [Artificial Analysis](https://artificialanalysis.ai/) every hour and surfaces the full competitive landscape through 10 interactive tabs — no sign-in, no API key required.

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| **Overview** | Bubble scatter of every model, togglable between Price and Speed on the x-axis (y = AA Intelligence Index either way). Dotted Pareto frontier highlights best value per dollar. Color and marker shape both encode provider for colorblind accessibility. |
| **Agent Stack** | Pick a workflow — API Only, Hybrid (fast local), Hybrid (fast + balanced local), or Local Only — plus a provider checklist, and get a three-tier model recommendation: Fast, Balanced, and Reasoning. Hybrid/Local modes add GPU, VRAM, and quantization controls. |
| **Landscape** | Treemap of the AI industry: tile area = model count per provider, color intensity = average intelligence. Followed by a provider leaderboard. |
| **Rankings** | Top 25 models by Intelligence, Value (score/$), or Speed, with tier-separator lines marking meaningful performance gaps — plus a "Value Leaders" chart ranking the top models by intelligence-per-dollar. |
| **Compare** | Head-to-head radar chart across five dimensions: Intelligence, Speed, Affordability, Context Window, and Latency. Supports up to 5 models simultaneously, with a raw-value table below the chart. |
| **Budget** | Enter a monthly token volume and see projected API cost per model, sorted cheapest-first. |
| **Table** | Full sortable table of all 300+ models with every available metric. |
| **Run Local** | Select a GPU, quantization level, and capability tags (code, reasoning, vision, multilingual) to see which open-weight models fit in VRAM, with estimated inference speed. |
| **Image Gen** | ELO-based leaderboard of image generation models, filterable by provider and style tag, from the AA Image Arena. |
| **Video Gen** | Ranked comparison of video generation models with pricing and quality data, filterable by provider and tag. |

---

## Screenshots

<table>
<tr>
<td width="50%"><img src="screenshots/agent-stack.png" alt="Agent Stack tab — three-tier model recommendations"></td>
<td width="50%"><img src="screenshots/landscape.png" alt="Landscape tab — provider treemap"></td>
</tr>
<tr>
<td width="50%"><img src="screenshots/rankings.png" alt="Rankings tab — top 25 models and Value Leaders"></td>
<td width="50%"><img src="screenshots/compare.png" alt="Compare tab — radar chart across five dimensions"></td>
</tr>
</table>

![Table tab — full sortable model table](screenshots/table.png)

---

## Data Sources

| Metric | Source | Notes |
|---|---|---|
| Intelligence | [Artificial Analysis](https://artificialanalysis.ai/) | Composite of reasoning, coding, math, and knowledge benchmarks |
| Price | Derived from Artificial Analysis | USD per 1M tokens, cheapest available host. **Our blend, not AA's**: 3 parts output to 1 part input. AA publishes the opposite weighting (3 parts input to 1 part output), so a price here reads roughly 1.9× theirs. Output-weighted is the honest basis for agentic and RAG workloads, where output dominates spend; the per-token `price_in` / `price_out` are shown beside it everywhere so you can compute either. |
| Speed | Artificial Analysis | Median tokens/second from the same host as the price |
| Latency | Artificial Analysis | Median time-to-first-token (TTFT) in seconds |
| Context Window | Artificial Analysis | Maximum input length **the model supports**, not what a given host serves — a host may cap it lower. Unlike price and speed this is not a cheapest-host figure. |
| Image ELO | AA Image Arena | Human-preference ranking, scraped live hourly |
| Open-weight / local models | Artificial Analysis | GPU VRAM fit, parameter counts, quantization levels, scraped live hourly |
| Video generation models | Curated dataset | Manually maintained pricing/quality data (`data/video_models.py`) — not live-scraped |

The LLM dataset currently covers **300+ models** across **30+ providers** (exact counts fluctuate hourly and are always visible live in the dashboard's header stats). Three separate scrapers (`data/scraper.py`, `data/image_scraper.py`, `data/local_scraper.py`) each pull their own Artificial Analysis endpoint on the same hourly cadence.

---

## Live Data Refresh

An **hourly GitHub Actions workflow** (`.github/workflows/refresh.yml`) is the sole source of truth for the deployed site — browsers can't call the Artificial Analysis API directly (CORS), so a cron job (`23 * * * *`, plus manual `workflow_dispatch`) does it server-side:

1. Runs all three scrapers, falling back to the last-known-good cache per source if a fetch fails.
2. Sanity-checks row counts before proceeding.
3. Skips the rebuild entirely if nothing changed (git-diff change guard).
4. Runs `python build_static.py --data-only` to swap the new CSVs into the static bundle without re-vendoring Plotly.
5. Commits and pushes the refreshed `docs/` folder as `github-actions[bot]`, which GitHub Pages auto-deploys.

The live site auto-loads the latest published snapshot on open and shows an "Updated X ago" freshness badge with a manual refresh button.

---

## Running Locally

Requires **Python 3.11+**.

```bash
# Clone the repo
git clone https://github.com/Avo-Sarkissian/AI-Frontier-Database.git
cd AI-Frontier-Database

# Create and activate a virtual environment.
# Use .venv, not a name containing spaces: this repo lives under a path with a
# space in it, and a venv whose activate script embeds that path silently
# prepends a directory that does not exist. The script still exits 0 and still
# sets the (name) prompt, so you get the SYSTEM python believing you are in a
# venv — which is how a build once ran under plotly 6.0 and tripled the bundle.
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
python -c "import sys; print(sys.prefix)"   # must print .../AI Frontier/.venv

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
python app.py
```

Open **http://localhost:8050**. The scrapers fetch fresh data on startup and run in the background every hour — no manual refresh needed.

---

## Static site (GitHub Pages)

The public site at **https://avo-sarkissian.github.io/AI-Frontier-Database/** is a static, server-free build of the same dashboard, served from the `docs/` folder on `main`.

```bash
# Regenerate the static site into docs/ (pre-rendered figures + Pyodide bundle)
python build_static.py

# Or, after a data-only refresh (see Live Data Refresh above):
python build_static.py --data-only

git add docs && git commit -m "rebuild static site" && git push   # Pages auto-deploys (~1 min)
```

How it works: `build_static.py` pre-renders every chart's default state to JSON and zips the project's Python (chart builders, data loaders, `static_api.py`) plus `plotly` into `docs/pybundle.zip`. The page loads the pre-rendered figures instantly, then boots Pyodide in the background and calls the **same** Python chart code in the browser for live filtering — so the static site is faithful to the Dash app with zero hosting cost. In normal operation the [hourly GitHub Actions bot](#live-data-refresh) handles the refresh + rebuild + deploy automatically; the manual commands above are only needed for local iteration or a full rebuild.

---

## Architecture

```
app.py                     # Dash app entry point, tab routing, background scrapers
components/
  charts/                  # One module per visualization (pareto, radar, treemap, bump chart, …)
  stack_recommender.py     # Agent Stack tab recommendation logic
data/
  scraper.py                # Live LLM data — Artificial Analysis
  image_scraper.py          # Live image-gen data — AA Image Arena
  local_scraper.py          # Live open-weight/local model data — Artificial Analysis
  ingest.py                  # Parses scraper output, manages CSV cache and daily history
  image_models.py, local_models.py, video_models.py   # Per-domain data access helpers
  raw/                      # Live cache + daily timestamped snapshots
utils/
  model_lookup.py           # Shared model metadata helpers
assets/
  style.css                 # Global dark theme, typography
static_helpers.py          # Pure dash-free helpers shared by app.py + the static build
static_api.py               # Browser bridge — ports the Dash callbacks to JSON-returning functions (runs in Pyodide)
build_static.py             # Pre-renders figures + bundles Python into docs/ for GitHub Pages
docs/                        # The static GitHub Pages site (index.html, app.js, figures/, pybundle.zip)
.github/workflows/refresh.yml   # Hourly bot: scrape → guard → data-only rebuild → commit + push
tests/                       # pytest suite covering the static build, scrapers, and site wiring
```

---

## Built With

| Tool | Role |
|---|---|
| [Plotly Dash](https://dash.plotly.com/) | Interactive web framework — Python functions rendered as a live site |
| [Plotly](https://plotly.com/python/) | All charts and visualizations |
| [Pandas](https://pandas.pydata.org/) | Data ingestion, filtering, and Pareto computation |
| [requests](https://docs.python-requests.org/) | Live data scraping from the Artificial Analysis API |
| [NumPy](https://numpy.org/) | Numerical operations (Pareto frontier, correlation) |
| [Pyodide](https://pyodide.org/) | Runs the Python chart code in the browser (WebAssembly) for the static site |
| [GitHub Actions](https://github.com/features/actions) | Hourly scrape → rebuild → deploy bot |
| [GitHub Pages](https://pages.github.com/) | Free static hosting for the live dashboard — auto-deploys on every push to `main` |

---

*Data Visualization — EECE 5642, Northeastern University, Spring 2026*
