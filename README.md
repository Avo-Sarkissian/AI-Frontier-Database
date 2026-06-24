# AI Frontier

> An interactive dashboard mapping the entire AI model landscape — comparing 255+ large language models on cost, speed, and intelligence in one place, updated hourly.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-Plotly-informational?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-Live%20%E2%80%93%20hourly-brightgreen)

### 🔗 Live dashboard → **https://avo-sarkissian.github.io/AI-Frontier-Database/**

Hosted free on GitHub Pages. The full interactive dashboard runs entirely in your browser — the same Python chart code is executed client-side via [Pyodide](https://pyodide.org/) (WebAssembly), so there is no server. Charts paint instantly from a pre-rendered snapshot; the live filters/calculators activate after a one-time background load (a few seconds, cached thereafter).

---

## Overview

Every week, new AI models ship from OpenAI, Google, Anthropic, Meta, and dozens of others. Each makes different tradeoffs — some excel at reasoning but cost more, some are fast but less capable, and some punch well above their price point.

**AI Frontier** makes those tradeoffs visible. It scrapes live benchmark data from [Artificial Analysis](https://artificialanalysis.ai/) every hour and surfaces the full competitive landscape through 11 interactive tabs — no sign-in, no API key required.

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| **Overview** | Bubble scatter of every model — x = price, y = intelligence, size = speed. Dotted Pareto frontier highlights best value per dollar. Color and marker shape both encode provider for colorblind accessibility. |
| **Agent Stack** | Pick a workflow (API-only, Hybrid, Local-only) and get a three-tier model recommendation: Fast, Balanced, and Reasoning. Filter by GPU and quantization in Hybrid/Local mode. |
| **Landscape** | Treemap of the AI industry: tile area = model count per provider, color intensity = average intelligence. Followed by a provider leaderboard. |
| **Rankings** | Top 25 models by Intelligence, Value (score/$), or Speed. Tier-separator lines mark meaningful performance gaps. |
| **Compare** | Head-to-head radar chart across five dimensions: Intelligence, Speed, Affordability, Context Window, and Latency. Supports up to 5 models simultaneously. |
| **Budget** | Enter a monthly token volume and see projected API cost per model, sorted cheapest-first. |
| **Table** | Full sortable table of all 255+ models with every available metric. |
| **Run Local** | Select a GPU and quantization level to see which open-weight models fit in VRAM, with estimated inference speed. |
| **Image Gen** | ELO-based leaderboard of image generation models across 15 style categories from the AA Image Arena. |
| **Video Gen** | Ranked comparison of video generation models with pricing and quality data. |
| **Trends** | Price-over-time chart tracking how API pricing has shifted across daily snapshots. |

---

## Data Sources

| Metric | Source | Notes |
|---|---|---|
| Intelligence | [Artificial Analysis](https://artificialanalysis.ai/) | Composite of reasoning, coding, math, and knowledge benchmarks |
| Price | Artificial Analysis | USD per 1M tokens — blended 3:1 output/input ratio, cheapest available host |
| Speed | Artificial Analysis | Median tokens/second from the same host as the price |
| Latency | Artificial Analysis | Median time-to-first-token (TTFT) in seconds |
| Context Window | Artificial Analysis | Maximum supported input length |
| Image ELO | AA Image Arena | Human-preference ranking across 15 style categories |
| Open-weight models | Manufacturer specs | GPU VRAM, parameter counts, quantization levels |

The LLM dataset currently covers **255+ models** across **29+ providers**. A timestamped snapshot is saved on each run and powers the Trends tab.

---

## Running Locally

Requires **Python 3.11+**.

```bash
# Clone the repo
git clone https://github.com/Avo-Sarkissian/AI-Frontier-Database.git
cd AI-Frontier-Database

# Create and activate a virtual environment
python -m venv ai-frontier
source ai-frontier/bin/activate   # Windows: ai-frontier\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
python app.py
```

Open **http://localhost:8050**. The scraper fetches fresh data on startup and runs in the background every hour — no manual refresh needed.

---

## Static site (GitHub Pages)

The public site at **https://avo-sarkissian.github.io/AI-Frontier-Database/** is a static, server-free build of the same dashboard, served from the `docs/` folder on `main`.

```bash
# Regenerate the static site into docs/ (pre-rendered figures + Pyodide bundle)
python build_static.py
git add docs && git commit -m "rebuild static site" && git push   # Pages auto-deploys (~1 min)
```

How it works: `build_static.py` pre-renders every chart's default state to JSON and zips the project's Python (chart builders, data loaders, `static_api.py`) plus `plotly` into `docs/pybundle.zip`. The page loads the pre-rendered figures instantly, then boots Pyodide in the background and calls the **same** Python chart code in the browser for live filtering — so the static site is faithful to the Dash app with zero hosting cost. The hourly scraper and `python app.py` remain the local source of truth; refreshing the data and re-running `build_static.py` is all it takes to update the live site.

---

## Architecture

```
app.py                   # Dash app entry point, tab routing, background scraper
components/
  charts/                # One module per visualization (pareto, radar, treemap, …)
  stack_recommender.py   # Agent stack recommendation logic
data/
  scraper.py             # Pulls live data from Artificial Analysis API
  ingest.py             # Parses API response, manages CSV cache and history
  raw/                   # Live cache + daily timestamped snapshots
utils/
  model_lookup.py        # Shared model metadata helpers
assets/
  style.css              # Global dark theme, typography
static_helpers.py        # Pure dash-free helpers shared by app.py + the static build
static_api.py            # Browser bridge — ports the Dash callbacks to JSON-returning functions (runs in Pyodide)
build_static.py          # Pre-renders figures + bundles Python into docs/ for GitHub Pages
docs/                    # The static GitHub Pages site (index.html, app.js, figures/, pybundle.zip)
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
| [GitHub Pages](https://pages.github.com/) | Free static hosting for the live dashboard — auto-deploys on every push to `main` |

---

*Data Visualization — EECE 5642, Northeastern University, Spring 2026*
