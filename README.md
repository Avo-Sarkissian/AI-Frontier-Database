# AI Frontier

> An interactive dashboard that maps the entire AI model landscape — comparing 255+ large language models on cost, speed, and intelligence in one place.

---

## What is this?

Every week, dozens of new AI models are released by companies like OpenAI, Google, Anthropic, Meta, and many others. Each model makes different tradeoffs: some are incredibly smart but expensive, some are blazing fast but less capable, and some are surprisingly powerful for almost no cost.

**AI Frontier** makes those tradeoffs visible. It pulls live data from [Artificial Analysis](https://artificialanalysis.ai/) and lets you explore the full competitive landscape through 11 interactive tabs.

---

## The Dashboard

### Overview — Cost vs. Intelligence
The landing page. Every bubble is one AI model — move left for cheaper models, move up for smarter ones. The dotted line traces the **Pareto frontier**: models that give you the best intelligence for their price. Bubble size encodes throughput (tok/s). Both color and marker shape encode provider, so the chart is readable without relying on color alone.

### Agent Stack — Model Recommendations
Pick a workflow (API-only, Hybrid, or Local-only) and get a three-tier stack: **Fast** (haiku-class sub-tasks), **Balanced** (coding and writing), and **Reasoning** (planning and delegation). In Hybrid or Local mode, filter by your GPU and quantization level.

### Landscape — Provider Ecosystem
A treemap of the AI industry: tile area = number of models released, color intensity = average intelligence. Followed by a provider leaderboard showing best model, provider average, and model count.

### Rankings — Top 25 Models
A leaderboard of the highest-scoring models, color-coded by company. Tier-separator lines mark meaningful performance gaps. Sort by **Intelligence**, **Value (score/$)**, or **Speed**. Below the leaderboard: a **Value Leaders** chart ranking the top 15 models by intelligence per dollar.

### Compare — Head-to-Head Radar
Pick up to 5 models and compare them across five dimensions simultaneously: Intelligence, Speed, Affordability, Context Window, and Latency. Raw values table below the chart.

### Budget — Monthly Cost Calculator
Enter a monthly token volume and see each model's projected API cost, sorted cheapest-first.

### Table — Full Data
Sortable table of all 255+ models. Columns: Score, Value (score/$), Price, Speed, Latency, and Context Window (in k tokens). Click any column header to sort in either direction.

### Run Local — Open-Weight Models
Select your GPU and quantization level to see which open-weight models fit in your VRAM, with estimated inference speed.

### Image Gen — Image Model Rankings
ELO-based leaderboard of image generation models across 15 style categories, sourced from the AA Image Arena. Includes a scatter view comparing ELO vs. price.

### Video Gen — Video Model Rankings
Ranked comparison of video generation models with pricing and quality data.

---

## Data

| Metric | Source | Description |
|---|---|---|
| **Intelligence** | [Artificial Analysis](https://artificialanalysis.ai/) | Composite benchmark score across reasoning, coding, and knowledge tasks |
| **Price** | Artificial Analysis | USD per 1M tokens — blended 3:1 output/input ratio, from the single cheapest API host |
| **Speed** | Artificial Analysis | Tokens/second from the same host as the price |
| **Latency** | Artificial Analysis | Median time-to-first-token (TTFT) in seconds |
| **Context Window** | Artificial Analysis | Maximum input length the model supports |
| **Image ELO** | AA Image Arena | Human-preference ranking across 15 style categories |

Data is scraped live from Artificial Analysis. The LLM dataset currently covers **255+ models** across **29+ providers**. A daily snapshot is saved each run and used to power the Rankings history.

---

## Running Locally

You'll need Python 3.11+.

```bash
# 1. Clone the repo
git clone https://github.com/Avo-Sarkissian/AI-Frontier-Database.git
cd AI-Frontier-Database

# 2. Create and activate a virtual environment
python -m venv ai-frontier
source ai-frontier/bin/activate   # Windows: ai-frontier\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
python app.py
```

Open **http://localhost:8050** in your browser. The scraper fetches fresh data on startup and every hour in the background.

---

## Built With

| Tool | Role |
|---|---|
| [Plotly Dash](https://dash.plotly.com/) | Interactive web framework |
| [Plotly](https://plotly.com/python/) | Charts and visualizations |
| [Pandas](https://pandas.pydata.org/) | Data processing |
| [requests](https://docs.python-requests.org/) | Live data scraping from AA API |
| [NumPy](https://numpy.org/) | Numerical operations (Pareto, correlation) |

---

*Data Visualization project — Northeastern University, Spring 2026*
