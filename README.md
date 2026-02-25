# AI Frontier

> An interactive dashboard that maps the entire AI model landscape — comparing 100+ large language models on cost, speed, and intelligence in one place.

---

## What is this?

Every week, dozens of new AI models are released by companies like OpenAI, Google, Anthropic, Meta, and many others. Each model makes different tradeoffs: some are incredibly smart but expensive, some are blazing fast but less capable, and some are surprisingly powerful for almost no cost.

**AI Frontier** makes those tradeoffs visible. It pulls live data from [Artificial Analysis](https://artificialanalysis.ai/) and lets you explore the full competitive landscape through six interactive visualizations.

---

## The Dashboard

### Overview — Cost vs. Intelligence
Every bubble is one AI model. Move left for cheaper models, move up for smarter ones. The dotted line traces the **Pareto frontier** — the models that give you the best intelligence for their price. Bubble size shows how fast the model generates text.

### Performance — Speed vs. Intelligence
The four quadrants tell the story at a glance: top-right is the sweet spot (fast *and* smart), top-left is smart-but-slow, bottom-right is fast-but-weak. Useful for understanding which models are actually production-ready.

### Value — Intelligence per Dollar
A simple question: which models give you the most *intelligence for your money*? This chart ranks every model by quality divided by price, surfacing hidden gems that outperform their cost tier.

### Landscape — Provider Ecosystem
A map of the AI industry. Each tile is a company — its size reflects how many models they've released, and the color shows how intelligent those models are on average. At a glance you can see who's leading, who's flooding the market, and who's punching above their weight.

### Rankings — Top 25 Models
A clean leaderboard of the highest-scoring models, color-coded by company. Hover any bar to see price and speed alongside the intelligence score.

### Compare — Head-to-Head Radar
Pick up to 5 models and compare them across four dimensions simultaneously: Intelligence, Speed, Affordability, and Context Window. The radar chart reveals at a glance why no single model wins on everything.

---

## Data

| Metric | Source | Description |
|---|---|---|
| **Intelligence** | [Artificial Analysis](https://artificialanalysis.ai/) | Composite benchmark score across reasoning, coding, and knowledge tasks |
| **Price** | Artificial Analysis | USD per 1 million tokens (blended input/output) |
| **Speed** | Artificial Analysis | Tokens generated per second |
| **Context Window** | Artificial Analysis | Maximum input length the model can process |

Data is scraped live from Artificial Analysis and covers **102 models** across **27 providers** as of the latest update.

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

Open **http://localhost:8050** in your browser.

---

## Built With

| Tool | Role |
|---|---|
| [Plotly Dash](https://dash.plotly.com/) | Interactive web framework |
| [Plotly](https://plotly.com/python/) | Charts and visualizations |
| [Pandas](https://pandas.pydata.org/) | Data processing |
| [Playwright](https://playwright.dev/) | Live data scraping |

---

*Data Visualization project — Northeastern University, Spring 2026*
