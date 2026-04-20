"""
Build the AI Frontier final report as a .docx file.
Run:  python3 build_report.py
Out:  FinalReport_Sarkissian.docx
"""
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ── Page margins (1 inch all around) ────────────────────────────────────────
section = doc.sections[0]
section.page_width  = Inches(8.5)
section.page_height = Inches(11)
section.left_margin   = Inches(1.25)
section.right_margin  = Inches(1.25)
section.top_margin    = Inches(1.0)
section.bottom_margin = Inches(1.0)


# ── Style helpers ────────────────────────────────────────────────────────────
def set_run_font(run, name="Times New Roman", size=12, bold=False,
                 italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)
    # Force font in complex script as well (needed for Times New Roman)
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:ascii"),    name)
    rFonts.set(qn("w:hAnsi"),    name)
    rFonts.set(qn("w:eastAsia"), name)
    rFonts.set(qn("w:cs"),       name)


def para_spacing(para, before=0, after=6, line=None):
    pPr = para._p.get_or_add_pPr()
    pSpacing = OxmlElement("w:spacing")
    pSpacing.set(qn("w:before"), str(int(before * 20)))
    pSpacing.set(qn("w:after"),  str(int(after * 20)))
    if line is not None:
        pSpacing.set(qn("w:line"),     str(int(line * 240)))
        pSpacing.set(qn("w:lineRule"), "auto")
    pPr.append(pSpacing)


def add_heading(doc, text, level=1):
    """Section heading — bold, slightly larger, left-aligned."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if level == 1:
        para_spacing(p, before=14, after=4)
        run = p.add_run(text)
        set_run_font(run, size=13, bold=True)
    else:
        para_spacing(p, before=8, after=2)
        run = p.add_run(text)
        set_run_font(run, size=12, bold=True, italic=True)
    return p


def add_body(doc, text, indent=False):
    """Body paragraph — 12pt Times New Roman, justified, first-line indent."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para_spacing(p, before=0, after=6, line=1.15)
    if indent:
        pPr = p._p.get_or_add_pPr()
        ind = OxmlElement("w:ind")
        ind.set(qn("w:firstLine"), str(int(0.3 * 1440)))
        pPr.append(ind)
    run = p.add_run(text)
    set_run_font(run, size=12)
    return p


def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para_spacing(p, before=0, after=4, line=1.15)
    pPr = p._p.get_or_add_pPr()
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"),      str(int(0.4 * 1440)))
    ind.set(qn("w:hanging"),   str(int(0.2 * 1440)))
    pPr.append(ind)
    if bold_prefix:
        r1 = p.add_run(bold_prefix + " ")
        set_run_font(r1, size=12, bold=True)
        r2 = p.add_run(text)
        set_run_font(r2, size=12)
    else:
        r = p.add_run(text)
        set_run_font(r, size=12)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para_spacing(p, before=2, after=10)
    run = p.add_run(text)
    set_run_font(run, size=10, italic=True)
    return p


# ── Title block ──────────────────────────────────────────────────────────────
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
para_spacing(title_p, before=0, after=4)
title_run = title_p.add_run("AI Frontier: An Interactive Dashboard for Mapping the LLM Landscape")
set_run_font(title_run, size=16, bold=True)

subtitle_p = doc.add_paragraph()
subtitle_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
para_spacing(subtitle_p, before=0, after=2)
sub_run = subtitle_p.add_run("EECE 5642 — Data Visualization  |  Final Project Report")
set_run_font(sub_run, size=11, italic=True, color=(80, 80, 80))

author_p = doc.add_paragraph()
author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
para_spacing(author_p, before=0, after=2)
auth_run = author_p.add_run("Avo Sarkissian")
set_run_font(auth_run, size=12, bold=True)

affil_p = doc.add_paragraph()
affil_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
para_spacing(affil_p, before=0, after=2)
affil_run = affil_p.add_run(
    "Department of Electrical and Computer Engineering  |  Northeastern University  |  Spring 2026")
set_run_font(affil_run, size=11, color=(80, 80, 80))

email_p = doc.add_paragraph()
email_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
para_spacing(email_p, before=0, after=12)
email_run = email_p.add_run("avohsarkissian@gmail.com")
set_run_font(email_run, size=11, color=(80, 80, 80))

# Divider line
div = doc.add_paragraph()
para_spacing(div, before=0, after=12)
pPr = div._p.get_or_add_pPr()
pBdr = OxmlElement("w:pBdr")
bottom = OxmlElement("w:bottom")
bottom.set(qn("w:val"),   "single")
bottom.set(qn("w:sz"),    "6")
bottom.set(qn("w:space"), "1")
bottom.set(qn("w:color"), "AAAAAA")
pBdr.append(bottom)
pPr.append(pBdr)


# ── Abstract ─────────────────────────────────────────────────────────────────
add_heading(doc, "Abstract")
abstract_text = (
    "The number of commercially available large language models has grown to the point where no "
    "single leaderboard covers all the dimensions that matter for practical use: price, throughput, "
    "latency, context window size, and benchmark performance. AI Frontier is an interactive "
    "dashboard that aggregates live data for 255+ models across 29 providers, refreshed every hour "
    "from the Artificial Analysis API, and makes the full competitive landscape navigable through "
    "eleven specialized views. The system runs entirely in Python using Plotly Dash and has been "
    "deployed at a public URL since March 2026. This report covers the data pipeline, composite "
    "intelligence scoring methodology, visualization design decisions, and the tradeoffs made when "
    "building a real-time data visualization tool without a JavaScript build step."
)
p = add_body(doc, abstract_text)


# ── 1. Introduction ───────────────────────────────────────────────────────────
add_heading(doc, "1.  Introduction")

add_body(doc,
    "Selecting a large language model for a real application used to be straightforward — a handful "
    "of options existed and the best one was obvious for most tasks. That changed fast. As of spring "
    "2026, over 29 companies offer models through hosted APIs, each making different tradeoffs in "
    "price, capability, and latency. A model that scores 70 on reasoning benchmarks might cost twenty "
    "times more per token than one that scores 65. A local model that fits on a consumer GPU might "
    "outperform a paid API for a narrow task. Without a way to see all of this together, model "
    "selection is largely guesswork.",
    indent=True)

add_body(doc,
    "Existing leaderboards do one thing well. The LMSYS Chatbot Arena ranks models on human "
    "preference. The Hugging Face Open LLM Leaderboard tracks open-weight models on academic "
    "benchmarks. But neither shows live pricing, and most don't update on a daily schedule. Provider "
    "dashboards show pricing but not quality scores. Nobody shows both, live, with a way to filter "
    "and compare across text, image, and video generation alongside locally runnable models.",
    indent=True)

add_body(doc,
    "AI Frontier fills that gap. The core idea is simple: pull current price and performance data "
    "for every major model on a regular schedule and surface it through enough visualization types "
    "that different kinds of users can answer different questions. A developer wants to know the "
    "cheapest model above a quality threshold. A researcher wants to see which provider is "
    "consistently leading. A student wants to know what will fit in their GPU's memory. Each of "
    "those questions needs a different view of the same data.",
    indent=True)

add_body(doc,
    "The project was built for EECE 5642 (Data Visualization) at Northeastern University. "
    "The dashboard is publicly accessible and all code is open-source at "
    "https://github.com/Avo-Sarkissian/AI-Frontier-Database.",
    indent=True)


# ── 2. Data Sources and Pipeline ─────────────────────────────────────────────
add_heading(doc, "2.  Data Sources and Pipeline")
add_heading(doc, "2.1  Cloud Model Data", level=2)

add_body(doc,
    "The primary data source is the Artificial Analysis API, which tracks pricing and benchmark "
    "data for commercially hosted LLMs. For each model it provides: price per million tokens "
    "(input and output, at the cheapest available provider), median tokens per second (throughput), "
    "median time-to-first-token (latency), context window size, and a composite intelligence score. "
    "The API is queried once per hour by a background thread that starts when the application "
    "launches.",
    indent=True)

add_body(doc,
    "Each successful API response is written to a CSV cache and also saved as a timestamped "
    "snapshot in a history directory. Over 30 daily snapshots have accumulated since the project "
    "launched; these power the Trends tab, which shows how model pricing has shifted over time. A "
    "blended price per million tokens is computed from raw input/output prices using a 3:1 "
    "output-to-input ratio, approximating real-world usage patterns and collapsing two numbers "
    "into one for display.",
    indent=True)

add_heading(doc, "2.2  Local Model and Hardware Data", level=2)

add_body(doc,
    "The second data source is a manually curated catalog of open-weight models and GPU hardware "
    "specifications. GPU memory bandwidth and VRAM capacity were pulled from official product pages "
    "(NVIDIA, AMD, Apple, Intel). Model parameter counts and quantization support were taken from "
    "Hugging Face model cards. This catalog is static rather than live — it is updated when "
    "significant new hardware generations or model families are released. It backs the Run Local "
    "tab, which estimates whether a given model fits in a selected GPU's memory and how fast it "
    "would run.",
    indent=True)


# ── 3. Intelligence Score and Benchmark Methodology ─────────────────────────
add_heading(doc, "3.  Intelligence Score and Benchmark Methodology")

add_body(doc,
    "The intelligence score displayed throughout the dashboard is inherited from Artificial "
    "Analysis, which aggregates results from multiple public evaluations into a single 0–100 "
    "composite. The score covers four categories:",
    indent=True)

add_bullet(doc,
    "MMLU-Pro and GPQA, testing factual knowledge and multi-step deductive reasoning.",
    bold_prefix="Reasoning:")
add_bullet(doc,
    "LiveCodeBench and SciCode, evaluating writing and debugging real programs.",
    bold_prefix="Coding:")
add_bullet(doc,
    "AIME and MATH-500, covering competition-style quantitative problems.",
    bold_prefix="Math:")
add_bullet(doc,
    "Humanity's Last Exam, a recently released benchmark of expert-level questions across "
    "scientific domains.",
    bold_prefix="Knowledge:")

add_body(doc,
    "Using multiple benchmarks matters here. A model fine-tuned heavily on MMLU training data "
    "might rank first on that test alone but fall to the middle on LiveCodeBench. Averaging across "
    "categories rewards models that are broadly capable rather than narrowly optimized, and it "
    "allows API-hosted and local open-weight models to be compared on the same scale.",
    indent=True)

add_body(doc,
    "The main limitation of composite scores is that they smooth over real differences. Two models "
    "with the same composite score can have completely different strength profiles — one excellent "
    "at coding and weak at math, the other the reverse. The Compare tab (radar chart) exists "
    "specifically to surface these within-score differences.",
    indent=True)


# ── 4. Dashboard Design ───────────────────────────────────────────────────────
add_heading(doc, "4.  Dashboard Design")
add_heading(doc, "4.1  Tab Structure", level=2)

add_body(doc,
    "The dashboard is organized into eleven tabs, each targeting a distinct question type:",
    indent=True)

tabs = [
    ("Overview",     "Bubble scatter of all 255+ models with a Pareto frontier overlay. "
                     "X-axis is log-scaled price, y-axis is intelligence score, bubble size encodes throughput."),
    ("Agent Stack",  "Recommends a three-tier model configuration (reasoning, balanced, fast) "
                     "for agentic workflows, filtered by available hardware."),
    ("Landscape",    "Treemap of the AI industry: tile area encodes model count per provider, "
                     "color intensity encodes average intelligence score."),
    ("Rankings",     "Top 25 models sorted by intelligence, value (score per dollar), or speed, "
                     "with tier-separator lines at meaningful performance gaps."),
    ("Compare",      "Radar chart for up to five models across five dimensions: intelligence, "
                     "speed, affordability, context window, and latency."),
    ("Budget",       "Estimates projected monthly API cost given a user-supplied token volume, "
                     "sorted cheapest-first."),
    ("Table",        "Full sortable table of all 255+ models with every available metric."),
    ("Run Local",    "VRAM compatibility calculator and inference speed estimator for "
                     "open-weight models, filtered by GPU and quantization level."),
    ("Image Gen",    "ELO-ranked image generation model leaderboard from the AA Image Arena."),
    ("Video Gen",    "Ranked comparison of video generation models with pricing and quality data."),
    ("Trends",       "Price-over-time line chart built from daily historical snapshots, "
                     "showing how API pricing has shifted since February 2026."),
]
for tab, desc in tabs:
    add_bullet(doc, desc, bold_prefix=tab + ":")

add_heading(doc, "4.2  Visualization Choices", level=2)

add_body(doc,
    "Overview — bubble scatter with Pareto frontier. The challenge with comparing 255 models "
    "simultaneously is that a ranked list hides tradeoffs. The bubble scatter encodes three "
    "variables at once: price on the x-axis, quality score on the y-axis, and throughput as bubble "
    "size. Log scale on price is necessary because model prices span roughly three orders of "
    "magnitude — from under $0.01 to over $15 per million tokens — and a linear axis would "
    "compress most models into one corner of the chart.",
    indent=True)

add_body(doc,
    "A dotted Pareto frontier connects the models that are undominated in the price-quality "
    "tradeoff: points where any improvement in quality requires paying more, and any cost reduction "
    "means accepting lower quality. The frontier is computed in Pandas on each data load using a "
    "standard non-dominated sorting pass. Models on or near this curve represent the best practical "
    "options for most use cases. Each provider is assigned both a color and a distinct marker shape, "
    "making the chart readable for colorblind viewers without a separate mode or toggle.",
    indent=True)

add_body(doc,
    "Rankings — horizontal bars with tier separators. For one-dimensional ranked comparisons, "
    "horizontal bar charts are clearer than scatter plots. Tier separator lines are drawn wherever "
    "the score gap between adjacent models exceeds a fixed threshold, marking meaningful performance "
    "breaks rather than implying uniform spacing across the range.",
    indent=True)

add_body(doc,
    "Compare — radar chart. Head-to-head comparison across five dimensions is a natural use case "
    "for a radar chart. Each axis is normalized so all five dimensions are comparable on the same "
    "scale. The limitation is that interpretation depends on axis ordering, which is fixed — "
    "reordering axes would change the visible shape without changing any underlying values.",
    indent=True)

add_body(doc,
    "Landscape — treemap. Tile area encodes model count per provider; color intensity encodes "
    "average intelligence. This communicates structural patterns that scatter plots miss — which "
    "providers dominate by volume, which are new entrants, which specialize in high-capability "
    "models. A ranked bar chart below it allows precise comparison of the same providers.",
    indent=True)

add_body(doc,
    "Trends — line chart over time. The historical snapshot system makes this tab possible. "
    "Plotting API prices over time shows a market that moves fast: several major providers dropped "
    "prices significantly between February and April 2026. A static dataset would miss this "
    "entirely.",
    indent=True)

add_body(doc,
    "Run Local — VRAM compatibility calculator. The user selects a GPU and quantization level; "
    "the dashboard filters the open-weight model catalog to models whose estimated VRAM requirement "
    "fits the selected GPU's memory and displays estimated inference speed in tokens per second. "
    "VRAM requirements scale with parameter count and bit-width; the formula accounts for model "
    "weights, KV-cache overhead, and activation memory. Quantization levels range from full "
    "precision (FP16/BF16) through 8-bit and 4-bit approximations.",
    indent=True)

add_heading(doc, "4.3  Design Principles", level=2)

add_body(doc,
    "Three decisions carried through all eleven tabs. First, data-ink ratio: gridlines, outer "
    "frames, and chart borders were stripped back wherever they add visual weight without encoding "
    "information. The dark background (#0a0a0a) with high-contrast data elements reduces perceived "
    "clutter compared to a white background at equivalent data density.",
    indent=True)

add_body(doc,
    "Second, colorblind accessibility: each provider gets both a color and a distinct marker shape, "
    "kept consistent across all tabs. A user who learns either the colors or the shapes can navigate "
    "between tabs without re-reading a legend.",
    indent=True)

add_body(doc,
    "Third, appropriate scales: price axes use log scale throughout. Context window sizes, which "
    "range from 8K to over one million tokens, also use log scale. Linear scales for heavily "
    "right-skewed distributions hide the variation at the low end, where most models sit.",
    indent=True)

add_heading(doc, "4.4  Technical Stack", level=2)

add_body(doc,
    "The dashboard runs on Plotly Dash, which compiles Python component trees into a live web "
    "interface without a JavaScript build step. All visualization logic is in Python using Plotly's "
    "graph objects API; Dash handles HTTP routing, callback wiring, and component state. Using a "
    "single language for the full stack — data pipeline, computation, and UI — simplified "
    "deployment significantly. There is no Node.js dependency, no webpack config, and no separate "
    "API server.",
    indent=True)

add_body(doc,
    "The application is hosted on Render, which starts it with python app.py and redeploys "
    "automatically on every push to the GitHub repository. The tradeoff is that some UI interactions "
    "that would be instant in a React application require a server round-trip in Dash. In practice "
    "this was acceptable because the dataset is under one megabyte and most callbacks return in "
    "under 200 milliseconds.",
    indent=True)


# ── 5. Contributions ──────────────────────────────────────────────────────────
add_heading(doc, "5.  Contributions")

add_body(doc,
    "What AI Frontier adds relative to existing tools is the combination of live data with "
    "multi-modal coverage. Static leaderboards go stale within days of a new model release. Most "
    "pricing tools do not include benchmark scores. AI Frontier scrapes both automatically, covers "
    "text generation, image generation, and video generation in one place, and includes local "
    "open-weight models alongside API services — a side-by-side comparison that does not exist "
    "elsewhere in a unified interface.",
    indent=True)

add_body(doc,
    "The Agent Stack tab translates the raw leaderboard data into a concrete configuration "
    "decision. Given a workflow type, it recommends a three-tier model setup — reasoning, balanced, "
    "and fast — and surfaces specific model options at each tier. For users building agentic "
    "pipelines, this answers the practical question of which model to use for which subtask.",
    indent=True)

add_body(doc,
    "The full data pipeline and all eleven visualization modules are open-source. The dashboard "
    "has served live data continuously since March 2026, with over 30 hourly snapshots preserved "
    "for trend analysis.",
    indent=True)


# ── 6. Future Work ────────────────────────────────────────────────────────────
add_heading(doc, "6.  Future Work")

add_body(doc,
    "Several extensions are worth building. Per-category breakdown scores — reasoning, coding, "
    "math, and knowledge separately — are collected by Artificial Analysis but not yet surfaced in "
    "the Compare tab. Adding them would allow more targeted model selection than a single composite "
    "number permits.",
    indent=True)

add_body(doc,
    "The Budget tab currently estimates cost from a token count alone. Prompt caching and multi-turn "
    "context accumulation both affect real production API costs and are straightforward to model "
    "given the existing pricing data. Fine-tuning cost estimation — factoring in training compute "
    "alongside inference pricing — is a related addition that would be useful for teams evaluating "
    "whether to fine-tune a cheaper model versus using a better one off-the-shelf.",
    indent=True)

add_body(doc,
    "Measured API latency by region is the other obvious addition. Published time-to-first-token "
    "medians from provider spec sheets are not always consistent with real-world performance under "
    "load. The hourly polling infrastructure is already in place; extending it to include synthetic "
    "request timing would give a more complete operational picture. A mobile-optimized layout is "
    "also on the list — several tabs assume a wide desktop viewport and would need reworking for "
    "smaller screens.",
    indent=True)


# ── 7. Conclusion ─────────────────────────────────────────────────────────────
add_heading(doc, "7.  Conclusion")

add_body(doc,
    "AI Frontier demonstrates that a practical, continuously updated data visualization tool can be "
    "built and deployed entirely in Python without sacrificing interactivity or visual quality. The "
    "dashboard aggregates live data from multiple sources, applies a composite benchmark scoring "
    "methodology, and presents the results through eleven distinct visualization types — each "
    "chosen to answer a specific class of user question. The project is publicly accessible, "
    "open-source, and has remained operational since March 2026.",
    indent=True)


# ── 8. References ─────────────────────────────────────────────────────────────
add_heading(doc, "References")

refs = [
    "[1]  Artificial Analysis. Artificial Analysis API: LLM performance and pricing benchmarks. "
    "https://artificialanalysis.ai, 2024.",

    "[2]  W.-L. Chiang et al. Chatbot Arena: An open platform for evaluating LLMs by human "
    "preference. arXiv preprint arXiv:2403.04132, 2024.",

    "[3]  Hugging Face. Open LLM Leaderboard. "
    "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard, 2023.",

    "[4]  Y. Wang et al. MMLU-Pro: A more robust and challenging multi-task language understanding "
    "benchmark. arXiv preprint arXiv:2406.01574, 2024.",

    "[5]  D. Rein et al. GPQA: A graduate-level google-proof Q&A benchmark. "
    "arXiv preprint arXiv:2311.12022, 2023.",

    "[6]  N. Jain et al. LiveCodeBench: Holistic and contamination free evaluation of large "
    "language models for code. arXiv preprint arXiv:2403.07974, 2024.",

    "[7]  M. Tian et al. SciCode: A research coding benchmark curated by scientists. "
    "arXiv preprint arXiv:2407.13168, 2024.",

    "[8]  D. Hendrycks et al. Measuring mathematical problem solving with the MATH dataset. "
    "In Proceedings of NeurIPS Track on Datasets and Benchmarks, 2021.",

    "[9]  D. Phan et al. Humanity's Last Exam. Scale AI, 2025. https://scale.com/hle.",

    "[10] T. Wolf et al. Transformers: State-of-the-art natural language processing. "
    "In Proceedings of EMNLP: System Demonstrations, pages 38–45, 2020.",

    "[11] E. R. Tufte. The Visual Display of Quantitative Information. "
    "Graphics Press, 2nd edition, 2001.",

    "[12] Render. Render cloud application platform. https://render.com, 2024.",
]

for ref in refs:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para_spacing(p, before=0, after=4, line=1.1)
    pPr = p._p.get_or_add_pPr()
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"),    str(int(0.35 * 1440)))
    ind.set(qn("w:hanging"), str(int(0.35 * 1440)))
    pPr.append(ind)
    run = p.add_run(ref)
    set_run_font(run, size=11)


# ── Save ─────────────────────────────────────────────────────────────────────
out = "FinalReport_Sarkissian.docx"
doc.save(out)
print(f"Wrote {out}")
