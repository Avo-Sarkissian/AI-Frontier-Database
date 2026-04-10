"""
Build the AI Frontier presentation deck.

Theme matches the live dashboard: near-black canvas (#0a0a0a), elevated cards
(#111111), cyan accent (#00d4ff), Inter typography, generous whitespace.
16:9, 8 slides, paced for ~3-4 min of talking + a long live demo within a
10-minute slot.

Run:  python3 build_presentation.py
Out:  AI_Frontier_Presentation.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ── Theme tokens (mirrors assets/style.css) ─────────────────────────────────
BG          = RGBColor(0x0A, 0x0A, 0x0A)
BG_CARD     = RGBColor(0x11, 0x11, 0x11)
BG_ELEVATED = RGBColor(0x16, 0x16, 0x16)
BORDER      = RGBColor(0x26, 0x26, 0x26)
TEXT_1      = RGBColor(0xED, 0xED, 0xED)
TEXT_2      = RGBColor(0x88, 0x88, 0x88)
TEXT_3      = RGBColor(0x66, 0x66, 0x66)
TEXT_4      = RGBColor(0x44, 0x44, 0x44)
ACCENT      = RGBColor(0x00, 0xD4, 0xFF)
ACCENT_DIM  = RGBColor(0x0E, 0x35, 0x44)
GREEN       = RGBColor(0x22, 0xC5, 0x5E)
PURPLE      = RGBColor(0xC0, 0x84, 0xFC)
ORANGE      = RGBColor(0xFB, 0x92, 0x3C)

FONT = "Inter"
FONT_FALLBACK = "Helvetica Neue"

# Slide geometry: 16:9 widescreen, 13.333" × 7.5"
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ── Helpers ─────────────────────────────────────────────────────────────────
def set_bg(slide, color=BG):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    bg.shadow.inherit = False
    return bg


def add_rect(slide, x, y, w, h, fill=BG_CARD, line=None, line_w=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = line_w or Pt(0.75)
    shp.shadow.inherit = False
    return shp


def add_line(slide, x1, y1, x2, y2, color=BORDER, width=Pt(0.75)):
    ln = slide.shapes.add_connector(1, x1, y1, x2, y2)
    ln.line.color.rgb = color
    ln.line.width = width
    return ln


def add_text(slide, x, y, w, h, text, *,
             size=12, color=TEXT_1, bold=False, font=FONT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             tracking=None):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_before = Pt(0)
    p.space_after = Pt(0)
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    if tracking is not None:
        # tracking in 1/100 of a point
        rPr = run._r.get_or_add_rPr()
        rPr.set("spc", str(tracking))
    return tb


def add_paragraphs(slide, x, y, w, h, lines, *,
                   size=12, color=TEXT_2, bold=False,
                   font=FONT, leading=Pt(6), align=PP_ALIGN.LEFT):
    """Add a multi-paragraph block. Each `line` is a string OR a dict
    {text, size, color, bold}."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = leading
        if isinstance(ln, dict):
            run = p.add_run()
            run.text = ln.get("text", "")
            run.font.name = ln.get("font", font)
            run.font.size = Pt(ln.get("size", size))
            run.font.bold = ln.get("bold", bold)
            run.font.color.rgb = ln.get("color", color)
        else:
            run = p.add_run()
            run.text = ln
            run.font.name = font
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
    return tb


def add_chip(slide, x, y, label, *, color=ACCENT, fill=ACCENT_DIM, size=9):
    """Pill-shaped tag matching the dashboard's header-badge style."""
    h = Inches(0.28)
    # Approximate width from label length
    w = Inches(0.18 + 0.085 * len(label))
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.adjustments[0] = 0.5
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = color
    shp.line.width = Pt(0.75)
    tf = shp.text_frame
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = label.upper()
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    rPr.set("spc", "120")
    return shp, w


TOTAL_SLIDES = 8


def add_header(slide, eyebrow, title, page_num):
    """Sticky-style header — small uppercase eyebrow over a tight title."""
    # Eyebrow chip
    add_chip(slide, Inches(0.6), Inches(0.45), eyebrow)

    # Page number, top right
    add_text(slide, SLIDE_W - Inches(1.2), Inches(0.5), Inches(0.6), Inches(0.3),
             f"{page_num:02d} / {TOTAL_SLIDES:02d}", size=9, color=TEXT_3,
             align=PP_ALIGN.RIGHT, tracking=120, bold=True)

    # Title row
    add_text(slide, Inches(0.6), Inches(0.85), Inches(12), Inches(0.7),
             title, size=30, color=TEXT_1, bold=True)

    # Divider line under header
    y = Inches(1.6)
    add_line(slide, Inches(0.6), y, SLIDE_W - Inches(0.6), y, color=BORDER, width=Pt(0.75))


def add_footer(slide):
    add_text(slide, Inches(0.6), SLIDE_H - Inches(0.4), Inches(6),
             Inches(0.3), "AI FRONTIER  ·  EECE 5642  ·  SPRING 2026",
             size=8, color=TEXT_3, tracking=160, bold=True)
    add_text(slide, SLIDE_W - Inches(4), SLIDE_H - Inches(0.4),
             Inches(3.4), Inches(0.3),
             "github.com/Avo-Sarkissian/AI-Frontier-Database",
             size=8, color=TEXT_3, tracking=80, align=PP_ALIGN.RIGHT)


def stat_card(slide, x, y, w, h, value, label, *, accent=False):
    add_rect(slide, x, y, w, h, fill=BG_CARD)
    add_text(slide, x + Inches(0.25), y + Inches(0.30), w - Inches(0.5),
             Inches(0.7), value, size=28, bold=True,
             color=ACCENT if accent else TEXT_1)
    add_text(slide, x + Inches(0.25), y + Inches(0.95), w - Inches(0.5),
             Inches(0.3), label, size=9, color=TEXT_3,
             tracking=120, bold=True)


# ════════════════════════════════════════════════════════════════════════════
#  BUILD
# ════════════════════════════════════════════════════════════════════════════
prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


# ── Slide 1 — TITLE ────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
set_bg(s)

# Subtle accent bar across the top
add_rect(s, 0, 0, SLIDE_W, Inches(0.06), fill=ACCENT)

# Cyan dot + brand mark
add_rect(s, Inches(0.6), Inches(0.55), Inches(0.12), Inches(0.12), fill=ACCENT)
add_text(s, Inches(0.85), Inches(0.5), Inches(6), Inches(0.3),
         "AI FRONTIER", size=11, bold=True, color=TEXT_1, tracking=200)

add_chip(s, SLIDE_W - Inches(2.0), Inches(0.5), "LIVE DEMO  ●", color=ACCENT, fill=ACCENT_DIM)

# Title block, vertically centered
add_text(s, Inches(0.6), Inches(2.4), Inches(12), Inches(0.4),
         "DATA VISUALIZATION  ·  EECE 5642", size=10, color=TEXT_3,
         bold=True, tracking=200)

add_text(s, Inches(0.6), Inches(2.9), Inches(12.2), Inches(1.4),
         "Mapping the AI Model Landscape", size=52, bold=True,
         color=TEXT_1)

add_text(s, Inches(0.6), Inches(4.25), Inches(12.2), Inches(0.7),
         "Comparing 255+ AI models on cost, speed, and intelligence,",
         size=20, color=TEXT_2)
add_text(s, Inches(0.6), Inches(4.65), Inches(12.2), Inches(0.7),
         "in one place, updated every hour.",
         size=20, color=TEXT_2)

# Bottom stat bar — mirrors the dashboard's stat strip
y0 = Inches(5.85)
add_line(s, Inches(0.6), y0, SLIDE_W - Inches(0.6), y0, color=BORDER)
stats = [
    ("255+", "MODELS TRACKED", False),
    ("29",   "PROVIDERS",      False),
    ("11",   "INTERACTIVE TABS", False),
    ("HOURLY", "DATA REFRESH",  True),
]
sw = (SLIDE_W - Inches(1.2)) / len(stats)
for i, (v, l, accent) in enumerate(stats):
    add_text(s, Inches(0.6) + sw * i, y0 + Inches(0.2), sw, Inches(0.6),
             v, size=26, bold=True, color=ACCENT if accent else TEXT_1)
    add_text(s, Inches(0.6) + sw * i, y0 + Inches(0.85), sw, Inches(0.3),
             l, size=9, color=TEXT_3, tracking=120, bold=True)

add_text(s, Inches(0.6), SLIDE_H - Inches(0.45), Inches(8), Inches(0.3),
         "Avo Sarkissian  ·  Northeastern University  ·  Spring 2026",
         size=10, color=TEXT_3)


# ── Slide 2 — WHY THIS MATTERS ─────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
set_bg(s)
add_header(s, "01 · WHY", "Why this matters", 2)
add_footer(s)

add_text(s, Inches(0.6), Inches(1.85), Inches(12), Inches(0.5),
         "New AI models ship every week. The right pick can be the difference between a $5 and a $5,000 monthly bill.",
         size=14, color=TEXT_2)

# Four audience cards — same layout, shorter copy
cards = [
    ("DEVELOPERS",
     "Pick the cheapest model that still does the job. Not every task needs the smartest model.",
     ACCENT),
    ("RESEARCHERS",
     "Track which models lead on what. The leaderboard shifts every few weeks.",
     PURPLE),
    ("BUSINESSES",
     "Estimate monthly cost before committing. Two similar models can differ 50× in price.",
     GREEN),
    ("STUDENTS",
     "Know what runs on your laptop versus what needs an API key.",
     ORANGE),
]
gap = Inches(0.18)
cw = (SLIDE_W - Inches(1.2) - gap * 3) / 4
ch = Inches(4.0)
cy = Inches(2.6)
for i, (title, body, col) in enumerate(cards):
    cx = Inches(0.6) + (cw + gap) * i
    add_rect(s, cx, cy, cw, ch, fill=BG_CARD)
    add_rect(s, cx, cy, cw, Inches(0.05), fill=col)
    add_text(s, cx + Inches(0.3), cy + Inches(0.35), cw - Inches(0.6),
             Inches(0.4), title, size=11, bold=True, color=col, tracking=160)
    add_text(s, cx + Inches(0.3), cy + Inches(0.95), cw - Inches(0.6),
             Inches(2.9), body, size=14, color=TEXT_1)


# ── Slide 3 — DATA SOURCE & FRESHNESS ──────────────────────────────────────
s = prs.slides.add_slide(BLANK)
set_bg(s)
add_header(s, "02 · DATA", "Where the data comes from", 3)
add_footer(s)

add_text(s, Inches(0.6), Inches(1.85), Inches(12), Inches(0.4),
         "Two separate data sources: a live API for cloud models, and a curated catalog for local hardware.",
         size=14, color=TEXT_2)

# Top row: two source cards side by side
lx = Inches(0.6); ly = Inches(2.45); lw = Inches(7.4); lh = Inches(2.3)
add_rect(s, lx, ly, lw, lh, fill=BG_CARD)
add_rect(s, lx, ly, lw, Inches(0.05), fill=ACCENT)
add_text(s, lx + Inches(0.35), ly + Inches(0.25), lw - Inches(0.7), Inches(0.3),
         "ARTIFICIAL ANALYSIS API  ·  artificialanalysis.ai", size=9, bold=True, color=ACCENT, tracking=140)
add_text(s, lx + Inches(0.35), ly + Inches(0.6), lw - Inches(0.7), Inches(0.4),
         "Cloud LLMs, image gen, video gen, embeddings", size=16, bold=True, color=TEXT_1)
aa_bullets = [
    "Price, speed, latency, context window for every public model",
    "Scraped hourly in the background. Each run saves a dated snapshot.",
]
for i, b in enumerate(aa_bullets):
    yy = ly + Inches(1.15) + Inches(0.42) * i
    add_rect(s, lx + Inches(0.4), yy + Inches(0.1), Inches(0.08), Inches(0.08), fill=ACCENT)
    add_text(s, lx + Inches(0.62), yy, lw - Inches(1.0), Inches(0.35), b, size=11, color=TEXT_2)

# Second source card below
lx2 = Inches(0.6); ly2 = Inches(4.9); lw2 = Inches(7.4); lh2 = Inches(1.8)
add_rect(s, lx2, ly2, lw2, lh2, fill=BG_CARD)
add_rect(s, lx2, ly2, lw2, Inches(0.05), fill=PURPLE)
add_text(s, lx2 + Inches(0.35), ly2 + Inches(0.25), lw2 - Inches(0.7), Inches(0.3),
         "MANUALLY CURATED  ·  Manufacturer spec sheets + HuggingFace", size=9, bold=True, color=PURPLE, tracking=140)
add_text(s, lx2 + Inches(0.35), ly2 + Inches(0.6), lw2 - Inches(0.7), Inches(0.4),
         "Local open-weight models + GPU hardware specs", size=16, bold=True, color=TEXT_1)
hw_bullets = [
    "GPU VRAM and memory bandwidth from official product pages (NVIDIA, AMD, Apple, Intel)",
    "Open-weight model catalog with parameter counts from HuggingFace model cards",
]
for i, b in enumerate(hw_bullets):
    yy = ly2 + Inches(1.1) + Inches(0.36) * i
    add_rect(s, lx2 + Inches(0.4), yy + Inches(0.08), Inches(0.08), Inches(0.08), fill=PURPLE)
    add_text(s, lx2 + Inches(0.62), yy, lw2 - Inches(1.0), Inches(0.32), b, size=11, color=TEXT_2)

# Right: freshness panel
rx = Inches(8.25); ry = Inches(2.45); rw = Inches(4.55); rh = Inches(4.25)
add_rect(s, rx, ry, rw, rh, fill=BG_CARD)
add_text(s, rx + Inches(0.3), ry + Inches(0.3), rw - Inches(0.6), Inches(0.3),
         "FRESHNESS", size=9, bold=True, color=TEXT_3, tracking=140)

add_text(s, rx + Inches(0.3), ry + Inches(0.7), rw - Inches(0.6), Inches(0.7),
         "1 hr", size=42, bold=True, color=ACCENT)
add_text(s, rx + Inches(0.3), ry + Inches(1.55), rw - Inches(0.6), Inches(0.3),
         "Cloud model refresh interval", size=11, color=TEXT_2)

add_line(s, rx + Inches(0.3), ry + Inches(2.05),
         rx + rw - Inches(0.3), ry + Inches(2.05), color=BORDER)

add_text(s, rx + Inches(0.3), ry + Inches(2.2), rw - Inches(0.6), Inches(0.7),
         "30+", size=42, bold=True, color=TEXT_1)
add_text(s, rx + Inches(0.3), ry + Inches(3.05), rw - Inches(0.6), Inches(0.3),
         "Daily snapshots on file", size=11, color=TEXT_2)

add_line(s, rx + Inches(0.3), ry + Inches(3.45),
         rx + rw - Inches(0.3), ry + Inches(3.45), color=BORDER)

add_text(s, rx + Inches(0.3), ry + Inches(3.6), rw - Inches(0.6), Inches(0.3),
         "LOCAL CATALOG", size=8, bold=True, color=TEXT_3, tracking=140)
add_text(s, rx + Inches(0.3), ry + Inches(3.82), rw - Inches(0.6), Inches(0.3),
         "Static  ·  updated when new models release", size=11, color=TEXT_1)


# ── Slide 4 — INTELLIGENCE SCORE ───────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
set_bg(s)
add_header(s, "03 · SCORING", "How the intelligence score works", 4)
add_footer(s)

add_text(s, Inches(0.6), Inches(1.85), Inches(12), Inches(0.4),
         "A single 0–100 number, made by averaging several public AI tests so no single benchmark dominates.",
         size=14, color=TEXT_2)

# Four broad category cards
cats = [
    ("REASONING",
     "General knowledge and step-by-step thinking",
     "MMLU-Pro, GPQA",
     ACCENT),
    ("CODING",
     "Writing and debugging real programs",
     "LiveCodeBench, SciCode",
     PURPLE),
    ("MATH",
     "Multi-step quantitative problems",
     "AIME, MATH-500",
     GREEN),
    ("KNOWLEDGE",
     "Recall and synthesis across many fields",
     "Humanity's Last Exam",
     ORANGE),
]
gap = Inches(0.18)
cw = (SLIDE_W - Inches(1.2) - gap * 3) / 4
ch = Inches(3.1)
cy = Inches(2.45)
for i, (cat, desc, tests, col) in enumerate(cats):
    cx = Inches(0.6) + (cw + gap) * i
    add_rect(s, cx, cy, cw, ch, fill=BG_CARD)
    add_rect(s, cx, cy, cw, Inches(0.05), fill=col)
    add_text(s, cx + Inches(0.3), cy + Inches(0.35), cw - Inches(0.6),
             Inches(0.3), cat, size=11, bold=True, color=col, tracking=160)
    add_text(s, cx + Inches(0.3), cy + Inches(0.85), cw - Inches(0.6),
             Inches(1.3), desc, size=15, color=TEXT_1)
    add_text(s, cx + Inches(0.3), cy + Inches(2.2), cw - Inches(0.6),
             Inches(0.25), "EXAMPLES", size=8, bold=True, color=TEXT_3, tracking=140)
    add_text(s, cx + Inches(0.3), cy + Inches(2.45), cw - Inches(0.6),
             Inches(0.4), tests, size=11, color=TEXT_2)

# Bottom note — full width
ny = Inches(5.85)
add_rect(s, Inches(0.6), ny, SLIDE_W - Inches(1.2), Inches(0.9), fill=BG_CARD)
add_text(s, Inches(0.85), ny + Inches(0.2), Inches(12), Inches(0.3),
         "WHY AVERAGE THEM?", size=9, bold=True, color=ACCENT, tracking=140)
add_text(s, Inches(0.85), ny + Inches(0.48), Inches(12), Inches(0.35),
         "One number rewards well-rounded models and lets you compare cloud and local models on the same scale.",
         size=12, color=TEXT_2)


# ── Slide 5 — TOOLS ────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
set_bg(s)
add_header(s, "04 · TOOLS", "How it's built", 5)
add_footer(s)

add_text(s, Inches(0.6), Inches(1.85), Inches(12), Inches(0.4),
         "Built in Python, hosted on Render. No JavaScript, no build step.",
         size=14, color=TEXT_2)

libs = [
    ("Dash",      "WEB FRAMEWORK",
     "Turns Python functions into a live interactive site. Handles tabs, filters, and callbacks.",
     ACCENT),
    ("Plotly",    "CHARTS",
     "Every chart in the dashboard. Interactive, hoverable, and styled from one config.",
     PURPLE),
    ("Pandas",    "DATA",
     "Loads the CSV, runs filters, and computes the Pareto frontier and value scores.",
     GREEN),
    ("Requests",  "LIVE SCRAPER",
     "Plain HTTP call to the Artificial Analysis API every hour. No browser needed.",
     ORANGE),
    ("Render",    "HOSTING",
     "Cloud platform that runs the app 24/7 from the GitHub repo. Redeploys on every push.",
     RGBColor(0x7C, 0x3A, 0xED)),
]
gap = Inches(0.16)
cw = (SLIDE_W - Inches(1.2) - gap * 4) / 5
ch = Inches(4.1)
ly = Inches(2.35)
for i, (name, sub, body, col) in enumerate(libs):
    cx = Inches(0.6) + (cw + gap) * i
    add_rect(s, cx, ly, cw, ch, fill=BG_CARD)
    add_rect(s, cx, ly, cw, Inches(0.05), fill=col)
    add_text(s, cx + Inches(0.25), ly + Inches(0.38), cw - Inches(0.5),
             Inches(0.6), name, size=22, bold=True, color=TEXT_1)
    add_text(s, cx + Inches(0.25), ly + Inches(1.05), cw - Inches(0.5),
             Inches(0.3), sub, size=9, bold=True, color=col, tracking=140)
    add_text(s, cx + Inches(0.25), ly + Inches(1.5), cw - Inches(0.5),
             Inches(2.5), body, size=12, color=TEXT_2)


# ── Slide 6 — VISUALIZATIONS (TYPES + PRINCIPLES) ──────────────────────────
s = prs.slides.add_slide(BLANK)
set_bg(s)
add_header(s, "05 · VISUALIZATIONS", "Charts and the choices behind them", 6)
add_footer(s)

# ── LEFT: chart types ──
lx = Inches(0.6); ly = Inches(1.95); lw = Inches(6.3); lh = Inches(4.85)
add_rect(s, lx, ly, lw, lh, fill=BG_CARD)
add_text(s, lx + Inches(0.3), ly + Inches(0.25), lw - Inches(0.6), Inches(0.3),
         "CHART TYPES USED", size=9, bold=True, color=ACCENT, tracking=140)

types = [
    ("Bubble scatter",     "Three things at once: price, quality, speed"),
    ("Pareto frontier",    "Highlights the best deals on the curve"),
    ("Treemap",            "One tile per provider, sized by model count"),
    ("Horizontal bars",    "Best for ranked comparisons"),
    ("Radar",              "Compare 5 models on 5 dimensions"),
    ("Bump chart",         "Tracks how rankings move over time"),
]
ty = ly + Inches(0.75)
for i, (name, desc) in enumerate(types):
    yy = ty + Inches(0.65) * i
    add_rect(s, lx + Inches(0.4), yy + Inches(0.1), Inches(0.08), Inches(0.45), fill=ACCENT)
    add_text(s, lx + Inches(0.6), yy + Inches(0.05), lw - Inches(1.0),
             Inches(0.35), name, size=15, bold=True, color=TEXT_1)
    add_text(s, lx + Inches(0.6), yy + Inches(0.35), lw - Inches(1.0),
             Inches(0.3), desc, size=12, color=TEXT_3)

# ── RIGHT: principles ──
rx = Inches(7.1); ry = Inches(1.95); rw = Inches(5.65); rh = Inches(4.85)
add_rect(s, rx, ry, rw, rh, fill=BG_CARD)
add_text(s, rx + Inches(0.3), ry + Inches(0.25), rw - Inches(0.6), Inches(0.3),
         "DESIGN CHOICES", size=9, bold=True, color=ACCENT, tracking=140)

principles = [
    ("Data-ink ratio",
     "Most pixels should encode data. Removed gridlines, frames, and a long legend."),
    ("Color with backup",
     "Provider has both a color AND a marker shape, readable for colorblind viewers."),
    ("Visual grouping",
     "Filters sit next to the chart they control. Shared colors carry across all 11 tabs."),
    ("Right chart for the job",
     "Bars for ranking, log scale for prices that span 1000×, small multiples to avoid clutter."),
]
py = ry + Inches(0.75)
for i, (title, body) in enumerate(principles):
    yy = py + Inches(1.0) * i
    add_text(s, rx + Inches(0.4), yy, Inches(0.5), Inches(0.3),
             f"0{i+1}", size=12, bold=True, color=ACCENT, tracking=140)
    add_text(s, rx + Inches(0.85), yy, rw - Inches(1.1), Inches(0.3),
             title, size=15, bold=True, color=TEXT_1)
    add_text(s, rx + Inches(0.85), yy + Inches(0.38), rw - Inches(1.1),
             Inches(0.6), body, size=12, color=TEXT_2)


# ── Slide 7 — LOCAL + HYBRID ───────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
set_bg(s)
add_header(s, "06 · LOCAL + HYBRID", "Open weights and Claude Code stacks", 7)
add_footer(s)

add_text(s, Inches(0.6), Inches(1.85), Inches(12), Inches(0.5),
         "Not everything has to live on the cloud. The dashboard also tracks open-weight models you can run yourself, and recommends hybrid setups for tools like Claude Code / openclaw.",
         size=13, color=TEXT_2)

# Left: Run Local card (concept only — no formulas)
lx = Inches(0.6); ly = Inches(2.85); lw = Inches(6.0); lh = Inches(3.85)
add_rect(s, lx, ly, lw, lh, fill=BG_CARD)
add_text(s, lx + Inches(0.3), ly + Inches(0.3), lw - Inches(0.6), Inches(0.3),
         "RUN LOCAL TAB", size=9, bold=True, color=ACCENT, tracking=140)
add_text(s, lx + Inches(0.3), ly + Inches(0.65), lw - Inches(0.6), Inches(0.5),
         "Will it run on my machine?", size=20, bold=True, color=TEXT_1)

local_pts = [
    "Pick your GPU and how much you want to compress the model",
    "See which open-weight models will actually fit in your VRAM",
    "Get an estimated speed in tokens/second for your hardware",
    "Compare every open model on the same intelligence score",
]
by = ly + Inches(1.45)
for i, b in enumerate(local_pts):
    yy = by + Inches(0.5) * i
    add_rect(s, lx + Inches(0.4), yy + Inches(0.13), Inches(0.1), Inches(0.1), fill=ACCENT)
    add_text(s, lx + Inches(0.65), yy, lw - Inches(1.0), Inches(0.4),
             b, size=12, color=TEXT_1)

# Right: Hybrid stack
rx = Inches(6.85); ry = Inches(2.85); rw = Inches(5.95); rh = Inches(3.85)
add_rect(s, rx, ry, rw, rh, fill=BG_CARD)
add_text(s, rx + Inches(0.3), ry + Inches(0.3), rw - Inches(0.6), Inches(0.3),
         "HYBRID STACK  ·  CLAUDE CODE / OPENCLAW", size=9, bold=True, color=ACCENT, tracking=140)
add_text(s, rx + Inches(0.3), ry + Inches(0.65), rw - Inches(0.6), Inches(0.5),
         "Three tiers, mix cloud + local", size=18, bold=True, color=TEXT_1)

tiers = [
    ("REASONING",  "Cloud API",      "Plans the work and delegates",     PURPLE),
    ("BALANCED",   "Cloud API",      "Writes the actual code",           ACCENT),
    ("FAST",       "Local model",    "Handles cheap, high-volume tasks", GREEN),
]
ty = ry + Inches(1.5)
th = Inches(0.7)
for i, (tier, where, role, col) in enumerate(tiers):
    yy = ty + (th + Inches(0.08)) * i
    add_rect(s, rx + Inches(0.3), yy, rw - Inches(0.6), th, fill=BG_ELEVATED)
    add_rect(s, rx + Inches(0.3), yy, Inches(0.06), th, fill=col)
    add_text(s, rx + Inches(0.5), yy + Inches(0.13), Inches(1.4), Inches(0.3),
             tier, size=9, bold=True, color=col, tracking=140)
    add_text(s, rx + Inches(0.5), yy + Inches(0.36), Inches(3.5), Inches(0.3),
             role, size=11, color=TEXT_1)
    add_text(s, rx + rw - Inches(1.6), yy + Inches(0.22), Inches(1.3), Inches(0.3),
             where, size=10, bold=True, color=TEXT_2, align=PP_ALIGN.RIGHT)


# ── Slide 8 — DEMO ─────────────────────────────────────────────────────────
s = prs.slides.add_slide(BLANK)
set_bg(s)

# Header chip + page number
add_chip(s, Inches(0.6), Inches(0.6), "07 · LIVE", color=ACCENT, fill=ACCENT_DIM)
add_text(s, SLIDE_W - Inches(1.2), Inches(0.65), Inches(0.6), Inches(0.3),
         f"08 / {TOTAL_SLIDES:02d}", size=9, color=TEXT_3, align=PP_ALIGN.RIGHT,
         tracking=120, bold=True)

add_text(s, Inches(0.6), Inches(1.6), Inches(12), Inches(2.0),
         "Live Demo", size=80, bold=True, color=TEXT_1)
add_text(s, Inches(0.6), Inches(3.05), Inches(12), Inches(0.6),
         "→ localhost:8050", size=24, color=ACCENT, bold=True)

# Walkthrough cards
items = [
    ("Overview",    "Cost vs. quality at a glance"),
    ("Rankings",    "Top models by score and value"),
    ("Compare",     "Five models head to head"),
    ("Run Local",   "Pick a GPU, see what fits"),
    ("Agent Stack", "Build a hybrid Claude Code stack"),
]
gap = Inches(0.15)
cw = (SLIDE_W - Inches(1.2) - gap * (len(items)-1)) / len(items)
ch = Inches(1.2)
iy = Inches(4.4)
for i, (tab, note) in enumerate(items):
    cx = Inches(0.6) + (cw + gap) * i
    add_rect(s, cx, iy, cw, ch, fill=BG_CARD)
    add_rect(s, cx, iy, cw, Inches(0.05), fill=ACCENT)
    add_text(s, cx + Inches(0.2), iy + Inches(0.25), cw - Inches(0.4),
             Inches(0.3), tab.upper(), size=10, bold=True, color=TEXT_1, tracking=120)
    add_text(s, cx + Inches(0.2), iy + Inches(0.6), cw - Inches(0.4),
             Inches(0.5), note, size=10, color=TEXT_3)

# Footer references — minimal
ry = Inches(6.25)
add_line(s, Inches(0.6), ry, SLIDE_W - Inches(0.6), ry, color=BORDER)
add_text(s, Inches(0.6), ry + Inches(0.25), Inches(3), Inches(0.3),
         "DATA  ·", size=9, bold=True, color=TEXT_3, tracking=140)
add_text(s, Inches(1.4), ry + Inches(0.25), Inches(8), Inches(0.3),
         "artificialanalysis.ai", size=10, color=TEXT_2)
add_text(s, Inches(0.6), ry + Inches(0.55), Inches(3), Inches(0.3),
         "CODE  ·", size=9, bold=True, color=TEXT_3, tracking=140)
add_text(s, Inches(1.4), ry + Inches(0.55), Inches(11), Inches(0.3),
         "github.com/Avo-Sarkissian/AI-Frontier-Database", size=10, color=TEXT_2)

# ── Save ────────────────────────────────────────────────────────────────────
out_path = "AI_Frontier_Presentation.pptx"
prs.save(out_path)
print(f"Wrote {out_path}  ·  {len(prs.slides)} slides")
