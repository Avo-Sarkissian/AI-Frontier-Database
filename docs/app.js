// ---- Tab + panel definitions (chart container ids match figures/<id>.json) ----
const TABS = [
  { id: "overview",  label: "Overview",    charts: ["pareto"] },
  { id: "recommend", label: "Agent Stack", charts: [] },
  { id: "landscape", label: "Landscape",   charts: ["treemap", "provider_leaderboard"] },
  { id: "rankings",  label: "Rankings",    charts: ["rankings", "value_leaders"] },
  { id: "compare",   label: "Compare",     charts: ["radar"] },
  { id: "budget",    label: "Budget",      charts: ["cost_calc"] },
  { id: "table",     label: "Table",       charts: [] },
  { id: "local",     label: "Run Local",   charts: ["local_scatter", "local_compat"] },
  { id: "image",     label: "Image Gen",   charts: ["image_faceted"] },
  { id: "video",     label: "Video Gen",   charts: ["video_rankings", "video_scatter"] },
];

// Which manifest.captions keys each panel shows, in order. The prose itself
// lives in captions.py and rides the manifest — this side used to render
// exactly one of the fourteen captions app.py renders, so nine tabs of the
// DEPLOYED site explained nothing at all.
const TAB_CAPTIONS = {
  overview:  ["overview_price"],
  recommend: ["recommend"],
  landscape: ["landscape_treemap", "landscape_leaderboard"],
  rankings:  ["rankings_intelligence", "rankings_value"],
  compare:   ["compare"],
  budget:    ["budget"],
  table:     ["table"],
  local:     ["local"],
  image:     ["image"],
  video:     ["video"],
};

// window.AF.state contract: { providers: string[], minQuality: number, search: string, tab: string }
const COMPARE_MAX = 5;   // mirrors static_helpers.COMPARE_MAX
// gpuMeta: preset name -> {vram_gb, bandwidth_gbps, hw_type, fp16_tflops},
// shipped with gpu_options at boot so a preset change resolves in-tick.
// drawn: chart div id -> "static" | "py", so a late pre-rendered figure can
// never paint over a live Python render.
window.AF = { pyReady: false, figCache: {}, manifest: null, localHwMeta: null,
  gpuMeta: {}, drawn: {}, compareOrder: [], state: {
  providers: [], minQuality: 0, search: "", tab: "overview" } };

const PLOT_CONFIG = { displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage"] };

// Model, provider and context strings come from the scraped Artificial Analysis
// feed, so they are third-party text. Anything interpolated into an HTML string
// that later reaches .innerHTML must go through this first. Prefer
// createElement + textContent where the shape of the code allows it.
const ESCAPE_MAP = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ESCAPE_MAP[c]);
}

// ---- Provider color map — mirrors PROVIDER_COLORS in
// components/charts/constants.py. Kept in sync by hand; tests/
// test_static_site_wiring.py asserts they match.
const PROVIDER_COLORS = {
  "Anthropic":               "#cc785c",
  "Meta":                    "#138df7",
  "OpenAI":                  "#efefef",
  "Alibaba":                 "#fb3eb2",
  "Google":                  "#6fcd81",
  "NVIDIA":                  "#a0f427",
  "Amazon":                  "#fea33d",
  "Mistral":                 "#ce1202",
  "DeepSeek":                "#274bfb",
  "SpaceXAI":                "#a2a2e8",
  "Kimi":                    "#af41c3",
  "Microsoft":               "#0078d5",
  "Cohere":                  "#d18ee2",
  "Z AI":                    "#1e710d",
  "MiniMax":                 "#eb3568",
  "InclusionAI":             "#4fb5ff",
  "Xiaomi":                  "#ff6900",
  "Baidu":                   "#2f4bec",
  "IBM":                     "#0f62fe",
  "LG AI Research":          "#df3b8a",
  "Nous Research":           "#006fa8",
  "Reka AI":                 "#546079",
  "AI21 Labs":               "#d63864",
  "Allen Institute for AI":  "#f0529c",
  "Inception":               "#47637c",
  "Upstage":                 "#7c59f5",
  "Perplexity":              "#1b818e",
  "KwaiKAT":                 "#489034",
  "Deep Cogito":             "#5080ec",
  "Thinking Machines":       "#5eead4",
  "Tencent":                 "#5cb9ff",
  "StepFun":                 "#017aff",
  "Arcee AI":                "#008c8d",
  "LongCat":                 "#2adb65",
  "Sapiens AI":              "#3152d8",
  "Nex AGI":                 "#125bc7",
  "Multiverse Computing":    "#f11338",
  "Apodex":                  "#0fd9d2",
  "Celeris":                 "#fdba74",
  "Sarvam":                  "#f472b6",
  "Swiss AI Initiative":     "#bee9fa",
};
const DEFAULT_PROVIDER_COLOR = "#6b7280";


// "148 models tracked" is only honest next to what upstream had that we cannot
// carry. Written as text, never markup — the names come from the scraped feed.
function renderCoverageNote(cov) {
  const label = document.querySelector("#stat-model-count + .stat-label");
  if (!label || !cov) return;
  const noScore = (cov.skipped_no_score || []).length;
  const noPrice = (cov.skipped_no_price || []).length;
  const total = noScore + noPrice;
  if (!total) { label.textContent = "Models tracked"; label.title = ""; return; }
  label.textContent = `Models tracked  ·  ${total} not carried`;
  const parts = [];
  if (noScore) parts.push(`${noScore} with no intelligence score: ` + (cov.skipped_no_score || []).join(", "));
  if (noPrice) parts.push(`${noPrice} with no published price: ` + (cov.skipped_no_price || []).join(", "));
  label.title =
    `${cov.kept} of ${(cov.kept || 0) + total} models upstream are shown.\n` + parts.join("\n");
}

function buildTabsAndPanels() {
  const tabsEl = document.getElementById("tabs");
  const panelsEl = document.getElementById("tab-panels");
  TABS.forEach((t, i) => {
    const b = document.createElement("button");
    b.className = "tab" + (i === 0 ? " tab--selected" : "");
    b.textContent = t.label; b.dataset.tab = t.id;
    b.onclick = () => switchTab(t.id);
    tabsEl.appendChild(b);

    const panel = document.createElement("div");
    panel.id = "panel-" + t.id;
    panel.style.display = i === 0 ? "block" : "none";

    // Special panels that need extra containers beyond charts
    if (t.id === "table") {
      panel.innerHTML = `
        <div class="chart-card" style="overflow-x:auto;padding:0;">
          <table id="model-table" style="width:100%;border-collapse:collapse;">
            <thead id="model-table-head"></thead>
            <tbody id="model-table-body"></tbody>
          </table>
        </div>`;
    } else if (t.id === "compare") {
      panel.innerHTML = `
        <div class="chart-card"><div id="chart-radar" style="min-height:400px"></div></div>
        <div id="compare-raw-table" class="chart-card" style="overflow-x:auto;padding:16px;"></div>`;
    } else if (t.id === "recommend") {
      panel.innerHTML = `<div id="recommend-cards" class="chart-card" style="padding:16px;"></div>`;
    } else {
      panel.innerHTML = t.charts.map(c =>
        `<div class="chart-card"><div id="chart-${c}" style="min-height:400px"></div></div>`
      ).join("");
    }
    panelsEl.appendChild(panel);
  });
}

// Tabs that read none of PROVIDER / MIN SCORE / SEARCH — mirrors
// static_helpers.TABS_WITHOUT_GLOBAL_FILTERS.
const TABS_WITHOUT_GLOBAL_FILTERS = ["recommend", "local", "image", "video"];

// Show/hide per-tab control rows
function showTabControls(id) {
  // The global filter bar lives outside #tab-panels, so nothing hid it on the
  // four tabs that ignore it: ?tab=image&p=Anthropic&q=45 showed
  // "Anthropic · ≥ 45" over a chart plotting 72 models from a dozen providers.
  // Only the filters go (.export-only, assets/style.css): ↓CSV sits in the
  // same bar, and hiding all of it made those tabs' exports unreachable.
  const globalFilters = document.querySelector(".filters");
  if (globalFilters) {
    globalFilters.classList.toggle("export-only",
      TABS_WITHOUT_GLOBAL_FILTERS.includes(id));
  }
  // Hide all tab control rows first
  document.querySelectorAll('[id^="tab-controls-"]').forEach(el => {
    el.style.display = "none";
  });
  // Hide recommend-specific rows
  const provRow = document.getElementById("recommend-providers-row");
  const hwRow = document.getElementById("recommend-hw-row");
  if (provRow) provRow.style.display = "none";
  if (hwRow) hwRow.style.display = "none";
  // The budget answer block lives outside #tab-panels, so it needs hiding too
  // Coming back to Budget with unchanged inputs skips the re-render (the
  // per-tab memo), so the card is re-shown here rather than by the render.
  const budgetAnswer = document.getElementById("budget-answer");
  if (budgetAnswer) {
    budgetAnswer.style.display =
      id === "budget" && budgetAnswer.hasChildNodes() ? "block" : "none";
  }

  // Show controls for the active tab
  const ctrl = document.getElementById("tab-controls-" + id);
  if (ctrl) ctrl.style.display = "flex";

  // For Agent Stack, also show providers/hw rows based on current mode
  if (id === "recommend") {
    const mode = document.querySelector('input[name="recommend-mode"]:checked')?.value || "api";
    if (provRow) provRow.style.display = mode !== "local" ? "flex" : "none";
    if (hwRow) hwRow.style.display = (mode === "hybrid" || mode === "hybrid2" || mode === "local") ? "flex" : "none";
  }
}

const VALID_TABS = TABS.map(t => t.id);

function switchTab(id) {
  // An unknown id used to hide EVERY panel — `display = t.id === id` matches
  // nothing — leaving the header, stat bar and filter row above a blank page
  // with no error. Not hypothetical: `insights`, `performance` and `embeddings`
  // were all live tab values this app once emitted into share URLs. The Dash
  // side has had _VALID_TABS since commit 82effa0; this side never got it.
  if (!VALID_TABS.includes(id)) id = VALID_TABS[0];
  // Close the model detail panel. It was only ever cleared by its own close
  // button, so it stayed open — and still armed to an LLM with a live "Add to
  // Compare" — over the Video Gen and Image Gen tabs.
  const panel = document.getElementById("detail-panel");
  if (panel) panel.className = "detail-panel";
  window.AF.detailModel = null;
  window.AF.state.tab = id;
  document.querySelectorAll(".tab").forEach(b =>
    b.classList.toggle("tab--selected", b.dataset.tab === id));
  TABS.forEach(t => {
    document.getElementById("panel-" + t.id).style.display = t.id === id ? "block" : "none";
  });
  showTabControls(id);
  // First visit: draw the pre-rendered figures now, into a panel that is
  // already visible, so Plotly measures the real width. Drawing all twelve
  // into display:none panels at init laid every one out at Plotly's 700 px
  // default and jumped to full width 60 ms after each tab switch.
  loadStaticFigures(id);
  // A plot drawn while hidden (a Python answer that landed after the user left
  // its tab) still needs a resize. Next frame, once layout has the width —
  // not a 60 ms timer, which showed the 700 px chart for four frames.
  requestAnimationFrame(() => fitPlotsIn(id));
  // Re-render active tab with current filters once Pyodide is ready. This also
  // writes the tab into the URL (syncUrl).
  rerenderActiveFilterCharts();
}

// Resize every drawn plot in a panel whose width no longer matches its box.
// Plotly.Plots.resize defers its work by 100 ms; relayout({autosize}) does the
// same thing now. A plot that already fits is left alone, so an ordinary tab
// switch costs nothing.
function fitPlotsIn(tabId) {
  document.querySelectorAll("#panel-" + tabId + " .js-plotly-plot").forEach(el => {
    const box = el.clientWidth;
    const drawn = el._fullLayout && el._fullLayout.width;
    if (!box || (drawn && Math.abs(drawn - box) < 2)) return;
    if (el.layout && el.layout.width) return;          // a fixed-width figure
    try { Plotly.relayout(el, { autosize: true }); } catch (e) { console.warn("resize:", e); }
  });
}

// Draw a tab's pre-rendered figures, once. Nothing is fetched for tabs the
// visitor never opens.
function loadStaticFigures(tabId) {
  const tab = TABS.find(t => t.id === tabId);
  if (!tab) return Promise.resolve();
  return Promise.all(tab.charts
    .filter(c => !window.AF.drawn["chart-" + c])
    .map(c => {
      window.AF.drawn["chart-" + c] = "loading";     // one fetch per chart
      return renderFigure("chart-" + c, c).catch(e => {
        if (window.AF.drawn["chart-" + c] === "loading") delete window.AF.drawn["chart-" + c];
        console.error(`figure ${c} failed:`, e);
      });
    }));
}

async function renderFigure(divId, figId) {
  let fig = window.AF.figCache[figId];
  if (!fig) {
    const r = await fetch(`figures/${figId}.json?v=${window.AF.version || ""}`);
    fig = await r.json();
    window.AF.figCache[figId] = fig;
  }
  // The fetch can outlive a Python render of the same div; the live answer wins.
  if (window.AF.drawn[divId] === "py") return;
  window.AF.drawn[divId] = "static";
  await Plotly.react(divId, fig.data, fig.layout, PLOT_CONFIG);
  // react() drops event handlers, and the Python render that re-attaches the
  // click may be skipped by its memo.
  if (divId === "chart-pareto") attachParetoClickHandler();
}

async function loadManifest() {
  const m = await (await fetch("figures/manifest.json", { cache: "no-store" })).json();
  window.AF.manifest = m;
  window.AF.version = m.version || "";
  window.AF.codeVersion = m.code_version || "";
  window.AF.generatedIso = m.generated_iso || null;
  document.getElementById("stat-model-count").textContent = m.model_count;
  renderCoverageNote(m.coverage);
  document.getElementById("stat-provider-count").textContent = m.provider_count;
  document.getElementById("stat-floor-price").textContent = m.floor_price;
  document.getElementById("stat-peak-quality").textContent = m.peak_quality;
  const sel = document.getElementById("filter-provider");
  m.provider_options.forEach(o => {
    const opt = document.createElement("option");
    opt.value = o.value; opt.textContent = o.label; sel.appendChild(opt);
  });
  // Populate radar-model-select from manifest
  const radarSel = document.getElementById("radar-model-select");
  if (radarSel && m.model_options) {
    m.model_options.forEach(o => {
      const opt = document.createElement("option");
      opt.value = o.value; opt.textContent = o.label;
      if (m.diverse5 && m.diverse5.includes(o.value)) opt.selected = true;
      radarSel.appendChild(opt);
    });
  }
  // Populate image/video provider filters from manifest (dynamic, survives catalog refreshes)
  const imgProvSel = document.getElementById("image-provider-filter");
  if (imgProvSel && m.image_providers) {
    imgProvSel.innerHTML = "";
    m.image_providers.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p; opt.textContent = p; imgProvSel.appendChild(opt);
    });
  }
  const vidProvSel = document.getElementById("video-provider-filter");
  if (vidProvSel && m.video_providers) {
    vidProvSel.innerHTML = "";
    m.video_providers.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p; opt.textContent = p; vidProvSel.appendChild(opt);
    });
  }
  // Tag filters come from the manifest for the same reason the provider lists
  // do: a hardcoded option outlives the data. The image "Fast" tag matched zero
  // models for months and the UI blamed the user for it.
  // MIN SCORE options ride the manifest so the <select> can hold the exact
  // percentile a preset sets. Snapping 52.7 down to 50 made "Top 10%" return
  // 15.5% of the catalogue under a label that is a precise numeric claim.
  const qSel = document.getElementById("filter-quality");
  if (qSel && Array.isArray(m.quality_options) && m.quality_options.length) {
    const current = qSel.value;
    qSel.replaceChildren(...m.quality_options.map(o => {
      const opt = document.createElement("option");
      opt.value = String(o.value); opt.textContent = o.label;
      return opt;
    }));
    if (current) qSel.value = current;
  }
  const effSel = document.getElementById("filter-effort");
  if (effSel && Array.isArray(m.effort_options) && m.effort_options.length) {
    const current = effSel.value;
    effSel.replaceChildren(...m.effort_options.map(o => {
      const opt = document.createElement("option");
      opt.value = o.value; opt.textContent = o.label;
      return opt;
    }));
    if (current) effSel.value = current;
  }
  fillTagSelect("image-tag-filter", m.image_tags);
  fillTagSelect("video-tag-filter", m.video_tags);
  // Video arenas come from the manifest for the same reason the tags do: a
  // hardcoded <option> outlives the data behind it.
  const vidModeSel = document.getElementById("video-mode-filter");
  if (vidModeSel && Array.isArray(m.video_modes) && m.video_modes.length) {
    vidModeSel.replaceChildren(...m.video_modes.map(o => {
      const opt = document.createElement("option");
      opt.value = o.value; opt.textContent = o.label;
      return opt;
    }));
  }
  renderCaptions(m.captions);
}

// Captions are written as text, never markup — they come from the manifest.
function renderCaptions(captions) {
  if (!captions) return;
  Object.entries(TAB_CAPTIONS).forEach(([tab, keys]) => {
    const panel = document.getElementById("panel-" + tab);
    if (!panel) return;
    keys.slice().reverse().forEach((key, revIdx) => {
      const text = captions[key];
      if (!text) return;
      // Overview already owns #overview-desc, which swaps with the x-axis radio.
      if (key === "overview_price" && document.getElementById("overview-desc")) return;
      const existing = panel.querySelector(`[data-caption="${key}"]`);
      const el = existing || document.createElement("div");
      el.className = "chart-caption";
      el.dataset.caption = key;
      el.textContent = text;
      if (!existing) {
        const cards = panel.querySelectorAll(".chart-card");
        const anchor = cards[keys.length - 1 - revIdx];
        panel.insertBefore(el, anchor || panel.firstChild);
      }
    });
  });
}

function fillQuantSelect(selectId, options, defaultValue) {
  const sel = document.getElementById(selectId);
  if (!sel || !Array.isArray(options)) return;
  sel.replaceChildren(...options.map(o => {
    const opt = document.createElement("option");
    // o is {label, value}: the label carries the "(lossy)" marker for Q3/Q2,
    // the value is what the Python side keys QUANT_BYTES by.
    opt.value = o.value; opt.textContent = o.label;
    // o.default lets Python carry the default with the options (CONTEXT,
    // SESSIONS) instead of this file naming a number. defaultValue stays for
    // the quant selects, whose default is a string literal in one place.
    if (o.default === true || (defaultValue != null && o.value === defaultValue)) {
      opt.selected = true;
    }
    return opt;
  }));
}

function fillTagSelect(selectId, tags) {
  const sel = document.getElementById(selectId);
  if (!sel) return;
  if (!Array.isArray(tags) || !tags.length) {
    // The <select> ships empty, so a manifest without this key is not a
    // no-op — it is a permanently blank control with nothing to say so.
    // Loud in the console beats a filter that looks broken.
    console.warn(`${selectId}: manifest carries no tag vocabulary; ` +
                 `the control will be empty until build_static.py is re-run`);
    return;
  }
  sel.innerHTML = "";
  tags.forEach(t => {
    const opt = document.createElement("option");
    opt.value = t.value;
    opt.textContent = t.label;
    sel.appendChild(opt);
  });
}

// ---- Pyodide boot ----

// Pyodide runs in a worker (docs/pyworker.js). On the main thread its boot
// blocked the UI for ~3.8s of a 4.7s startup — single stalls up to 1.8s — while
// the pre-rendered charts were already interactive at ~450ms, so hovering a
// bubble during startup felt like the mouse had frozen.
function bootPyodide() {
  const statusEl = document.getElementById("py-status");
  const setStatus = (text, show = true) => {
    if (!statusEl) return;
    statusEl.style.display = show ? "block" : "none";
    if (text) statusEl.textContent = text;
  };
  setStatus("warming up interactivity…");

  let worker;
  try {
    worker = new Worker("pyworker.js");
  } catch (err) {
    console.error("Pyodide worker failed to start:", err);
    setStatus("interactivity unavailable: " + err.message);
    return;
  }

  const pending = new Map();
  let nextId = 1;

  const rpc = (fn, args) => new Promise((resolve, reject) => {
    if (!window.AF.pyReady) { reject(new Error("python not ready")); return; }
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: "call", id, fn, args });
  });

  window.AF.callPy = async (fn, ...args) => JSON.parse(await rpc(fn, args));
  window.AF.callPyRaw = (fn, ...args) => rpc(fn, args);

  worker.onmessage = (ev) => {
    const msg = ev.data || {};
    if (msg.type === "status") { setStatus(msg.text); return; }

    if (msg.type === "ready") {
      window.AF.pyReady = true;
      setStatus("", false);
      // Boot now starts before the controls are wired (it only needs the
      // manifest version), so wait for init() to finish before touching them.
      (window.AF.uiReady || Promise.resolve())
        .then(() => { setExportPending(false); return populateDynamicSelects(); })
        .then(rerenderActiveFilterCharts)
        .catch((e) => console.error("post-boot refresh failed:", e));
      return;
    }

    if (msg.type === "bootError") {
      console.error("Pyodide boot failed:", msg.message);
      setStatus("interactivity unavailable: " + msg.message);
      for (const { reject } of pending.values()) reject(new Error(msg.message));
      pending.clear();
      return;
    }

    if (msg.type === "result") {
      const entry = pending.get(msg.id);
      if (!entry) return;
      pending.delete(msg.id);
      if (msg.ok) entry.resolve(msg.value);
      else entry.reject(new Error(msg.error));
    }
  };

  worker.onerror = (err) => {
    console.error("Pyodide worker error:", err.message || err);
    setStatus("interactivity unavailable: " + (err.message || "worker error"));
  };

  worker.postMessage({ type: "boot", version: window.AF.version || "",
                       codeVersion: window.AF.codeVersion || "" });
}

// Write a preset's VRAM into a box. `userChange` is true only for a preset the
// visitor picked; the boot default must not clobber a figure they typed.
function setVramFromPreset(inputId, hw, userChange) {
  const el = document.getElementById(inputId);
  if (!el || !hw || hw.vram_gb == null) return;
  if (!userChange && el.dataset.userTyped === "1") return;
  el.value = hw.vram_gb;
  delete el.dataset.userTyped;
}

// ---- Populate selects that need data from Python ----
async function populateDynamicSelects() {
  try {
    const gpuOptions = await window.AF.callPy("gpu_options");
    const quantLevels = await window.AF.callPy("quant_options");
    const DEFAULT_GPU = "NVIDIA RTX 5090";
    const DEFAULT_QUANT = "Q4";
    // Every option carries its preset's hardware. Both preset handlers resolve
    // a change from this table in the same tick; the old per-change
    // local_hw_for_gpu call queued behind a running update_local for 1.2 s+.
    window.AF.gpuMeta = {};
    gpuOptions.forEach(o => {
      window.AF.gpuMeta[o.value] = {
        vram_gb: o.vram_gb, bandwidth_gbps: o.bandwidth_gbps,
        hw_type: o.hw_type, fp16_tflops: o.fp16_tflops ?? null,
      };
    });

    // local-gpu-preset
    const localGpu = document.getElementById("local-gpu-preset");
    if (localGpu) {
      localGpu.innerHTML = "";
      gpuOptions.forEach(o => {
        const opt = document.createElement("option");
        opt.value = o.value; opt.textContent = o.label;
        if (o.value === DEFAULT_GPU) opt.selected = true;
        localGpu.appendChild(opt);
      });
    }
    // recommend-gpu-preset
    const recGpu = document.getElementById("recommend-gpu-preset");
    if (recGpu) {
      recGpu.innerHTML = "";
      gpuOptions.forEach(o => {
        const opt = document.createElement("option");
        opt.value = o.value; opt.textContent = o.label;
        if (o.value === DEFAULT_GPU) opt.selected = true;
        recGpu.appendChild(opt);
      });
    }
    // Both quant selects, filled by ONE function.
    //
    // They were two copies of the same loop. When quant_levels became
    // quant_options and started returning {label, value} so Q3/Q2 could be
    // marked lossy, only the local-quant copy was updated — recommend-quant
    // kept assigning the whole object, so its <option value> serialised to the
    // string "[object Object]". Agent Stack then sent that as the quantisation
    // to update_recommend, calc_vram_gb raised KeyError('[object Object]'), and
    // every workflow with a local tier silently kept the previous API cards
    // while the radio and the hardware row both moved. One function now, so a
    // third select cannot drift.
    fillQuantSelect("local-quant", quantLevels, DEFAULT_QUANT);
    fillQuantSelect("recommend-quant", quantLevels, DEFAULT_QUANT);

    // CONTEXT and SESSIONS, filled the same way and for the same reason: the
    // choices and their defaults live in data/local_models.py, so neither shell
    // declares a hardware default of its own.
    try {
      const ctxOpts = await window.AF.callPy("context_options");
      fillQuantSelect("local-context", ctxOpts, null);
      const speedOpts = await window.AF.callPy("speed_mode_options");
      fillQuantSelect("local-speed-mode", speedOpts, null);
    } catch (e) { console.warn("context/slo options:", e); }

    // Default local HW meta, from the same table. A VRAM figure the visitor
    // typed while Python was still loading is theirs: only an actual preset
    // change may overwrite it.
    const hw = window.AF.gpuMeta[DEFAULT_GPU];
    if (hw) {
      window.AF.localHwMeta = hw;
      setVramFromPreset("local-vram", hw, false);
      // Same default preset feeds the Agent Stack tab's VRAM cap.
      setVramFromPreset("recommend-vram", hw, false);
    }

  } catch (e) {
    console.error("populateDynamicSelects failed:", e);
  }
}

// ---- Debounce helper ----
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
// Settle time for the per-tab controls (Run Local, Image, Video, Agent Stack).
const LOCAL_DEBOUNCE_MS = 150;

// ---- Relative time + freshness badge ----
function relativeTime(iso) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (secs < 60) return "just now";
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs} hr ago`;
  const days = Math.round(hrs / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

// ---- Toast ----
let _toastTimer;
function toast(msg) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
}

// ---- Refresh: pull latest published snapshot ----
async function doRefresh() {
  const btn = document.getElementById("btn-refresh");
  btn.classList.add("is-loading");
  try {
    const m = await (await fetch("figures/manifest.json", { cache: "no-store" })).json();
    if (m.version && m.version !== window.AF.version) {
      const u = new URL(location.href);
      u.searchParams.set("v", m.version);
      location.replace(u.toString());   // fresh figures + bundles + Pyodide reboot
      return;                            // navigating away; leave spinner on
    }
    toast("Already up to date — updated " + relativeTime(window.AF.generatedIso));
  } catch (e) {
    console.error("refresh failed:", e);
    location.reload();                   // safe fallback
    return;
  }
  btn.classList.remove("is-loading");
}

function renderFreshness() {
  const el = document.getElementById("data-freshness");
  if (!el) return;
  const m = window.AF.manifest || {};
  // The DATA's age, not the build's. generatedIso is when the site was built,
  // and the hourly job rebuilds whenever any one of three datasets moves — so
  // on CI run 31558618824 two scrapers failed, only the image arena refreshed,
  // and a 5h-stale catalogue published under a badge reading "just now".
  const iso = m.data_fetched_iso || window.AF.generatedIso;
  if (!iso) { el.textContent = ""; el.title = ""; return; }

  // Recompute staleness in the BROWSER, not just at build time. The manifest's
  // stale_datasets is a snapshot from the moment the site was built, so on a
  // page left open — or a day when the hourly job stops running entirely — the
  // relative time aged honestly while the warning never appeared. A dataset
  // that has not refreshed in STALE_AFTER_HOURS is stale no matter what the
  // build thought.
  // Shipped by build_static.py from data/scrape_status.STALE_AFTER_HOURS. It
  // was hardcoded as 3 here, against a cron that claims hourly but whose p90
  // gap is 6.9h — so the badge warned about GitHub's scheduler, not the data.
  const STALE_AFTER_HOURS = Number(m.stale_after_hours) || 12;
  const ds0 = m.datasets || {};
  const stale = Array.from(new Set([
    ...(Array.isArray(m.stale_datasets) ? m.stale_datasets : []),
    ...Object.keys(ds0).filter(k => {
      const e = ds0[k] || {};
      if (e.ok === false || e.ok === null || !e.fetched_at) return true;
      const age = (Date.now() - new Date(e.fetched_at).getTime()) / 3600000;
      return !(age < STALE_AFTER_HOURS);
    }),
  ]));
  el.textContent = "Updated " + relativeTime(iso) + (stale.length ? "  ·  ⚠" : "");
  el.style.color = stale.length ? "#fbbf24" : "";

  const lines = [];
  const ds = ds0;
  const LABELS = { hosted: "Hosted LLMs", local: "Open-weight models", image: "Image arena", video: "Video arena" };
  Object.keys(LABELS).forEach(k => {
    const e = ds[k] || {};
    const when = e.fetched_at ? new Date(e.fetched_at).toUTCString() : "never";
    const flag = e.ok === false ? "  (last scrape FAILED)" : e.ok === null ? "  (unknown)" : "";
    lines.push(`${LABELS[k]}: ${when}${flag}`);
  });
  if (stale.length) {
    lines.unshift(`Stale or failing: ${stale.join(", ")}`, "");
  }
  lines.push("", "Badge shows the oldest successful fetch across all four.");
  el.title = lines.join("\n");
}

// ---- Read current filter state ----
function readGlobalFilters() {
  const providers = Array.from(document.getElementById("filter-provider").selectedOptions).map(o => o.value);
  const minQuality = Number(document.getElementById("filter-quality").value);
  const search = document.getElementById("model-search").value;
  // "" means every variant, which is the default and the state the dashboard
  // has always shown. Anything else selects one row per model at that tier.
  const effort = document.getElementById("filter-effort")?.value || "";
  Object.assign(window.AF.state, { providers, minQuality, search, effort });
  return [providers, minQuality, search, effort];
}

async function renderJsonFig(divId, figObj) {
  window.AF.drawn[divId] = "py";
  return Plotly.react(divId, figObj.data, figObj.layout, PLOT_CONFIG);
}

// ---- Coalescing refresher: one per tab ----
//
// Every control change used to post its own worker call, and the single
// Pyodide worker ran them strictly first-in-first-out. Five GPU presets picked
// 150 ms apart gave ten renders and settled 6.3 s after the last click, showing
// cards that were no longer selected for 3.5 s of it; a tab switch behind three
// queued QUANT changes waited 8.6 s.
//
// makeRefresher(name, readArgs, call, draw) returns refresh(), which:
//   - keeps at most ONE call in flight for its tab;
//   - when the inputs moved while the call ran, drops the answer unrendered
//     and runs once more with the LATEST inputs read from the DOM — never the
//     intermediate ones. It compares the DOM, not a request count: every
//     control reaches refresh() through a 150 ms debounce, so the inputs can
//     move before a newer request is counted;
//   - remembers the arguments of its last successful render and does nothing
//     when asked again with the same ones, so a tab switch with unchanged
//     inputs skips both Python and Plotly.react (Run Local: 1.3-3.4 s).
// readArgs() reads the controls and returns a JSON-able list; call(...args)
// asks Python and resolves to its answer; draw(out, args) puts it on screen.
// refresh.invalidate() forgets the memo.
function makeRefresher(name, readArgs, call, draw) {
  let busy = false;          // a loop is running
  let again = false;         // a request arrived while it ran
  let lastKey = null;        // JSON of the args last rendered
  let current = Promise.resolve();

  async function loop() {
    try {
      do {
        again = false;
        const args = readArgs();
        const key = JSON.stringify(args);
        if (key === lastKey) continue;
        let out;
        try {
          out = await call(...args);
        } catch (e) { console.error(`${name} failed:`, e); continue; }
        // Superseded: the inputs are no longer the ones this answer was
        // computed for, whether or not the debounced refresh() has fired yet.
        // Rendering it would flash a state the user has already left; when
        // that refresh() does fire it finds the loop busy or hits the memo.
        if (JSON.stringify(readArgs()) !== key) {
          again = true;
          continue;
        }
        try {
          await draw(out, args);
          lastKey = key;
        } catch (e) { console.error(`${name} render failed:`, e); }
      } while (again);
    } finally {
      busy = false;
    }
  }

  const refresh = () => {
    if (!window.AF.pyReady) return Promise.resolve();
    if (busy) { again = true; return current; }
    busy = true;
    current = loop();
    return current;
  };
  refresh.invalidate = () => { lastKey = null; };
  return refresh;
}

// ---- Helper: read multiselect values ----
function multiVals(id) {
  const el = document.getElementById(id);
  if (!el) return [];
  return Array.from(el.selectedOptions).map(o => o.value);
}

// ---- Per-tab refresh functions ----

// Args of the last successful Compare render, for the tab-switch memo.
let lastCompareKey = null;

// `wanted`, when given, is the selection to chart even if some of it has no
// <option> yet: "Add to Compare" can name a model the list dropped under an
// earlier, narrower filter, and reading only the DOM silently lost it.
async function refreshCompare(triggered, wanted) {
  if (!window.AF.pyReady) return;
  const [p, q, s, e] = readGlobalFilters();
  const sel = document.getElementById("radar-model-select");
  const inDom = Array.from(sel.selectedOptions).map(o => o.value);
  // Prefer the recency order the onchange handler maintains; fall back to
  // document order for the paths that set .selected programmatically.
  const tracked = (window.AF.compareOrder || []).filter(v => inDom.includes(v));
  let selected = Array.isArray(wanted) ? wanted.slice()
    : tracked.concat(inDom.filter(v => !tracked.includes(v)));
  // A tab switch with the same filters and picks as the last render has
  // nothing new to draw; skip Python and Plotly.react.
  const key = JSON.stringify([p, q, s, e, selected]);
  if (triggered === "tab-switch" && key === lastCompareKey) return;
  if (selected.length > COMPARE_MAX) {
    // Shift-selecting 9 left 8 highlighted while 5 were charted, and the
    // <select> was never corrected — the control disagreed with the chart.
    selected = selected.slice(-COMPARE_MAX);
    Array.from(sel.options).forEach(o => { o.selected = selected.includes(o.value); });
  }
  window.AF.compareOrder = selected;
  try {
    const out = await window.AF.callPy("update_compare", p, q, s, selected, triggered || "", e);
    renderJsonFig("chart-radar", out.figure);
    document.getElementById("compare-raw-table").innerHTML = out.raw_table_html;
    // Sync select options/value only when not triggered by the select itself
    if (triggered !== "radar-model-select") {
      // Built with createElement so scraped labels can never be parsed as markup,
      // and so option.value keeps the exact raw string the selection match needs.
      // Mirrors how this same select is populated at boot.
      sel.replaceChildren(...out.options.map(o => {
        const opt = document.createElement("option");
        opt.value = o.value; opt.textContent = o.label;
        return opt;
      }));
      Array.from(sel.options).forEach(o => { o.selected = out.value.includes(o.value); });
    }
    window.AF.compareOrder = out.value.slice();
    lastCompareKey = JSON.stringify([p, q, s, e, out.value]);
  } catch (e) { console.error("refreshCompare failed:", e); }
}

// The Budget tab answers one question: what is the cheapest model smarter than X.
// Python decides the winner (same frame the chart is built from) and this renders
// it; nothing here re-derives it from the figure.
function renderBudgetAnswer(best, floor) {
  const host = document.getElementById("budget-answer");
  if (!host) return;
  host.replaceChildren();
  if (!floor) { host.style.display = "none"; return; }
  // A late answer must not appear over another tab; showTabControls re-shows
  // the card when Budget is opened again.
  host.style.display = window.AF.state.tab === "budget" ? "block" : "none";

  const card = document.createElement("div");
  card.style.cssText =
    "margin:0 24px 4px;padding:12px 16px;border:1px solid var(--border);" +
    "border-left:2px solid #00d4ff;border-radius:4px;background:var(--bg-card);" +
    "font-family:Inter,sans-serif;";

  const label = document.createElement("div");
  label.style.cssText =
    "font-size:9px;letter-spacing:0.1em;color:#555;font-weight:600;margin-bottom:5px;";
  label.textContent = `CHEAPEST MODEL SCORING ${floor}+`;
  card.appendChild(label);

  if (!best) {
    const none = document.createElement("div");
    none.style.cssText = "font-size:13px;color:#999;";
    none.textContent = "No model in the current filters reaches that score.";
    card.appendChild(none);
    host.appendChild(card);
    return;
  }

  const line = document.createElement("div");
  line.style.cssText = "display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;";

  const name = document.createElement("span");
  name.style.cssText = "font-size:15px;color:#f2f2f2;";
  name.textContent = best.model;                    // textContent: never markup

  const prov = document.createElement("span");
  prov.style.cssText =
    "font-size:11px;color:" + (PROVIDER_COLORS[best.provider] || DEFAULT_PROVIDER_COLOR) + ";";
  prov.textContent = best.provider;

  const cost = document.createElement("span");
  cost.style.cssText = "font-size:15px;color:#00d4ff;margin-left:auto;";
  const c = best.monthly_cost;
  cost.textContent = (c >= 1 ? "$" + c.toFixed(2) : "$" + c.toFixed(3)) + " / mo";

  const meta = document.createElement("span");
  meta.style.cssText = "font-size:11px;color:#777;";
  meta.textContent =
    `${best.quality.toFixed(1)} pts · $${best.price.toFixed(4)}/M tok · ` +
    `${best.n_qualifying} model${best.n_qualifying === 1 ? "" : "s"} qualify`;

  line.append(name, prov, cost);
  card.append(line, meta);
  host.appendChild(card);
}

async function refreshBudget() { return refreshers.budget(); }

async function refreshTable() { return refreshers.table(); }

// Send a blank box through as null and let Python apply the shared default
// (data/local_models.DEFAULT_VRAM_GB). The old `Number(el.value || 32)` did not
// itself swallow a typed 0 — `.value` is a string and "0" is truthy — but it
// hardcoded a fallback that the Python behind it also hardcoded, *differently*:
// 32 here, 8 in static_api. One cleared box, two answers to "which models fit?".
// One default, in Python, is the fix; this function's only job is to say
// "blank" without inventing a number.
function numOrNull(id) {
  const el = document.getElementById(id);
  const raw = el ? el.value : "";
  if (raw === "" || raw === null || raw === undefined) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

// The Run Local hardware, in static_helpers.LOCAL_ARG_NAMES order. One
// builder feeds both update_local and the tab's ↓CSV, so the export describes
// the hardware on screen rather than the default card.
function localArgs() {
  // Same rule as the numeric boxes: send nothing and let Python own the
  // fallback (DEFAULT_BANDWIDTH_GBPS / "nvidia"). A literal bandwidth figure
  // here is a second copy of a constant that lives in data/local_models.py.
  const hw = window.AF.localHwMeta || {};
  const vram = numOrNull("local-vram");
  const gpus = numOrNull("local-num-gpus");
  const quant = document.getElementById("local-quant")?.value || "Q4";
  // numOrNull, never `|| 8192`: a literal numeric fallback on a local-* box is
  // a second copy of a constant that lives in data/local_models.py, and
  // test_neither_rendering_declares_its_own_hardware_default greps for exactly
  // that shape.
  const ctx = numOrNull("local-context");
  const tags = multiVals("local-tags");
  return [vram, gpus, quant, hw.bandwidth_gbps ?? null, hw.hw_type ?? null,
    tags.length ? tags : null, ctx, hw.fp16_tflops ?? null];
}

async function refreshLocal() { return refreshers.local(); }
async function refreshImage() { return refreshers.image(); }
async function refreshVideo() { return refreshers.video(); }
async function refreshRecommend() { return refreshers.recommend(); }

// One coalescing refresher per tab (see makeRefresher). Compare is not here:
// its trigger decides how Python treats the selection, so it keeps its own
// function with a tab-switch memo.
const refreshers = {
  overview: makeRefresher("overview",
    () => {
      const [p, q, s, e] = readGlobalFilters();
      const x = document.querySelector('input[name="overview-xaxis"]:checked')?.value || "price";
      return [p, q, s, x, e];
    },
    (p, q, s, x, e) => window.AF.callPy("update_overview", p, q, s, x, e),
    async (fig) => { await renderJsonFig("chart-pareto", fig); attachParetoClickHandler(); }),

  landscape: makeRefresher("landscape",
    () => readGlobalFilters(),
    async (p, q, s, e) => ({
      treemap: await window.AF.callPy("update_treemap", p, q, s, e),
      leaderboard: await window.AF.callPy("update_provider_leaderboard", p, q, s, e),
    }),
    (out) => {
      renderJsonFig("chart-treemap", out.treemap);
      return renderJsonFig("chart-provider_leaderboard", out.leaderboard);
    }),

  rankings: makeRefresher("rankings",
    () => {
      const [p, q, s, e] = readGlobalFilters();
      const sort = document.querySelector('input[name="rankings-sort"]:checked')?.value || "intelligence";
      return [p, q, s, sort, e];
    },
    // Value Leaders follows the same global filters (Dash's update_value_leaders);
    // the sort toggle does not touch it, but both share one coalesced refresh.
    async (p, q, s, sort, e) => ({
      rankings: await window.AF.callPy("update_rankings", p, q, s, sort, e),
      leaders: await window.AF.callPy("update_value_leaders", p, q, s, e),
    }),
    (out) => {
      renderJsonFig("chart-value_leaders", out.leaders);
      return renderJsonFig("chart-rankings", out.rankings);
    }),

  budget: makeRefresher("budget",
    () => {
      const [p, q, s, e] = readGlobalFilters();
      const tok = Number(document.getElementById("budget-tokens")?.value || 1);
      const floor = Number(document.getElementById("budget-min-intelligence")?.value || 0);
      return [tok, p, q, s, floor, e];
    },
    (tok, p, q, s, floor, e) => window.AF.callPy("update_cost_calc", tok, p, q, s, floor, e),
    (out) => {
      renderBudgetAnswer(out.best, out.floor);
      return renderJsonFig("chart-cost_calc", out.figure);
    }),

  table: makeRefresher("table",
    () => {
      const [p, q, s, e] = readGlobalFilters();
      const col = document.getElementById("table-sort-col").value;
      const dir = document.getElementById("table-sort-dir").value;
      return [p, q, s, col, dir, e];
    },
    (p, q, s, col, dir, e) => window.AF.callPy("update_table", p, q, s, col, dir, e),
    (rows) => renderTableRows(rows)),

  local: makeRefresher("refreshLocal",
    () => [...localArgs(), document.getElementById("local-speed-mode")?.value || null],
    // update_local's late arguments were APPENDED (ctx, speed mode, fp16).
    // pyworker.js spreads this list straight into the Python function with no
    // arity check, so an argument inserted mid-list shifts every one after it
    // and fails silently.
    (vram, gpus, quant, bw, hwType, tags, ctx, fp16, speedMode) =>
      window.AF.callPy("update_local", vram, gpus, quant, bw, hwType, tags, ctx, speedMode, fp16),
    (out) => {
      renderJsonFig("chart-local_scatter", out.scatter);
      return renderJsonFig("chart-local_compat", out.compat);
    }),

  image: makeRefresher("refreshImage",
    () => {
      const providers = multiVals("image-provider-filter");
      const tags = multiVals("image-tag-filter");
      return [providers.length ? providers : null, tags.length ? tags : null];
    },
    (providers, tags) => window.AF.callPy("update_image", providers, tags),
    (fig) => renderJsonFig("chart-image_faceted", fig)),

  video: makeRefresher("refreshVideo",
    () => {
      const providers = multiVals("video-provider-filter");
      const tags = multiVals("video-tag-filter");
      const modeSel = document.getElementById("video-mode-filter");
      return [providers.length ? providers : null, tags.length ? tags : null,
        modeSel && modeSel.value ? modeSel.value : null];
    },
    (providers, tags, mode) => window.AF.callPy("update_video", providers, tags, mode),
    (out) => {
      renderJsonFig("chart-video_rankings", out.rankings);
      return renderJsonFig("chart-video_scatter", out.scatter);
    }),

  recommend: makeRefresher("refreshRecommend",
    () => {
      const mode = document.querySelector('input[name="recommend-mode"]:checked')?.value || "api";
      const providers = Array.from(document.querySelectorAll('input[name="recommend-providers"]:checked')).map(c => c.value);
      const gpu = document.getElementById("recommend-gpu-preset")?.value || "NVIDIA RTX 5090";
      const vram = numOrNull("recommend-vram");
      const gpus = numOrNull("recommend-num-gpus");
      const quant = document.getElementById("recommend-quant")?.value || "Q4";
      return [providers, mode, gpu, vram, gpus, quant];
    },
    (providers, mode, gpu, vram, gpus, quant) =>
      window.AF.callPy("update_recommend", providers, mode, gpu, vram, gpus, quant),
    (out) => {
      document.getElementById("recommend-cards").innerHTML = out.cards_html;
      // Only while Agent Stack is showing: a late answer must not reveal its
      // rows over another tab.
      if (window.AF.state.tab !== "recommend") return;
      const provRow = document.getElementById("recommend-providers-row");
      const hwRow = document.getElementById("recommend-hw-row");
      if (provRow) provRow.style.display = out.show_providers ? "flex" : "none";
      if (hwRow) hwRow.style.display = out.show_hw ? "flex" : "none";
    }),
};

// ---- Build model-table HTML from records ----
function renderTableRows(records) {
  const thead = document.getElementById("model-table-head");
  const tbody = document.getElementById("model-table-body");
  if (!thead || !tbody) return;

  const hStyle = "padding:8px 14px;font-size:9px;letter-spacing:0.08em;color:#555;" +
    "font-family:Inter,sans-serif;font-weight:600;text-transform:uppercase;" +
    "text-align:right;white-space:nowrap;";
  const hStyleLeft = hStyle.replace("text-align:right", "text-align:left");

  if (!thead.hasChildNodes()) {
    thead.innerHTML = `<tr>
      <th style="${hStyleLeft}">Model</th>
      <th style="${hStyleLeft}">Provider</th>
      <th style="${hStyle}">Intelligence</th>
      <th style="${hStyle}">Value (score/$)</th>
      <th style="${hStyle}" title="Our blend, computed from Artificial Analysis's per-token prices: 3 parts output to 1 part input, cheapest available host. Artificial Analysis publishes the opposite weighting; this basis is ours, so their site will quote a lower number.">Price ($/M tok, 3:1)</th>
      <th style="${hStyle}">Speed (tok/s)</th>
      <th style="${hStyle}">Latency (s)</th>
      <th style="${hStyle}">Context</th>
    </tr>`;
  }

  const cellBase = "padding:7px 14px;font-size:11px;font-family:Inter,sans-serif;" +
    "border-bottom:1px solid rgba(255,255,255,0.04);text-align:right;white-space:nowrap;";

  const rows = records.map(r => {
    const pcolor = PROVIDER_COLORS[r.provider] || DEFAULT_PROVIDER_COLOR;
    const qual = r.quality != null ? r.quality.toFixed(1) : "—";
    const val  = r.value  != null && r.value > 0  ? r.value.toFixed(2) : "—";
    const price = r.price != null && r.price > 0   ? "$" + r.price.toFixed(4) : "—";
    // "en-US", not the visitor's locale: a bare toLocaleString renders 1560 as
  // "1.560" in de-DE — a 1000x ambiguity in a column sitting next to $0.1580.
  const speed = r.speed != null && r.speed > 0
    ? Math.round(r.speed).toLocaleString("en-US") : "—";
    const lat   = r.latency != null && r.latency > 0 ? r.latency.toFixed(2) + "s" : "—";
    const ctx   = r.context != null ? escapeHtml(r.context) : "—";

    return `<tr>
      <td style="${cellBase}text-align:left;color:#ccc;max-width:260px;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(r.model)}</td>
      <td style="${cellBase}text-align:left;color:${pcolor};">${escapeHtml(r.provider)}</td>
      <td style="${cellBase}color:#f2f2f2;">${qual}</td>
      <td style="${cellBase}color:#34d399;">${val}</td>
      <td style="${cellBase}color:#888;">${price}</td>
      <td style="${cellBase}color:#888;">${speed}</td>
      <td style="${cellBase}color:#888;">${lat}</td>
      <td style="${cellBase}color:#888;">${ctx}</td>
    </tr>`;
  });
  tbody.innerHTML = rows.join("");
}

// Called by the three global filter controls so the Compare tab can tell a
// real filter change from an incidental re-render.
function rerenderAfterFilterChange(which) {
  window.AF._compareTrigger = which || "filter-provider";
  return rerenderActiveFilterCharts();
}

// ---- Re-render active tab's filter-driven charts ----
async function rerenderActiveFilterCharts() {
  // Read the controls BEFORE the guard. readGlobalFilters is what populates
  // window.AF.state, and it sat one line below this return — so pre-boot the
  // state said {providers: [], minQuality: 0} while the DOM said Anthropic /
  // >= 40, and Share copied a filter-less link AND rewrote the address bar
  // with it. Pyodide is CDN-loaded, so a cold cache widens that window.
  readGlobalFilters();
  // Every tab switch and filter change lands here, so this is where the
  // address bar follows the state — pre-boot included.
  syncUrl();
  if (!window.AF.pyReady) return;
  const tab = window.AF.state.tab;
  if (tab === "compare") {
    // "tab-switch", not "filter-provider": this path runs on every tab change,
    // and passing a filter trigger made Python discard the user's picks and
    // substitute the diverse-5 defaults. Go to Table and back and a curated
    // comparison was gone. Dash has no tabs input on this callback and never
    // did this — the static site drifted.
    const trigger = window.AF._compareTrigger || "tab-switch";
    const wanted = window.AF._compareWanted;
    window.AF._compareTrigger = null;
    window.AF._compareWanted = null;
    await refreshCompare(trigger, wanted || undefined);
  } else if (refreshers[tab]) {
    await refreshers[tab]();
  }
}

// ---- Preset helper ----
// `providers === null` means "leave the provider selection alone".
// "Top 25%" and "Top 10%" are labelled purely in quality terms and sit in the
// same bar as the PROVIDER dropdown, so clearing it made them answer a
// different question than the one asked: PROVIDER=Anthropic + Top 10% silently
// became "top 10% of everything".
function setPreset(minQ, providers, clearSearch) {
  const sel = document.getElementById("filter-quality");
  ensureQualityOption(sel, minQ);
  sel.value = String(minQ);
  if (providers !== null) {
    const pSel = document.getElementById("filter-provider");
    Array.from(pSel.options).forEach(o => { o.selected = providers.includes(o.value); });
  }
  if (clearSearch) document.getElementById("model-search").value = "";
  rerenderAfterFilterChange("filter-quality");
}

// A <select> silently refuses a value it has no option for: assigning 42.1
// left selectedIndex -1 and value "", which Number() read as 0 — the filter
// dropped with nothing to say so. Insert the option instead of snapping.
function ensureQualityOption(sel, value) {
  if (!sel || !Number.isFinite(Number(value))) return;
  const v = Number(value);
  if (Array.from(sel.options).some(o => Number(o.value) === v)) return;
  const opt = document.createElement("option");
  opt.value = String(v);
  opt.textContent = `\u2265 ${v}`;
  const after = Array.from(sel.options).find(o => Number(o.value) > v);
  sel.insertBefore(opt, after || null);
}

// ↓CSV and Share need Python. Say so instead of doing nothing: the export
// button returned silently — no download, no message, no console line — and
// Pyodide is CDN-loaded, so on a cold cache or a blocked jsDelivr that window
// stays open indefinitely.
function setExportPending(pending) {
  const btn = document.getElementById("btn-export");
  if (!btn) return;
  btn.title = pending
    ? "Preparing… the CSV needs the in-browser Python runtime, which is still loading."
    : "Download the data on screen as CSV";
  btn.textContent = pending ? "↓ CSV  ·  loading…" : "↓ CSV";
  btn.disabled = !!pending;
  btn.style.opacity = pending ? "0.5" : "";
}

// ---- Wire all global controls ----
function wireGlobalControls() {
  // Each control names itself, so the Compare tab can tell a real filter change
  // from an incidental re-render and stop discarding the user's picks.
  const onFilter = id => debounce(() => rerenderAfterFilterChange(id), 200);
  document.getElementById("filter-provider").onchange = onFilter("filter-provider");
  document.getElementById("filter-quality").onchange = onFilter("filter-quality");
  const effEl = document.getElementById("filter-effort");
  if (effEl) effEl.onchange = onFilter("filter-effort");
  document.getElementById("model-search").oninput = onFilter("model-search");
  // "Reset filters" clears everything; the two quality presets touch quality only.
  document.getElementById("preset-all").onclick = () => setPreset(0, [], true);
  document.getElementById("preset-strong").onclick = () => setPreset(window.AF.manifest.p75, null, false);
  document.getElementById("preset-elite").onclick = () => setPreset(window.AF.manifest.p90, null, false);

  // CSV export — get filtered CSV text from Python, trigger download.
  document.getElementById("btn-export").onclick = async () => {
    if (!window.AF.pyReady) {
      // Returned silently: no download, no message, no console output.
      setExportPending(true);
      return;
    }
    const [p, q, s, e] = readGlobalFilters();
    try {
      // Export what is on screen. This always sent the hosted-LLM catalogue
      // through the global filters, so ↓CSV on Image Gen handed back the LLM
      // header and seven text models — a different dataset from the chart.
      // EFFORT and the Run Local hardware ride along too: the effort was read
      // and dropped (113 rows on screen, 195 in the file), and Run Local
      // exported "fits" for the default 32 GB card whatever was selected.
      const tab = window.AF.state.tab;
      const local = tab === "local" ? localArgs() : null;
      const csv = await window.AF.callPyRaw("export_csv", p, q, s, tab, e, local);
      const name = await window.AF.callPyRaw("export_csv_filename", tab);
      const blob = new Blob([csv], { type: "text/csv" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = name || "ai_frontier_export.csv"; a.click();
    } catch (e) { console.error("export_csv failed:", e); }
  };

  // Share — copy the current state's URL (?tab=&p=&q=&e=)
  document.getElementById("btn-share").onclick = () => {
    readGlobalFilters();
    const url = syncUrl();
    navigator.clipboard?.writeText(url);
  };

  document.getElementById("btn-refresh").onclick = doRefresh;
}

// ---- Keep the URL in step with the state ----
// Mirrors Dash's url-sync clientside callback: tab, p, q and e. The URL used to
// change only on Share, so ⟳ (which navigates to location.href) and a plain
// reload both dropped the tab and every filter, and EFFORT was never written.
// Parameters this function does not own (the ⟳ cache-buster `v`) are kept.
function stateUrl() {
  const { tab, providers, minQuality, effort } = window.AF.state;
  const params = new URLSearchParams(location.search);
  ["tab", "p", "q", "e"].forEach(k => params.delete(k));
  if (tab) params.set("tab", tab);
  if (providers && providers.length) params.set("p", providers.join(","));
  if (minQuality > 0) params.set("q", minQuality);
  if (effort) params.set("e", effort);
  const qs = params.toString();
  return location.origin + location.pathname + (qs ? "?" + qs : "");
}

function syncUrl() {
  const url = stateUrl();
  if (url !== location.href) {
    try { history.replaceState(null, "", url); } catch (e) { console.warn("url sync:", e); }
  }
  return url;
}

// ---- Restore state from URL on page load ----
function applyUrlState() {
  const u = new URLSearchParams(location.search);
  if (u.get("q")) {
    const qEl = document.getElementById("filter-quality");
    const q = Number(u.get("q"));
    if (qEl && Number.isFinite(q)) {
      ensureQualityOption(qEl, q);   // ?q=42.1 used to select nothing at all
      qEl.value = String(q);
    }
  }
  if (u.get("p")) {
    // Resolve retired provider spellings before matching. A ?p=xAI link shared
    // before Artificial Analysis renamed the provider selected no option at
    // all, and an empty selection means "all providers" — so the link silently
    // showed the whole catalogue instead of the one provider it named. The
    // alias map rides the manifest so it stays derived from PROVIDER_ALIASES
    // rather than hand-copied here.
    const aliases = (window.AF.manifest && window.AF.manifest.provider_aliases) || {};
    const set = new Set(u.get("p").split(",").map(p => aliases[p] || p));
    const pEl = document.getElementById("filter-provider");
    if (pEl) Array.from(pEl.options).forEach(o => { o.selected = set.has(o.value); });
  }
  if (u.has("e")) {
    // A URL is user input: only an effort the control offers is applied, and
    // anything else falls back to "All variants" — the same rule as Dash's
    // init_from_url.
    const effEl = document.getElementById("filter-effort");
    const valid = ((window.AF.manifest && window.AF.manifest.effort_options) || []).map(o => o.value);
    const want = u.get("e");
    if (effEl) effEl.value = valid.includes(want) ? want : "";
  }
  readGlobalFilters();
  if (u.get("tab")) switchTab(u.get("tab"));   // switchTab validates
}

// ---- Detail panel — Plotly click on pareto → model_detail HTML ----
function wireDetailPanel() {
  document.getElementById("detail-close").onclick = () =>
    document.getElementById("detail-panel").className = "detail-panel";
  document.getElementById("detail-add-compare").onclick = async () => {
    const m = window.AF.detailModel; if (!m) return;
    const sel = document.getElementById("radar-model-select");
    const chosen = Array.from(sel.selectedOptions).map(o => o.value);
    const ordered = (window.AF.compareOrder || []).filter(v => chosen.includes(v));
    const base = ordered.length ? ordered : chosen;
    // Evict the oldest pick rather than refusing the new one. Both renderings
    // default to exactly 5 models, so on a fresh page load this button — the
    // detail panel's only call to action — was a silent no-op that still
    // switched tabs.
    const keep = base.includes(m) ? base : base.slice(-(COMPARE_MAX - 1)).concat([m]);
    window.AF.compareOrder = keep;
    // Hand the selection over explicitly and go through switchTab. Only
    // setting .selected did nothing when the option list was stale (built
    // under a narrower filter, so `m` had no <option>) while still evicting a
    // pick; and toggling panels by hand skipped switchTab's resize, leaving
    // the radar at 700 px with the detail panel still open over it. The
    // tab-switch trigger re-syncs the options from the current filters.
    window.AF._compareWanted = keep;
    switchTab("compare");
  };
  attachParetoClickHandler();
}

// Attach (or re-attach) plotly_click on the Pareto chart.
// Must be called after every Plotly.react on that div because react() clears handlers.
function attachParetoClickHandler() {
  const div = document.getElementById("chart-pareto");
  if (!div || !div.on) return;
  div.removeAllListeners && div.removeAllListeners("plotly_click");
  div.on("plotly_click", async (ev) => {
    if (!window.AF.pyReady) return;
    const cd = ev.points?.[0]?.customdata;
    if (!cd) return;
    // cd[0]/cd[1] are HTML-escaped for the hover template; the raw model and
    // provider ride at cd[5]/cd[6], and model_detail looks up by the raw name
    // ("A&B" escaped to "A&amp;B" matches nothing). Older figures lack them.
    const model = Array.isArray(cd) ? (cd[5] ?? cd[0]) : cd;
    const provider = Array.isArray(cd) ? (cd[6] ?? cd[1]) : "";
    try {
      const html = await window.AF.callPyRaw("model_detail", model, provider);
      if (!html) return;
      document.getElementById("detail-panel-body").innerHTML = html;
      document.getElementById("detail-panel").className = "detail-panel open";
      window.AF.detailModel = model;
    } catch (e) { console.error("model_detail failed:", e); }
  });
}

// ---- Overview caption text (mirrors app.py) ----
const OVERVIEW_CAPTIONS = {
  price: "Each bubble is one model FAMILY — the best-scoring variant, so five Claude Opus 5 effort tiers collapse to one point rather than five near-duplicates at the same price. X = price per 1M tokens, blended 3:1 output:input, our weighting not AA's (log scale), Y = AA Intelligence Index. Bubble size = throughput (tok/s). Dotted line = Pareto frontier. Click any bubble for full details.",
  speed: "Speed (tok/s) vs. AA Intelligence Index, one bubble per model family (best-scoring variant). Top-right = fast and smart. Bubble size = affordability (larger = cheaper, log-scaled). Click any bubble for full details.",
};
function updateOverviewCaption() {
  const x = document.querySelector('input[name="overview-xaxis"]:checked')?.value || "price";
  const el = document.getElementById("overview-desc");
  if (el) el.textContent = OVERVIEW_CAPTIONS[x] || OVERVIEW_CAPTIONS.price;
}

// ---- Wire per-tab controls ----
function wireTabControls() {
  // Overview X-axis
  document.querySelectorAll('input[name="overview-xaxis"]').forEach(r => {
    r.onchange = async () => {
      // Pre-boot, rerenderActiveFilterCharts returns early — so the caption
      // flipped to "Bubble size = affordability (larger = cheaper)" over a
      // chart whose x-axis still read "Price (USD / 1M tokens)": an exactly
      // inverted reading of the same picture. The Speed view has a pre-built
      // figure (docs/figures/quadrant.json) that was never fetched; use it, and
      // only then move the caption.
      const x = document.querySelector('input[name="overview-xaxis"]:checked')?.value || "price";
      if (!window.AF.pyReady) {
        try {
          await renderFigure("chart-pareto", x === "speed" ? "quadrant" : "pareto");
        } catch (e) { console.error("pre-boot overview render failed:", e); }
      }
      updateOverviewCaption();
      rerenderActiveFilterCharts();
    };
  });

  // Rankings sort
  document.querySelectorAll('input[name="rankings-sort"]').forEach(r => {
    r.onchange = () => rerenderActiveFilterCharts();
  });

  // Compare model select
  const radarSel = document.getElementById("radar-model-select");
  if (radarSel) {
    radarSel.onchange = () => {
      // Track WHEN each model was picked, not where it sits in the list.
      // selectedOptions is document order, so `opts[opts.length - 1]` was the
      // bottom-most option, not the newest: picking a 6th model below the
      // current five silently deselected the one just clicked (a no-op), and
      // picking one above it made an unrelated model vanish instead.
      const now = Array.from(radarSel.selectedOptions).map(o => o.value);
      const previous = (window.AF.compareOrder || []).filter(v => now.includes(v));
      const added = now.filter(v => !previous.includes(v));
      let order = previous.concat(added);
      if (order.length > COMPARE_MAX) order = order.slice(-COMPARE_MAX);
      window.AF.compareOrder = order;
      Array.from(radarSel.options).forEach(o => { o.selected = order.includes(o.value); });
      refreshCompare("radar-model-select");
    };
  }

  // Budget tokens
  const budgetInput = document.getElementById("budget-tokens");
  if (budgetInput) {
    budgetInput.oninput = debounce(() => rerenderActiveFilterCharts(), 300);
  }

  // Budget minimum intelligence. The readout tracks the drag immediately; the
  // re-render is debounced so dragging does not queue a call per pixel.
  const minIntel = document.getElementById("budget-min-intelligence");
  const minIntelOut = document.getElementById("budget-min-intelligence-value");
  if (minIntel) {
    const debouncedBudget = debounce(() => rerenderActiveFilterCharts(), 200);
    minIntel.oninput = () => {
      if (minIntelOut) minIntelOut.textContent = minIntel.value;
      debouncedBudget();
    };
  }

  // Table sort controls
  const tableSortCol = document.getElementById("table-sort-col");
  const tableSortDir = document.getElementById("table-sort-dir");
  if (tableSortCol) tableSortCol.onchange = () => refreshTable();
  if (tableSortDir) tableSortDir.onchange = () => refreshTable();

  // Every Run Local control goes through ONE 150 ms debounce into the
  // coalescing refresher, so arrowing through a select or ctrl-clicking three
  // tags is one update_local, not one per change.
  const scheduleLocal = debounce(() => refreshLocal(), LOCAL_DEBOUNCE_MS);

  // Local GPU preset — resolved from the table gpu_options shipped at boot, in
  // this tick: localHwMeta and the VRAM box change together with the dropdown,
  // before any refresh can read them. This is a real preset change, so it is
  // the one place allowed to overwrite a typed VRAM figure.
  const localGpu = document.getElementById("local-gpu-preset");
  if (localGpu) {
    localGpu.onchange = () => {
      const hw = window.AF.gpuMeta[localGpu.value];
      if (hw) {
        window.AF.localHwMeta = hw;
        setVramFromPreset("local-vram", hw, true);
      }
      scheduleLocal();
    };
  }
  // Local VRAM, num GPUs, quant, context, speed, tags
  const localVram = document.getElementById("local-vram");
  if (localVram) {
    localVram.oninput = () => { localVram.dataset.userTyped = "1"; scheduleLocal(); };
  }
  ["local-num-gpus", "local-quant", "local-context", "local-speed-mode", "local-tags"]
    .forEach(id => {
      const el = document.getElementById(id);
      if (el) el.onchange = scheduleLocal;
    });

  // Image Gen filters
  const scheduleImage = debounce(() => refreshImage(), LOCAL_DEBOUNCE_MS);
  ["image-provider-filter", "image-tag-filter"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.onchange = scheduleImage;
  });

  // Video Gen filters
  const scheduleVideo = debounce(() => refreshVideo(), LOCAL_DEBOUNCE_MS);
  ["video-mode-filter", "video-provider-filter", "video-tag-filter"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.onchange = scheduleVideo;
  });

  // Agent Stack mode radios — update row visibility immediately (pre-boot), then refresh data
  const scheduleRecommend = debounce(() => refreshRecommend(), LOCAL_DEBOUNCE_MS);
  document.querySelectorAll('input[name="recommend-mode"]').forEach(r => {
    r.onchange = () => { showTabControls("recommend"); scheduleRecommend(); };
  });
  // Agent Stack providers checkboxes
  document.querySelectorAll('input[name="recommend-providers"]').forEach(c => {
    c.onchange = scheduleRecommend;
  });
  // Agent Stack hardware controls
  const recGpu = document.getElementById("recommend-gpu-preset");
  const recVram = document.getElementById("recommend-vram");
  const recNumGpus = document.getElementById("recommend-num-gpus");
  const recQuant = document.getElementById("recommend-quant");
  if (recGpu) recGpu.onchange = () => {
    // Sync VRAM to the selected preset so the 'fits' filter uses the real
    // hardware capacity (mirrors the Local tab + Dash update_recommend_hw).
    // Same in-tick table lookup as the Local preset.
    setVramFromPreset("recommend-vram", window.AF.gpuMeta[recGpu.value], true);
    scheduleRecommend();
  };
  if (recVram) {
    recVram.oninput = () => { recVram.dataset.userTyped = "1"; scheduleRecommend(); };
  }
  if (recNumGpus) recNumGpus.onchange = scheduleRecommend;
  if (recQuant) recQuant.onchange = scheduleRecommend;
}

async function init() {
  buildTabsAndPanels();
  // Inject overview caption just above the chart-pareto card (matches Dash app layout)
  const overviewPanel = document.getElementById("panel-overview");
  if (overviewPanel) {
    const cap = document.createElement("p");
    cap.id = "overview-desc";
    cap.className = "chart-caption";
    overviewPanel.insertBefore(cap, overviewPanel.firstChild);
  }
  await loadManifest();
  // Pyodide first: its boot only needs the manifest version, and it used to
  // wait for all twelve figures to be fetched and drawn. The worker's "ready"
  // handler waits on uiReady before touching any control.
  let markUiReady;
  window.AF.uiReady = new Promise(r => { markUiReady = r; });
  bootPyodide();      // fire-and-forget; pyReady gate protects filter calls
  try {
    renderFreshness();
    setInterval(renderFreshness, 60000);
    // Show controls for the initial tab
    showTabControls("overview");
    wireGlobalControls();
    setExportPending(!window.AF.pyReady);
    wireTabControls();
    updateOverviewCaption();  // set caption for initial price x-axis
    wireDetailPanel();  // wires close/add-compare immediately; pareto click re-attached after each render
    applyUrlState();    // restore tab + filters from URL params (pure JS, no Pyodide needed)
  } finally {
    markUiReady();      // never strand the worker's ready handler
  }
  // Only the visible tab's pre-rendered figures; the rest load on first visit
  // (switchTab), into a panel that is visible and so measured correctly.
  await loadStaticFigures(window.AF.state.tab);
}
init().catch(err => {
  const s = document.getElementById("py-status");
  if (s) { s.style.display = "block"; s.textContent = "Failed to load: " + err.message; }
});
