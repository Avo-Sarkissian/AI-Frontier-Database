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

// window.AF.state contract: { providers: string[], minQuality: number, search: string, tab: string }
window.AF = { pyReady: false, figCache: {}, manifest: null, localHwMeta: null, state: {
  providers: [], minQuality: 0, search: "", tab: "overview" } };

const PLOT_CONFIG = { displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage"] };

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js";

// ---- Provider color map (mirrors PROVIDER_COLORS in constants.py) ----
const PROVIDER_COLORS = {
  "Anthropic": "#c084fc",
  "OpenAI": "#34d399",
  "Google": "#60a5fa",
  "Meta": "#fb923c",
  "DeepSeek": "#f472b6",
  "Mistral": "#facc15",
  "xAI": "#a3e635",
  "Alibaba": "#38bdf8",
  "Amazon": "#ff9900",
  "NVIDIA": "#22d3ee",
  "Microsoft Azure": "#818cf8",
  "Cohere": "#f87171",
  "Kimi": "#d4a1f5",
  "Z AI": "#7dd3fc",
  "MiniMax": "#86efac",
  "InclusionAI": "#fca5a5",
  "Xiaomi": "#6ee7b7",
  "Baidu": "#fde68a",
  "IBM": "#93c5fd",
  "LG AI Research": "#c4b5fd",
  "Nous Research": "#f9a8d4",
  "Reka AI": "#a78bfa",
  "AI21 Labs": "#34d399",
  "Allen Institute for AI": "#67e8f9",
  "Inception": "#fb7185",
  "Upstage": "#fbbf24",
  "Perplexity": "#a3a3a3",
  "KwaiKAT": "#f97316",
  "Deep Cogito": "#a8a29e",
};
const DEFAULT_PROVIDER_COLOR = "#555555";

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

// Show/hide per-tab control rows
function showTabControls(id) {
  // Hide all tab control rows first
  document.querySelectorAll('[id^="tab-controls-"]').forEach(el => {
    el.style.display = "none";
  });
  // Hide recommend-specific rows
  const provRow = document.getElementById("recommend-providers-row");
  const hwRow = document.getElementById("recommend-hw-row");
  if (provRow) provRow.style.display = "none";
  if (hwRow) hwRow.style.display = "none";

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

function switchTab(id) {
  window.AF.state.tab = id;
  document.querySelectorAll(".tab").forEach(b =>
    b.classList.toggle("tab--selected", b.dataset.tab === id));
  TABS.forEach(t => {
    document.getElementById("panel-" + t.id).style.display = t.id === id ? "block" : "none";
  });
  showTabControls(id);
  // Plotly needs a resize when a hidden plot becomes visible.
  setTimeout(() => document.querySelectorAll("#panel-" + id + " .js-plotly-plot")
    .forEach(el => { try { Plotly.Plots.resize(el); } catch (e) { console.warn("resize:", e); } }), 60);
  // Re-render active tab with current filters once Pyodide is ready
  rerenderActiveFilterCharts();
}

async function renderFigure(divId, figId) {
  let fig = window.AF.figCache[figId];
  if (!fig) {
    const r = await fetch(`figures/${figId}.json`);
    fig = await r.json();
    window.AF.figCache[figId] = fig;
  }
  Plotly.react(divId, fig.data, fig.layout, PLOT_CONFIG);
}

async function loadManifest() {
  const m = await (await fetch("figures/manifest.json")).json();
  window.AF.manifest = m;
  document.getElementById("stat-model-count").textContent = m.model_count;
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
}

// ---- Pyodide boot ----

async function bootPyodide() {
  const statusEl = document.getElementById("py-status");
  if (statusEl) { statusEl.style.display = "block"; statusEl.textContent = "warming up interactivity…"; }

  try {
    await new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = PYODIDE_URL; s.onload = res; s.onerror = rej; document.head.appendChild(s);
    });
    const pyodide = await loadPyodide();
    if (statusEl) statusEl.textContent = "loading packages…";
    await pyodide.loadPackage(["pandas", "numpy", "narwhals"]);
    if (statusEl) statusEl.textContent = "loading bundle…";
    const buf = await (await fetch("pybundle.zip")).arrayBuffer();
    pyodide.unpackArchive(buf, "zip", { extractDir: "/bundle" });
    await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/bundle")
import importlib.metadata as _im
_orig_version = _im.version
def _version_shim(name):
    _VERSIONS = {"plotly": "6.5.2", "narwhals": "1.0.0", "tenacity": "8.2.3"}
    if name in _VERSIONS:
        return _VERSIONS[name]
    return _orig_version(name)
_im.version = _version_shim
import static_api
`);
    window.AF.pyodide = pyodide;
    window.AF.callPy = async (fn, ...args) => {
      const py = pyodide.globals.get("static_api");
      const res = py[fn](...args);
      const s = res.toString();
      if (typeof res.destroy === "function") res.destroy();
      return JSON.parse(s);
    };
    window.AF.callPyRaw = async (fn, ...args) => {
      const py = pyodide.globals.get("static_api");
      const res = py[fn](...args);
      const s = res.toString();
      if (typeof res.destroy === "function") res.destroy();
      return s;
    };
    window.AF.pyReady = true;
    if (statusEl) statusEl.style.display = "none";
    await populateDynamicSelects();
    rerenderActiveFilterCharts();
  } catch (err) {
    console.error("Pyodide boot failed:", err);
    if (statusEl) { statusEl.style.display = "block"; statusEl.textContent = "interactivity unavailable: " + err.message; }
  }
}

// ---- Populate selects that need data from Python ----
async function populateDynamicSelects() {
  try {
    const gpuOptions = await window.AF.callPy("gpu_options");
    const quantLevels = await window.AF.callPy("quant_levels");
    const DEFAULT_GPU = "NVIDIA RTX 5090";
    const DEFAULT_QUANT = "Q4";

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
    // local-quant
    const localQuant = document.getElementById("local-quant");
    if (localQuant) {
      localQuant.innerHTML = "";
      quantLevels.forEach(q => {
        const opt = document.createElement("option");
        opt.value = q; opt.textContent = q;
        if (q === DEFAULT_QUANT) opt.selected = true;
        localQuant.appendChild(opt);
      });
    }
    // recommend-quant
    const recQuant = document.getElementById("recommend-quant");
    if (recQuant) {
      recQuant.innerHTML = "";
      quantLevels.forEach(q => {
        const opt = document.createElement("option");
        opt.value = q; opt.textContent = q;
        if (q === DEFAULT_QUANT) opt.selected = true;
        recQuant.appendChild(opt);
      });
    }

    // Set default local HW meta for RTX 5090
    try {
      const hw = await window.AF.callPy("local_hw_for_gpu", DEFAULT_GPU);
      if (hw) {
        window.AF.localHwMeta = hw;
        const vramInput = document.getElementById("local-vram");
        if (vramInput) vramInput.value = hw.vram_gb;
      }
    } catch (e) { console.warn("local_hw_for_gpu init:", e); }

  } catch (e) {
    console.error("populateDynamicSelects failed:", e);
  }
}

// ---- Debounce helper ----
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

// ---- Read current filter state ----
function readGlobalFilters() {
  const providers = Array.from(document.getElementById("filter-provider").selectedOptions).map(o => o.value);
  const minQuality = Number(document.getElementById("filter-quality").value);
  const search = document.getElementById("model-search").value;
  Object.assign(window.AF.state, { providers, minQuality, search });
  return [providers, minQuality, search];
}

async function renderJsonFig(divId, figObj) { Plotly.react(divId, figObj.data, figObj.layout, PLOT_CONFIG); }

// ---- Helper: read multiselect values ----
function multiVals(id) {
  const el = document.getElementById(id);
  if (!el) return [];
  return Array.from(el.selectedOptions).map(o => o.value);
}

// ---- Per-tab refresh functions ----

async function refreshCompare(triggered) {
  if (!window.AF.pyReady) return;
  const [p, q, s] = readGlobalFilters();
  const sel = document.getElementById("radar-model-select");
  const selected = Array.from(sel.selectedOptions).map(o => o.value);
  try {
    const out = await window.AF.callPy("update_compare", p, q, s, selected, triggered || "");
    renderJsonFig("chart-radar", out.figure);
    document.getElementById("compare-raw-table").innerHTML = out.raw_table_html;
    // Sync select options/value only when not triggered by the select itself
    if (triggered !== "radar-model-select") {
      sel.innerHTML = out.options.map(o => `<option value="${o.value}">${o.label}</option>`).join("");
      Array.from(sel.options).forEach(o => { o.selected = out.value.includes(o.value); });
    }
  } catch (e) { console.error("refreshCompare failed:", e); }
}

async function refreshTable() {
  if (!window.AF.pyReady) return;
  const [p, q, s] = readGlobalFilters();
  const col = document.getElementById("table-sort-col").value;
  const dir = document.getElementById("table-sort-dir").value;
  try {
    const rows = await window.AF.callPy("update_table", p, q, s, col, dir);
    renderTableRows(rows);
  } catch (e) { console.error("refreshTable failed:", e); }
}

async function refreshLocal() {
  if (!window.AF.pyReady) return;
  const hw = window.AF.localHwMeta || { bandwidth_gbps: 1792, hw_type: "nvidia" };
  const vram = Number(document.getElementById("local-vram").value || 32);
  const gpus = Number(document.getElementById("local-num-gpus").value || 1);
  const quant = document.getElementById("local-quant").value || "Q4";
  const tags = multiVals("local-tags");
  try {
    const out = await window.AF.callPy("update_local", vram, gpus, quant, hw.bandwidth_gbps, hw.hw_type, tags.length ? tags : null);
    renderJsonFig("chart-local_scatter", out.scatter);
    renderJsonFig("chart-local_compat", out.compat);
  } catch (e) { console.error("refreshLocal failed:", e); }
}

async function refreshImage() {
  if (!window.AF.pyReady) return;
  const providers = multiVals("image-provider-filter");
  const tags = multiVals("image-tag-filter");
  try {
    renderJsonFig("chart-image_faceted", await window.AF.callPy("update_image", providers.length ? providers : null, tags.length ? tags : null));
  } catch (e) { console.error("refreshImage failed:", e); }
}

async function refreshVideo() {
  if (!window.AF.pyReady) return;
  const providers = multiVals("video-provider-filter");
  const tags = multiVals("video-tag-filter");
  try {
    const out = await window.AF.callPy("update_video", providers.length ? providers : null, tags.length ? tags : null);
    renderJsonFig("chart-video_rankings", out.rankings);
    renderJsonFig("chart-video_scatter", out.scatter);
  } catch (e) { console.error("refreshVideo failed:", e); }
}

async function refreshRecommend() {
  if (!window.AF.pyReady) return;
  const mode = document.querySelector('input[name="recommend-mode"]:checked')?.value || "api";
  const providers = Array.from(document.querySelectorAll('input[name="recommend-providers"]:checked')).map(c => c.value);
  const gpu = document.getElementById("recommend-gpu-preset")?.value || "NVIDIA RTX 5090";
  const vram = Number(document.getElementById("recommend-vram")?.value || 32);
  const gpus = Number(document.getElementById("recommend-num-gpus")?.value || 1);
  const quant = document.getElementById("recommend-quant")?.value || "Q4";
  try {
    const out = await window.AF.callPy("update_recommend", providers, mode, gpu, vram, gpus, quant);
    document.getElementById("recommend-cards").innerHTML = out.cards_html;
    const provRow = document.getElementById("recommend-providers-row");
    const hwRow = document.getElementById("recommend-hw-row");
    if (provRow) provRow.style.display = out.show_providers ? "flex" : "none";
    if (hwRow) hwRow.style.display = out.show_hw ? "flex" : "none";
  } catch (e) { console.error("refreshRecommend failed:", e); }
}

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
      <th style="${hStyle}">Price ($/M tok)</th>
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
    const speed = r.speed != null && r.speed > 0   ? Math.round(r.speed).toLocaleString() : "—";
    const lat   = r.latency != null && r.latency > 0 ? r.latency.toFixed(2) + "s" : "—";
    const ctx   = r.context != null ? String(r.context) : "—";

    return `<tr>
      <td style="${cellBase}text-align:left;color:#ccc;max-width:260px;overflow:hidden;text-overflow:ellipsis;">${r.model}</td>
      <td style="${cellBase}text-align:left;color:${pcolor};">${r.provider}</td>
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

// ---- Re-render active tab's filter-driven charts ----
async function rerenderActiveFilterCharts() {
  if (!window.AF.pyReady) return;
  const [p, q, s] = readGlobalFilters();
  const tab = window.AF.state.tab;
  if (tab === "overview") {
    const x = document.querySelector('input[name="overview-xaxis"]:checked')?.value || "price";
    try {
      renderJsonFig("chart-pareto", await window.AF.callPy("update_overview", p, q, s, x));
    } catch (e) { console.error("overview render failed:", e); }
  } else if (tab === "landscape") {
    try {
      renderJsonFig("chart-treemap", await window.AF.callPy("update_treemap", p, q, s));
      renderJsonFig("chart-provider_leaderboard", await window.AF.callPy("update_provider_leaderboard", p, q, s));
    } catch (e) { console.error("landscape render failed:", e); }
  } else if (tab === "rankings") {
    const sort = document.querySelector('input[name="rankings-sort"]:checked')?.value || "intelligence";
    try {
      renderJsonFig("chart-rankings", await window.AF.callPy("update_rankings", p, q, s, sort));
      renderJsonFig("chart-value_leaders", await window.AF.callPy("update_value_leaders", p, q, s));
    } catch (e) { console.error("rankings render failed:", e); }
  } else if (tab === "compare") {
    await refreshCompare("filter-provider");
  } else if (tab === "budget") {
    const tok = Number(document.getElementById("budget-tokens")?.value || 1);
    try {
      renderJsonFig("chart-cost_calc", await window.AF.callPy("update_cost_calc", tok, p, q, s));
    } catch (e) { console.error("budget render failed:", e); }
  } else if (tab === "table") {
    await refreshTable();
  } else if (tab === "local") {
    await refreshLocal();
  } else if (tab === "image") {
    await refreshImage();
  } else if (tab === "video") {
    await refreshVideo();
  } else if (tab === "recommend") {
    await refreshRecommend();
  }
}

// ---- Preset helper ----
function setPreset(minQ, providers) {
  const sel = document.getElementById("filter-quality");
  const opts = Array.from(sel.options).map(o => Number(o.value));
  const snapped = opts.reduce((best, v) => (Math.abs(v - minQ) < Math.abs(best - minQ) ? v : best), opts[0]);
  sel.value = String(snapped);
  const pSel = document.getElementById("filter-provider");
  Array.from(pSel.options).forEach(o => { o.selected = providers.includes(o.value); });
  rerenderActiveFilterCharts();
}

// ---- Wire all global controls ----
function wireGlobalControls() {
  const trigger = debounce(rerenderActiveFilterCharts, 200);
  document.getElementById("filter-provider").onchange = trigger;
  document.getElementById("filter-quality").onchange = trigger;
  document.getElementById("model-search").oninput = trigger;
  document.getElementById("preset-all").onclick = () => setPreset(0, []);
  document.getElementById("preset-strong").onclick = () => setPreset(window.AF.manifest.p75, []);
  document.getElementById("preset-elite").onclick = () => setPreset(window.AF.manifest.p90, []);
}

// ---- Wire per-tab controls ----
function wireTabControls() {
  // Overview X-axis
  document.querySelectorAll('input[name="overview-xaxis"]').forEach(r => {
    r.onchange = () => rerenderActiveFilterCharts();
  });

  // Rankings sort
  document.querySelectorAll('input[name="rankings-sort"]').forEach(r => {
    r.onchange = () => rerenderActiveFilterCharts();
  });

  // Compare model select
  const radarSel = document.getElementById("radar-model-select");
  if (radarSel) {
    radarSel.onchange = () => {
      // Cap at 5
      const opts = Array.from(radarSel.selectedOptions);
      if (opts.length > 5) {
        const last = opts[opts.length - 1];
        last.selected = false;
      }
      refreshCompare("radar-model-select");
    };
  }

  // Budget tokens
  const budgetInput = document.getElementById("budget-tokens");
  if (budgetInput) {
    budgetInput.oninput = debounce(() => rerenderActiveFilterCharts(), 300);
  }

  // Table sort controls
  const tableSortCol = document.getElementById("table-sort-col");
  const tableSortDir = document.getElementById("table-sort-dir");
  if (tableSortCol) tableSortCol.onchange = () => refreshTable();
  if (tableSortDir) tableSortDir.onchange = () => refreshTable();

  // Local GPU preset
  const localGpu = document.getElementById("local-gpu-preset");
  if (localGpu) {
    localGpu.onchange = async () => {
      try {
        const hw = await window.AF.callPy("local_hw_for_gpu", localGpu.value);
        if (hw) {
          window.AF.localHwMeta = hw;
          const vramInput = document.getElementById("local-vram");
          if (vramInput) vramInput.value = hw.vram_gb;
        }
        refreshLocal();
      } catch (e) { console.error("local_hw_for_gpu change:", e); }
    };
  }
  // Local VRAM, num GPUs, quant, tags
  const localVram = document.getElementById("local-vram");
  const localNumGpus = document.getElementById("local-num-gpus");
  const localQuant = document.getElementById("local-quant");
  const localTags = document.getElementById("local-tags");
  if (localVram) localVram.oninput = debounce(() => refreshLocal(), 300);
  if (localNumGpus) localNumGpus.onchange = () => refreshLocal();
  if (localQuant) localQuant.onchange = () => refreshLocal();
  if (localTags) localTags.onchange = () => refreshLocal();

  // Image Gen filters
  const imgProvider = document.getElementById("image-provider-filter");
  const imgTags = document.getElementById("image-tag-filter");
  if (imgProvider) imgProvider.onchange = () => refreshImage();
  if (imgTags) imgTags.onchange = () => refreshImage();

  // Video Gen filters
  const vidProvider = document.getElementById("video-provider-filter");
  const vidTags = document.getElementById("video-tag-filter");
  if (vidProvider) vidProvider.onchange = () => refreshVideo();
  if (vidTags) vidTags.onchange = () => refreshVideo();

  // Agent Stack mode radios
  document.querySelectorAll('input[name="recommend-mode"]').forEach(r => {
    r.onchange = () => refreshRecommend();
  });
  // Agent Stack providers checkboxes
  document.querySelectorAll('input[name="recommend-providers"]').forEach(c => {
    c.onchange = () => refreshRecommend();
  });
  // Agent Stack hardware controls
  const recGpu = document.getElementById("recommend-gpu-preset");
  const recVram = document.getElementById("recommend-vram");
  const recNumGpus = document.getElementById("recommend-num-gpus");
  const recQuant = document.getElementById("recommend-quant");
  if (recGpu) recGpu.onchange = () => refreshRecommend();
  if (recVram) recVram.oninput = debounce(() => refreshRecommend(), 300);
  if (recNumGpus) recNumGpus.onchange = () => refreshRecommend();
  if (recQuant) recQuant.onchange = () => refreshRecommend();
}

async function init() {
  buildTabsAndPanels();
  await loadManifest();
  // Show controls for the initial tab
  showTabControls("overview");
  // Load static figures for all chart tabs
  const chartLoads = TABS.flatMap(t => t.charts.map(c => renderFigure("chart-" + c, c)));
  await Promise.all(chartLoads);
  wireGlobalControls();
  wireTabControls();
  bootPyodide();  // fire-and-forget; pyReady gate protects filter calls
}
init().catch(err => {
  const s = document.getElementById("py-status");
  if (s) { s.style.display = "block"; s.textContent = "Failed to load: " + err.message; }
});
