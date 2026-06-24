// ---- Tab + panel definitions (chart container ids match figures/<id>.json) ----
const TABS = [
  { id: "overview",  label: "Overview",    charts: ["pareto"] },
  { id: "recommend", label: "Agent Stack", charts: [] },           // cards, Phase 4
  { id: "landscape", label: "Landscape",   charts: ["treemap", "provider_leaderboard"] },
  { id: "rankings",  label: "Rankings",    charts: ["rankings", "value_leaders"] },
  { id: "compare",   label: "Compare",     charts: ["radar"] },
  { id: "budget",    label: "Budget",      charts: ["cost_calc"] },
  { id: "table",     label: "Table",       charts: [] },           // table, Phase 4
  { id: "local",     label: "Run Local",   charts: ["local_scatter", "local_compat"] },
  { id: "image",     label: "Image Gen",   charts: ["image_faceted"] },
  { id: "video",     label: "Video Gen",   charts: ["video_rankings", "video_scatter"] },
];

// window.AF.state contract: { providers: string[], minQuality: number, search: string, tab: string }
window.AF = { pyReady: false, figCache: {}, manifest: null, state: {
  providers: [], minQuality: 0, search: "", tab: "overview" } };

const PLOT_CONFIG = { displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage"] };

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js";

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
    panel.innerHTML = t.charts.map(c =>
      `<div class="chart-card"><div id="chart-${c}" style="min-height:400px"></div></div>`
    ).join("");
    panelsEl.appendChild(panel);
  });
}

function switchTab(id) {
  window.AF.state.tab = id;
  document.querySelectorAll(".tab").forEach(b =>
    b.classList.toggle("tab--selected", b.dataset.tab === id));
  TABS.forEach(t => {
    document.getElementById("panel-" + t.id).style.display = t.id === id ? "block" : "none";
  });
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
}

// ---- Pyodide boot ----

async function bootPyodide() {
  // Show booting status
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
    // plotly + _plotly_utils are vendored inside the bundle — no micropip needed.
    // Shim importlib.metadata.version so vendored plotly doesn't need dist-info.
    await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/bundle")
import importlib.metadata as _im
_orig_version = _im.version
def _version_shim(name):
    _VERSIONS = {"plotly": "6.5.2", "narwhals": "1.0.0"}
    if name in _VERSIONS:
        return _VERSIONS[name]
    return _orig_version(name)
_im.version = _version_shim
import static_api
`);
    window.AF.pyodide = pyodide;
    window.AF.callPy = async (fn, ...args) => {
      const py = pyodide.globals.get("static_api");
      const res = py[fn](...args);   // JS args auto-convert; returns JSON string
      const s = res.toString();
      if (typeof res.destroy === "function") res.destroy();
      return JSON.parse(s);
    };
    window.AF.callPyRaw = async (fn, ...args) => {
      const py = pyodide.globals.get("static_api");
      const res = py[fn](...args);
      const s = res.toString();
      if (typeof res.destroy === "function") res.destroy();
      return s;  // raw string, no JSON.parse
    };
    window.AF.pyReady = true;
    if (statusEl) statusEl.style.display = "none";
    rerenderActiveFilterCharts();   // refresh with live (identical) data once ready
  } catch (err) {
    console.error("Pyodide boot failed:", err);
    if (statusEl) { statusEl.style.display = "block"; statusEl.textContent = "interactivity unavailable: " + err.message; }
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
    // Compare uses radar — basic re-render when filters change
    try {
      const result = await window.AF.callPy("update_compare", p, q, s, [], "filter-provider");
      renderJsonFig("chart-radar", result.figure);
    } catch (e) { console.error("compare render failed:", e); }
  } else if (tab === "budget") {
    const tok = Number(document.getElementById("budget-tokens")?.value || 1);
    try {
      renderJsonFig("chart-cost_calc", await window.AF.callPy("update_cost_calc", tok, p, q, s));
    } catch (e) { console.error("budget render failed:", e); }
  }
}

// ---- Preset helper ----
function setPreset(minQ, providers) {
  // Snap minQ to the nearest available option value (handles p75/p90 non-integer thresholds)
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

async function init() {
  buildTabsAndPanels();
  await loadManifest();
  await Promise.all(TABS.flatMap(t => t.charts.map(c => renderFigure("chart-" + c, c))));
  wireGlobalControls();
  bootPyodide();  // fire-and-forget; pyReady gate protects filter calls
}
init().catch(err => {
  const s = document.getElementById("py-status");
  if (s) { s.style.display = "block"; s.textContent = "Failed to load: " + err.message; }
});
