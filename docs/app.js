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

async function init() {
  buildTabsAndPanels();
  await loadManifest();
  await Promise.all(TABS.flatMap(t => t.charts.map(c => renderFigure("chart-" + c, c))));
}
init().catch(err => {
  const s = document.getElementById("py-status");
  if (s) { s.style.display = "block"; s.textContent = "Failed to load: " + err.message; }
});
