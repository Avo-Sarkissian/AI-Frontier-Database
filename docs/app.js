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
window.AF = { pyReady: false, figCache: {}, manifest: null, localHwMeta: null,
  compareOrder: [], state: {
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
  "Anthropic":              "#d97757",
  "Meta":                   "#0566db",
  "OpenAI":                 "#52d678",
  "Alibaba":                "#b46eb6",
  "Google":                 "#fea92f",
  "NVIDIA":                 "#777221",
  "Amazon":                 "#bd088c",
  "Mistral":                "#dcb1f2",
  "DeepSeek":               "#7d9bff",
  "Kimi":                   "#b200c1",
  "SpaceXAI":               "#ffffff",
  "Microsoft":              "#818cf8",
  "Cohere":                 "#f87171",
  "Z AI":                   "#7dd3fc",
  "MiniMax":                "#86efac",
  "InclusionAI":            "#fca5a5",
  "Xiaomi":                 "#6ee7b7",
  "Baidu":                  "#fde68a",
  "IBM":                    "#93c5fd",
  "LG AI Research":         "#c4b5fd",
  "Nous Research":          "#f9a8d4",
  "Reka AI":                "#a78bfa",
  "AI21 Labs":              "#2dd4bf",
  "Allen Institute for AI": "#67e8f9",
  "Inception":              "#fb7185",
  "Upstage":                "#fbbf24",
  "Perplexity":             "#a3a3a3",
  "KwaiKAT":                "#f97316",
  "Deep Cogito":            "#a8a29e",
  "Thinking Machines":      "#5eead4",
  "Tencent":                "#4ade80",
  "StepFun":                "#e879f9",
  "Arcee AI":               "#fcd34d",
  "LongCat":                "#bef264",
  "Sapiens AI":             "#f0abfc",
  "Nex AGI":                "#94a3b8",
  "Multiverse Computing":   "#d6d3d1",
  "Celeris":                "#fdba74",
  "Sarvam":                 "#f472b6",
  "Swiss AI Initiative":    "#84cc16",
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
  const globalFilters = document.querySelector(".filters");
  if (globalFilters) {
    globalFilters.style.display =
      TABS_WITHOUT_GLOBAL_FILTERS.includes(id) ? "none" : "";
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
  const budgetAnswer = document.getElementById("budget-answer");
  if (budgetAnswer && id !== "budget") budgetAnswer.style.display = "none";

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
  // Plotly needs a resize when a hidden plot becomes visible.
  setTimeout(() => document.querySelectorAll("#panel-" + id + " .js-plotly-plot")
    .forEach(el => { try { Plotly.Plots.resize(el); } catch (e) { console.warn("resize:", e); } }), 60);
  // Re-render active tab with current filters once Pyodide is ready
  rerenderActiveFilterCharts();
}

async function renderFigure(divId, figId) {
  let fig = window.AF.figCache[figId];
  if (!fig) {
    const r = await fetch(`figures/${figId}.json?v=${window.AF.version || ""}`);
    fig = await r.json();
    window.AF.figCache[figId] = fig;
  }
  Plotly.react(divId, fig.data, fig.layout, PLOT_CONFIG);
}

async function loadManifest() {
  const m = await (await fetch("figures/manifest.json", { cache: "no-store" })).json();
  window.AF.manifest = m;
  window.AF.version = m.version || "";
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
      setExportPending(false);
      populateDynamicSelects()
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

  worker.postMessage({ type: "boot", version: window.AF.version || "" });
}

// ---- Populate selects that need data from Python ----
async function populateDynamicSelects() {
  try {
    const gpuOptions = await window.AF.callPy("gpu_options");
    const quantLevels = await window.AF.callPy("quant_options");
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

    // Set default local HW meta for RTX 5090
    try {
      const hw = await window.AF.callPy("local_hw_for_gpu", DEFAULT_GPU);
      if (hw) {
        window.AF.localHwMeta = hw;
        const vramInput = document.getElementById("local-vram");
        if (vramInput) vramInput.value = hw.vram_gb;
        // Same default preset feeds the Agent Stack tab's VRAM cap.
        const recVramInput = document.getElementById("recommend-vram");
        if (recVramInput) recVramInput.value = hw.vram_gb;
      }
    } catch (e) { console.warn("local_hw_for_gpu init:", e); }

  } catch (e) {
    console.error("populateDynamicSelects failed:", e);
  }
}

// ---- Debounce helper ----
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

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
      location.replace(u.toString());   // fresh figures + pybundle + Pyodide reboot
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
  const STALE_AFTER_HOURS = 3;
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
  const [p, q, s, e] = readGlobalFilters();
  const sel = document.getElementById("radar-model-select");
  const inDom = Array.from(sel.selectedOptions).map(o => o.value);
  // Prefer the recency order the onchange handler maintains; fall back to
  // document order for the paths that set .selected programmatically.
  const tracked = (window.AF.compareOrder || []).filter(v => inDom.includes(v));
  let selected = tracked.concat(inDom.filter(v => !tracked.includes(v)));
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
  host.style.display = "block";

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

async function refreshBudget() {
  if (!window.AF.pyReady) return;
  const [p, q, s, e] = readGlobalFilters();
  const tok = Number(document.getElementById("budget-tokens")?.value || 1);
  const floor = Number(document.getElementById("budget-min-intelligence")?.value || 0);
  try {
    const out = await window.AF.callPy("update_cost_calc", tok, p, q, s, floor, e);
    renderJsonFig("chart-cost_calc", out.figure);
    renderBudgetAnswer(out.best, out.floor);
  } catch (e) { console.error("budget render failed:", e); }
}

async function refreshTable() {
  if (!window.AF.pyReady) return;
  const [p, q, s, e] = readGlobalFilters();
  const col = document.getElementById("table-sort-col").value;
  const dir = document.getElementById("table-sort-dir").value;
  try {
    const rows = await window.AF.callPy("update_table", p, q, s, col, dir, e);
    renderTableRows(rows);
  } catch (e) { console.error("refreshTable failed:", e); }
}

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

async function refreshLocal() {
  if (!window.AF.pyReady) return;
  // Same rule as the numeric boxes: send nothing and let Python own the
  // fallback (DEFAULT_BANDWIDTH_GBPS / "nvidia"). A literal bandwidth figure
  // here is a second copy of a constant that lives in data/local_models.py.
  const hw = window.AF.localHwMeta || {};
  const vram = numOrNull("local-vram");
  const gpus = numOrNull("local-num-gpus");
  const quant = document.getElementById("local-quant").value || "Q4";
  // numOrNull, never `|| 8192`: a literal numeric fallback on a local-* box is
  // a second copy of a constant that lives in data/local_models.py, and
  // test_neither_rendering_declares_its_own_hardware_default greps for exactly
  // that shape.
  const ctx = numOrNull("local-context");
  const speedMode = document.getElementById("local-speed-mode")?.value || null;
  const tags = multiVals("local-tags");
  try {
    // The three new args are APPENDED. pyworker.js spreads this list straight
    // into the Python function with no arity check, so an argument inserted
    // mid-list shifts every one after it and fails silently.
    const out = await window.AF.callPy("update_local", vram, gpus, quant,
      hw.bandwidth_gbps ?? null, hw.hw_type ?? null, tags.length ? tags : null,
      ctx, speedMode, hw.fp16_tflops ?? null);
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
  const modeSel = document.getElementById("video-mode-filter");
  const mode = modeSel && modeSel.value ? modeSel.value : null;
  try {
    const out = await window.AF.callPy("update_video", providers.length ? providers : null, tags.length ? tags : null, mode);
    renderJsonFig("chart-video_rankings", out.rankings);
    renderJsonFig("chart-video_scatter", out.scatter);
  } catch (e) { console.error("refreshVideo failed:", e); }
}

async function refreshRecommend() {
  if (!window.AF.pyReady) return;
  const mode = document.querySelector('input[name="recommend-mode"]:checked')?.value || "api";
  const providers = Array.from(document.querySelectorAll('input[name="recommend-providers"]:checked')).map(c => c.value);
  const gpu = document.getElementById("recommend-gpu-preset")?.value || "NVIDIA RTX 5090";
  const vram = numOrNull("recommend-vram");
  const gpus = numOrNull("recommend-num-gpus");
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
  const [p, q, s, e] = readGlobalFilters();
  if (!window.AF.pyReady) return;
  const tab = window.AF.state.tab;
  if (tab === "overview") {
    const x = document.querySelector('input[name="overview-xaxis"]:checked')?.value || "price";
    try {
      await renderJsonFig("chart-pareto", await window.AF.callPy("update_overview", p, q, s, x, e));
      attachParetoClickHandler();
    } catch (e) { console.error("overview render failed:", e); }
  } else if (tab === "landscape") {
    try {
      renderJsonFig("chart-treemap", await window.AF.callPy("update_treemap", p, q, s, e));
      renderJsonFig("chart-provider_leaderboard", await window.AF.callPy("update_provider_leaderboard", p, q, s, e));
    } catch (e) { console.error("landscape render failed:", e); }
  } else if (tab === "rankings") {
    const sort = document.querySelector('input[name="rankings-sort"]:checked')?.value || "intelligence";
    try {
      renderJsonFig("chart-rankings", await window.AF.callPy("update_rankings", p, q, s, sort, e));
      // Value Leaders is built once on the full dataset (matches Dash app — no callback).
      // Do NOT re-render it here; it keeps the pre-loaded figures/value_leaders.json figure.
    } catch (e) { console.error("rankings render failed:", e); }
  } else if (tab === "compare") {
    // "tab-switch", not "filter-provider": this path runs on every tab change,
    // and passing a filter trigger made Python discard the user's picks and
    // substitute the diverse-5 defaults. Go to Table and back and a curated
    // comparison was gone. Dash has no tabs input on this callback and never
    // did this — the static site drifted.
    await refreshCompare(window.AF._compareTrigger || "tab-switch");
    window.AF._compareTrigger = null;
  } else if (tab === "budget") {
    await refreshBudget();
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
      const tab = window.AF.state.tab;
      const csv = await window.AF.callPyRaw("export_csv", p, q, s, tab);
      const name = await window.AF.callPyRaw("export_csv_filename", tab);
      const blob = new Blob([csv], { type: "text/csv" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = name || "ai_frontier_export.csv"; a.click();
    } catch (e) { console.error("export_csv failed:", e); }
  };

  // Share — copy URL with ?tab=&p=&q=
  document.getElementById("btn-share").onclick = () => {
    const { tab, providers, minQuality } = window.AF.state;
    const params = new URLSearchParams();
    if (tab) params.set("tab", tab);
    if (providers && providers.length) params.set("p", providers.join(","));
    if (minQuality > 0) params.set("q", minQuality);
    const url = location.origin + location.pathname + (params.toString() ? "?" + params : "");
    navigator.clipboard?.writeText(url);
    history.replaceState(null, "", url);
  };

  document.getElementById("btn-refresh").onclick = doRefresh;
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
    if (!chosen.includes(m)) {
      // Evict the oldest pick rather than refusing the new one. Both renderings
      // default to exactly 5 models, so on a fresh page load this button — the
      // detail panel's only call to action — was a silent no-op that still
      // switched tabs.
      const ordered = (window.AF.compareOrder || []).filter(v => chosen.includes(v));
      const base = ordered.length ? ordered : chosen;
      const keep = base.slice(-(COMPARE_MAX - 1)).concat([m]);
      window.AF.compareOrder = keep;
      Array.from(sel.options).forEach(o => { o.selected = keep.includes(o.value); });
    }
    // Switch to compare tab without triggering auto-rerender, then refresh with correct selection.
    window.AF.state.tab = "compare";
    document.querySelectorAll(".tab").forEach(b => b.classList.toggle("tab--selected", b.dataset.tab === "compare"));
    TABS.forEach(t => { document.getElementById("panel-" + t.id).style.display = t.id === "compare" ? "block" : "none"; });
    showTabControls("compare");
    await refreshCompare("radar-model-select");
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
    // customdata is [model, provider] array
    const model = Array.isArray(cd) ? cd[0] : cd;
    const provider = Array.isArray(cd) ? cd[1] : "";
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
  const localContext = document.getElementById("local-context");
  const localSpeedMode = document.getElementById("local-speed-mode");
  const localTags = document.getElementById("local-tags");
  if (localVram) localVram.oninput = debounce(() => refreshLocal(), 300);
  if (localNumGpus) localNumGpus.onchange = () => refreshLocal();
  if (localQuant) localQuant.onchange = () => refreshLocal();
  if (localContext) localContext.onchange = () => refreshLocal();
  if (localSpeedMode) localSpeedMode.onchange = () => refreshLocal();
  if (localTags) localTags.onchange = () => refreshLocal();

  // Image Gen filters
  const imgProvider = document.getElementById("image-provider-filter");
  const imgTags = document.getElementById("image-tag-filter");
  if (imgProvider) imgProvider.onchange = () => refreshImage();
  if (imgTags) imgTags.onchange = () => refreshImage();

  // Video Gen filters
  const vidMode = document.getElementById("video-mode-filter");
  const vidProvider = document.getElementById("video-provider-filter");
  const vidTags = document.getElementById("video-tag-filter");
  if (vidMode) vidMode.onchange = () => refreshVideo();
  if (vidProvider) vidProvider.onchange = () => refreshVideo();
  if (vidTags) vidTags.onchange = () => refreshVideo();

  // Agent Stack mode radios — update row visibility immediately (pre-boot), then refresh data
  document.querySelectorAll('input[name="recommend-mode"]').forEach(r => {
    r.onchange = () => { showTabControls("recommend"); refreshRecommend(); };
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
  if (recGpu) recGpu.onchange = async () => {
    // Sync VRAM to the selected preset so the 'fits' filter uses the real
    // hardware capacity (mirrors the Local tab + Dash update_recommend_hw).
    try {
      const hw = await window.AF.callPy("local_hw_for_gpu", recGpu.value);
      if (hw && recVram) recVram.value = hw.vram_gb;
    } catch (e) { console.error("recommend gpu preset sync:", e); }
    refreshRecommend();
  };
  if (recVram) recVram.oninput = debounce(() => refreshRecommend(), 300);
  if (recNumGpus) recNumGpus.onchange = () => refreshRecommend();
  if (recQuant) recQuant.onchange = () => refreshRecommend();
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
  renderFreshness();
  setInterval(renderFreshness, 60000);
  // Show controls for the initial tab
  showTabControls("overview");
  // Load static figures for all chart tabs
  const chartLoads = TABS.flatMap(t => t.charts.map(c => renderFigure("chart-" + c, c)));
  await Promise.all(chartLoads);
  wireGlobalControls();
  setExportPending(!window.AF.pyReady);
  wireTabControls();
  updateOverviewCaption();  // set caption for initial price x-axis
  wireDetailPanel();  // wires close/add-compare immediately; pareto click re-attached after each render
  applyUrlState();    // restore tab + filters from URL params (pure JS, no Pyodide needed)
  bootPyodide();      // fire-and-forget; pyReady gate protects filter calls
}
init().catch(err => {
  const s = document.getElementById("py-status");
  if (s) { s.style.display = "block"; s.textContent = "Failed to load: " + err.message; }
});
