// Pyodide host — runs off the main thread.
//
// Booting Pyodide on the UI thread froze the tab: loadPackage, an 11MB
// unpackArchive and `import static_api` (pandas + plotly) are synchronous, and
// they blocked the main thread for ~3.8s of a 4.7s boot, with single stalls up
// to 1.8s. The pre-rendered figures are interactive at ~450ms, so the user is
// already hovering bubbles when the freeze lands and the pointer appears stuck.
//
// Everything here is the same work; it just happens on a worker thread, so the
// page keeps painting and hovering throughout. app.js talks to it over a tiny
// request/response protocol keyed by call id.

const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";

let pyodide = null;
let staticApi = null;

const post = (type, payload) => self.postMessage({ type, ...payload });

// plotly is vendored into the zip rather than pip-installed, so
// importlib.metadata.version() can't see it. Read the version out of the
// vendored package instead of hardcoding one — a hardcoded string silently
// went stale when the bundle was rebuilt against a different plotly.
const BOOTSTRAP = `
import sys
sys.path.insert(0, "/bundle")

import pathlib as _pl, re as _re

def _vendored_plotly_version(default="6.0.0"):
    for _p in ("/bundle/plotly/_version.py", "/bundle/plotly/version.py"):
        try:
            _m = _re.search(r'"version"\\s*:\\s*"([^"]+)"|__version__\\s*=\\s*"([^"]+)"',
                            _pl.Path(_p).read_text())
            if _m:
                return _m.group(1) or _m.group(2)
        except Exception:
            pass
    return default

import importlib.metadata as _im
_orig_version = _im.version
_VERSIONS = {"plotly": _vendored_plotly_version(), "narwhals": "1.0.0", "tenacity": "8.2.3"}
def _version_shim(name):
    if name in _VERSIONS:
        return _VERSIONS[name]
    return _orig_version(name)
_im.version = _version_shim

import static_api
`;

// pycode.zip (project modules + vendored plotly, ~4 MB) is keyed by a hash of
// its contents and kept in CacheStorage, so a returning visitor downloads it
// once per code change instead of once per hourly data refresh. The HTTP cache
// cannot do this on its own: Pages serves max-age=600 with an mtime-based ETag,
// and every deploy resets the mtime, so a revalidation always re-downloads.
const CODE_CACHE = "af-pycode";

async function openCodeCache() {
  // CacheStorage is absent outside secure contexts and throws in some private
  // modes; without it the zip is simply fetched every time, as before.
  try { return self.caches ? await self.caches.open(CODE_CACHE) : null; }
  catch (_) { return null; }
}

async function fetchZip(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url.split("?")[0]} ${res.status}`);
  return res.arrayBuffer();
}

async function fetchCode(codeVersion, { skipCache = false } = {}) {
  const url = `pycode.zip?v=${codeVersion}`;
  const cache = await openCodeCache();
  if (cache && !skipCache) {
    try {
      const hit = await cache.match(url);
      if (hit) return { buf: await hit.arrayBuffer(), fromCache: true };
    } catch (_) { /* fall through to the network */ }
  }
  const buf = await fetchZip(url);
  if (cache) {
    try {
      const keep = new URL(url, self.location.href).href;
      await cache.put(url, new Response(buf.slice(0), {
        headers: { "Content-Type": "application/zip" },
      }));
      // One version at a time: an old build's 4 MB is dead weight once replaced.
      for (const req of await cache.keys()) {
        if (req.url !== keep) await cache.delete(req);
      }
    } catch (_) { /* quota or eviction: the fetched copy still boots */ }
  }
  return { buf, fromCache: false };
}

async function dropCachedCode() {
  const cache = await openCodeCache();
  if (!cache) return;
  try {
    for (const req of await cache.keys()) await cache.delete(req);
  } catch (_) { /* nothing to clean */ }
}

async function boot(version, codeVersion) {
  // Both zips are started before the runtime: they need nothing from Pyodide,
  // and fetching them after loadPackage put ~4 MB behind the runtime download.
  // An old app.js (cached for up to 10 minutes across a deploy) sends no
  // codeVersion; keying on the data version then just skips the cache.
  const codeKey = codeVersion || version || "";
  const codeP = fetchCode(codeKey);
  const dataP = fetchZip(`pydata.zip?v=${version || ""}`);
  // Awaited below; this only keeps a rejection that lands while the runtime is
  // still loading from being reported as unhandled.
  codeP.catch(() => {});
  dataP.catch(() => {});

  post("status", { text: "loading runtime…" });
  self.importScripts(PYODIDE_INDEX + "pyodide.js");
  pyodide = await self.loadPyodide({ indexURL: PYODIDE_INDEX });

  post("status", { text: "loading packages…" });
  await pyodide.loadPackage(["pandas", "numpy", "narwhals"]);

  post("status", { text: "loading bundle…" });
  const code = await codeP;
  try {
    pyodide.unpackArchive(code.buf, "zip", { extractDir: "/bundle" });
  } catch (err) {
    // A cached copy can be truncated by an eviction mid-write. Drop it and
    // take the network copy once; a network copy that fails is a real error.
    if (!code.fromCache) throw err;
    await dropCachedCode();
    const fresh = await fetchCode(codeKey, { skipCache: true });
    pyodide.unpackArchive(fresh.buf, "zip", { extractDir: "/bundle" });
  }
  pyodide.unpackArchive(await dataP, "zip", { extractDir: "/bundle" });

  post("status", { text: "starting analysis engine…" });
  await pyodide.runPythonAsync(BOOTSTRAP);
  staticApi = pyodide.globals.get("static_api");

  post("ready", {});
}

// Arrays/objects have to cross into Python as native lists/dicts so json.dumps
// works on the far side. Primitives pass through untouched.
function toPy(value) {
  if (value === null || value === undefined) return { value, proxy: null };
  if (typeof value === "object") {
    const proxy = pyodide.toPy(value);
    return { value: proxy, proxy };
  }
  return { value, proxy: null };
}

function call(fn, args) {
  if (!staticApi) throw new Error("python not ready");
  const converted = args.map(toPy);
  let result;
  try {
    result = staticApi[fn](...converted.map((c) => c.value));
    const text = result === null || result === undefined ? "" : result.toString();
    return text;
  } finally {
    if (result && typeof result.destroy === "function") result.destroy();
    for (const c of converted) {
      if (c.proxy && typeof c.proxy.destroy === "function") c.proxy.destroy();
    }
  }
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};

  if (msg.type === "boot") {
    try {
      await boot(msg.version, msg.codeVersion);
    } catch (err) {
      post("bootError", { message: String((err && err.message) || err) });
    }
    return;
  }

  if (msg.type === "call") {
    try {
      post("result", { id: msg.id, ok: true, value: call(msg.fn, msg.args || []) });
    } catch (err) {
      post("result", { id: msg.id, ok: false, error: String((err && err.message) || err) });
    }
  }
};
