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

async function boot(version) {
  post("status", { text: "loading runtime…" });
  self.importScripts(PYODIDE_INDEX + "pyodide.js");
  pyodide = await self.loadPyodide({ indexURL: PYODIDE_INDEX });

  post("status", { text: "loading packages…" });
  await pyodide.loadPackage(["pandas", "numpy", "narwhals"]);

  post("status", { text: "loading bundle…" });
  const res = await fetch(`pybundle.zip?v=${version || ""}`);
  if (!res.ok) throw new Error(`pybundle.zip ${res.status}`);
  pyodide.unpackArchive(await res.arrayBuffer(), "zip", { extractDir: "/bundle" });

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
      await boot(msg.version);
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
