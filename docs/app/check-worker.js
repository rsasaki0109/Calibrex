// Web Worker of app/check.html: Pyodide + the Calibrex wheel, planning `calibrex check --plan`.
//
// The user's bag is never copied into the Python heap. Each dropped file is mounted
// with Emscripten WORKERFS, which serves reads from the File/Blob on demand
// (FileReaderSync on slices), so a multi-gigabyte .db3 or .mcap is read lazily by
// sqlite3 / the MCAP reader exactly where Python asks for it. The planner itself is
// Python (calibrex.check.browser.plan_request_json); this file only mounts files.
//
// Protocol, page -> worker:
//   {type: "plan", bag: {files: File[]} | {urls: string[]}, tf: File[], request: {...}}
// worker -> page:
//   {type: "log", text}  {type: "ready"}  {type: "result", payload}  {type: "failed", text}
"use strict";

const PYODIDE_VERSION = "0.27.2";
const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v" + PYODIDE_VERSION + "/full/";
importScripts(PYODIDE_INDEX + "pyodide.js");

const log = (text) => postMessage({ type: "log", text });
let pyodide = null;
let mounts = [];

const ready = (async () => {
  try {
    log("Loading Python…");
    pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });
    log("Loading NumPy, SciPy, Pydantic, PyYAML, SQLite…");
    // zstandard lets the MCAP reader open zstd-compressed chunks (the ros2 bag default).
    await pyodide.loadPackage(["numpy", "scipy", "pydantic", "pyyaml", "sqlite3", "zstandard", "micropip"]);
    const manifest = await (await fetch(new URL("wheels/manifest.json", self.location.href))).json();
    log("Installing Calibrex " + manifest.version + "…");
    await pyodide.pyimport("micropip").install(new URL("wheels/" + manifest.wheel, self.location.href).href, { deps: false });
    await pyodide.runPythonAsync(`
import json
from calibrex.check.browser import plan_request_json
`);
    postMessage({ type: "ready", version: manifest.version });
  } catch (error) {
    postMessage({ type: "failed", text: "Failed to load Python: " + error });
    throw error;
  }
})();

function unmountAll() {
  for (const path of mounts.reverse()) {
    try { pyodide.FS.unmount(path); } catch (_) { /* already gone */ }
  }
  mounts = [];
}

function mountFiles(path, { files = [], blobs = [] }) {
  pyodide.FS.mkdirTree(path);
  pyodide.FS.mount(pyodide.FS.filesystems.WORKERFS, { files, blobs }, path);
  mounts.push(path);
}

async function fetchBlobs(urls) {
  return Promise.all(urls.map(async (url) => {
    const response = await fetch(new URL(url, self.location.href));
    if (!response.ok) throw new Error("could not fetch " + url + " (" + response.status + ")");
    return { name: url.split("/").pop(), data: await response.blob() };
  }));
}

onmessage = async (event) => {
  const message = event.data;
  if (message.type !== "plan") return;
  try {
    await ready;
    unmountAll();
    // Bag: loose File objects (user files, read lazily) or fetched blobs (the sample).
    const bag = message.bag.urls ? { blobs: await fetchBlobs(message.bag.urls) } : { files: message.bag.files };
    mountFiles("/bag", bag);
    // Each calibration file gets its own mount, so equal names (calib.yaml) cannot collide.
    const tfPaths = message.tf.map((file, index) => {
      mountFiles("/tf/" + index, { files: [file] });
      return "/tf/" + index + "/" + file.name;
    });
    log("Reading the bag in place (nothing is uploaded)…");
    await new Promise((resolve) => setTimeout(resolve, 20));
    const started = performance.now();
    pyodide.globals.set("request_json", JSON.stringify({
      ...message.request, bag_dir: "/bag", tf_files: tfPaths,
    }));
    const output = JSON.parse(await pyodide.runPythonAsync("plan_request_json(request_json)"));
    output.seconds = (performance.now() - started) / 1000;
    postMessage({ type: "result", payload: output });
  } catch (error) {
    postMessage({ type: "result", payload: { ok: false, error: String(error).split("\n").slice(-2).join(" ") } });
  } finally {
    if (pyodide) unmountAll();
  }
};
