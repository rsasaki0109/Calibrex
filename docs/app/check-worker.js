// Web Worker of app/check.html: Pyodide + the Calibrex wheel, planning and running `calibrex check`.
//
// The user's bag is never copied into the Python heap. Each dropped file is mounted
// with Emscripten WORKERFS, which serves reads from the File/Blob on demand
// (FileReaderSync on slices), so a multi-gigabyte .db3 or .mcap is read lazily by
// sqlite3 / the MCAP reader exactly where Python asks for it. The planner itself is
// Python (calibrex.check.browser.plan_request_json / run_request_json); this file only
// mounts files and forwards the progress events of the run.
//
// Protocol, page -> worker:
//   {type: "plan" | "run", bag: {files: File[]} | {urls: string[]}, tf: File[], request: {...}}
//   (a "run" request adds mode "check" | "estimate", pairs, max_duration_s)
// worker -> page:
//   {type: "log", text}  {type: "ready"}  {type: "result", kind, payload}  {type: "failed", text}
//   {type: "progress", event}   one CheckProgress event of a run (see EventProgress)
"use strict";

const PYODIDE_VERSION = "0.27.2";
const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v" + PYODIDE_VERSION + "/full/";
importScripts(PYODIDE_INDEX + "pyodide.js");

const log = (text) => postMessage({ type: "log", text });
let pyodide = null;
let mounts = [];
let opencvLoaded = false;
const CAMERA_PAIRS = ["camera-imu", "camera-focal"];

const ready = (async () => {
  try {
    log("Loading Python…");
    pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });
    installReadAhead();
    log("Loading NumPy, SciPy, Pydantic, PyYAML, SQLite…");
    // zstandard lets the MCAP reader open zstd-compressed chunks (the ros2 bag default).
    await pyodide.loadPackage(["numpy", "scipy", "pydantic", "pyyaml", "sqlite3", "zstandard", "micropip"]);
    const manifest = await (await fetch(new URL("wheels/manifest.json", self.location.href))).json();
    log("Installing Calibrex " + manifest.version + "…");
    await pyodide.pyimport("micropip").install(new URL("wheels/" + manifest.wheel, self.location.href).href, { deps: false });
    await pyodide.runPythonAsync(`
import json
from calibrex.check.browser import plan_request_json, run_request_json
`);
    postMessage({ type: "ready", version: manifest.version });
  } catch (error) {
    postMessage({ type: "failed", text: "Failed to load Python: " + error });
    throw error;
  }
})();

// WORKERFS reads one File slice per read() call, and sqlite asks for one 4 KiB page at a time:
// a 1 MB LiDAR message is hundreds of synchronous slice reads. Serve reads from 1 MiB blocks
// that are read once and kept (most recent 32 MiB per worker), so the bag is still read lazily
// and never held whole, but with a few hundred times fewer FileReaderSync calls.
function installReadAhead() {
  const ops = pyodide.FS.filesystems.WORKERFS.stream_ops;
  const reader = new FileReaderSync();
  const BLOCK = 1 << 20, KEEP = 32;
  const blocks = new Map(); // "<id>:<index>" -> Uint8Array, oldest first
  const ids = new WeakMap();
  let nextId = 0;
  ops.read = (stream, buffer, offset, length, position) => {
    const node = stream.node;
    if (position >= node.size) return 0;
    const end = Math.min(position + length, node.size);
    if (!ids.has(node.contents)) ids.set(node.contents, nextId++);
    const id = ids.get(node.contents);
    for (let at = position; at < end;) {
      const index = Math.floor(at / BLOCK), key = id + ":" + index;
      let block = blocks.get(key);
      if (block) { blocks.delete(key); } else {
        const from = index * BLOCK;
        block = new Uint8Array(reader.readAsArrayBuffer(node.contents.slice(from, Math.min(from + BLOCK, node.size))));
        if (blocks.size >= KEEP) blocks.delete(blocks.keys().next().value);
      }
      blocks.set(key, block);
      const take = Math.min(end, (index + 1) * BLOCK) - at;
      buffer.set(block.subarray(at - index * BLOCK, at - index * BLOCK + take), offset + (at - position));
      at += take;
    }
    return end - position;
  };
}

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
  if (message.type !== "plan" && message.type !== "run") return;
  const kind = message.type;
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
    if (kind === "run" && (message.request.pairs || []).some((pair) => CAMERA_PAIRS.includes(pair)) && !opencvLoaded) {
      log("Loading OpenCV for the camera pairs (about 50 MB the first time)…");
      await pyodide.loadPackage("opencv-python");
      opencvLoaded = true;
    }
    log("Reading the bag in place (nothing is uploaded)…");
    await new Promise((resolve) => setTimeout(resolve, 20));
    const started = performance.now();
    pyodide.globals.set("request_json", JSON.stringify({
      ...message.request, bag_dir: "/bag", tf_files: tfPaths,
    }));
    let output;
    if (kind === "run") {
      pyodide.globals.set("emit_event", (text) => postMessage({ type: "progress", event: JSON.parse(text) }));
      output = JSON.parse(await pyodide.runPythonAsync("run_request_json(request_json, emit_event)"));
    } else {
      output = JSON.parse(await pyodide.runPythonAsync("plan_request_json(request_json)"));
    }
    output.seconds = (performance.now() - started) / 1000;
    postMessage({ type: "result", kind, payload: output });
  } catch (error) {
    postMessage({ type: "result", kind, payload: { ok: false, error: String(error).split("\n").slice(-2).join(" ") } });
  } finally {
    if (pyodide) unmountAll();
  }
};
