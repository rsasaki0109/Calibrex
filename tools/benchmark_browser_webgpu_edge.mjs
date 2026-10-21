// Explicit developer probe: not invoked by the normal check page.
// node tools/benchmark_browser_webgpu_edge.mjs --fixture prepared.npz --out NEW_DIR
//   --chrome CHROME_PATH [--playwright-module IMPORT_SPECIFIER] [--fixture ...]
// The NPZ contains frames, intrinsics, distortion_model, distortion, start, and
// edge_i/points_i/weights_i/scale_i; no object arrays or pickle are accepted.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFileSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import { createServer } from "node:http";
import { resolve, basename, dirname } from "node:path";

const argv = process.argv.slice(2);
const option = (name, fallback = null) => { const at = argv.indexOf(name); return at < 0 ? fallback : argv[at + 1]; };
const fixtures = argv.flatMap((value, at) => value === "--fixture" ? [argv[at + 1]] : []);
const out = option("--out");
const calibration = argv.includes("--calibration");
const preparationReuse = argv.includes("--preparation-reuse");
if (preparationReuse && calibration) throw new Error("choose --preparation-reuse or --calibration");
const fullWorkflow = calibration || preparationReuse;
const profileCpu = argv.includes("--profile-cpu");
if (profileCpu && !calibration) throw new Error("--profile-cpu requires --calibration");
const probePath = preparationReuse ? "tools/browser_edge_preparation_probe.py" : calibration ? "tools/browser_webgpu_calibration_probe.py" : "tools/browser_webgpu_edge_probe.py";
const schemaPath = preparationReuse ? "tools/schemas/browser_edge_preparation_probe.schema.json" : calibration ? "tools/schemas/browser_webgpu_calibration_probe.schema.json" : "tools/schemas/browser_webgpu_edge_probe.schema.json";
if (!out || !fixtures.length || !option("--chrome")) throw new Error("--out NEW_DIR, --fixture NPZ and --chrome PATH are required");
if (existsSync(out)) throw new Error("output already exists; preserve earlier attempts and choose a new directory");
mkdirSync(out, { recursive: true });
const hash = (data) => createHash("sha256").update(data).digest("hex");
const baselineHead = preparationReuse ? execFileSync("git", ["rev-parse", "--verify", option("--baseline-ref", "d050acca2ded92938ceacf0149a053917e21f54c") + "^{commit}"], {encoding:"utf8"}).trim() : null;
if (preparationReuse) assert.match(baselineHead, /^[0-9a-f]{40}$/);
const baselineBytes = preparationReuse ? execFileSync("git", ["show", baselineHead + ":src/calibrex/evaluation/camera_lidar_edge.py"]) : null;
const manifestPath = "docs/app/wheels/manifest.json";
const manifest = JSON.parse(readFileSync(manifestPath, "utf8").replace(/^\uFEFF/, ""));
const sources = ["docs/app/edge-gpu.js", "src/calibrex/solvers/browser_webgpu_edge_adapter.py",
  "src/calibrex/evaluation/camera_lidar_edge.py", probePath,
  "tools/benchmark_browser_webgpu_edge.mjs", schemaPath,
  ...(preparationReuse ? ["src/calibrex/evaluation/edge_preparation_cache.py", "src/calibrex/check/browser.py",
    "src/calibrex/check/runner.py", "src/calibrex/check/estimators.py", "docs/app/check-worker.js"] : []),
  manifestPath, "docs/app/wheels/" + manifest.wheel, ...fixtures];
const snapshots = new Map(sources.map((path) => [path, readFileSync(path)]));
const sourceHashes = Object.fromEntries([...snapshots].map(([path, data]) => [path, hash(data)]));
assert.equal(sourceHashes["docs/app/wheels/" + manifest.wheel], manifest.sha256, "wheel differs from its manifest");
const report = {
  schema_version: preparationReuse ? "calibrex.browser_edge_preparation_probe/v0.1" : calibration ? "calibrex.browser_webgpu_calibration_probe/v0.2" : "calibrex.browser_webgpu_edge_probe/v0.1", status: "RUNNING",
  provenance: { created_at: new Date().toISOString(), command: process.argv,
    source_head: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
    ...(fullWorkflow ? {source_worktree_dirty: Boolean(execFileSync("git", ["status", "--porcelain", "--untracked-files=normal"], {encoding:"utf8"}).trim())} : {}),
    ...(preparationReuse ? {baseline_source_head:baselineHead, baseline_source_sha256:hash(baselineBytes)} : {}),
    sources: sourceHashes, pyodide_version: "0.27.2", whole_calibration_claim: false },
  runs: [], logs: [], errors: [], checks: {},
};
const save = () => writeFileSync(resolve(out, "report.json"), JSON.stringify(report, null, 2) + "\n");
save();
// Preserve the exact measured implementation, not only its hashes. Large NPZ
// inputs remain in their original immutable fixture directories, bound by SHA256.
for (const [path, data] of snapshots) {
  if (path.endsWith(".npz")) continue;
  const destination = resolve(out, "sources", path);
  mkdirSync(dirname(destination), { recursive: true });
  writeFileSync(destination, data);
}
if (preparationReuse) writeFileSync(resolve(out, "sources", "baseline_edge.py"), baselineBytes);
const content = new Map([
  ["/", ["text/html", Buffer.from("<!doctype html><title>Calibrex objective probe</title>")]],
  ["/edge-gpu.js", ["application/javascript", snapshots.get("docs/app/edge-gpu.js")]],
  ["/probe.py", ["text/plain", snapshots.get(probePath)]],
  ["/" + manifest.wheel, ["application/octet-stream", snapshots.get("docs/app/wheels/" + manifest.wheel)]],
  ...fixtures.map((path, index) => ["/fixture-" + index + ".npz", ["application/octet-stream", snapshots.get(path)]]),
]);
if (preparationReuse) content.set("/baseline_edge.py", ["text/plain", baselineBytes]);
const measurementKeywords = (fullWorkflow ? ", source_head='" + report.provenance.source_head + "'" : "")
  + (preparationReuse ? ", baseline_source_head='" + baselineHead + "'" : "")
  + (profileCpu ? ", profile_cpu=True" : "");
const worker = `
importScripts("/edge-gpu.js", "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js");
onmessage = async ({data}) => {
  try {
    const started = performance.now();
    const py = await loadPyodide({indexURL:"https://cdn.jsdelivr.net/pyodide/v0.27.2/full/"});
    py.setStdout({batched:text=>postMessage({type:"log", text})});
    await py.loadPackage(["numpy", "scipy", "pydantic", "pyyaml", "sqlite3", "micropip"]);
    const installer = py.pyimport("micropip");
    try { await installer.install(new URL("/${manifest.wheel}?sha256=${manifest.sha256}", location.href).href, {deps:false}); }
    finally { installer.destroy(); }
    py.FS.writeFile("/probe.py", await (await fetch("/probe.py")).text());
    ${preparationReuse ? 'py.FS.writeFile("/baseline_edge.py", await (await fetch("/baseline_edge.py")).text());' : ''}
    await py.runPythonAsync("import sys; sys.path.insert(0, '/'); from probe import measure; import json; from js import CalibrexEdgeGpu");
    postMessage({type:"ready", seconds:(performance.now()-started)/1000, isolated:crossOriginIsolated});
    for (let index=0; index<data.fixtures; index++) {
      const bytes = new Uint8Array(await (await fetch("/fixture-"+index+".npz")).arrayBuffer());
      py.FS.writeFile("/fixture.npz", bytes);
      const value = JSON.parse(await py.runPythonAsync("json.dumps(await measure('/fixture.npz', CalibrexEdgeGpu${measurementKeywords}))"));
      value.fixture_index = index;
      postMessage({type:"run", value});
      py.FS.unlink("/fixture.npz");
    }
    postMessage({type:"done"});
  } catch(error) { postMessage({type:"failed", text:String(error)}); }
};`;
content.set("/worker.js", ["application/javascript", Buffer.from(worker)]);
const server = createServer((req, res) => {
  const item = content.get(new URL(req.url, "http://127.0.0.1").pathname);
  if (!item) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "Content-Type": item[0], "Cache-Control": "no-store" }); res.end(item[1]);
});
let browser;
const deadline = setTimeout(async () => {
  report.status = "TIMEOUT"; save();
  await browser?.close().catch(() => {}); server.closeAllConnections(); server.close();
}, 600000);
try {
  const { chromium } = await import(option("--playwright-module", "playwright"));
  await new Promise((done) => server.listen(0, "127.0.0.1", done));
  browser = await chromium.launch({ executablePath: option("--chrome"), headless: true, args: ["--no-sandbox"] });
  report.provenance.browser_version = browser.version();
  const page = await browser.newPage();
  await page.exposeFunction("record", (message) => {
    if (message.type === "run") {
      message.value.fixture = basename(fixtures[message.value.fixture_index]); report.runs.push(message.value);
      console.log(JSON.stringify(preparationReuse ? {
        fixture:message.value.fixture, status:message.value.status,
        cold_ratio:message.value.paired_median_cold_ratio,
        warm_ratio:message.value.paired_median_warm_ratio,
        three_request_ratio:message.value.paired_median_three_request_ratio,
        every_score_and_decision_equal:message.value.every_score_and_decision_equal,
      } : calibration ? {
        fixture:message.value.fixture, status:message.value.status,
        first_estimator_ratio:message.value.first_estimator_ratio,
        paired_median_ratio:message.value.paired_median_ratio, comparison:message.value.first_comparison,
        first_untraced_estimator_ratio:message.value.first_untraced_estimator_ratio,
        paired_median_untraced_ratio:message.value.paired_median_untraced_ratio,
      } : {fixture:message.value.fixture, status:message.value.status,
        cold_batch_ratio:message.value.cold_batch_ratio, resident_median_ratio:message.value.resident_median_ratio,
        max_absolute_error:message.value.first_comparison.max_absolute_error}));
    } else if (message.type === "ready") report.checks.runtime = message;
    else if (message.type === "log") report.logs.push(message.text);
    else if (message.type === "failed") report.errors.push(message.text);
    save();
  });
  page.on("pageerror", (error) => { report.errors.push(error.message); save(); });
  await page.goto("http://127.0.0.1:" + server.address().port);
  await page.evaluate((count) => {
    globalThis.probeDone = false;
    const job = new Worker("/worker.js");
    job.onmessage = async ({data}) => {
      await globalThis.record(data);
      if (["done", "failed"].includes(data.type)) { job.terminate(); globalThis.probeDone = true; }
    };
    job.onerror = async (error) => { await globalThis.record({type:"failed", text:error.message}); job.terminate(); globalThis.probeDone = true; };
    job.postMessage({fixtures:count});
  }, fixtures.length);
  await page.waitForFunction(() => globalThis.probeDone, null, { timeout: 590000 });
  assert.equal(report.errors.length, 0, report.errors.join("\n"));
  assert.equal(report.runs.length, fixtures.length);
  assert.ok(report.runs.every((row) => row.status === "PASS"));
  assert.equal(report.checks.runtime.isolated, false, "probe should work without cross-origin isolation");
  report.checks.sources_unchanged = sources.every((path) => hash(readFileSync(path)) === sourceHashes[path]);
  assert.equal(report.checks.sources_unchanged, true);
  report.status = "PASS";
} catch (error) {
  report.status = report.status === "TIMEOUT" ? "TIMEOUT" : "FAIL";
  report.errors.push(String(error)); process.exitCode = 1;
} finally {
  save(); clearTimeout(deadline); await browser?.close().catch(() => {});
  server.closeAllConnections(); await new Promise((done) => server.close(done));
}
