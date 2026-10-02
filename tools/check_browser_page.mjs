// Smoke-test the browser pages: run the exact Python snippets embedded in
// docs/app/index.html (IMU calibration) and the plan call of docs/app/check-worker.js (bag check)
// under Pyodide in Node.  The headless-Chrome test of the check page itself is
// tools/check_browser_check_page.mjs.  Usage (from the repository root):
//
//   python tools/build_browser_wheel.py
//   npm install --no-save pyodide@0.27.2
//   node tools/check_browser_page.mjs
//
import { loadPyodide } from "pyodide";
import { readFileSync, readdirSync } from "fs";
const root = new URL("../docs/app/", import.meta.url).pathname;
const html = readFileSync(root + "index.html", "utf8");
const snippets = [...html.matchAll(/runPythonAsync\(`([\s\S]*?)`\)/g)].map((m) => m[1]);
console.log("python snippets in page:", snippets.length);
const pyodide = await loadPyodide({ packageBaseUrl: "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/" });
await pyodide.loadPackage(["numpy", "scipy", "pydantic", "pyyaml", "sqlite3", "micropip"]);
const manifest = JSON.parse(readFileSync(root + "wheels/manifest.json"));
pyodide.FS.writeFile("/tmp/" + manifest.wheel, readFileSync(root + "wheels/" + manifest.wheel));
await pyodide.pyimport("micropip").install("emfs:/tmp/" + manifest.wheel, { deps: false });
await pyodide.runPythonAsync(snippets[0]);                       // boot imports
const data = JSON.parse(await pyodide.runPythonAsync(snippets[2])); // synthetic example
for (const [unit, reference, translation] of [["mps2", "none", true], ["mps2", "mid360", false]]) {
  pyodide.globals.set("imu_text", data.imu);
  pyodide.globals.set("trajectory_text", data.trajectory);
  pyodide.globals.set("options_json", JSON.stringify({ acceleration_unit: unit, reference, estimate_translation: translation, sequence_id: "synthetic_example" }));
  const result = JSON.parse(await pyodide.runPythonAsync(snippets[1]));        // calibrate
  const statuses = [result.summary.rotation.policy_status].concat(
    result.summary.translation ? [result.summary.translation.policy_status] : []);
  if (statuses.some((status) => status !== "pass")) { console.error("FAIL", statuses); process.exit(1); }
  console.log(reference, translation, result.summary.rotation.policy_status,
    result.summary.translation ? result.summary.translation.policy_status : "no-translation",
    result.rotation_yaml.length, result.translation_yaml ? result.translation_yaml.length : null);
}

// The bag-check page: the worker's boot import, then a plan of the committed sample bag
// (mounted from disk with NODEFS; the browser mounts the same files with WORKERFS).
const workerSource = readFileSync(root + "check-worker.js", "utf8");
const workerSnippet = workerSource.match(/runPythonAsync\(`([\s\S]*?)`\)/)[1];
await pyodide.runPythonAsync(workerSnippet);
const sampleDir = root + "samples/check_sample";
pyodide.FS.mkdirTree("/bag");
pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: sampleDir }, "/bag");
pyodide.globals.set("request_json", JSON.stringify({ bag_dir: "/bag", bag_label: "check_sample" }));
const plan = JSON.parse(await pyodide.runPythonAsync("plan_request_json(request_json)"));
if (!plan.ok) { console.error("FAIL check plan", plan.error); process.exit(1); }
const planned = plan.artifact.pairs.filter((pair) => pair.status === "planned").length;
console.log("check plan of", readdirSync(sampleDir).join(","), "->", planned, "planned,", plan.artifact.summary.skipped_by_reason);
if (planned !== 8) { console.error("FAIL expected 8 planned pairs"); process.exit(1); }
// A result that is not "pass" on the synthetic example is a regression.
