// Smoke-test the browser calibration page: run the exact Python snippets embedded in
// docs/app/index.html under Pyodide in Node.  Usage (from the repository root):
//
//   python tools/build_browser_wheel.py
//   npm install --no-save pyodide@0.27.2
//   node tools/check_browser_page.mjs
//
import { loadPyodide } from "pyodide";
import { readFileSync } from "fs";
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
// A result that is not "pass" on the synthetic example is a regression.
