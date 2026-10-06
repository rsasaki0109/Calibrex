// End-to-end test of the browser bag-check page (docs/app/check.html) in headless Chrome.
//
// Drives the real page: the Pyodide worker, WORKERFS mounts, and the rendered plan. Usage
// (from the repository root, with the site served on any port):
//
//   python tools/build_browser_wheel.py
//   python -m http.server 8124 --directory docs &
//   npm install --no-save --no-package-lock puppeteer-core
//   node tools/check_browser_check_page.mjs --url http://127.0.0.1:8124/app/check.html \
//        --out /tmp/checkpage [--chrome /usr/bin/google-chrome] \
//        [--bag PATH ...] [--tf FILE ...] [--vehicle-frame base_link] \
//        [--topic-kind TOPIC=wheel ...] [--camera TOPIC] \
//        [--run [--cap 60] [--pairs lidar-vehicle,imu-vehicle] [--mode check|estimate]]
//
// Without --bag it clicks "Try a sample bag" and asserts the plan (8 pairs can be checked, 4
// skipped), then screenshots the page in light and dark themes and at phone width. With --bag
// (files of a rosbag2: metadata.yaml and the .db3/.mcap) it uploads them through the file
// picker, which hands the worker real File objects, so a multi-GB bag exercises WORKERFS lazy
// reads exactly as a user's drop does; it prints the plan and does not assert counts. With --run it
// then presses "Run in the browser" (duration cap --cap seconds, default 60; the planned pairs, or
// --pairs) and writes the run's JSON, text and timings to --out/run_<mode>.json. The sample is
// always run in both modes (check, estimate) and asserted.
import puppeteer from "puppeteer-core";
import { mkdirSync, writeFileSync } from "fs";

const args = process.argv.slice(2);
const option = (name, fallback = null) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
};
const many = (name) => args.flatMap((arg, index) => (arg === name ? [args[index + 1]] : []));
const url = option("--url", "http://127.0.0.1:8124/app/check.html");
const out = option("--out", "/tmp/checkpage");
const chrome = option("--chrome", "/usr/bin/google-chrome");
const bagFiles = many("--bag");
const tfFiles = many("--tf");
const vehicle = option("--vehicle-frame");
const doRun = args.includes("--run");
const cap = option("--cap", "60");
const runPairs = option("--pairs");
const topicKinds = many("--topic-kind");
const cameraOption = option("--camera");
const runModeOption = option("--mode", "check");
const timeoutMs = Number(option("--timeout-ms", "900000"));
mkdirSync(out, { recursive: true });

const browser = await puppeteer.launch({ executablePath: chrome, headless: true, protocolTimeout: timeoutMs, args: ["--no-sandbox"] });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 900 });
page.on("pageerror", (error) => console.log("pageerror:", error.message));
page.on("console", (message) => { if (message.type() === "error") console.log("console error:", message.text()); });
const fail = (text) => { console.error("FAIL", text); process.exitCode = 1; };

await page.goto(url);
await page.waitForFunction(() => !document.getElementById("sample").disabled, { timeout: 300000 });
console.log("python ready");

async function waitResult() {
  await page.waitForFunction(
    () => !document.getElementById("results").hidden || !document.getElementById("error").hidden,
    { timeout: timeoutMs },
  );
  const error = await page.$eval("#error", (element) => (element.hidden ? null : element.textContent));
  if (error) { fail("page reported: " + error); return null; }
  return page.evaluate(() => ({
    stats: [...document.querySelectorAll("#stats .stat")].map((e) => e.textContent.trim().replace(/\s+/g, " ")),
    pairs: [...document.querySelectorAll("#pairs tr")].slice(1).map((r) => [...r.children].map((c) => c.textContent.trim()).slice(0, 3).join(" | ")),
    topics: document.querySelectorAll("#topics tr").length - 1,
    treeText: document.getElementById("tree").innerText,
    sources: document.getElementById("sources").innerText,
    command: document.getElementById("cmd").textContent,
    log: document.getElementById("log").textContent.split("\n").slice(-3).join(" / "),
  }));
}

// Press "Run in the browser" and wait for the result; returns what the page shows and its payload.
async function runInPage(mode, capSeconds, pairs) {
  await page.select("#run-mode", mode);
  await page.$eval("#run-cap", (element, value) => { element.value = value; }, String(capSeconds));
  if (pairs) {
    await page.$$eval("#run-pairs input[name=pair]", (boxes, wanted) => {
      for (const box of boxes) box.checked = wanted.includes(box.value) && !box.disabled;
    }, pairs);
  }
  if (cameraOption) await page.$eval("#run-camera", (element, value) => { element.value = value; }, cameraOption);
  const started = Date.now();
  await page.click("#run-estimators");
  let lastLine = "";
  const watcher = setInterval(async () => {
    try {
      const line = await page.evaluate(() => document.getElementById("prog-line").textContent + " | " + document.getElementById("prog-detail").textContent);
      if (line !== lastLine) { lastLine = line; console.log(`[${((Date.now() - started) / 1000).toFixed(0)} s]`, line); }
    } catch (_) { /* page busy */ }
  }, 30000);
  await page.waitForFunction(
    () => !document.getElementById("run-results").hidden || !document.getElementById("run-error").hidden,
    { timeout: timeoutMs, polling: 250 },
  );
  clearInterval(watcher);
  const error = await page.$eval("#run-error", (element) => (element.hidden ? null : element.textContent));
  if (error) { fail("run reported: " + error); return null; }
  const shown = await page.evaluate(() => ({
    title: document.getElementById("run-title").textContent,
    stats: [...document.querySelectorAll("#run-stats .stat")].map((e) => e.textContent.trim().replace(/\s+/g, " ")),
    text: document.getElementById("run-text").textContent,
    downloads: [...document.querySelectorAll("#run-downloads button")].map((e) => e.textContent),
    reportLength: document.getElementById("run-report").hidden ? 0 : document.getElementById("run-report").srcdoc.length,
    finished: [...document.querySelectorAll("#prog-done li")].map((e) => e.textContent.trim().replace(/\s+/g, " ")),
    payload: lastRun,
  }));
  shown.wallSeconds = (Date.now() - started) / 1000;
  return shown;
}

if (bagFiles.length === 0) {
  await page.click("#sample");
  const result = await waitResult();
  if (result) {
    console.log(JSON.stringify(result, null, 1));
    const planned = result.pairs.filter((row) => row.includes("can be checked")).length;
    const skipped = result.pairs.filter((row) => row.includes("skipped")).length;
    if (planned !== 8 || skipped !== 4) fail(`expected 8 planned and 4 skipped pairs, got ${planned} and ${skipped}`);
    if (!result.command.startsWith("calibrex check check_sample")) fail("unexpected command " + result.command);
    await page.screenshot({ path: out + "/sample_desktop.png", fullPage: true });
    // Re-plan the sample as a ground vehicle: the vehicle pairs become plannable.
    await page.click("#open-vehicle");
    await page.type("#vehicle-frame", "base_link");
    await page.click("#run");
    await page.waitForFunction(() => document.getElementById("cmd").textContent.includes("--vehicle-frame"), { timeout: 60000 });
    const again = await waitResult();
    const plannedAgain = again.pairs.filter((row) => row.includes("can be checked")).length;
    console.log("with --vehicle-frame base_link:", plannedAgain, "pairs can be checked");
    if (plannedAgain <= planned) fail("vehicle frame did not enable the vehicle pairs");
    // Run mode: the planned pairs of the sample as a ground vehicle, check then estimate.
    const checkRun = await runInPage("check", 20, null);
    if (checkRun) {
      console.log("run (check):", checkRun.wallSeconds.toFixed(1), "s;", checkRun.stats.join(" | "));
      console.log(checkRun.finished.join("\n"));
      if (!checkRun.finished.length) fail("no pair finished in the progress list");
      if (!checkRun.downloads.some((text) => text.includes("check JSON"))) fail("no check JSON download");
      if (checkRun.reportLength < 1000) fail("the HTML report was not rendered");
      if (!checkRun.payload.ok || checkRun.payload.artifact.plan_only) fail("run payload is a plan");
      if (!checkRun.payload.artifact.provenance.notes.some((note) => note.includes("browser"))) fail("browser note missing from provenance");
      if (!checkRun.payload.artifact.overall_verdict) fail("no overall verdict");
    }
    const estimateRun = await runInPage("estimate", 20, null);
    if (estimateRun) {
      console.log("run (estimate):", estimateRun.wallSeconds.toFixed(1), "s;", estimateRun.stats.join(" | "));
      if (!estimateRun.payload.ok || estimateRun.payload.mode !== "estimate") fail("estimate run did not return an estimate");
      if (!Object.keys(estimateRun.payload.files).length && estimateRun.payload.artifact.frames.entries.length) fail("estimate exported no files");
    }
    await page.screenshot({ path: out + "/sample_run.png", fullPage: true });
    await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
    await new Promise((resolve) => setTimeout(resolve, 500)); // theme transitions
    await page.screenshot({ path: out + "/sample_dark.png", fullPage: true });
    await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "light" }]);
    await page.setViewport({ width: 390, height: 844 }); // not isMobile: that reloads the page
    await new Promise((resolve) => setTimeout(resolve, 500));
    await page.screenshot({ path: out + "/sample_mobile.png", fullPage: true });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    if (overflow > 1) fail("horizontal page scroll at phone width: " + overflow + "px");
  }
} else {
  if (vehicle || topicKinds.length) await page.click("#vehicle-details > summary");
  if (vehicle) await page.type("#vehicle-frame", vehicle);
  if (topicKinds.length) await page.type("#topic-kind", topicKinds.join("\n"));
  await (await page.$("#bag-files")).uploadFile(...bagFiles);
  if (tfFiles.length) await (await page.$("#tf-files")).uploadFile(...tfFiles);
  const started = Date.now();
  await page.click("#run");
  const result = await waitResult();
  console.log("planned in", ((Date.now() - started) / 1000).toFixed(1), "s");
  if (result) {
    console.log(JSON.stringify(result, null, 1));
    await page.screenshot({ path: out + "/own_bag_desktop.png", fullPage: true });
    if (doRun) {
      const run = await runInPage(runModeOption, cap, runPairs ? runPairs.split(",") : null);
      if (run) {
        const heap = await page.evaluate(() => (performance.memory ? performance.memory.usedJSHeapSize : null));
        console.log("main-thread JS heap bytes:", heap);
        console.log("run:", run.wallSeconds.toFixed(1), "s wall;", run.payload.seconds.toFixed(1), "s in Python");
        console.log(run.finished.join("\n"));
        console.log(run.text);
        writeFileSync(out + "/run_" + runModeOption + ".json", JSON.stringify(run.payload, null, 1));
        await page.screenshot({ path: out + "/own_bag_run.png", fullPage: true });
      }
    }
  }
}
await browser.close();
