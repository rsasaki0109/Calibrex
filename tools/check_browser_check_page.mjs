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
//        [--bag PATH ...] [--tf FILE ...] [--vehicle-frame base_link]
//
// Without --bag it clicks "Try a sample bag" and asserts the plan (8 pairs can be checked, 4
// skipped), then screenshots the page in light and dark themes and at phone width. With --bag
// (files of a rosbag2: metadata.yaml and the .db3/.mcap) it uploads them through the file
// picker, which hands the worker real File objects, so a multi-GB bag exercises WORKERFS lazy
// reads exactly as a user's drop does; it prints the plan and does not assert counts.
import puppeteer from "puppeteer-core";
import { mkdirSync } from "fs";

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
const timeoutMs = Number(option("--timeout-ms", "900000"));
mkdirSync(out, { recursive: true });

const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ["--no-sandbox"] });
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
  if (vehicle) await page.click("#vehicle-details > summary");
  if (vehicle) await page.type("#vehicle-frame", vehicle);
  await (await page.$("#bag-files")).uploadFile(...bagFiles);
  if (tfFiles.length) await (await page.$("#tf-files")).uploadFile(...tfFiles);
  const started = Date.now();
  await page.click("#run");
  const result = await waitResult();
  console.log("planned in", ((Date.now() - started) / 1000).toFixed(1), "s");
  if (result) {
    console.log(JSON.stringify(result, null, 1));
    await page.screenshot({ path: out + "/own_bag_desktop.png", fullPage: true });
  }
}
await browser.close();
