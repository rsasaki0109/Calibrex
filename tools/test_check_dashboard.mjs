import assert from "node:assert/strict";
import test from "node:test";
import "../docs/app/check-dashboard.js";
const { LiveRun, resultHTML } = globalThis.CalibrexDashboard;

test("interleaved workers retain separate stages and reset counts on a new pass", () => {
  const live = new LiveRun();
  live.consume({event: "pair_started", worker: 1, label: "imu-lidar"});
  live.consume({event: "stage", worker: 1, text: "Read scans"});
  live.consume({event: "tick", worker: 1, done: 50, total: 100});
  live.consume({event: "pair_started", worker: 2, label: "camera-imu"});
  assert.equal(live.workers.get(1).done, 50);
  live.consume({event: "stage", worker: 1, text: "Holdout"});
  assert.equal(live.workers.get(1).done, null);
  assert.deepEqual(live.workers.get(1).points, []);
  live.consume({event: "tick", worker: 1, done: 20});
  live.consume({event: "tick", worker: 1, done: 1});
  assert.deepEqual(live.workers.get(1).points, [1]);
  live.consume({event: "pair_finished", worker: 1, status: "pass"});
  live.stop("cancelled");
  assert.equal(live.workers.get(1).status, "pass");
  assert.equal(live.workers.get(2).status, "cancelled");
  live.reset();
  assert.equal(live.workers.size, 0);
});

test("check graph preserves verdicts, unclipped numeric ratios, unchecked reasons and escaped labels", () => {
  const html = resultHTML({pairs: [{pair: '<img src=x onerror="bad()">', status: "fail", axes: [
    {name: "yaw", status: "fail", unit: "deg", candidate_error: 5, tolerance: .5, ratio: 10},
  ], unchecked_axes: [{name: "x", reason: "No <excitation>"}]}]}, "check");
  assert.ok(html.includes("10× tolerance"));
  assert.ok(html.includes('width:100%'));
  assert.ok(html.includes("UNCHECKED"));
  assert.ok(html.includes("No &lt;excitation&gt;"));
  assert.ok(!html.includes("<img"));
});

test("estimate never shows unobserved prior values as measurements", () => {
  const html = resultHTML({pairs: [{pair: "imu-lidar", status: "partial", axes: [
    {name: "roll", value: 2, std: .01, unit: "deg", status: "observed"},
    {name: "x", value: 999, unit: "m", status: "unobservable", reason: "Turn about y"},
  ]}]}, "estimate");
  assert.ok(html.includes("2 ± 0.01 deg"));
  assert.ok(html.includes("NOT MEASURED"));
  assert.ok(!html.includes("999"));
  assert.ok(html.includes("Turn about y"));
});

test("residual traces isolate fits and workers, preserve rising values and reject invalid data", () => {
  const live = new LiveRun();
  const metric = (worker, step, value) => live.consume({event: "metric", worker,
    name: "Rotation training residual", unit: "noise-normalized RMS", step, value});
  live.consume({event: "pair_started", worker: 1, label: "imu-lidar"});
  live.consume({event: "pair_started", worker: 2, label: "camera-imu"});
  metric(1, 0, 4); metric(2, 0, 10); metric(1, 1, 2); metric(1, 2, 3);
  assert.deepEqual(live.workers.get(1).metric.points.map((p) => p.value), [4, 2, 3]);
  metric(1, 3, NaN); metric(1, 3, Infinity); metric(1, 3, -1); metric(1, .5, 1);
  assert.equal(live.workers.get(1).metric.points.length, 3);
  metric(1, 0, 8); metric(1, 1, 5);
  assert.equal(live.workers.get(1).fits, 2);
  assert.deepEqual(live.workers.get(1).metric.points.map((p) => p.value), [8, 5]);
  assert.deepEqual(live.workers.get(2).metric.points.map((p) => p.value), [10]);
  const html = live.html();
  assert.ok(html.includes("FIT 2"));
  assert.ok(html.includes("Training residuals can rise"));
  assert.ok(!html.includes("NaN"));
  live.consume({event: "pair_started", worker: 1, label: "next"});
  assert.equal(live.workers.get(1).metric, null);
});
