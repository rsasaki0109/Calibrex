// Deterministic controller tests: no Python downloads, browser, or GPU required.
import assert from "node:assert/strict";
import test from "node:test";
import "../docs/app/check-pool.js";

const { CheckWorkerPool, partitionPairs } = globalThis.CalibrexCheckWorkers;
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const message = (pairs = ["lidar-lidar", "imu-lidar"]) => ({
  type: "run", bag: { files: [{ name: "bag.db3" }] }, tf: [],
  request: { mode: "check", pairs, bag_label: "bag" },
});

function fixture(handler = null, { holdSecond = false } = {}) {
  const workers = [], sent = [], progress = [];
  const factory = () => {
    const index = workers.length;
    const worker = {
      closed: false,
      emit(data) { this.onmessage?.({ data }); },
      terminate() { this.closed = true; },
      postMessage(data) {
        sent.push({ index, data });
        if (handler) { handler(this, data, index); return; }
        queueMicrotask(() => {
          if (data.type === "run") {
            this.emit({ type: "progress", id: data.id, event: { event: "run_started", total_pairs: 1 } });
            this.emit({ type: "progress", id: data.id, event: { event: "pair_finished", label: "test", status: "pass", runtime_s: 1 } });
            this.emit({ type: "progress", id: data.id, event: { event: "run_finished" } });
          }
          this.emit({ type: "result", id: data.id, payload: {
            ok: true, seconds: 1, allocated_wasm_bytes: 100, artifact: {}, files: {},
          } });
        });
      },
    };
    workers.push(worker);
    if (index !== 1 || !holdSecond) queueMicrotask(() => worker.emit({ type: "ready", version: "test" }));
    return worker;
  };
  const pool = new CheckWorkerPool({ factory, progress: (event) => progress.push(event) });
  return { pool, workers, sent, progress };
}

test("partition keeps estimator dependencies and LiDAR odometry together", () => {
  const pairs = ["camera-focal", "imu-lidar", "gnss-imu", "camera-imu", "gnss-lidar",
    "lidar-vehicle", "ins-lidar", "lidar-wheel_odometry", "lidar-lidar", "imu-vehicle"];
  const groups = partitionPairs(pairs);
  assert.equal(groups.length, 2);
  assert.deepEqual(groups.flat().sort(), pairs.slice().sort());
  for (const names of [["camera-focal", "camera-imu"], ["imu-lidar", "gnss-lidar", "gnss-imu"],
    ["lidar-vehicle", "ins-lidar", "lidar-wheel_odometry"]]) {
    assert.ok(groups.some((group) => names.every((name) => group.includes(name))));
  }
  assert.deepEqual(partitionPairs(["camera-imu", "camera-focal"]), [["camera-imu", "camera-focal"]]);
  assert.deepEqual(partitionPairs(["gnss-imu"]), [["gnss-imu"]]);
});

test("parallel work starts on both workers before either reply; merge waits for both", async () => {
  const replies = [];
  const f = fixture((worker, data) => {
    if (data.type === "run") replies.push(() => worker.emit({ type: "result", id: data.id,
      payload: { ok: true, seconds: 1, allocated_wasm_bytes: 100 } }));
    else worker.emit({ type: "result", id: data.id, payload: { ok: true } });
  });
  try {
    const input = message();
    const running = f.pool.run(input, 2);
    await tick();
    assert.equal(replies.length, 2);
    assert.equal(f.sent.length, 2);
    assert.ok(f.sent.every(({ data }) => data.request.scan_memory_mb === 256 && data.request.parallel_shard));
    assert.ok(f.sent.every(({ data }) => data.bag.files === input.bag.files));
    replies[1](); await tick();
    assert.equal(f.sent.length, 2);
    replies[0]();
    const result = await running;
    assert.equal(f.sent[2].data.type, "merge");
    assert.equal(f.sent[2].data.shards.length, 2);
    assert.equal(result.execution.workers, 2);
    assert.equal(result.execution.scan_cache_budget_mb, 512);
    assert.ok(result.seconds >= result.execution.second_worker_ready_seconds);
    assert.equal(f.progress.filter((event) => event.event === "run_finished").length, 1);
  } finally { f.pool.close(); }
});

test("single-worker, dependent-only and estimate jobs do not start an extra runtime", async () => {
  for (const [pairs, mode, requested] of [
    [["lidar-lidar", "imu-lidar"], "check", 1],
    [["camera-imu", "camera-focal"], "check", 2],
    [["lidar-lidar", "imu-lidar"], "estimate", 2],
  ]) {
    const f = fixture();
    try {
      const input = message(pairs); input.request.mode = mode;
      const result = await f.pool.run(input, requested);
      assert.equal(result.execution.workers, 1);
      assert.deepEqual(result.execution.shard_seconds, [1]);
      assert.equal(f.workers.length, 1);
      assert.equal(f.sent[0].data.request.scan_memory_mb, 512);
      assert.equal(f.sent.length, 1);
    } finally { f.pool.close(); }
  }
});

test("both worker progress streams contribute to one total and one finish", async () => {
  const f = fixture();
  try {
    await f.pool.run(message(), 2);
    assert.equal(f.progress.filter((event) => event.event === "pair_finished").length, 2);
    assert.equal(f.progress.filter((event) => event.event === "run_finished").length, 1);
    assert.equal(f.progress.filter((event) => event.event === "run_started").at(-1).total_pairs, 2);
    assert.deepEqual(f.progress.filter((event) => event.event === "pair_finished").map((event) => event.worker).sort(), [1, 2]);
  } finally { f.pool.close(); }
});

test("cancellation terminates all runtimes and ignores late replies", async () => {
  const f = fixture(() => {});
  const running = f.pool.run(message(), 2);
  const rejected = assert.rejects(running, /Cancelled/);
  await tick();
  f.pool.close();
  await rejected;
  assert.ok(f.workers.every((worker) => worker.closed));
  for (const worker of f.workers) worker.emit({ type: "result", id: 1, payload: { ok: true } });
  assert.equal(f.sent.length, 2);
  assert.ok(!f.progress.some((event) => event.event === "run_finished"));
});

test("worker errors, unreadable messages, and failed shards stop both workers", async () => {
  for (const failure of ["error", "messageerror", "result"]) {
    const f = fixture(() => {});
    const running = f.pool.run(message(), 2);
    const rejected = assert.rejects(running, /broken|could not be read/);
    await tick();
    const worker = f.workers[1];
    if (failure === "error") worker.onerror({ message: "broken worker" });
    else if (failure === "messageerror") worker.onmessageerror();
    else worker.emit({ type: "result", id: 1, payload: { ok: false, error: "broken shard" } });
    await rejected;
    assert.ok(f.workers.every((item) => item.closed));
    assert.ok(!f.sent.some(({ data }) => data.type === "merge"));
  }
});

test("a previous request ID cannot complete the next request", async () => {
  const f = fixture((worker, data) => {
    worker.emit({ type: "result", id: data.id - 1, payload: { ok: false, error: "stale" } });
    worker.emit({ type: "result", id: data.id, payload: { ok: true, seconds: 0 } });
  });
  try {
    assert.ok((await f.pool.plan({ ...message(), type: "plan" })).ok);
    assert.ok((await f.pool.run(message())).ok);
    assert.equal(f.sent[1].data.id, 2);
  } finally { f.pool.close(); }
});

test("switching back to one worker releases the second runtime", async () => {
  const f = fixture();
  try {
    await f.pool.run(message(), 2);
    await f.pool.run(message(), 1);
    assert.ok(f.workers[1].closed);
    assert.ok(!f.workers[0].closed);
  } finally { f.pool.close(); }
});

test("the sample blobs are fetched once and reused across workers", async () => {
  const f = fixture();
  const urls = ["samples/bag.db3", "samples/metadata.yaml"];
  const fetched = [];
  f.pool.baseURL = "https://example.test/app/check.html";
  f.pool.fetcher = async (url) => {
    fetched.push(String(url)); return { ok: true, blob: async () => new Blob(["test"]) };
  };
  try {
    const input = { ...message(), bag: { urls } };
    await f.pool.plan({ ...input, type: "plan" });
    await f.pool.run(input, 2);
    assert.equal(fetched.length, 2);
    assert.ok(f.sent.filter(({ data }) => data.type !== "merge").every(({ data }) => data.bag.blobs.length === 2));
  } finally { f.pool.close(); }
});

test("the default browser fetch keeps the Window receiver", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async function () {
    assert.equal(this, globalThis);
    return { ok: true, blob: async () => new Blob(["test"]) };
  };
  const f = fixture();
  f.pool.baseURL = "https://example.test/app/check.html";
  try {
    assert.equal((await f.pool.bag({ urls: ["bag.db3"] })).blobs.length, 1);
  } finally { f.pool.close(); globalThis.fetch = original; }
});

test("a short check uses the first worker without waiting for a second startup", async () => {
  const f = fixture(null, { holdSecond: true });
  try {
    const result = await f.pool.run(message(), 2);
    assert.equal(result.execution.workers, 1);
    assert.equal(result.execution.requested_workers, 2);
    assert.equal(result.execution.runtime_count, 2);
    assert.equal(result.execution.second_worker_ready_seconds, null);
    assert.deepEqual(result.execution.partition_workers, [1, 1]);
    assert.ok(f.sent.every(({ index }) => index === 0));
    assert.equal(f.sent.at(-1).data.request.workers_used, 1);
    // Once the idle runtime is ready, the next check can use both immediately.
    f.workers[1].emit({ type: "ready", version: "test" });
    assert.equal((await f.pool.run(message(), 2)).execution.workers, 2);
  } finally { f.pool.close(); }
});

test("an unavailable auxiliary runtime falls back to the first worker", async () => {
  const f = fixture(null, { holdSecond: true });
  try {
    const running = f.pool.run(message(), 2);
    f.workers[1].emit({ type: "failed", text: "could not load second Python" });
    const result = await running;
    assert.equal(result.execution.workers, 1);
    assert.equal(result.execution.runtime_count, 1);
    assert.ok(!f.pool.closed && !f.workers[0].closed && f.workers[1].closed);
    assert.deepEqual(result.execution.partition_workers, [1, 1]);
  } finally { f.pool.close(); }
});

test("a late idle startup failure does not disable a completed check or its primary worker", async () => {
  const f = fixture(null, { holdSecond: true });
  try {
    assert.ok((await f.pool.run(message(), 2)).ok);
    f.workers[1].emit({ type: "failed", text: "late startup failure" });
    assert.ok(!f.pool.closed && !f.workers[0].closed);
    assert.ok((await f.pool.run(message(), 1)).ok);
  } finally { f.pool.close(); }
});
