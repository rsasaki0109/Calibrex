// Independent Pyodide runtimes need neither SharedArrayBuffer nor special HTTP headers.
// Keep this controller independent of the DOM so cancellation and worker failures can be tested.
"use strict";
(() => {
  function partitionPairs(pairs, plan = []) {
    const remaining = new Set(pairs);
    const groups = [];
    const take = (names) => {
      const group = names.filter((name) => remaining.delete(name));
      if (group.length) groups.push(group);
    };
    if (remaining.has("gnss-imu")) take(["imu-lidar", "gnss-lidar", "gnss-imu"]);
    if (remaining.has("camera-focal")) take(["camera-imu", "camera-focal"]);
    // These pairs reuse one worker's LiDAR-odometry memo instead of computing it twice.
    take(["lidar-vehicle", "ins-lidar", "lidar-wheel_odometry"]);
    for (const pair of remaining) groups.push([pair]);
    if (groups.length < 2) return [pairs.slice()];
    const weight = (group) => Math.max(1, plan.filter(
      (pair) => pair.status === "planned" && group.includes(pair.pair),
    ).length);
    groups.sort((a, b) => weight(b) - weight(a));
    const bins = [[], []], loads = [0, 0];
    for (const group of groups) {
      const at = loads[0] <= loads[1] ? 0 : 1;
      bins[at].push(...group); loads[at] += weight(group);
    }
    // A partition changes execution order only between independent groups.
    return bins.map((bin) => pairs.filter((pair) => bin.includes(pair)));
  }

  class WorkerClient {
    constructor(factory, index, callbacks) {
      this.index = index;
      this.callbacks = callbacks;
      this.pending = new Map();
      this.nextId = 0;
      this.closed = false;
      this.worker = factory();
      this.ready = new Promise((resolve, reject) => {
        this.resolveReady = resolve; this.rejectReady = reject;
      });
      this.ready.catch(() => {});
      this.bootTimer = setTimeout(() => this.fail(new Error("Python loading timed out")), 300000);
      this.worker.onmessage = ({ data }) => {
        if (this.closed) return;
        if (data.type === "ready") {
          clearTimeout(this.bootTimer);
          this.resolveReady(data.version);
          if (index === 0) callbacks.ready(data.version);
        } else if (data.type === "failed") {
          this.fail(new Error(data.text));
        } else if (data.type === "log") {
          callbacks.log((index ? "Worker 2: " : "") + data.text);
        } else {
          const pending = this.pending.get(data.id);
          if (!pending) return; // Messages from cancelled/finished requests cannot affect a new run.
          if (data.type === "progress") pending.progress?.(data.event);
          else if (data.type === "result") {
            this.pending.delete(data.id);
            pending.resolve(data.payload);
          }
        }
      };
      this.worker.onerror = (event) => this.fail(new Error(event.message || "Worker failed"));
      this.worker.onmessageerror = () => this.fail(new Error("Worker message could not be read"));
    }

    async call(message, progress) {
      await this.ready;
      if (this.closed) throw new Error("Worker is closed");
      if (this.pending.size) throw new Error("Worker is already running a request");
      const id = ++this.nextId;
      return new Promise((resolve, reject) => {
        this.pending.set(id, { resolve, reject, progress });
        try { this.worker.postMessage({ ...message, id }); }
        catch (error) { this.pending.delete(id); reject(error); }
      });
    }

    fail(error) {
      if (this.closed) return;
      this.close(error);
      this.callbacks.failed(error, this.index);
    }

    close(error = new Error("Cancelled")) {
      if (this.closed) return;
      this.closed = true;
      clearTimeout(this.bootTimer);
      this.worker.terminate();
      this.rejectReady(error);
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    }
  }

  class CheckWorkerPool {
    constructor({ factory = () => new Worker("check-worker.js"), log = () => {},
      ready = () => {}, failed = () => {}, progress = () => {}, baseURL = globalThis.location?.href,
      fetcher = globalThis.fetch?.bind(globalThis) } = {}) {
      this.factory = factory;
      this.auxiliaryBusy = false;
      this.callbacks = { log, ready, failed: (error, index) => {
        if (index === 1 && !this.auxiliaryBusy) {
          log("Second Python worker unavailable; continuing with the first worker.");
          return;
        }
        this.close(error); failed(error, index);
      } };
      this.progress = progress;
      this.baseURL = baseURL;
      this.fetcher = fetcher;
      this.clients = [new WorkerClient(factory, 0, this.callbacks)];
      this.sampleBlobs = new Map();
      this.running = false;
      this.closed = false;
    }

    async bag(input) {
      if (!input.urls) return input;
      const key = JSON.stringify(input.urls);
      if (!this.sampleBlobs.has(key)) {
        const pending = Promise.all(input.urls.map(async (url) => {
          const absolute = new URL(url, this.baseURL);
          const response = await this.fetcher(absolute);
          if (!response.ok) throw new Error("Could not fetch " + url + " (" + response.status + ")");
          return { name: absolute.pathname.split("/").pop(), data: await response.blob() };
        }));
        this.sampleBlobs.set(key, pending);
        pending.catch(() => this.sampleBlobs.delete(key));
      }
      return { blobs: await this.sampleBlobs.get(key) };
    }

    async plan(message) {
      return this.clients[0].call({ ...message, bag: await this.bag(message.bag) });
    }

    async run(message, workers = 1, plan = []) {
      if (this.running || this.closed) throw new Error("The worker pool is not available");
      this.running = true;
      const started = performance.now();
      try {
        const groups = workers === 2 && message.request.mode !== "estimate"
          ? partitionPairs(message.request.pairs, plan) : [message.request.pairs];
        if (groups.length === 1 && this.clients[1]) { this.clients.pop().close(); }
        const bootStarted = performance.now();
        if (groups.length === 2 && (!this.clients[1] || this.clients[1].closed)) {
          this.callbacks.log("Starting a second Python worker (extra memory and startup time)…");
          this.clients[1] = new WorkerClient(this.factory, 1, this.callbacks);
        }
        let secondReadySeconds = groups.length === 2 ? null : 0;
        const secondReady = groups.length === 2 ? this.clients[1].ready.then(() => {
          secondReadySeconds = (performance.now() - bootStarted) / 1000;
          return true;
        }, () => false) : Promise.resolve(false);
        const bag = await this.bag(message.bag);
        if (this.closed) throw new Error("Cancelled");
        const totals = groups.map((group) => plan.filter(
          (pair) => pair.status === "planned" && group.includes(pair.pair),
        ).length);
        const taskWorkers = [];
        const execute = (index, clientIndex) => {
          taskWorkers[index] = clientIndex;
          if (clientIndex === 1) this.auxiliaryBusy = true;
          return this.clients[clientIndex].call({
            ...message, bag, request: { ...message.request, pairs: groups[index],
              scan_memory_mb: Math.floor(512 / groups.length), parallel_shard: groups.length === 2 },
          }, (event) => {
            if (event.event === "run_started") {
              totals[index] = event.total_pairs;
              this.progress({ ...event, total_pairs: totals.reduce((a, b) => a + b, 0) });
            } else if (event.event !== "run_finished") {
              this.progress({ ...event, worker: groups.length === 2 ? clientIndex + 1 : null });
            }
          }).then((reply) => {
            if (!reply.ok) throw new Error(reply.error || "Worker returned no result");
            return reply;
          }).finally(() => { if (clientIndex === 1) this.auxiliaryBusy = false; });
        };
        const first = execute(0, 0);
        let replies;
        if (groups.length === 2) {
          // Whichever becomes available first takes the remaining partition.
          // A short check never waits solely for a second Python startup.
          const useSecond = await Promise.race([secondReady, first.then(() => false)]);
          const second = useSecond ? execute(1, 1) : first.then(() => execute(1, 0));
          if (!useSecond) this.callbacks.log("The first worker continued the remaining pair group while the second was preparing.");
          replies = await Promise.all([first, second]);
        } else replies = [await first];
        const workersUsed = new Set(taskWorkers).size;
        const shardSeconds = replies.map((reply) => reply.seconds);
        let payload = replies[0];
        if (groups.length === 2) {
          payload = await this.clients[0].call({ type: "merge",
            request: { ...message.request, workers_used: workersUsed }, shards: replies });
          if (!payload.ok) throw new Error(payload.error || "Could not combine the check results");
        }
        payload.seconds = (performance.now() - started) / 1000;
        const memory = this.clients.map(() => null);
        replies.forEach((reply, index) => {
          const clientIndex = taskWorkers[index];
          if (reply.allocated_wasm_bytes != null) memory[clientIndex] = Math.max(
            memory[clientIndex] || 0, reply.allocated_wasm_bytes);
        });
        if (payload.allocated_wasm_bytes != null) memory[0] = Math.max(memory[0] || 0, payload.allocated_wasm_bytes);
        payload.execution = {
          workers: workersUsed, requested_workers: workers,
          runtime_count: this.clients.filter((client) => !client.closed).length,
          partitions: groups, partition_workers: taskWorkers.map((index) => index + 1),
          wall_seconds: payload.seconds, second_worker_ready_seconds: secondReadySeconds,
          shard_seconds: shardSeconds,
          allocated_wasm_bytes: memory,
          scan_cache_budget_mb: 512, scan_cache_mb_per_worker: Math.floor(512 / groups.length),
        };
        this.progress({ event: "run_finished" });
        return payload;
      } catch (error) {
        // A failed shard must not leave the other worker consuming CPU in the background.
        this.close(error);
        throw error;
      } finally { this.running = false; }
    }

    close(error) {
      this.closed = true;
      for (const client of this.clients) client.close(error);
      this.sampleBlobs.clear();
    }
  }

  globalThis.CalibrexCheckWorkers = { CheckWorkerPool, partitionPairs };
})();
