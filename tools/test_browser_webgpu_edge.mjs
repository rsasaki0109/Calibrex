// Lifecycle and transport tests. Real shader arithmetic is checked by the browser probe.
import assert from "node:assert/strict";
import test from "node:test";
import "../docs/app/edge-gpu.js";
globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, UNIFORM: 4, COPY_SRC: 8, MAP_READ: 16 };
globalThis.GPUShaderStage = { COMPUTE: 1 };
globalThis.GPUMapMode = { READ: 1 };
const { create, createFromPython, createContext } = globalThis.CalibrexEdgeGpu;

function input() {
  const samples = new Float32Array(12);
  samples.set([0, 0, 2, 1]);
  new Uint32Array(samples.buffer).set([0, 2, 2, 0], 4);
  samples[9] = 1;
  const camera = new Float32Array(16);
  camera.set([1, 1, 0.5, 0.5]);
  return { samples, images: new Float32Array([1, 2, 3, 4]), camera, totalWeight: 1 };
}
function fixture({ map = null, compile = null, validation = null, requestDevice = null } = {}) {
  const buffers = [], writes = [], dispatches = [];
  let resolveLost;
  const device = {
    destroyed: 0, lost: new Promise((resolve) => { resolveLost = resolve; }),
    queue: { writeBuffer(...args) { writes.push(args); }, submit() {} },
    createBuffer({ size }) {
      const buffer = { size, destroyed: 0, destroy() { this.destroyed++; },
        mapAsync: map || (async () => {}), getMappedRange(_offset, bytes) { return new ArrayBuffer(bytes); }, unmap() {},
      };
      buffers.push(buffer);
      return buffer;
    },
    createBindGroupLayout: (x) => x, createPipelineLayout: (x) => x,
    createBindGroup: (x) => x, createShaderModule: (x) => x,
    createComputePipelineAsync: compile || (async (x) => x),
    pushErrorScope() {}, async popErrorScope() { return validation; },
    createCommandEncoder: () => ({
      beginComputePass: () => ({ setBindGroup() {}, setPipeline() {},
        dispatchWorkgroups(...args) { dispatches.push(args); }, end() {} }),
      copyBufferToBuffer() {}, finish() { return {}; },
    }),
    destroy() { this.destroyed++; resolveLost({}); },
  };
  const adapter = { limits: { maxStorageBufferBindingSize: 128 * 1024 ** 2 }, requestDevice: requestDevice || (async () => device) };
  const gpu = { requests: 0, async requestAdapter() { this.requests++; return adapter; } };
  return { gpu, device, buffers, writes, dispatches, lose: () => resolveLost({}) };
}

test("rejects unsupported GPU, invalid arrays, denominator, address and resource limits before submission", async () => {
  await assert.rejects(create(input(), { gpu: null }), /unavailable/);
  for (const mutate of [
    (x) => { x.images = [1]; }, (x) => { x.samples = new Float32Array(11); },
    (x) => { x.totalWeight = 2; }, (x) => { x.samples[3] = -1; },
    (x) => { x.images[0] = NaN; }, (x) => { x.camera[12] = 4; },
    (x) => { new Uint32Array(x.samples.buffer)[4] = 10; },
  ]) {
    const x = input(), f = fixture(); mutate(x);
    await assert.rejects(create(x, { gpu: f.gpu }));
    assert.equal(f.gpu.requests, 0);
  }
  await assert.rejects(create(input(), { gpu: fixture().gpu, maxResidentBytes: 1 }), /budget/);
  await assert.rejects(create(input(), { gpu: fixture().gpu, timeoutMs: 0 }), /limits/);
});

test("uploads inputs once, chunks every pose in order, reads compact scores, and closes once", async () => {
  const f = fixture(), session = await create(input(), { gpu: f.gpu });
  assert.equal(f.writes.length, 3);
  const rows = new Float32Array(131 * 12);
  const scores = await session.scoreMany(rows);
  assert.equal(scores.length, 131);
  assert.deepEqual(f.dispatches, [[1, 64], [64], [1, 64], [64], [1, 3], [3]]);
  assert.equal(f.writes.length, 6);
  assert.equal(session.allocatedBytes, f.buffers.reduce((sum, x) => sum + x.size, 0));
  assert.equal((await session.scoreMany(new Float32Array())).length, 0);
  await assert.rejects(session.scoreMany(new Float32Array([NaN])), /poses/);
  session.destroy(); session.destroy();
  assert.equal(f.device.destroyed, 1);
  assert.ok(f.buffers.every((x) => x.destroyed === 1));
  await assert.rejects(session.scoreMany(rows), /closed/);
});

test("readback failure destroys owned buffers and propagates", async () => {
  const f = fixture({ map: async () => { throw new Error("readback failed"); } });
  const session = await create(input(), { gpu: f.gpu });
  await assert.rejects(session.scoreMany(new Float32Array(12)), /readback failed/);
  assert.equal(f.device.destroyed, 1);
  assert.ok(f.buffers.every((x) => x.destroyed === 1));
});

test("finite readback timeout closes device and rejects overlapping batches", async () => {
  const f = fixture({ map: () => new Promise(() => {}) });
  const session = await create(input(), { gpu: f.gpu, timeoutMs: 10 });
  const pending = session.scoreMany(new Float32Array(12));
  await assert.rejects(session.scoreMany(new Float32Array(12)), /overlapping/);
  await assert.rejects(pending, /timed out/);
  assert.equal(f.device.destroyed, 1);
});

test("device loss is an explicit error", async () => {
  const f = fixture(), session = await create(input(), { gpu: f.gpu });
  f.lose(); await Promise.resolve();
  await assert.rejects(session.scoreMany(new Float32Array(12)), /device was lost/);
  session.destroy();
});

test("compile and initialization validation failures release resources", async () => {
  for (const options of [{ compile: async () => { throw new Error("bad shader"); } }, { validation: { message: "invalid" } }]) {
    const f = fixture(options);
    await assert.rejects(create(input(), { gpu: f.gpu }));
    assert.equal(f.device.destroyed, 1);
    assert.ok(f.buffers.every((x) => x.destroyed === 1));
  }
});

test("a device returned after initialization timeout is destroyed", async () => {
  let release;
  const f = fixture({ requestDevice: () => new Promise((resolve) => { release = resolve; }) });
  await assert.rejects(create(input(), { gpu: f.gpu, timeoutMs: 10 }), /timed out/);
  release(f.device); await Promise.resolve(); await Promise.resolve();
  assert.equal(f.device.destroyed, 1);
});

test("Python views are copied using offsets and released even on invalid strides", async () => {
  const x = input(), f = fixture(), releases = [], destroyed = [];
  const payload = { get(key) {
    if (key === "totalWeight") return x[key];
    const padded = new Float32Array(x[key].length + 4); padded.set(x[key], 2);
    return { getBuffer() { return { data: padded, offset: 2, shape: [x[key].length], strides: [1], release() { releases.push(key); } }; },
      destroy() { destroyed.push(key); } };
  } };
  const session = await createFromPython(payload, { gpu: f.gpu });
  assert.deepEqual(releases, ["samples", "images", "camera"]);
  assert.deepEqual(destroyed, releases);
  session.destroy();
  let released = false;
  assert.throws(() => createFromPython({ get() { return { getBuffer() { return {
    shape: [1], strides: [2], release() { released = true; },
  }; }, destroy() {} }; } }), /contiguous/);
  assert.equal(released, true);
});

test("calibration context reuses one device and pipeline pair across released folds", async () => {
  let compiles = 0;
  const f = fixture({ compile: async (x) => { compiles++; return x; } });
  const context = await createContext({ gpu: f.gpu });
  const first = await context.create(input()), bytes = first.allocatedBytes;
  await first.scoreMany(new Float32Array(12));
  first.destroy();
  assert.equal(context.allocatedBytes, 0);
  assert.equal(f.device.destroyed, 0);
  const second = await context.create(input());
  assert.equal(second.layout, first.layout);
  assert.equal(compiles, 2);
  assert.equal(f.gpu.requests, 1);
  assert.equal(context.peakAllocatedBytes, bytes);
  context.destroy(); context.destroy();
  assert.equal(context.allocatedBytes, 0);
  assert.ok(f.buffers.every((buffer) => buffer.destroyed === 1));
  assert.equal(f.device.destroyed, 1);
  await assert.rejects(second.scoreMany(new Float32Array(12)), /closed/);
  await assert.rejects(context.create(input()), /closed/);
});

test("context budget includes every live objective and recovers when buffers are released", async () => {
  const f = fixture(), context = await createContext({ gpu: f.gpu, maxResidentBytes: 4000 });
  const first = await context.create(input());
  await assert.rejects(context.create(input()), /budget/);
  assert.equal(context.closed, false);
  first.destroy();
  await context.create(input());
  context.destroy();
  assert.equal(context.allocatedBytes, 0);
});

test("context readback failure and device loss close every live objective", async () => {
  for (const lose of [false, true]) {
    const f = fixture({ map: lose ? null : async () => { throw new Error("readback failure"); } });
    const context = await createContext({ gpu: f.gpu });
    const first = await context.create(input()), second = await context.create(input());
    if (lose) { f.lose(); await Promise.resolve(); }
    await assert.rejects(first.scoreMany(new Float32Array(12)), lose ? /lost/ : /readback failure/);
    await assert.rejects(second.scoreMany(new Float32Array(12)), lose ? /lost/ : /readback failure/);
    assert.equal(context.closed, true);
    assert.equal(context.allocatedBytes, 0);
    assert.equal(f.device.destroyed, 1);
  }
});

test("context rejects overlapping operations and releases partial initialization", async () => {
  const f = fixture({ map: () => new Promise(() => {}) });
  const context = await createContext({ gpu: f.gpu, timeoutMs: 10 });
  const first = await context.create(input()), second = await context.create(input());
  const pending = first.scoreMany(new Float32Array(12));
  await assert.rejects(second.scoreMany(new Float32Array(12)), /overlapping/);
  await assert.rejects(context.create(input()), /overlapping/);
  await assert.rejects(pending, /timed out/);
  const broken = fixture({ compile: async () => { throw new Error("bad shader"); } });
  const failed = await createContext({ gpu: broken.gpu });
  await assert.rejects(failed.create(input()), /bad shader/);
  assert.equal(failed.allocatedBytes, 0);
  assert.ok(broken.buffers.every((buffer) => buffer.destroyed === 1));
  assert.equal(broken.device.destroyed, 1);
});

test("resident frame masks retain high bits, subset denominator and full-objective reset", async () => {
  const x = input(), f = fixture();
  const samples = new Float32Array(4 * 12);
  const groups = [0, 31, 32, 63];
  for (let i = 0; i < 4; i++) {
    samples.set(x.samples, i * 12); samples[i * 12 + 3] = i + 1;
    new Uint32Array(samples.buffer)[i * 12 + 7] = groups[i];
  }
  const session = await create({...x, samples, totalWeight:10}, {gpu:f.gpu});
  session.readBuffer.getMappedRange = (_offset, bytes) => new Float32Array(bytes / 4).fill(48).buffer;
  const result = await session.scoreMany(new Float32Array(12), {low:1, high:0x80000000, totalWeight:5});
  assert.deepEqual([...result], [9.6]);
  assert.deepEqual([...f.writes.find((row) => row[1] === 68)[2]], [1, 1, 0x80000000]);
  for (const selection of [{low:1, high:0, totalWeight:2}, {low:-1, high:0, totalWeight:1}]) {
    await assert.rejects(session.scoreMany(new Float32Array(12), selection), /selection|denominator/);
  }
  assert.deepEqual([...(await session.scoreMany(new Float32Array(12)))], [4.8]);
  assert.deepEqual([...f.writes.filter((row) => row[1] === 68).at(-1)[2]], [0, 0, 0]);
  assert.equal(session.closed, false);
  session.destroy();
});
