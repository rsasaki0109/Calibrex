// Experimental asynchronous batches for evaluation.camera_lidar_edge.EdgeObjective.
// No frame dropping, shared memory, service worker, or changes to the check solver.
// Inputs stay on the device. Two reductions return only one float per candidate.
"use strict";

(() => {
  const GROUP = 128, CAPACITY = 64, MIB = 1024 * 1024;
  const shader = `
struct Sample { point: vec4f, image: vec4u, sampling: vec4f }
struct Pose { r0: vec4f, r1: vec4f, r2: vec4f }
struct Camera { intrinsics: vec4f, k0: vec4f, k1: vec4f, sizes: vec4u, flags: vec4u }
@group(0) @binding(0) var<storage, read> samples: array<Sample>;
@group(0) @binding(1) var<storage, read> images: array<f32>;
@group(0) @binding(2) var<storage, read> poses: array<Pose>;
@group(0) @binding(3) var<uniform> camera: Camera;
@group(0) @binding(4) var<storage, read_write> partial: array<f32>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>;
var<workgroup> sums: array<f32, 128>;

// WGSL's builtin atan permits 4096 ULP of f32 error, enough to move a
// fisheye projection across an image boundary. Reduce to |t| <= tan(pi/8)
// and evaluate the alternating series through t^17 instead. Its exact-real
// truncation error is below 3e-9 rad; f32 rounding is checked by the probe.
// https://www.w3.org/TR/WGSL/#accuracy-of-builtin-functions
fn projection_atan(radius: f32) -> f32 {
  var t = radius;
  var base = 0.0;
  if (radius > 2.414213562373095) {
    t = -1.0 / radius;
    base = 1.5707963267948966;
  } else if (radius > 0.41421356237309503) {
    t = (radius - 1.0) / (radius + 1.0);
    base = 0.7853981633974483;
  }
  let t2 = t * t;
  let series = -1.0/3.0 + t2 * (1.0/5.0 + t2 * (-1.0/7.0 + t2 *
    (1.0/9.0 + t2 * (-1.0/11.0 + t2 * (1.0/13.0 + t2 * (-1.0/15.0 + t2/17.0))))));
  return base + (t + t * t2 * series);
}

fn response(index: u32, pose_index: u32) -> f32 {
  let s = samples[index];
  if (camera.flags.y != 0u) {
    let mask = select(camera.flags.z, camera.flags.w, s.image.w >= 32u);
    if (((mask >> (s.image.w & 31u)) & 1u) == 0u) { return 0.0; }
  }
  let p = poses[pose_index];
  let v = vec4f(s.point.xyz, 1.0);
  let moved = vec3f(dot(p.r0, v), dot(p.r1, v), dot(p.r2, v));
  if (moved.z <= 0.001) { return 0.0; }
  var xy = moved.xy / moved.z;
  let r2 = dot(xy, xy);
  if (camera.sizes.w == 2u) {
    let radius = sqrt(r2);
    let theta = projection_atan(radius);
    let t2 = theta * theta;
    let k = camera.k0;
    let distorted = theta * (1.0 + t2 * (k.x + t2 * (k.y + t2 * (k.z + t2 * k.w))));
    if (radius > 1e-12) { xy *= distorted / radius; }
  } else if (camera.sizes.w == 1u) {
    let k = camera.k0;
    var radial = 1.0 + r2 * (k.x + r2 * (k.y + r2 * camera.k1.x));
    if (camera.flags.x != 0u) {
      radial /= 1.0 + r2 * (camera.k1.y + r2 * (camera.k1.z + r2 * camera.k1.w));
    }
    xy = vec2f(xy.x * radial + 2.0 * k.z * xy.x * xy.y + k.w * (r2 + 2.0 * xy.x * xy.x),
               xy.y * radial + k.z * (r2 + 2.0 * xy.y * xy.y) + 2.0 * k.w * xy.x * xy.y);
  }
  let uv = (camera.intrinsics.xy * xy + camera.intrinsics.zw - s.sampling.x) * s.sampling.y;
  // Positive tests also reject NaN/overflow from a singular distortion denominator.
  if (!(uv.x >= 0.0 && uv.y >= 0.0 && uv.x < f32(s.image.y - 1u) && uv.y < f32(s.image.z - 1u))) {
    return 0.0;
  }
  let left = u32(uv.x);
  let top = u32(uv.y);
  let d = uv - vec2f(f32(left), f32(top));
  let base = s.image.x + top * s.image.y + left;
  let value = (1.0 - d.x) * ((1.0 - d.y) * images[base] + d.y * images[base + s.image.y])
            + d.x * ((1.0 - d.y) * images[base + 1u] + d.y * images[base + s.image.y + 1u]);
  return value * s.point.w;
}

@compute @workgroup_size(128)
fn project(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
  let index = group.x * 128u + lane;
  var value = 0.0;
  if (index < camera.sizes.x) { value = response(index, group.y); }
  sums[lane] = value;
  workgroupBarrier();
  for (var step = 64u; step > 0u; step /= 2u) {
    if (lane < step) { sums[lane] += sums[lane + step]; }
    workgroupBarrier();
  }
  if (lane == 0u) { partial[group.y * camera.sizes.y + group.x] = sums[0]; }
}

@compute @workgroup_size(128)
fn reduce(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
  var value = 0.0;
  for (var index = lane; index < camera.sizes.y; index += 128u) {
    value += partial[group.x * camera.sizes.y + index];
  }
  sums[lane] = value;
  workgroupBarrier();
  for (var step = 64u; step > 0u; step /= 2u) {
    if (lane < step) { sums[lane] += sums[lane + step]; }
    workgroupBarrier();
  }
  if (lane == 0u) { scores[group.x] = sums[0]; }
}`;

  function validateInput(input) {
    for (const key of ["samples", "images", "camera"]) {
      if (!(input?.[key] instanceof Float32Array)) throw new TypeError(key + " must be Float32Array");
    }
    const { samples, images, camera, totalWeight } = input;
    if (!samples.length || samples.length % 12 || camera.length !== 16 || !images.length) {
      throw new RangeError("invalid packed edge dimensions");
    }
    if (!Number.isFinite(totalWeight) || totalWeight <= 0 || !camera.every(Number.isFinite)) {
      throw new RangeError("edge camera and total weight must be finite, with positive total weight");
    }
    if (![0, 1, 2].includes(camera[12]) || ![0, 1].includes(camera[13])) throw new RangeError("unsupported camera model");
    if (!images.every(Number.isFinite)) throw new RangeError("edge images must be finite");
    const words = new Uint32Array(samples.buffer, samples.byteOffset, samples.length);
    let sum = 0;
    for (let at = 0; at < samples.length; at += 12) {
      if (![0, 1, 2, 3, 8, 9].every((i) => Number.isFinite(samples[at + i])) || samples[at + 3] < 0 || samples[at + 9] <= 0) {
        throw new RangeError("invalid edge sample");
      }
      const [offset, width, height] = words.subarray(at + 4, at + 7);
      if (width < 2 || height < 2 || offset + width * height > images.length) throw new RangeError("edge sample is outside image storage");
      sum += samples[at + 3];
    }
    if (Math.abs(sum - totalWeight) > Math.max(1e-12, sum * 1e-12)) throw new RangeError("edge denominator differs from complete input weights");
    return samples.length / 12;
  }

  // Make owned copies before awaiting anything: WASM heap growth must not detach
  // a borrowed NumPy view while GPU initialization/readback is pending.
  function copyPythonArray(proxy) {
    const view = proxy.getBuffer("f32");
    try {
      if (view.shape.length !== 1 || view.strides[0] !== 1) throw new TypeError("expected a contiguous one-dimensional Python buffer");
      return new Float32Array(view.data.subarray(view.offset, view.offset + view.shape[0]));
    }
    finally { view.release(); }
  }
  function withTimeout(promise, milliseconds) {
    let timer;
    return Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("WebGPU edge operation timed out")), milliseconds);
    })]).finally(() => clearTimeout(timer));
  }

  class EdgeBatch {
    constructor(device, input, count, timeoutMs, owner = null) {
      this.device = device;
      this.owner = owner;
      this.totalWeight = input.totalWeight;
      this.count = count;
      this.groups = Math.ceil(count / GROUP);
      this.timeoutMs = timeoutMs;
      this.buffers = [];
      this.closed = false;
      this.busy = false;
      this.failure = null;
      this.selectionKey = null;
      this.frameWeights = new Float64Array(64);
      this.groupedInput = true;
      const words = new Uint32Array(input.samples.buffer, input.samples.byteOffset, input.samples.length);
      for (let at = 0; at < input.samples.length; at += 12) {
        if (words[at + 7] >= 64) this.groupedInput = false;
        else this.frameWeights[words[at + 7]] += input.samples[at + 3];
      }
      owner?.sessions.add(this);
      if (!owner) device.lost.then(() => {
        if (!this.closed) { this.failure = new Error("WebGPU edge device was lost"); this.destroy(); }
      });
      if (!owner) device.addEventListener?.("uncapturederror", (event) => {
        event.preventDefault();
        if (!this.closed) { this.failure = new Error("WebGPU edge uncaptured device error"); this.destroy(); }
      });
      const make = (size, usage) => {
        const buffer = device.createBuffer({ size: Math.max(4, size), usage });
        this.buffers.push(buffer);
        if (owner) {
          owner.allocatedBytes += buffer.size;
          owner.peakAllocatedBytes = Math.max(owner.peakAllocatedBytes, owner.allocatedBytes);
        }
        return buffer;
      };
      const upload = (data, usage) => {
        const buffer = make(data.byteLength, usage | GPUBufferUsage.COPY_DST);
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      const pointBuffer = upload(input.samples, GPUBufferUsage.STORAGE);
      const imageBuffer = upload(input.images, GPUBufferUsage.STORAGE);
      this.poseBuffer = make(CAPACITY * 48, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const parameters = new ArrayBuffer(80);
      new Float32Array(parameters, 0, 12).set(input.camera.subarray(0, 12));
      new Uint32Array(parameters, 48, 8).set([count, this.groups, 0, input.camera[12], input.camera[13], 0, 0, 0]);
      const cameraBuffer = upload(new Uint8Array(parameters), GPUBufferUsage.UNIFORM);
      this.cameraBuffer = cameraBuffer;
      this.partialBuffer = make(CAPACITY * this.groups * 4, GPUBufferUsage.STORAGE);
      this.scoreBuffer = make(CAPACITY * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      this.readBuffer = make(CAPACITY * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const layout = owner?.bindingLayout || device.createBindGroupLayout({ entries: [
        ...[0, 1, 2].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } })),
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ...[4, 5].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } })),
      ] });
      this.layout = owner?.layout || device.createPipelineLayout({ bindGroupLayouts: [layout] });
      if (owner) { owner.bindingLayout = layout; owner.layout = this.layout; }
      this.bindings = device.createBindGroup({ layout, entries: [pointBuffer, imageBuffer, this.poseBuffer, cameraBuffer, this.partialBuffer, this.scoreBuffer]
        .map((buffer, binding) => ({ binding, resource: { buffer } })) });
      this.allocatedBytes = this.buffers.reduce((sum, buffer) => sum + buffer.size, 0);
    }

    async initialize() {
      let pipelines = this.owner?.pipelines;
      if (!pipelines) {
        const module = this.device.createShaderModule({ code: shader });
        pipelines = Promise.all(["project", "reduce"].map((entryPoint) =>
          this.device.createComputePipelineAsync({ layout: this.layout, compute: { module, entryPoint } })));
        if (this.owner) this.owner.pipelines = pipelines;
      }
      [this.projectPipeline, this.reducePipeline] = await pipelines;
    }

    async scoreMany(rows, selection = null) {
      if (this.closed || this.failure) throw this.failure || new Error("WebGPU edge objective is closed");
      if (this.busy || this.owner?.busy) throw new Error("overlapping WebGPU edge batches are not supported");
      if (!(rows instanceof Float32Array) || rows.length % 12 || !rows.every(Number.isFinite)) {
        throw new TypeError("poses must be finite Float32Array rows of 3 by 4 matrices");
      }
      let denominator = this.totalWeight;
      if (selection) {
        const { low, high, totalWeight } = selection;
        if (!this.groupedInput || ![low, high].every((x) => Number.isInteger(x) && x >= 0 && x <= 0xffffffff) || !Number.isFinite(totalWeight) || totalWeight <= 0) throw new RangeError("invalid WebGPU frame selection");
        let sum = 0;
        for (let frame = 0; frame < 64; frame++) {
          if (((frame < 32 ? low : high) >>> (frame & 31)) & 1) sum += this.frameWeights[frame];
        }
        if (Math.abs(sum - totalWeight) > Math.max(1e-12, sum * 1e-12)) throw new RangeError("subset denominator differs from selected frame weights");
        denominator = totalWeight;
      }
      const output = new Float64Array(rows.length / 12);
      this.busy = true;
      if (this.owner) this.owner.busy = true;
      try {
        const key = selection ? selection.low + ":" + selection.high : null;
        if (key !== this.selectionKey) {
          this.device.queue.writeBuffer(this.cameraBuffer, 68,
            new Uint32Array(selection ? [1, selection.low, selection.high] : [0, 0, 0]));
          this.selectionKey = key;
        }
        for (let from = 0; from < output.length; from += CAPACITY) {
          if (this.closed || this.failure) throw this.failure || new Error("WebGPU edge objective is closed");
          const size = Math.min(CAPACITY, output.length - from);
          this.device.pushErrorScope("validation");
          let scopeOpen = true;
          try {
            this.device.queue.writeBuffer(this.poseBuffer, 0, rows.subarray(from * 12, (from + size) * 12));
            const encoder = this.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setBindGroup(0, this.bindings);
            pass.setPipeline(this.projectPipeline);
            pass.dispatchWorkgroups(this.groups, size);
            pass.setPipeline(this.reducePipeline);
            pass.dispatchWorkgroups(size);
            pass.end();
            encoder.copyBufferToBuffer(this.scoreBuffer, 0, this.readBuffer, 0, size * 4);
            this.device.queue.submit([encoder.finish()]);
            await withTimeout(this.readBuffer.mapAsync(GPUMapMode.READ, 0, size * 4), this.timeoutMs);
            const values = new Float32Array(this.readBuffer.getMappedRange(0, size * 4));
            for (let index = 0; index < size; index++) output[from + index] = values[index] / denominator;
            this.readBuffer.unmap();
            const errorPromise = this.device.popErrorScope();
            scopeOpen = false;
            const error = await withTimeout(errorPromise, this.timeoutMs);
            if (error) throw new Error("WebGPU edge validation failed: " + error.message);
          } finally {
            if (scopeOpen) this.device.popErrorScope().catch(() => {});
          }
        }
        if (!output.every(Number.isFinite)) throw new Error("WebGPU edge returned nonfinite scores");
        return output;
      } catch (error) {
        this.failure = error;
        if (this.owner) this.owner.fail(error);
        else this.destroy();
        throw error;
      } finally { this.busy = false; if (this.owner) this.owner.busy = false; }
    }

    scoreFromPython(rows) { return this.scoreMany(copyPythonArray(rows)); }
    scoreSubsetFromPython(rows, low, high, totalWeight) {
      return this.scoreMany(copyPythonArray(rows), {low, high, totalWeight});
    }

    destroy() {
      if (this.closed) return;
      this.closed = true;
      for (const buffer of this.buffers) buffer.destroy();
      if (this.owner) {
        this.owner.sessions.delete(this);
        this.owner.allocatedBytes -= this.buffers.reduce((sum, buffer) => sum + buffer.size, 0);
      } else this.device.destroy();
    }
  }

  async function create(input, { gpu = globalThis.navigator?.gpu, timeoutMs = 15000, maxResidentBytes = 256 * MIB } = {}) {
    const count = validateInput(input);
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || !Number.isFinite(maxResidentBytes) || maxResidentBytes <= 0) throw new RangeError("invalid WebGPU resource limits");
    if (!gpu) throw new Error("WebGPU is unavailable");
    const groups = Math.ceil(count / GROUP);
    const largestStorage = Math.max(input.samples.byteLength, input.images.byteLength, CAPACITY * groups * 4, CAPACITY * 48);
    const bytes = input.samples.byteLength + input.images.byteLength + CAPACITY * groups * 4 + CAPACITY * (48 + 8) + 80;
    if (bytes > maxResidentBytes) throw new RangeError("WebGPU edge resident memory budget exceeded");
    const adapter = await withTimeout(gpu.requestAdapter({ powerPreference: "high-performance" }), timeoutMs);
    if (!adapter) throw new Error("WebGPU adapter is unavailable");
    // Use the portable default device limits rather than requesting adapter maxima.
    if (largestStorage > Math.min(adapter.limits.maxStorageBufferBindingSize, 128 * MIB) || groups > 65535) throw new RangeError("WebGPU edge input exceeds portable device limits");
    let session, device;
    const devicePromise = adapter.requestDevice();
    try {
      device = await withTimeout(devicePromise, timeoutMs);
      device.pushErrorScope("validation");
      session = new EdgeBatch(device, input, count, timeoutMs);
      await withTimeout(session.initialize(), timeoutMs);
      const error = await withTimeout(device.popErrorScope(), timeoutMs);
      if (error) throw new Error("WebGPU edge initialization failed: " + error.message);
      if (session.closed || session.failure) throw session.failure || new Error("WebGPU edge objective is closed");
      // Retain only this capability flag, never adapter names or driver details.
      session.fallbackAdapter = adapter.info?.isFallbackAdapter ?? null;
      return session;
    } catch (error) {
      if (session) session.destroy();
      else if (device) device.destroy();
      // A device acquired after the timeout still belongs to this failed attempt.
      else devicePromise.then((late) => late.destroy(), () => {});
      throw error;
    }
  }

  function unpackPythonInput(payload) {
    const input = { totalWeight: payload.get("totalWeight") };
    for (const key of ["samples", "images", "camera"]) {
      const proxy = payload.get(key);
      try { input[key] = copyPythonArray(proxy); }
      finally { proxy.destroy(); }
    }
    return input;
  }
  function createFromPython(payload, options) { return create(unpackPythonInput(payload), options); }

  // One device and pipeline pair for a complete asynchronous calibration. The
  // caller releases objective buffers before uploading the next fold. A shared
  // budget covers all live objectives; failures close the entire owned context.
  async function createContext({ gpu = globalThis.navigator?.gpu, timeoutMs = 15000, maxResidentBytes = 256 * MIB } = {}) {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || !Number.isFinite(maxResidentBytes) || maxResidentBytes <= 0) throw new RangeError("invalid WebGPU resource limits");
    if (!gpu) throw new Error("WebGPU is unavailable");
    const adapter = await withTimeout(gpu.requestAdapter({ powerPreference: "high-performance" }), timeoutMs);
    if (!adapter) throw new Error("WebGPU adapter is unavailable");
    const pending = adapter.requestDevice();
    let device;
    try { device = await withTimeout(pending, timeoutMs); }
    catch (error) { pending.then((late) => late.destroy(), () => {}); throw error; }
    const context = {
      device, timeoutMs, maxResidentBytes, sessions: new Set(), allocatedBytes: 0, peakAllocatedBytes: 0,
      closed: false, busy: false, failure: null,
      fallbackAdapter: adapter.info?.isFallbackAdapter ?? null,
      async create(input) {
        if (this.closed) throw this.failure || new Error("WebGPU edge context is closed");
        if (this.busy) throw new Error("overlapping WebGPU edge operations are not supported");
        const count = validateInput(input), groups = Math.ceil(count / GROUP);
        const largest = Math.max(input.samples.byteLength, input.images.byteLength, CAPACITY * groups * 4, CAPACITY * 48);
        const bytes = input.samples.byteLength + input.images.byteLength + CAPACITY * groups * 4 + CAPACITY * (48 + 8) + 80;
        if (bytes + this.allocatedBytes > maxResidentBytes) throw new RangeError("WebGPU edge resident memory budget exceeded");
        if (largest > Math.min(adapter.limits.maxStorageBufferBindingSize, 128 * MIB) || groups > 65535) throw new RangeError("WebGPU edge input exceeds portable device limits");
        this.busy = true;
        device.pushErrorScope("validation");
        let scopeOpen = true;
        try {
          const session = new EdgeBatch(device, input, count, timeoutMs, this);
          await withTimeout(session.initialize(), timeoutMs);
          const errorPromise = device.popErrorScope(); scopeOpen = false;
          const error = await withTimeout(errorPromise, timeoutMs);
          if (error) throw new Error("WebGPU edge initialization failed: " + error.message);
          if (this.closed || session.closed) throw this.failure || new Error("WebGPU edge context is closed");
          session.fallbackAdapter = this.fallbackAdapter;
          return session;
        } catch (error) { this.fail(error); throw error; }
        finally { if (scopeOpen) device.popErrorScope().catch(() => {}); this.busy = false; }
      },
      createFromPython(payload) { return this.create(unpackPythonInput(payload)); },
      fail(error) {
        this.failure = error;
        for (const session of this.sessions) session.failure = error;
        this.destroy();
      },
      destroy() {
        if (this.closed) return;
        this.closed = true;
        for (const session of [...this.sessions]) session.destroy();
        device.destroy();
      },
    };
    device.lost.then(() => { if (!context.closed) context.fail(new Error("WebGPU edge device was lost")); });
    device.addEventListener?.("uncapturederror", (event) => {
      event.preventDefault();
      if (!context.closed) context.fail(new Error("WebGPU edge uncaptured device error"));
    });
    return context;
  }
  globalThis.CalibrexEdgeGpu = { create, createFromPython, createContext };
})();
