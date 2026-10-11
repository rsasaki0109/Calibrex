# CUDA D2D profiling evidence

This is a diagnostic of the existing CUDA v0.3 backend, **not a new speedup**.
No projection kernels, solver settings, thresholds or calibration results change.
The measured tool commit is `36dedc3`; follow-up commits publish this evidence.

The public KITTI raw drive 0018 problem is the same digest-bound input used by
the [verified snapshot experiment](../verified_input_snapshot/README.md).
Exclude the five frozen holdout frame IDs to profile 20 fit frames,
2,503,637 points, and 1226 x 370 pinhole images at the declared initial pose.
Resolved histogram settings are unchanged. This does not measure recovery or
held-out accuracy; the benchmark schema's holdout anchor is the NumPy evaluation
of the same pose, not an independent accuracy split.

```powershell
python tools/profile_d2d_cuda.py `
  --problem outputs/public-kitti-six-dof-fused-cpu-v1/demo.problem.json `
  --output-dir outputs/public-kitti-cuda-stage-profile-v1 `
  --exclude-frame-id 0000000460 --exclude-frame-id 0000001035 `
  --exclude-frame-id 0000001611 --exclude-frame-id 0000002186 `
  --exclude-frame-id 0000002761 --repeats 10 --poses-per-repeat 20
```

Ten repetitions alternate ordinary/instrumented order. Each mode evaluates
20 poses per repetition, after three ordinary warmups. All 403 evaluations
exactly match one native NumPy evaluation, including frame records, visibility
and projected counts, skip records, MI and NMI. Every instrumented repetition
records 400 calls per kernel stage; stage means are retained in trial provenance.
This is one repeatedly evaluated pose, not 403 independent calibration trials.

On this shared i7-9750H / GTX 1660 Ti 6 GiB Windows host, the median per-pose
ordinary wall time is **5.558 ms**, versus **6.527 ms** with event instrumentation.
The median stage spans are:

| CUDA event stream span | Median per pose |
| --- | ---: |
| Initialization | 0.647 ms |
| Projection | 1.772 ms |
| Visibility selection | 0.297 ms |
| Histogram | 1.162 ms |

Events bracket launches in the current stream. Their spans include scheduling
and possible host enqueue gaps, and instrumentation changes latency. These are
**not pure kernel execution times**. Do not subtract them from ordinary wall time
or use them as an Amdahl speedup ceiling. The current evidence prioritizes further
investigation of projection and histogram work; it does not establish that either
proposed optimization will improve full pipeline latency.

Verification/loading (2.396 s), CUDA preparation (0.229 s), and warmups are
excluded from samples and recorded separately. The prepared cache holds
152,695,584 bytes and uploads no input arrays during evaluation. Working memory
and allocator pools are additional. Background user CPU work remains active;
all raw timings are retained, without a confidence or portability claim.

The uncached smoke run uses the same command with `--cuda-cache-mib 0`,
`--repeats 1 --poses-per-repeat 1 --warmup-poses 0` and a different output directory.
Both evaluations match NumPy. It validates instrumentation with per-call uploads;
its cold, single-sample timings are **not a cache speed comparison**.

Both definitions and aggregates validate against the existing benchmark schemas.
The aggregates are recomputed from their definitions before publication. Source
hashes bind actual measured worktree bytes (some existing sources have CRLF);
the tool's measured Git LF bytes match the worktree bytes. Publication replaces
the absolute workspace prefix in commands with `<workspace>` only.

Original report SHA-256 before path normalization:

| Published file | Original SHA-256 |
| --- | --- |
| `cached.definition.json` | `146e5e63ec1250a482dd53b5d9ac516cf95587617c0a97f7b2f347e4e4235c69` |
| `cached.benchmark.json` | `eab6827a807508c99695f2c93af1e7e5ea56389ff028e3a0a2fc63f1fa2648b0` |
| `uncached-smoke.definition.json` | `d458472dc91cd920c27539d854e6467c2780e15b8e82b4af480a5797ec82e2a1` |
| `uncached-smoke.benchmark.json` | `2607ceea0d8aca16f42482fc851c9cc40a11ecff5fed25bc1a1eaee22c73084e` |
