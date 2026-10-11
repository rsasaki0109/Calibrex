# Ordered CUDA frame batches on public KITTI

Seven schema-valid JSON artifacts preserve comparisons measured at commit
`0d5f753` (`0d5f7531e1a29437013f01c43fbfabaefdc37e8d`). Inputs are the frozen
public KITTI raw `2011_09_30_drive_0018_sync/image_02` problem from the
[fused CPU comparison](../fused_cpu_histogram/README.md), original Velodyne
scans, and FeatDepth provider SHA256
`7e3ef953a360a7c01ed0cdc3cf947db97fb0aff5912ced805d80bebe210d3360`.
The same 25 frames contain 3,130,279 points: 20 fit and five holdout, fixed
before optimization and histogram-range selection. The prior adds local x
rotation 0.5 degrees and camera-frame x translation 0.5 m to the dataset
reference. Default search bounds, steps, and the 800-evaluation cap are retained;
the reference is never an optimization target.

After measurement, the transfer-statistics docstring was clarified to describe
both upload and histogram-download counters; executable adapter code is unchanged.
Measured source-file SHA256 values bind the Windows CRLF working-tree bytes;
separate `committed_*` source hashes bind canonical LF Git blobs at `0d5f753`.

Three methods run three times each. Fused Numba CPU uses two persistent frame
threads. The CUDA single-frame reference wraps the adapter's unchanged
single-frame histogram API and omits the batch protocol. CUDA batches share
pose/edge uploads and one invocation-local scratch workspace, enqueue frame
kernels in current-stream order, and download at most 32 histograms together.
Both GPU methods use the same FP64 kernel source, disabled FMA, and input-cache
budget; source hashes and the reference wrapper's launcher hash are bound.

Order rotates so every method occupies each execution position once:
CPU / serial GPU / batched GPU; batched GPU / CPU / serial GPU; serial GPU /
batched GPU / CPU. Each invocation verifies and loads all inputs and prepares a
fresh projector. Absent Numba and CuPy disk cache directories were selected
before startup. The first CPU solve includes JIT compilation; the first serial
GPU preparation includes NVRTC preparation. Later invocations share compiled
caches in the same process. There is no separate warm-up, and all invocations
are retained.

| Repetition | CPU solve (s) | Serial GPU solve (s) | Batched GPU solve (s) | CPU pipeline (s) | Serial GPU pipeline (s) | Batched GPU pipeline (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.162837 | 2.707314 | 1.227273 | 8.272800 | 6.129189 | 3.795574 |
| 2 | 3.476237 | 3.158479 | 1.403745 | 6.160327 | 5.526306 | 3.795514 |
| 3 | 6.000503 | 3.303651 | 1.855467 | 11.919513 | 6.349387 | 5.909505 |
| Median | 5.162837 | 3.158479 | 1.403745 | 8.272800 | 6.129189 | 3.795574 |

Ratios of solve medians are **2.2500x** versus serial GPU and **3.6779x** versus
fused CPU. Ratios of pipeline phase medians are **1.6148x** and **2.1796x**,
respectively. Pipeline phases include input verification/loading, backend
preparation, solve, and holdout/pool shutdown; shared setup, serialization, and
report rendering are excluded. Verification remains enabled, including the
provider checkpoint hash. Measurements ran on i7-9750H / GTX 1660 Ti 6 GiB,
Windows, Python 3.12.10, NumPy 2.3.4, Numba 0.65.1, and CuPy 14.2.0. The host
was not isolated from other running work, and substantial variation is visible,
especially the last CPU invocation. Cold compilation contributes to the CPU
population. Three repetitions do not establish statistical confidence or a
portable guarantee; artifact bootstrap fields are schema summaries only.

All nine invocations converged after exactly 229 evaluations. Their complete
solver records, metrics, poses, skipped frames, and initial/final holdout
evaluations match exactly, including the original CPU/CUDA x-positive trace.
The canonical complete numerical payload SHA256 is
`fdf4d36e8cbe9b8a88a5cb765f4c387f67fa2021df8e51271ce461bb4f9e3160`.
No subsampling, approximate arithmetic, or search-budget reduction is involved.

Every serial GPU invocation performs **4,590** successful histogram downloads;
every batched invocation performs **231** (229 fit evaluations plus two
holdout evaluations). Histogram byte counts are identical: batching reduces
calls and synchronization rather than removing samples. Original input uploads
and the fixed cache budget remain recorded. Each group result buffer is bounded
to 32 rows and scratch is private to one iterator. Closing or failing iteration
releases it; partial iteration does not retain a trial cache in the projector.

Separately, all six previously frozen signed-axis problems were rerun with
batched CUDA and compared against their original full CPU numerical payloads.

| Case | Evaluations | Histogram downloads | Complete numerical equality |
| --- | ---: | ---: | --- |
| x-positive | 229 | 231 | Exact |
| x-negative | 313 | 315 | Exact |
| y-positive | 277 | 279 | Exact |
| y-negative | 241 | 243 | Exact |
| z-positive | 265 | 267 | Exact |
| z-negative | 241 | 243 | Exact |

All six converge. `axis-validation.definition.json` and
`axis-validation.benchmark.json` retain six candidates and six imported
historical CPU numerical oracles. Timings are omitted from this matrix; the
historical results are numerical comparisons, not paired speed measurements.
Parity preserves the existing outcome and does not establish accuracy: quality
remains `warn`, covariance and observability are unknown, raw scans are not
motion compensated, and FeatDepth may have KITTI training overlap.

`comparison.definition.json` and `comparison.benchmark.json` retain all nine
timed attempts and original external result hashes. `numba_cpu.result.json`,
`cuda_serial.result.json`, and `cuda_batched.result.json` retain the first
repetition's complete traces and holdout records. All other complete payloads
were checked against those representatives. Original output hashes bind
external bytes; published representative hashes are separately recorded in
`source_artifacts`, and each representative retains its original hash under
`run.provenance.publication`. Workspace/dataset paths are normalized to
`${WORKSPACE_ROOT}` and `${DATASET_ROOT}`. Raw inputs, weights, original results,
and generated reports remain outside Git.

For normal CLI use, batching is automatic:

```bash
pip install 'calibrex[cuda]'
calibrex camera-lidar estimate-six-dof problem.json \
  --projection-backend cuda --frame-workers 1 \
  --output cuda-batched.result.json
```

The nine-invocation comparison was generated by the bound manual launcher,
which selects the single-frame reference through an explicit wrapper; that
reference mode is not a CLI option. Original measurement and six-axis outputs
remain in `outputs/public-kitti-cuda-frame-batching-v1` and
`outputs/public-kitti-cuda-frame-batch-axis-validation-v1`.
