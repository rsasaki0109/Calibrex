# GPU acceleration for Camera–LiDAR six-DoF search

The optional CUDA backend accelerates repeated LiDAR projection, nearest-range
visibility and histogram counting in the Borer depth-to-depth (D2D) six-DoF search.
This is a useful first
GPU target: the frozen recovery protocols evaluate hundreds of candidate poses
per trial, across many frames and perturbations. Existing published development
runs in the [Camera–LiDAR plan](../research/camera_lidar_sota_plan.md) took minutes
per trial. On a 200,000-point synthetic workload, profiling the NumPy evaluator
put about 80% of its time in projection and visibility.

LiDAR odometry and map registration already use vectorized arrays and SciPy
KD-trees. Moving their small linear solves to a GPU would leave correspondence
search on the CPU. D2D provides a narrower adapter boundary and a large repeated
parallel workload, so it was selected first. This is a throughput optimization;
the estimator, histogram objective, calibration thresholds and evidence contract
retain their existing definitions.

## Install

From a checkout containing this change:

```bash
python -m pip install -e ".[cuda]"
```

The extra installs `cupy-cuda12x`; it requires an NVIDIA GPU, a compatible driver
and a CUDA 12.x toolkit with NVRTC. The driver may support a newer CUDA version
than the installed toolkit. See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for CUDA component wheels, conda installations and platform requirements. With
an existing CUDA 13 toolkit, install `cupy-cuda13x>=14,<15` instead of the CUDA 12
extra; both provide the same `cupy` adapter interface.

## Estimate one calibration

For a public-data comparison without your own rosbag, use the
[KITTI raw CPU/GPU demo](public_six_dof_demo.md). It builds verified inputs from
a raw drive and frozen provider, applies a declared initial perturbation, and
writes a 20-fit/5-holdout comparison report with image overlays.

The single-run command refines the prior declared in
`problem.initial_transform_camera_lidar` and saves a standard
`slac.result/v0.1` calibration result:

```bash
calibrex camera-lidar estimate-six-dof problem.yaml \
  --projection-backend cuda --output outputs/d2d-result.yaml
calibrex validate outputs/d2d-result.yaml --kind result
```

The problem binds synchronized LiDAR scans to previously generated camera depth
maps, with camera models and file/checkpoint digests. Use the existing
`build-kitti-problem`, `build-kitti360-problem` or `build-a2d2-problem` commands
after preparing a depth-provider artifact; this command verifies the bound
files and does not generate learned depth. Declare your measured rough prior in
the problem. Some dataset builders initialize the prior from the dataset
reference, which is useful for evaluation but is not an independently measured
prior. The estimator uses the declared initial pose directly and does not seed
from `reference_transform_camera_lidar` or synthesize benchmark perturbations.

To reserve frames for validation, repeat `--holdout-frame-id` with IDs from the
problem. Remaining frames are used for optimization and histogram range
selection; the frozen ranges then apply to the holdout frames:

```bash
calibrex camera-lidar estimate-six-dof problem.yaml \
  --projection-backend cuda --holdout-frame-id 0000002761 \
  --output outputs/d2d-with-holdout.yaml
```

`--initial-result previous.yaml` uses the matching pose from that standard
result's `transforms` as the prior and records the source digest. It requires the
same camera/LiDAR frames and never substitutes a reference-only pose.
`--rotation-bound-deg`, `--translation-bound-m`, initial/minimum step options,
`--histogram-bins`, `--min-visible-points`, `--max-evaluations` and `--objective`
control the search. Bounds cannot exceed those declared in the problem.

The default backend is `numpy`; use `numba_cpu` for the optional CPU accelerator.
CPU execution requires neither CuPy nor a GPU. CUDA failure is reported without
silently switching implementations. The output records full backend identity,
verified input and effective configuration digests, exact command arguments,
separate input-loading/backend-preparation/solve timings and the complete solver
trace in `run.provenance`. Initial, output and dataset-reference poses remain
separate. Training and optional holdout MI values are evidence of the objective,
not calibration accuracy or an acceptance gate.

Exit code `0` means the optimizer converged. Exit code `2` also writes a result
when the evaluation limit is reached (a partial candidate with `warning` status)
or observations are insufficient (`failed`, without an output transform).
Invalid inputs or unavailable requested backends return `2` without a new
result. Result quality stays `warn` after convergence because observability,
covariance and independent accuracy acceptance have not been established.

## Run a frozen protocol

Use your digest-verified problem and six-DoF protocol artifacts:

```bash
calibrex camera-lidar benchmark-six-dof problem.yaml protocol.yaml \
  --projection-backend cuda --workers 1 \
  --trace-dir outputs/cuda-traces \
  --definition-output outputs/cuda-definition.yaml \
  --output outputs/cuda-benchmark.yaml
```

The chunk runner also accepts `--projection-backend cuda`. Use one process per
GPU. The CLI rejects multiple workers for this backend because the GPU already
parallelizes the points. `--resume` validates the full backend identity, including
CuPy/NumPy versions, CUDA runtime/driver, GPU model, compute capability, NVRTC,
kernel source hash and FP64 policy. Use separate trace directories for CPU and
GPU runs.

The default remains `numpy`; `numba_cpu` is also available. Select CUDA explicitly
for `estimate-six-dof` or this six-DoF benchmark. The bag-level `estimate`,
`check` and `drift` commands
do not currently expose this backend. Missing GPU dependencies or an unusable
runtime produce an error when CUDA is requested.

## Numerical behavior and memory

Pinhole, double-sphere and MEI cameras are supported. The CUDA kernels use FP64
with fused multiply-add disabled. They round pixels to nearest-even, preserve
row-major pixel ordering, choose the first input point on equal camera ranges,
and retain the pre-visibility projected-point denominator. With the z-buffer
disabled, point order and duplicate pixels are preserved. The fused evaluator
samples camera-depth/LiDAR-range features and applies finite/positive gates on
the GPU. Integer histogram counts use the exact NumPy bin edges, including the
rightmost edge. Only the histogram and two denominators return to the CPU;
entropy, MI arithmetic and optimization use the original NumPy definitions.
The projector's direct call still returns full feature arrays for inspection.

Coordinates, depth maps and original LiDAR ranges are prepared once, with a
default 256 MiB input-cache capacity. Frames that do not fit are transferred per
call. `--cuda-cache-mib N` changes this capacity for `estimate-six-dof`,
`benchmark-six-dof`, the chunk runner and the timing tool; `0` disables caching.
The setting requires the CUDA backend. For larger inputs that fit your GPU,
for example:

```bash
calibrex camera-lidar estimate-six-dof problem-integrated.yaml \
  --projection-backend cuda --cuda-cache-mib 2048 \
  --output outputs/d2d-large-result.yaml
```

Capacity is an upper bound on retained inputs, not a reservation of that amount
of VRAM. Scratch arrays and
CuPy's reusable allocation pool are additional memory. Prepared arrays are
snapshots and must remain unchanged during a run. Direct Python callers can
construct `CudaDepthPairProjector(observations, cache_bytes=0)` to disable input
caching, and pass it through `BorerSixDofOptions(projector=..., projection_backend=...)`.
Use `projector.identity()` for the backend identity.

Non-default capacities are part of the backend identity. A frozen benchmark
cannot resume traces produced with a different capacity; an explicitly supplied
`256` retains the existing default identity. An input-cache allocation failure
reports an error suggesting a smaller capacity.
`run.provenance.projection_transfer_statistics` in a single-estimation result
records cache capacity, retained input bytes, prepared frame count and separate
preparation/evaluation upload counts and bytes. These counters cover point
coordinates, depth maps and original LiDAR ranges; they exclude the small
pose/bin-edge uploads and returned histogram. Counters add no GPU
synchronization to normal evaluation.

GPU/CPU floating-point implementations can differ at extremely close rounding or
range boundaries. The parity tests cover projection counts, feature arrays,
visibility ties, image/bin boundaries, histogram denominators and full recovery
traces. The timing tool
also compares every measured projection and MI value to NumPy and aborts on a
mismatch. A successful report establishes equivalence for its recorded inputs.

## Measure your workload

For the CPU comparison, install the optional Numba adapter:

```bash
python -m pip install -e ".[cuda,numba]"
python tools/benchmark_d2d_projection.py \
  --points 200000 --width 1280 --height 720 --repeats 25 \
  --solve-evaluations 80 --solver-repeats 2 \
  --output-dir outputs/d2d-timing
```

Use `--projection double_sphere` or `--projection mei` for fish-eye workloads, or
`--problem problem.yaml` to time verified real input arrays. The reports include
input/configuration/code hashes, backend identities, setup and first-call
times, paired warm timings, CUDA allocation snapshots and numerical equivalence
checks. Projection and solver runs rotate backend order between repetitions.
GPU timing includes
execution, synchronization and host/device transfers. Synthetic random inputs
measure throughput; recovery accuracy is tested separately.

Outputs reuse the existing `benchmark-definition` and `benchmark` schemas:

```bash
calibrex validate outputs/d2d-timing/projection.benchmark.json --kind benchmark
calibrex validate outputs/d2d-timing/solver.benchmark.json --kind benchmark
```

Small clouds can run faster on the CPU because CUDA launch and transfer costs
dominate. Fish-eye models do more FP64 arithmetic, so their improvement depends
on the GPU's double-precision throughput. Compare with `numba_cpu` as well as
NumPy when selecting a backend.

### Compare input cache capacities

For a large verified problem, compare the default capacity with a capacity that
can retain the full input:

```bash
python tools/benchmark_d2d_projection.py \
  --problem problem-integrated.yaml --backends numpy cuda \
  --cuda-cache-mib 256 --compare-cuda-cache-mib 2048 \
  --repeats 3 --solve-evaluations 13 --solver-repeats 3 \
  --output-dir outputs/d2d-cache-comparison
```

The tool rotates NumPy and both CUDA implementations through the same timing
order, checking every measured feature array, MI record and complete solver
result/trace against NumPy. Both projectors retain their cached inputs in the
same process. Reported cached bytes and upload counters belong to each
projector; allocation-pool/device snapshots include all resident projectors.
These snapshots are not an isolated per-method VRAM peak. Input transfers and
synchronization are included in latency; input verification/loading and cache
preparation are outside the warm timing. Short timing runs with 13 evaluations
can stop at the evaluation limit and do not establish convergence or accuracy.

Choose capacity with room for scratch allocations and other GPU users. The
cache comparison itself needs room for both resident input snapshots; use
separate `--cuda-cache-mib` timing invocations if they do not fit together.

### Large integrated-cloud measurement

On the same NVIDIA CUDA GPU, the existing KITTI raw 0018 problem with scan-integration
radius 5 contains 33,173,268 points across 25 frames, using frozen FeatDepth
depth maps and 1226×370 pinhole grids. Its complete prepared inputs occupy
1,152,268,576 bytes (1098.89 MiB). The paired command above measured three
repetitions of each 13-evaluation search:

| CUDA input-cache capacity | Retained frames | Retained input bytes | Search median | Range across three repetitions |
| --- | ---: | ---: | ---: | ---: |
| 256 MiB (default) | 6 / 25 | 268,045,632 | 2.384 s | 2.184–2.390 s |
| 2048 MiB | 25 / 25 | 1,152,268,576 | 0.404 s | 0.397–0.451 s |

The larger cache reduced the search median by 5.9× in this measurement. The
default cache uploaded 884,222,944 input bytes per full candidate evaluation;
the larger cache uploaded all inputs during preparation and uploaded zero
large input bytes during evaluation. Small pose/bin-edge uploads and returned
histograms remain. All measured feature arrays, MI records and complete solver
results/traces were exactly equal to NumPy, with unchanged FP64 kernels and
optimizer settings. This short search reaches its evaluation limit and is a
latency/equivalence check, not a convergence or accuracy result.

The [paired search report](../assets/gpu_acceleration/kitti_raw_cache_six_dof.benchmark.json)
and [projection/evaluation report](../assets/gpu_acceleration/kitti_raw_cache_projection.benchmark.json)
retain every timing, preparation/first-call costs, per-projector input-upload
counters and original input/code digests. NumPy search times varied substantially
(38.847, 51.829 and 188.774 s), so the cache comparison above uses the two CUDA
measurements from the same process. Input loading and cache preparation are
excluded from these search timings; full CLI latency also includes those phases.

## Measured development workload

Warm medians on a GeForce NVIDIA CUDA GPU (6 GiB), Core i7-9750H and Windows,
using Python 3.12.10, NumPy 2.3.4, Numba 0.65.1 and CuPy 14.2.0. Each workload
has one 1280×720 depth map and 200,000 seeded synthetic points. Evaluation is
projection, visibility, finite/positive gates, histogram counting and MI.
The timing runs used one worker and included host/device synchronization.

| Workload | NumPy | Numba CPU | CUDA | NumPy / CUDA |
| --- | ---: | ---: | ---: | ---: |
| [Pinhole evaluation](../assets/gpu_acceleration/pinhole.benchmark.json), 25 poses | 45.01 ms | 27.24 ms | 2.85 ms | 15.8× |
| [Double-sphere evaluation](../assets/gpu_acceleration/double_sphere.benchmark.json), 25 poses | 65.06 ms | 34.63 ms | 3.10 ms | 21.0× |
| [MEI evaluation](../assets/gpu_acceleration/mei.benchmark.json), 25 poses | 65.35 ms | 30.73 ms | 3.20 ms | 20.4× |
| [Pinhole six-DoF search](../assets/gpu_acceleration/six_dof.benchmark.json), 80 evaluations, 3 repeats | 3.225 s | 2.067 s | 0.352 s | 9.2× |

The complete search was also 5.9× faster than the existing Numba CPU backend.
Every sampled projection and MI record, and every full search trace and output
transform, matched the NumPy reference exactly. These are synthetic throughput
results on this machine; the reports retain raw timings, startup costs, input
digests and implementation/hardware identities. Measure your real problem with
`--problem` before adopting a latency expectation.

## Real input validation (11 October 2026)

The same GPU and environment were also tested on existing, digest-verified
[KITTI raw](https://www.cvlibs.net/datasets/kitti/raw_data.php) and
[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) depth-provider artifacts.
No depth maps were regenerated and no objective
or optimizer thresholds were tuned. Each timing repetition used all 25 frames;
the six-DoF search was capped at 80 evaluations and repeated three times.
These runs test backend equivalence and latency, rather than a recovery hit-rate
or completed 200-perturbation paper reproduction.

| Real workload | NumPy | Numba CPU | CUDA | NumPy / CUDA |
| --- | ---: | ---: | ---: | ---: |
| [KITTI raw 0018 six-DoF search](../assets/gpu_acceleration/kitti_raw_six_dof.benchmark.json) | 19.128 s | 12.632 s | 1.257 s | 15.2× |
| [KITTI-360 0002 six-DoF search](../assets/gpu_acceleration/kitti360_six_dof.benchmark.json) | 55.709 s | 40.594 s | 4.934 s | 11.3× |

KITTI raw `2011_09_30_drive_0018_sync/image_02` uses frozen FeatDepth inverse
depth, 1226×370 pinhole grids and 3,130,279 LiDAR points across the 25 frames.
Its search was also 10.0× faster than Numba CPU. All measured feature arrays,
visibility counts, MI records, full search results and traces matched NumPy
exactly. See the existing [provider reproduction instructions](../research/camera_lidar_sota_plan.md#phase-1-kitti-reproduction-evidence-31-july-2026)
for the model/checkpoint lock and declared possible training-data overlap.

KITTI-360 `2013_05_28_drive_0002_sync/image_03` uses frozen MiDaS v3.1
`dpt_beit_large_512` inverse depth, official 1400×1400 MEI grids and 3,021,947
points across 25 motion-compensated scans. Its search was also 8.2× faster than
Numba CPU, with the same exact equivalence checks. This is the existing
development capture, with possible KITTI-family provider-training overlap
declared in its provider artifact.

The separate [KITTI raw evaluator report](../assets/gpu_acceleration/kitti_raw_projection.benchmark.json)
and [KITTI-360 evaluator report](../assets/gpu_acceleration/kitti360_projection.benchmark.json)
retain 25 paired pose samples. Their interleaved, short evaluation timings
include CPU pauses between GPU calls; continuous-search timings above measure
the repeated workload the adapter is intended to accelerate.

The KITTI raw run retained 182.0 MiB of prepared inputs. CuPy's allocation pool
reserved 194.4 MiB after projection and solver measurements. These figures
exclude CUDA context/driver allocations and are allocation snapshots, not
process peak VRAM. The report also retains the CUDA runtime's device-wide
memory reading, which includes other applications and may use different
accounting from `nvidia-smi` under Windows.
KITTI-360 retained 242.6 MiB of inputs and reserved 304.2 MiB in the pool.
Its complete inputs exceed the 256 MiB cache budget, so uncached frames are
transferred on each evaluation; those transfers are included in the timing.

An additional KITTI raw check reran trial 000 from its original 200-perturbation
frozen protocol, retaining the original 800-evaluation cap and stopping rules.
NumPy and CUDA both converged after 325 evaluations, with identical candidate
parameters, objectives, acceptance decisions, final transform and outcome.
The complete [NumPy trace](../assets/gpu_acceleration/kitti_raw_converged_numpy.trace.json)
and [CUDA trace](../assets/gpu_acceleration/kitti_raw_converged_cuda.trace.json)
are retained with [original input/protocol/trace digests](../assets/gpu_acceleration/README.md).
This additional check covers one of the 200 frozen trials.

To reproduce the workload with an existing frozen problem:

```bash
python tools/benchmark_d2d_projection.py \
  --problem /path/to/kitti-raw-0018/problem.yaml \
  --repeats 25 --solve-evaluations 80 --solver-repeats 3 \
  --output-dir outputs/d2d-real-kitti-raw
```

The public timing reports contain metadata only. Local interpreter/dataset
paths and host names are normalized before publication; the SHA-256 of each
original report is recorded alongside the original input/configuration/code
digests. Raw sensor data, depth maps and model checkpoints remain external.

## Single-estimation CLI validation

The standard CLI was also checked end to end on KITTI raw 0018 with 20 training
frames and five reserved holdout frames. Both backends converged after 157
evaluations. Output transforms, every solver-trace record and initial/final
train and holdout MI values matched exactly. The standard
[NumPy result](../assets/gpu_acceleration/kitti_raw_estimate_numpy.result.json) and
[CUDA result](../assets/gpu_acceleration/kitti_raw_estimate_cuda.result.json)
passed result-schema validation; the CUDA result also rendered through
`calibrex render` and exported through `calibrex export --format ros-tf`.

The solve phase took 29.591 s on NumPy and 1.847 s on CUDA. Input verification
and loading took about 2.35 s on each; CUDA preparation took another 0.618 s.
This is one CLI invocation per backend, not a repeated latency benchmark.
The problem's declared initial pose came from the dataset reference; this
check demonstrates execution equivalence, not independent calibration accuracy.
See the [evidence notes](../assets/gpu_acceleration/README.md#single-estimation-cli-with-holdout)
for commands, selected frame IDs, original result hashes and timing scope.

## Ordered CUDA frame batching

The CUDA adapter automatically supplies ordered batch histograms to the D2D
evaluator when no frame executor is supplied. Each pose uploads its transform
and histogram edges once, enqueues frame kernels in the current CUDA stream,
and downloads a group of at most 32 histograms together. A private projection
workspace, sized for the largest input frame, is reused within that invocation.
It is released when iteration finishes, is closed, or fails. Each device result
buffer has at most 32 rows; the input snapshot budget remains independent of
scratch and CuPy's allocator pool. Uncached frames still upload their inputs
on every evaluation.

Native NumPy computes each frame's entropy/MI from the returned integer counts
and reduces frames in input order. Empty and insufficiently visible frames keep
their original positions and skip records. Pinhole, double-sphere, and Mei
frames can share a batch with different image sizes and point counts. The
single-frame histogram and pair-array APIs remain available. CUDA continues to
require `--frame-workers 1`.

Backend identity `calibrex.cuda_depth_pair_projector/v0.3` records
`histogram_batch_frames=32`. Standard result provenance includes
`projection_transfer_statistics.histogram_download_calls` and
`histogram_download_bytes`, counting successful histogram downloads from both
single-frame and batch calls. Pair-array downloads are outside those counters.
The streaming batch contract requires exactly one histogram per observation;
the evaluator rejects missing or extra results.

[CuPy's host copy](https://docs.cupy.dev/en/stable/reference/generated/cupy.asnumpy.html)
blocks until each group's result is ready. The existing FP64 kernels, disabled
FMA policy, nearest-point ties, histogram edges, and search settings are retained.

On public KITTI raw drive 0018, three invocations per method all retain the exact
complete 229-evaluation payload. Solve medians were 3.158 s for single-frame
CUDA and 1.404 s for batched CUDA (**2.25x**); pipeline phase medians were
6.129 / 3.796 s (**1.61x**). The fused CPU reference with two frame threads
measured 5.163 s solve / 8.273 s pipeline, with cold compilation included and
substantial shared-host variation. Histogram downloads fall from 4,590 to 231
per solve-plus-holdout invocation while byte counts remain identical. All six
frozen signed-axis problems also retain full historical CPU numerical equality.
[Published CUDA batch evidence](../assets/cuda_frame_batching/README.md) retains
all timings, original hashes, representative full traces, six-axis validation,
cold-cache conditions, and timing/accuracy limitations.

## Reusing verified inputs across estimates

After GPU batching, checkpoint/file verification dominates repeated public-data
estimates. Python callers can explicitly own a
`CameraLidarSixDofInputSnapshot`; public demos/sweeps expose the same behavior
with `--reuse-verified-inputs`. Inputs are fully verified and captured once in
immutable host arrays, and every estimate checks the problem SHA-256 again.
External files are not reverified after capture, as recorded in provenance;
creating a new snapshot performs fresh verification. Each call still prepares
its own accelerator and solver.

Ten alternating public KITTI invocations per method measured pipeline medians
of **4.405 / 1.657 s (2.66x)** and phase totals of **60.524 / 18.663 s**.
The snapshot's full initial **2.797 s** verification/loading/capture cost is
included once across its ten calls. The shared host showed substantial variation;
all raw timings and ranges are retained. All twenty complete timed payloads and
twelve signed-axis snapshot payloads match their bound CPU oracles exactly.
See the [snapshot evidence](../assets/verified_input_snapshot/README.md) and
[usage/timing contract](public_six_dof_demo.md) before comparing repeated runs.

## Bounded CUDA candidate batches

CUDA v0.4 evaluates the independent pattern-search candidates together. A search
step still uses the same six axes, direction order, bounds, evaluation budget,
winner tie break, and trace numbering. Four candidates at a time share four
kernel launches across up to 32 training frames, followed by one histogram
download. Entropy uses native NumPy operations with the original reduction
order; grouped positive values avoid changing reductions by padding zeros.

For pinhole cameras, the adapter also computes a conservative frustum mask on
the GPU once per search. A point is removed only when it cannot enter the image
under any allowed rotation or translation. The bound uses the sum of the three
Euler angle limits, the full translation limits, half-pixel image boundaries,
and a floating-point guard. Retained points preserve input order and their
original LiDAR ranges. Other camera models retain their full clouds. Poses
outside the prepared bound and different frame selections use the full-input
path.

Each candidate slot keeps private visibility scratch across iterations. A
separate reset kernel clears only pixels touched by that slot's previous pose;
visibility and histogram kernels visit the retained points instead of scanning
every image pixel. Integer histogram atomics combine equal warp bin indices.
These changes remove repeated work without changing the objective or the sensor
samples that can contribute to it.

The original cache budget includes both ordinary cached inputs and additional
compact input copies. Candidate scratch has a separate **1 GiB** default limit;
the adapter reduces batch capacity to fit. Uncached inputs, more than 32 frames,
insufficient scratch, or failed allocation use ordinary ordered frame batches.
CuPy allocator reservations and result buffers are additional. Use one worker
per GPU, keep observation arrays unchanged during a search, and release the
projector when finished. `candidate_workspace_bytes=0` disables the prepared
path through the Python adapter API. Standard result provenance records
candidate input bytes, retained/original point counts, and peak scratch bytes.

The normal `estimate-six-dof --projection-backend cuda` command selects this
path automatically. The measurement tool compares against the pinned existing
CUDA implementation and checks each complete result against its frozen NumPy
oracle:

```bash
python tools/benchmark_d2d_cuda_candidates.py \
  --sweep-root outputs/public-kitti-six-dof-sweep-cuda-v1 \
  --output-dir outputs/public-kitti-cuda-candidates-v2 --repeats 3
```

Use `--input-mode fresh_files` to include fresh verification in every invocation.
The default verified snapshot records its one-time capture cost separately.
Each method still constructs a fresh accelerator and solver. See the
[candidate batch evidence](../assets/cuda_candidates/README.md) for matched
GPU-to-GPU timings, complete signed-axis traces, and timing scope.

On the six frozen public KITTI initializations, three-repeat solve medians
improve **2.18–2.47x** against the existing CUDA implementation, with complete
numerical equality in all 36 trials. The x-positive solve changes from
1.226 s to 0.534 s. With fresh external-file verification on every invocation,
a separate x-positive comparison improves solve time 1.94x and pipeline time
**1.17x** (4.008 s to 3.421 s); input handling remains the main cost.
Snapshot capture costs are charged separately, and these measurements do not
include depth generation or establish independent calibration accuracy.

## Profiling the remaining CUDA work

The subsequent fresh-input optimization overlaps checkpoint verification with
verified observation loading and estimation, and reuses a bounded buffer for
large SHA256 reads. Every ordinary invocation still verifies every external
file, and no result is returned before verification succeeds. The same CUDA
v0.4 solver is used in all methods. Six public KITTI cases show three-repeat
complete API-call medians improving **1.52–2.63x**, with full numerical parity
in all 54 final trials. This includes verification and result construction,
and excludes JSON serialization and depth generation. The
[input verification evidence](../assets/input_verification_overlap/README.md)
retains all timings, the serial buffered control and the earlier experiment's
regression; these shared-host observations are not a portable speed guarantee.

`python tools/profile_d2d_cuda.py --problem problem.json --output-dir profile`
measures ordinary evaluation latency alongside an instrumented run. Use
`--exclude-frame-id ID` repeatedly to retain the same training frame selection,
and set `--repeats`, `--poses-per-repeat`, `--warmup-poses`, or `--cuda-cache-mib`
explicitly when comparing diagnostic runs. Compilation, input verification and
preparation are excluded from samples and recorded separately.

The tool checks every complete evaluation against NumPy, brackets initialization,
projection, visibility selection and histogram launches with CUDA events, and
restores original kernel handles even after an exception. It retains one pose's
events at a time. The two schema-valid reports alternate ordinary/instrumented
order and show observer overhead; stage spans and call counts are retained in
instrumented trial provenance. CUDA events can include scheduling and host
enqueue gaps, so these spans are not pure kernel execution times and must not be
subtracted from ordinary wall time.

The [public KITTI diagnostic](../assets/cuda_stage_profile/README.md) retains
ten repetitions of 20 evaluations per mode, numerical parity, source hashes,
stage timings and an uncached smoke run. It is profiling evidence, not a new
solver acceleration or an accuracy benchmark.

## Generating public depth maps on GPU

The external FeatDepth adapter accepts `--device cuda --batch-size 2
--preprocess-workers 2` for FP32 inference with bounded parallel PNG decoding.
It hashes and decodes one checkpoint byte snapshot and reuses its manifest
reference. CPU inference remains the default; `--cpu-threads N` controls its
thread count. PyTorch and FeatDepth stay outside the Calibrex core dependencies.

On the public 25-frame KITTI workload, a single comparison using six default
CPU threads measured 89.5 s CPU versus 17.8 s batched CUDA (**5.02x**), including
fresh interpreter startup and artifact writes. A separate three-repetition
comparison with two CPU threads measured 123.9 / 13.1 s medians (**9.48x**).
Batching alone gives a small gain on this short clip; moving inference to GPU
provides the larger difference. Both comparisons use the same FP32 checkpoint.
All six frozen initializations retain identical final poses and evaluation
counts; complete traces have small score differences, with a maximum final
holdout MI difference of 6.2e-7 nat. See the
[provider commands](public_six_dof_demo.md#prepare-public-inputs) and
[full inference evidence](../assets/featdepth_inference/README.md) for all
timings, source hashes, memory counters, output differences and limitations.
