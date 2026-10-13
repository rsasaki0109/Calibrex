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

## CUDA LiDAR-LiDAR map registration

LiDAR-LiDAR calibration against local maps can now use
`--registration-backend cuda` on `calibrex lidar-lidar ros2`. The optional
adapter moves 3D neighborhoods, normal covariance/eigensolves, point transforms,
point-to-plane residuals, analytic Jacobians and robust normal equations onto
the GPU. Captured spatial topology and six-dimensional updates stay on CPU;
one device snapshot is reused for block jackknife, holdout and known-bad controls.
CPU remains the default, and execution identity is recorded in artifact options.

Five pairs on a public KITTI noisy copied-scan execution control measure fresh
process medians of **16.951654 → 6.646739 s (2.55x)**, including imports, inputs,
GPU preparation/compilation, all evidence calls, hashes and saves. The first
CUDA process is slower (17.157697 versus 12.621121 s) and is included. Complete
map-evaluation API medians are **13.896949 → 3.800446 s (3.66x)**. All statuses,
counts and decisions match; numerical fields use declared FP64 tolerances.

The [full LiDAR CUDA evidence](../assets/cuda_lidar_map/README.md) retains every
trial, full results, original/publication hashes and the earlier exploratory
attempts separately. Generated sample blocks reuse underlying scans and are
an execution control, not independent temporal or two-sensor accuracy evidence.
ROS-bag collection and reference odometry remain outside this timing scope.

## CUDA LiDAR scan odometry

Reference LiDAR odometry can now use `--odometry-backend cuda` on
`calibrex lidar-lidar ros2`. The adapter retains target geometry on device and
fuses point projection, exact spatial queries and robust point-to-plane systems.
It also prepares normal covariances on GPU. Native small solves, time-gap scaling,
deskew policies and recorded uncertainty repairs remain in place.

Five fresh process pairs on 120 real temporal public AIST GLIM scans measure
**64.164142 → 21.092158 s (3.04x)** for a dense 0.03 m voxel / three-scan map.
Timing includes startup, decoding, preprocessing, GPU initialization/compilation,
all registrations, source/input hashes and saves. The first CUDA compilation is
included. Replay API medians are **58.755552 → 16.283616 s (3.61x)**.
The 0.1 m workload measures **21.198051 → 15.866211 s (1.34x)** instead.

Both final methods explicitly use a 1 nm coordinate grid
(`geometry_quantum_m=1e-9`) to stabilize spatial-tree choices after tiny pose
roundoff. Zero remains the default. All poses, transforms and numerical quality
fields match under abs 1e-9 / rel 1e-6; iterations, counts and convergence
decisions match exactly. Unconverged steps remain in every timing/result.
This checks execution agreement, rather than independent trajectory accuracy.

The [full odometry CUDA evidence](../assets/cuda_scan_odometry/README.md) retains
all final trials, the smaller whole-process improvement of the preceding version,
and the rejected unquantized dense trace. CPU remains the default; the dense
speedup is not a guarantee for default CLI point budgets. The Python API exposes
custom preprocessing and coordinate-grid options, and saved calibration
artifacts record the full odometry options and execution identity.

## Exact CPU neighbors for native ICP

Native robust point-to-point ICP and Chen–Medioni point-to-plane ICP now use
spatial indexes for neighbor matching, target normals and diagnostics. Candidate
distances are recomputed with the original float64 NumPy operations; boundary
ties retain the original point-index order. Mutual checking, stable trimming,
all iterations, holdout, rematching, multi-start trials and pose probes keep
their existing definitions. Both solver APIs and the native registration
comparison select this path automatically, using the existing SciPy dependency.

On a public KITTI scan pair with fixed 500/1,500/3,000-point caps, three-repeat
complete solver-call medians improve **3.53–32.87x** against the pinned previous
implementations. All 36 final full numerical payloads match exactly. For
3,000-point point-to-point ICP, the median changes from 123.731 s to 4.058 s;
point-to-plane changes from 18.512 s to 0.563 s. These calls include preparation
of indexes/normals and every diagnostic, but exclude input capture/downsampling
and full CLI overhead. The [ICP neighbor evidence](../assets/icp_neighbors/README.md)
retains every timing, full payload and pilot, including unchanged iteration-limit
statuses. These measurements describe one shared-host workload and preserve
existing calibration behavior; they do not establish independent accuracy.

Native registration also uses array voxel centroids, batched target-normal PCA,
vectorized information matrices and reused training coordinates. Point ICP's
six independent multi-start trials use up to four CPU threads automatically
for source clouds with at least 1,500 points; smaller clouds stay sequential.
Set `RobustPointToPointIcpOptions(multi_start_workers=1)` or the same
`registration_backend_comparison` factor option to select sequential execution.
All trial results retain their original direction order and diagnostic settings.

Against the already indexed implementation, five-repeat medians on two public
KITTI scan pairs improve **1.95–3.17x** for fresh input reads/verification,
voxel preprocessing and both complete native ICP calls together. All 90 final
payloads match exactly, with bitwise prepared-position digests. The 3,000-point
cases change from 4.410 to 1.850 s and 2.369 to 1.046 s. Most of the gain comes
from vectorization; sequential controls retain the smaller incremental threading
gain. This scope excludes the full calibration CLI and external backends.
The [registration preparation evidence](../assets/registration_preparation/README.md)
retains every timing, range, full payload and exploratory attempt, including
an early control-metadata mismatch and unchanged iteration-limit statuses.

## Complete public Livox calibration workload

Livox candidate and perturbation evaluation now reuses decoded records,
invariant statistics and source planes within one invocation. Exact 27-voxel
associations use bounded batches and checked sparse integer keys; coordinate
transforms preserve the original Hamilton-product arithmetic. New invocations
read fresh inputs. No GPU or manual tuning is required.

Five fresh-process comparisons on the official public Horizon pair measure
**53.535 → 14.245 s wall medians (3.76x)**, including startup, dataset inspection,
all 24 controls, native point/plane ICP, Open3D GICP and report/artifact saves.
The public calibration API itself measures **50.854 → 11.611 s (4.38x)**;
its timing is a subset of wall. Workers call that API, so Calibrex CLI argument
dispatch is excluded. All ten complete normalized result payloads match exactly,
including unchanged failed quality and inconclusive comparison decisions.

See the [complete Livox evidence](../assets/livox_evaluation/README.md) for every
timing, shared-host range, source binding, numerical payload and reproduction
command. It also retains ten intermediate and two pilot executions. These
single-pair timings do not establish independent calibration accuracy, and
must not be multiplied by earlier ICP-only ratios.

## Sharing Livox geometry across the complete run

Inspection, candidate/control evaluation and native registration now share one
captured input context per calibration API invocation. Source voxel sums and
read-only plane arrays are prepared once. Compact maps use direct position
tables capped at 8 MiB each; larger maps retain sparse or scalar lookup.
Empty neighbors avoid distance calculations while keeping the same 27-candidate
order, ties, arithmetic, population and control settings. New runs read fresh
inputs automatically.

The direct cumulative comparison against the pre-Livox-acceleration version
measures **40.685 → 10.690 s wall medians (3.81x)** over five fresh-process
repetitions per method. A separate comparison against the preceding accelerated
version measures **12.172 → 10.728 s (1.13x)**; its API subset measures
**9.699 → 7.639 s (1.27x)**. The additional complete-process gain is modest.
Both comparisons include the same inspection, full input points, 24 controls,
native ICP diagnostics, Open3D GICP, NDT availability and output saves. CLI
argument dispatch is excluded; worker measurement writes are included in wall.

All twenty final formal saved results and API payloads are exactly equal after
run-ID/timestamp/destination normalization. Quality remains `fail`, and the
registration comparison remains `inconclusive`. These are single-pair execution
and numerical-parity measurements. The [geometry reuse evidence](../assets/livox_geometry/README.md)
retains all timings, source bytes, ranges, ten intermediate workers and two
completed API workers from an interrupted attempt. The interrupted attempt's
wall timing and final parent guards are unavailable; its API times remain in a
separate protocol. Prior ratios are not multiplied.

## Packing Livox neighborhoods and shared coordinates

Livox evaluation now prepares occupied voxel neighbors in the original 27-offset
order and shares one read-only transformed array between voxel and plane checks.
Checked integer encoding and unbuffered voxel additions preserve scalar sums and
first-seen key order. Large extents and unusual inputs retain the scalar or sparse
paths. The original direct table and packed neighborhood lookup share an 8 MiB
budget per index; only the latest transformed array per file is retained.

Against the preceding accelerated version, five fresh-process repetitions per
method measure **3.238 -> 0.954 s (3.40x)** for the candidate plus all 24 controls,
timed inside the same complete run. Prior inspection is outside that phase.
Complete-process medians are **8.593 -> 6.201 s (1.39x)**, and the calibration API
subset is **6.169 -> 3.912 s (1.58x)**. The phase ratio is not the whole-process ratio.
A separate direct cumulative comparison to the version before Livox acceleration
measures **54.026 -> 9.270 s (5.83x)**. Earlier ratios are not multiplied.

All twenty final saved results and API payloads match after volatile-field
normalization. Quality remains `fail`, and registration remains `inconclusive`.
The [packed evaluation evidence](../assets/livox_packed/README.md) retains every
timing, source binding, artifact hash and an earlier ten-worker comparison.
This CPU change adds no dependency or GPU backend. The single-pair measurements
do not establish calibration accuracy or portable speed guarantees.

## Batching the native point-to-plane optimizer

The native fixed-rig factor now computes residuals in chunks of up to 4,096
observations. It retains the original Hamilton products, quaternion normalization
counts, weighted dot-product order and numerical finite differences. Small and
nonfinite inputs use the scalar path; observation arrays belong to the factor
and refresh when its observation tuple is replaced. Optimizer rules, robust
weights and normal-equation accumulation remain unchanged.

On the existing public Livox `online_accumulation_config.yaml`, normal offline
PCD calibration with 2,732 correspondences measures **12.384 -> 5.724 s (2.16x)**
for the complete fresh process. The optimizer subset is **6.369 -> 0.524 s
(12.15x)**, and the API subset is **10.011 -> 3.421 s (2.93x)**. These are five-pair
medians against the preceding accelerated version, with reversed order on
alternate repeats and no discarded timings.

All ten complete results match, including the nine-iteration convergence history,
diagnostics and all 24 candidate controls. Quality remains `warn` and rank remains
3/6. Despite the config filename, these measurements are offline PCD runs; they
use a different workload from the ICP/GICP/NDT comparison above. Their speedups
must not be combined with those earlier ratios. The
[full factor evidence](../assets/point_to_plane_factor/README.md) retains every
timing, complete canonical payload, artifact hash and exact source archive.

```bash
python tools/benchmark_livox_evaluation.py --workflow fixed-rig \
  --baseline-commit d017dfa03c6275924c8da44f55b2d3707ca58311 \
  --repetitions 5 --omp-threads 1 --output-dir outputs/livox-fixed-rig-rerun
```

## Removing repeated input work and batching native operations

Captured Livox records now reach the native solver and distance-bin quality
checks. The same two input files are decoded twice per complete run, replacing
eight calls. Checked strided decoding and ordered statistics retain scalar bytes,
extrema, sums and geometry. Independent controls use at most four CPU workers with
isolated transform caches; case order is unchanged.

Static uniform-plane associations reuse the exact occupied-neighbor lookup.
Bit-identical per-point poses are composed once per residual call, and normal
equations retain the original multiply/addition order in bounded row chunks.
The spatial tree is imported when actually needed. No point, control, finite
difference or output artifact is omitted.

Five fresh-process pairs against the preceding accelerated version measure:

| Offline public PCD config | Correspondences | Complete process medians | Ratio |
|---|---:|---:|---:|
| All finite target points | 16,884 | 11.249 -> 5.451 s | 2.06x |
| Existing 4,000-point cap | 2,732 | 7.208 -> 3.970 s | 1.82x |

The added `native_full_points_config.yaml` sets the existing
`max_target_points: 0` option to remove the target reader cap. Both methods use
the same config within each row. These rows and the previous measurement cohorts
have distinct timing conditions; ratios are not multiplied. All twenty complete
results match within their respective configurations, including nine-iteration
histories, diagnostics, quality and all 24 controls. Quality remains `warn` and
rank remains 3/6. The [full runtime evidence](../assets/livox_runtime/README.md)
retains every sample, range, payload, artifact hash and exact source binding.

```bash
python tools/benchmark_livox_evaluation.py --workflow fixed-rig --baseline-scope runtime \
  --baseline-commit 6b30cdf92ce710b0810396deeee45d651fd308d5 \
  --config examples/public_datasets/livox_horizon_horizon_pcd_sample/native_full_points_config.yaml \
  --repetitions 5 --omp-threads 1 --output-dir outputs/livox-runtime-full-rerun
```

## Searching occupied adaptive voxel cells

Continuous-time LiDAR refinement repeats plane association for each clock
candidate and extrinsic solve. The adaptive map formerly enumerated every cell
in each level's gate box, including empty cells. It now binary-searches occupied
x/y/z rows and visits their planes in the same level and lexical key order.
Scalar distances, first ties, inclusive gates and the resulting correspondences
are preserved. This CPU optimization is automatic for built adaptive maps.

The complete public GLIM Livox comparison uses the existing bounded
continuous-time configuration and supplied trajectory. Every process executes
the same 26 clock-candidate/refinement evaluations and 62,400 adaptive queries,
with the same source/target budgets, robust filtering and holdout diagnostics.
The [archived five-pair evidence](../assets/adaptive_plane_search/README.md)
includes whole-process and query medians, every trial, exact full-artifact
parity, actual train/holdout point identities and verified conversion lineage.
The complete-process median falls from 67.38 s to 14.59 s (4.62×); adaptive
query time within those same runs falls from 53.21 s to 1.42 s (37.54×).

The copied topic is a single-sensor identity control. Its unchanged result has
rank 6, 894 train and 876 holdout correspondences, and holdout RMSE
0.053133620013224026 m. These results establish execution consistency on public
data; the recording does not supply a second-sensor absolute extrinsic reference.

```bash
python tools/benchmark_adaptive_plane_search.py \
  --repetitions 5 --output-dir outputs/glim-adaptive-search
calibrex continuous-time-lidar-pair \
  examples/public_datasets/glim_versatile/online_continuous_time_benchmark_config.yaml \
  --output outputs/glim-continuous-time.yaml
```

Prepare the official recording with the archive's checksum-verified instructions
before running these commands.

## Reusing continuous-time geometry and interpolation

Continuous-time refinement now captures uniform voxel geometry once per solve,
reuses occupied adaptive-neighbor tuples and prepares odometry interpolation
intervals. Moving world-coordinate transforms run in bounded array chunks,
retaining the scalar quaternion and normalization order. Independent complete
input hashing overlaps replay and optimization; changed inputs invalidate the
result. Query counters and estimator behavior remain unchanged.

Against the preceding sparse-query version, five fresh-process pairs per
configuration measure:

| Public GLIM sampling | Train / holdout targets | Complete-process medians | Ratio |
|---|---:|---:|---:|
| Existing bounded configuration | 1,200 / 1,200 | 15.542 -> 8.756 s | 1.78x |
| Solver default sampling budgets | 6,000 / 3,200 | 32.153 -> 17.196 s | 1.87x |

All twenty complete API payloads and saved YAML artifacts match within their
configurations, including clock-candidate scores, refinement histories, train
and holdout diagnostics, and provenance. The larger workload retains rank 6,
4,773 train/2,526 holdout correspondences and holdout RMSE
0.05078387183780445 m. This uses the same copied-topic identity control and
supplied public trajectory. The [full reuse evidence](../assets/continuous_time_query_reuse/README.md)
retains all timings, ranges, calculation counts, exact source archives and
input lineage. Ratios compare complete runs within each cohort and do not
multiply earlier gains. This CPU improvement applies automatically.

```bash
python tools/benchmark_adaptive_plane_search.py --baseline-scope runtime \
  --baseline-commit 3f1e1f412d5d373493395e53b2636e079184e457 \
  --config examples/public_datasets/glim_versatile/online_continuous_time_default_budget_config.yaml \
  --repetitions 5 --output-dir outputs/glim-query-reuse-default
```

## Compiling continuous-time CPU kernels

Continuous-time LiDAR refinement can use fused CPU kernels for interpolation,
world coordinates and repeated point-to-plane residuals. Exact batched
adaptive-centroid tree queries also reduce association work. The implementation
keeps quaternion/product order, scalar gate and tie decisions, numerical finite
differences, robust filtering and the temporal holdout policy.

Install the optional compiler and request its backend in factor options:

```bash
python -m pip install -e ".[numba]"
```

```yaml
continuous_time_query_backend: numba_cpu
```

The Python backend remains the default. Actual optional execution and its
runtime version are saved in provenance. Supported kernels retain a disk cache;
the first compilation and later cache loading both contribute to elapsed time.

<!-- compiled-queries:measurements:start -->
Five paired fresh processes per configuration, including the first empty-cache
compilation, measure the following incremental gains against the preceding
geometry-reuse version:

| Public GLIM sampling | Train / holdout targets | Complete-process medians | Ratio |
|---|---:|---:|---:|
| Default sampling budgets | 6,000 / 3,200 | 26.255 -> 15.643 s | 1.68x |
| All captured targets, 120 frames | 38,400 / 9,600 | 154.501 -> 78.869 s | 1.96x |

<!-- compiled-queries:measurements:end -->

The [full compiled-query evidence](../assets/compiled_continuous_time_queries/README.md)
retains every timing, initial empty-cache trial, cache file hashes, exact source
archives and complete saved result payloads. Results match within each cohort
except the explicit optional execution identity note. The larger configuration
uses all 48,000 captured targets, with the same within-cohort point identities
and solver budgets. Its unchanged status is `max_iterations`; this acceleration
does not claim convergence or absolute two-sensor calibration accuracy for the
copied-topic public-data control.

```bash
calibrex continuous-time-lidar-pair \
  examples/public_datasets/glim_versatile/online_continuous_time_compiled_full_capture_config.yaml \
  --output outputs/glim-compiled-continuous-time.yaml
```

## Keeping continuous-time refinement in numeric arrays

The same optional CPU backend now retains all thirteen numerical
finite-difference residual evaluations and ordered robust normal equations in
numeric arrays. Observation preparation reuses captured native fields and
packed rows around MAD filtering. Repeated timestamp batches reuse prepared
poses within a conservative 128 MiB retained-payload budget; changed points
and extrinsics still recompute world coordinates. Exact adaptive queries use
bulk decisions only for unambiguous gates, retaining scalar boundary/tie repair.

<!-- compiled-linearization:measurements:start -->
Five paired fresh interpreters per configuration; both methods start with
their own empty compiler cache. Every first run and later run is included.
Incremental complete-process wall comparisons against the compiled-query version:

| Public GLIM sampling | Train / holdout targets | Medians | Ratio |
|---|---:|---:|---:|
| Default sampling budgets | 6,000 / 3,200 | 11.419 -> 8.323 s | 1.37x |
| All captured targets, 120 frames | 38,400 / 9,600 | 76.194 -> 32.280 s | 2.36x |

All-thread worker CPU diagnostic:

| Public GLIM sampling | Train / holdout targets | Medians | Ratio |
|---|---:|---:|---:|
| Default sampling budgets | 6,000 / 3,200 | 12.422 -> 9.297 s | 1.34x |
| All captured targets, 120 frames | 38,400 / 9,600 | 75.984 -> 33.266 s | 2.28x |

Worker CPU excludes interpreter startup and measurement-JSON serialization;
complete wall includes both. All 20 current saved/API results retain every
value and provenance field except the exact declared execution identity note.
<!-- compiled-linearization:measurements:end -->

These incremental comparisons use the preceding compiled-query version as
their baseline and independent initially empty compiler caches for both
methods. Every first run, later run and complete saved result is retained in
the [linearization evidence](../assets/compiled_lidar_linearization/README.md).
That archive also retains the weaker preceding candidate's separate cohorts.
Sampling, temporal holdout, robust weights and bounded optimization are fixed;
the public copied-topic control supplies execution parity rather than an
independent two-sensor accuracy reference. No new configuration is required.

## Keeping continuous-time observations packed and parallel

The optional `numba_cpu` path keeps pose/observation rows in read-only arrays
through correspondence preparation and MAD filtering. It computes independent
clock candidates with up to three threads for large populations, retaining
sequential Hessian/gradient accumulation and every existing evaluation field.
This improves CPU work; it adds no GPU execution.

| Public GLIM workload | Complete-process medians | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|
| 120 frames, all captured targets | 27.096 → 12.723 s | 2.13× | 2.12× |
| Default sampling budgets | 8.402 → 6.582 s | 1.28× | 1.29× |

Five pairs per workload include initial compilation and every outlier/slower run.
All 20 final saved/API results match apart from the exact declared execution note.
The copied-topic recording provides execution parity, without independent sensor
accuracy evidence. See [all results, ranges, prior attempts and source bindings](../assets/packed_continuous_time_lidar/README.md)
for the limitations and commands.

## CUDA finite differences for continuous-time LiDAR

Install the optional adapters and set the continuous-time factor option:

```bash
pip install -e ".[cuda,numba]"
```

```yaml
continuous_time_query_backend: cuda
```

For capture budgets of at least 8,192 rows, retained native packed observations stay
on the captured CUDA device across optimizer iterations. Independent clock
candidates own separate streams and observation buffers, sharing compiled
kernels. The kernels evaluate the same residuals and all thirteen central
finite-difference transforms in FP64 with fused multiply-add disabled.
Single residual evaluations, trajectory interpolation and ordered
Hessian/gradient accumulation remain on CPU. Capturable uniform/adaptive
plane maps also query on CUDA, with native CPU repair for adaptive ties and
gate boundaries. Selected indices and
native timestamp rows are prepared once; offsets retain the same integer
rounding and bounds. Small capture budgets use the compiled CPU path
without importing CuPy; provenance records the backend that actually ran.
The Python default and result schemas remain unchanged.

The preceding finite-difference-only stage kept associations on CPU.
On 120 public GLIM frames with up to 8,000 points per frame (768,000 train /
192,000 holdout targets), five fresh paired subprocesses measured complete
wall medians of 169.696 -> 97.911 s (1.73x), solver medians of
130.850 -> 65.554 s (2.00x), and all-thread CPU medians of
273.813 -> 147.656 s (about 46% less CPU). The median paired whole-wall
ratio is 1.47x; individual ratios span 1.17x to 1.96x. Thus these runs do not
establish a stable twofold complete-process gain. The 400-point/frame capture
measured only 1.14x whole-wall improvement; default sampling caps use CPU
dispatch and measured 1.04x. Every first/slower run is retained.

All 30 final saved/API results match apart from the declared execution note.
Full/dense workloads retain `max_iterations`, rank 6; the default workload
retains `converged`. A separate guarded diagnostic records 129 actual GPU
linearizations, 26 retained-batch uploads and 207 CPU standalone residual
calls. Input transfer falls from 255,403,104 bytes in the preceding shared
GPU candidate to 103,523,608 bytes; the comparison is diagnostic, not a
pooled timing estimate.

Compare against the preceding packed/parallel CPU version with identical
capture, clock, solver, robust-loss and holdout settings:

```bash
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope compiled \
  --baseline-commit eb50c99878706a45eb0e584f3573507313cdc172 \
  --candidate-query-backend cuda \
  --config examples/public_datasets/glim_versatile/online_continuous_time_compiled_dense_capture_config.yaml \
  --repetitions 5 \
  --output-dir outputs/public-glim-cuda-continuous-dense
```

The dense config retains 120 messages and permits up to 8,000 points per
message. The benchmark records the sole candidate execution override while
retaining the shared physical config and its checksum. Each method has its
own initially empty compiler cache; process wall includes imports, CUDA
initialization, compilation, transfers, replay, refinement, diagnostics,
provenance hashing and saves. First runs, slower runs and earlier candidates
are retained in the [complete CUDA evidence](../assets/cuda_continuous_time_lidar/README.md).
The duplicated public topic supplies execution parity, without independent
two-sensor accuracy evidence.

## Captured GPU associations and large-frame CPU replay

The same optional backend captures native uniform voxel neighborhoods and
adaptive spatial topology once, then uploads each world-coordinate batch once
to query both maps. Neighborhood order, normal-less nearest rejection and
inclusive gates stay native. Adaptive close ties, tiny distances and gate
boundaries use the existing CPU repair. Unsupported maps retain CPU queries.
Diagnostics stream local XYZ without building every observation and pose.

With the compiled continuous-time option selected, replay frames of at least
4,096 selected points also batch their capture-time trajectory/base calculations
on CPU. Source world points, target poses/extrapolation values, timestamp rounding
and failure prefixes match the scalar path. Small declared frame budgets keep
scalar replay without importing or compiling another adapter. This CPU work
adds no GPU coverage.

On the dense public GLIM control, five fresh paired subprocesses against the
preceding GPU version measured whole-wall medians of 113.101 -> 61.549 s
(1.84x), with a 1.73x median paired ratio. All-thread CPU medians fell from
160.484 to 85.203 s (about 47% less CPU). A separate three-pair comparison
against the packed/parallel CPU version measured 167.320 -> 66.829 s
(2.50x), with a 2.82x median paired ratio. These cohorts remain separate.

The standard 400-point/frame workload measured only 1.27x ratio of whole-wall
medians and 1.11x median paired ratio. Its first candidate took 72.676 s versus
28.353 s for the preceding GPU version; that slower cold case is retained.
Default sampling caps remain CPU-only and measured 5.672 -> 5.456 s (1.04x).
The earlier 64-row replay dispatch regressed those defaults to 0.90x, motivating
the final frame crossover; all preceding trials remain in the evidence.

A separate final dense diagnostic records 52 GPU query batches, 24,960,000
uniform and 24,960,000 adaptive point queries, and no complete query fallback.
It also records 129 GPU linearizations over 75,503,015 correspondence rows,
207 CPU standalone residual calls and 129 ordered CPU normal calls. Compiled
CPU replay processes 120 batches and 960,000 target rows. Every complete
saved/API result retains estimates, traces, statuses and counters apart from
the exact execution notes. Dense/standard results retain `max_iterations`,
rank 6; default results retain `converged`.

```bash
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope cuda \
  --baseline-commit a1ff874ab0dbc39c2dacc9607f40841c610997d7 \
  --baseline-query-backend cuda --candidate-query-backend cuda \
  --config examples/public_datasets/glim_versatile/online_continuous_time_compiled_dense_capture_config.yaml \
  --repetitions 5 \
  --output-dir outputs/public-glim-cuda-associations-dense-reproduction
```

Both methods own initially empty persistent compiler caches, and wall includes
initialization, compilation, replay, transfers, all profile/refinement work,
diagnostics, hashes and saves. No warmup, cold case or outlier is discarded.
The copied public topic supplies execution parity, without independent
two-sensor accuracy evidence. See the [complete association/replay evidence](../assets/cuda_continuous_time_associations/README.md)
for all trials, source/input bindings, actual work and preserved preceding candidates.

## Parallel CPU summaries and finite differences

For native immutable batches of at least 32,768 observations, eligible compiled
CPU-backend solves share three additional helpers across candidate workers. Each request
computes two disjoint row blocks. Summary accumulation and robust normal equations
keep their exact serial order. CPU finite differences also retain their last exact
batch within a 64 MiB cache; callers receive independent writable arrays.
Six logical CPUs and a sufficiently large sampling budget are required to create
the helper pool. Smaller/default sampling follows its previous execution route.
CUDA finite-difference and query kernels stay unchanged.

| Workload | Previous whole median | Candidate whole median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|
| CPU: 960,000 captured points | 88.192810 s | 50.415180 s | 1.749x | 1.728x |
| CUDA policy smoke: one pair | 46.185255 s | 51.526152 s | 0.896x | 0.896x |
| Default sampling: unchanged small-batch route | 5.601387 s | 5.556325 s | 1.008x | 0.980x |

The primary CPU comparison uses three alternating fresh-process pairs; the CUDA
policy smoke uses one pair and does not establish a GPU speedup. Default sampling
retains five pairs on the preceding experimental source with identical small-batch
dispatch. CUDA method cache directories were separate. CPU scopes used a shared
directory with distinct code-cache namespaces: the first candidate inventory
contained only baseline files and left those files unchanged. Cache hashes are
audited; the final harness also separates CPU method directories explicitly.
The directory correction was not formally retimed. First compilation and every slower observation
remain inside the measurements. Wall includes startup, replay, every candidate
and refinement, transfers, diagnostics, hashing and saves.
An initial CPU cohort observed another project benchmark and is retained separately.
The table uses the repeated CPU cohort after that observed benchmark ended.
Other OS/application load is not controlled; these are local observations.
Ratios are not pooled with or multiplied by preceding speedups.

The helper-enabled CUDA experiment had a 0.865x ratio of whole medians
(43.256966 s to 50.028804 s, about 16% slower). The final policy creates
extra helpers only for the CPU backend and retains the preceding CUDA route.
The exact one-condition source change and all unsuccessful observations are retained.

All complete results match within their cohorts, including histories and diagnostics.
A separate CPU diagnostic confirms 56 parallel finite-difference misses and 73 exact
cache hits across 129 logical normal-equation calls, without importing CuPy.
The fourteen-module `cpu-full` baseline scope requires explicit CPU overrides and
verifies packed observation and compiled replay bindings.

```bash
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope cpu-full \
  --baseline-commit aa19ec5d4bf01e1167e70f48332e6e13d51263c0 \
  --baseline-query-backend numba_cpu --candidate-query-backend numba_cpu \
  --config examples/public_datasets/glim_versatile/online_continuous_time_compiled_dense_capture_config.yaml \
  --repetitions 3 \
  --output-dir outputs/public-glim-parallel-cpu-reproduction
```

The public sensor copied into two topics provides execution parity without
independent two-sensor accuracy evidence. See the
[complete parallel CPU evidence](../assets/parallel_continuous_time_lidar/README.md)
for all 44 complete cohort trials and 55 complete saved results, stopped/failed attempts,
seven kernel probes, exact source/input hashes and independent work counters.

## Experimental captured CUDA residual summaries

`ContinuousTimeLidarPairOptions.cuda_captured_summary=True` explicitly enables
CUDA residual summaries for native factors whose immutable observation arrays
are already uploaded by finite differences. The default is `False`: this
experiment did not establish a reliable whole-pipeline speedup. The option is
available through the direct Python solver API; YAML and result schemas are
unchanged.

Fresh summaries reuse device observations, upload the transform, download
residual values and retain ordered CPU summation. Initial and holdout summaries,
uncaptured data, small factors and unsupported summation runtimes keep the CPU
route. Cached values remain read-only; public outputs and histories are preserved.

Three alternating fresh-process pairs each for dense and full capture, plus five
for default sampling, retain every initial compilation and slower trial. Each
method owns initially empty persistent compiler caches. Whole wall includes
startup, replay, all candidates/refinements, transfers, diagnostics and saves.

| Workload | Previous GPU median | Experimental median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|
| 960,000 captured points | 35.314477 s | 38.644438 s | 0.914x | 1.130x |
| 48,000 captured points | 19.251234 s | 17.538761 s | 1.098x | 1.098x |
| Default sampling | 7.490853 s | 6.837300 s | 1.096x | 0.982x |

The dense ratio of medians is about 9% slower. Default sampling runs no CUDA
summary and shows inconsistent timing ratios. All 22 complete results match
after validating execution identity notes. No new CPU-only timing cohort was
run; these observations are not pooled with preceding speedup ratios.

An independent diagnostic confirms 102 additional GPU residual launches over
59,724,347 rows without extra observation uploads. Reduction remains on CPU;
finite-difference and query work remains unchanged. More GPU work does not by
itself establish effective acceleration. The copied public sensor topic is an
execution parity control without independent two-sensor accuracy evidence.
See the [complete experimental evidence](../assets/cuda_captured_residual_summary/README.md)
for all trials, exact measured sources and the separately tested final policy
that keeps this operation disabled by default.

## Fusing ordered CPU residual error summaries

Native compiled factors with at least 8,192 observations compute residuals and
their squared-error sum in one CPU Numba pass, releasing the GIL and avoiding
Python row objects. Fastmath remains disabled. CPython 3.10/3.11 use ordered
addition; CPython 3.12 uses its native compensation policy, including overflow
handling. Python performs the final division and square root. Smaller factors,
other runtimes and custom residual/correction or reduction hooks retain the
original path. Public residuals and final diagnostics remain complete.

The native optimizer also requests normal equations without converting their
discarded residual vector to a list. Direct callers and custom compiled-system
hooks retain complete residuals and their existing argument contract.

Three alternating fresh-process pairs retain compilation, startup, replay, all
candidates/refinements, transfers, diagnostics, hashes and saves. Each method
owns initially empty persistent compiler caches. Every trial is retained.

| Comparison | Baseline whole median | Candidate whole median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|
| Previous GPU, 960,000 captured points | 69.805556 s | 50.653481 s | 1.378x | 1.380x |
| Previous GPU, 48,000 captured points | 13.870382 s | 13.226075 s | 1.049x | 1.034x |
| Previous GPU, default sampling | 7.703580 s | 8.765065 s | 0.879x | 0.965x |
| Packed CPU, separate dense control | 196.076865 s | 39.026530 s | 5.024x | 5.024x |

Default timing regresses by about 14% in the ratio of medians, while keeping the
original residual/RMSE route. The cause of timing variation remains unresolved.
Cold CPU wall of 289.691684 s is retained. Separate cohorts are not pooled or
multiplied by previous ratios. Every complete numerical result, optimization
history and diagnostic matches after validating execution identity notes.

An independent diagnostic records 154 summary calls over 76,674,615 rows and
102 discarded residual lists avoided over another 59,724,347 rows. Existing
GPU work remains 52 plane queries, 56 finite-difference launches and 73 cache
hits. GPU coverage remains unchanged. The public sensor is copied into two
topics as an execution parity control, without independent two-sensor accuracy
evidence. See the [complete summary evidence](../assets/continuous_time_residual_summary/README.md)
for all trials, preserved profiles, summation policy, source/input bindings
and limits.

## Fusing CPU observation construction around GPU queries

Large compiled LiDAR batches combine raw-normal scores, plane selection and
ordered gathers in two CPU Numba kernels. They release the GIL with fastmath
disabled and keep the existing three candidate workers. Batches below 8,192
rows retain NumPy construction. Equal-centroid score ties keep the first plane;
other exact score ties retain Python `math.dist`. Outputs use independent
read-only allocations, and public query/observation guards remain active.

Three fresh alternating pairs per workload retain compilation, startup, replay,
all candidates/refinements, transfers, diagnostics, hashes and saves. Initial
compiler caches are empty and separate for each method; all trials are retained.

| Comparison | Baseline whole median | Candidate whole median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|
| Previous GPU, 960,000 captured points | 53.362673 s | 50.009781 s | 1.067x | 1.100x |
| Previous GPU, 48,000 captured points | 10.123346 s | 9.228069 s | 1.097x | 1.061x |
| Previous GPU, default sampling | 5.450336 s | 5.448372 s | 1.000x | 0.992x |
| Packed CPU, separate dense control | 136.841544 s | 45.582851 s | 3.002x | 2.930x |

The incremental whole-pipeline improvement is modest; GPU coverage remains
unchanged. The separate CPU control describes the cumulative hybrid pipeline.
These cohorts must not be pooled or multiplied by earlier results. Every
complete result value, history and diagnostic matches after validating
execution notes. The copied public sensor topic is an execution parity control,
without independent two-sensor accuracy evidence.

An independent diagnostic records 52 compiled construction calls covering
24,960,000 rows and 307,670 distinct-centroid ties requiring native distance
comparisons. It also confirms the existing 52 GPU queries, 56 actual GPU
linearizations and 73 linearization cache hits. See the
[complete construction evidence](../assets/continuous_time_observation_packing/README.md)
for every trial, preserved diagnostics, exact source/input bindings and limits.

## Reusing exact continuous-time LiDAR differences

The CUDA adapter keeps one exact residual/Jacobian entry per owned context,
keyed by the FP64 bits of all thirteen transforms and six difference steps.
Replacing immutable observations invalidates it. Public return arrays stay
independently writable, forks own their caches, and retained entries are capped
at 64 MiB. Native compiled factors with at least 8,192 residuals compute only
initial rank at solver entry; full final diagnostics and public evaluation
hooks retain their existing behavior.

The final public dense control records 129 difference requests, 56 GPU launches
and 73 cache hits, with no CPU difference fallback. A separate instrumented run
matches the complete final calibration artifact. GPU plane queries, CPU replay,
standalone CPU residuals and ordered CPU normal reductions keep their existing
work and arithmetic.

| Complete public workload | Baseline / candidate wall medians | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|
| Dense, previous GPU, three pairs | 81.842589 / 58.798766 s | 1.392x | 1.392x |
| Dense, packed/parallel CPU, separate three pairs | 149.580531 / 52.385271 s | 2.855x | 3.231x |
| Standard, previous GPU, three pairs | 17.541021 / 13.136499 s | 1.335x | 0.974x |
| Default CPU dispatch, five pairs | 7.633336 / 8.759261 s | 0.871x | 0.911x |

Every pair preserves complete API/saved results, iteration histories, status,
correspondences, holdout diagnostics and logical counters. Fresh processes,
initially empty method-specific compiler caches, first compilation, slow runs
and outliers remain included. Whole wall includes startup, replay, transfers,
all candidates/refinements, diagnostics, hashes and saves; parent audits are
outside it. Separate cohort ratios are never multiplied.

The dense timings support a local gain. Standard paired timing does not show a
consistent gain, and default timing regresses despite retaining native small
factor evaluation. The cause of timing variation remains unresolved. The copied
public sensor topic provides execution parity without independent two-sensor
accuracy evidence. See the [complete reuse evidence](../assets/continuous_time_linearization_reuse/README.md)
for all final trials, exact sources, actual counts, the preceding small-workload
regression and both stopped comparisons.

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

## Exact mutual point-to-point ICP on CUDA

Use the optional `CudaPointToPointIcpAdapter` with the same `IcpPoint` inputs and
options as the native robust solver. It accelerates both nearest-neighbor
directions and retains native pose updates, holdout and calibration diagnostics.
It shares static source/target snapshots across the six diagnostic starts.

```python
from calibrex.solvers.cuda_point_to_point_icp_adapter import CudaPointToPointIcpAdapter

answer = CudaPointToPointIcpAdapter().solve(source_points, target_points)
transform = answer.result.transform_target_source
provenance = answer.provenance
```

Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before
starting Python for the measured parallel diagnostic route. The adapter keeps
process-global BLAS settings unchanged. Small and extreme-coordinate searches
fall back to native arithmetic; provenance records actual GPU query counts.

The public KITTI control measured **1.92x** whole-worker acceleration (46.8 → 24.4 s medians), with exact complete results across three alternating pairs. This is a copied
public-cloud execution control. See the [complete ICP evidence](../assets/cuda_point_to_point_icp/README.md)
for every trial, thread budget, retained failed workers and reproduction commands.

The preceding initial-projection cache was rejected: GPU dense improved only
1.049x and CPU dense regressed by about 5%. Its code was reverted; [all retained
trials and diagnostics](../assets/initial_lidar_projection_reuse/README.md) remain available.

The follow-up comparison uses actual consecutive KITTI frames with navigation
initialization and the default 50-iteration budget. All three CPU/CUDA pairs
converged with identical complete native results; whole-worker medians were
**56.0 → 25.3 s (2.21x)**.
Two additional frame pairs retained result parity. The 768-point control used
native fallback and was slower with cold CUDA initialization; choose CPU for
small clouds. See the [actual-frame evidence](../assets/cuda_point_to_point_icp/actual_frames/README.md)
for all trials, holdout scores, caps and reproducible fixture commands.
