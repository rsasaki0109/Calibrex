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
for this six-DoF benchmark. The bag-level `estimate`, `check` and `drift` commands
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

Coordinates, depth maps and original LiDAR ranges are prepared once for the
benchmark, with at most 256 MiB of cached input arrays. Larger inputs are
transferred per call. Scratch arrays and
CuPy's reusable allocation pool are additional memory. Prepared arrays are
snapshots and must remain unchanged during a run. Direct Python callers can
construct `CudaDepthPairProjector(observations, cache_bytes=0)` to disable input
caching, and pass it through `BorerSixDofOptions(projector=..., projection_backend=...)`.
Use `projector.identity()` for the backend identity.

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

## Measured development workload

Warm medians on a GeForce GTX 1660 Ti (6 GiB), Core i7-9750H and Windows,
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
