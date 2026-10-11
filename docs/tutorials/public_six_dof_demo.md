# KITTI raw CPU/GPU calibration demo

Run a Camera–LiDAR six-DoF comparison using public KITTI raw data and a frozen
depth-provider artifact. No personal recording or rosbag conversion is needed.
One command verifies inputs, selects 25 frames, estimates on 20, evaluates the
other 5, and writes an offline HTML report with image overlays.

## Prepare public inputs

Get a synced, rectified drive and date-level calibration files from the
[official KITTI raw dataset](https://www.cvlibs.net/datasets/kitti/raw_data.php).
This example uses `2011_09_30_drive_0018_sync`, `image_02`, and original
Velodyne scans. Follow the dataset's download and license terms. Keep this layout:

```text
2011_09_30/
  calib_cam_to_cam.txt
  calib_velo_to_cam.txt
  2011_09_30_drive_0018_sync/
    image_02/data/*.png
    velodyne_points/data/*.bin
    velodyne_points/timestamps.txt
```

Supply an existing `slac.depth_provider/v0.1` manifest with at least 25 frames
from that drive. Image, depth, and checkpoint hashes are verified. This demo
requires a rectified pinhole camera and depth returned to the source image grid:
identical source/output dimensions, unit scales, and zero crop offsets.

To generate a provider once, use the existing external FeatDepth runner from a
checkout and a separate environment with the pinned FeatDepth dependencies:

```bash
PYTHONPATH=src /path/to/featdepth-env/bin/python \
  tools/run_featdepth_provider.py /path/to/2011_09_30_drive_0018_sync \
  --featdepth-repository /path/to/FeatDepth \
  --checkpoint /path/to/fm_depth.pth --frame-count 25 \
  --output-directory /path/to/depth \
  --artifact-output /path/to/featdepth-provider.json
```

See the [Camera–LiDAR input lock](../research/camera_lidar_sota_plan.md) for the
repository/checkpoint identity and reproduction limits. The demo does not
download data, infer new depth, or refine the provider on holdout frames. Raw
data, images, and weights remain external to the repository.

## Run the comparison

Install the [optional CUDA backend](gpu_acceleration.md#install), then run:

```bash
calibrex demo kitti-six-dof \
  /path/to/2011_09_30_drive_0018_sync /path/to/featdepth-provider.json \
  --output-dir outputs/kitti-six-dof-demo --json
```

The candidate defaults to `cuda`; the reference always uses NumPy CPU. Without
CUDA, explicitly choose `--candidate-backend numpy` for a CPU replay, or
`numba_cpu` for the optional CPU accelerator. An unavailable CUDA backend
produces a failed trial, without substituting a CPU candidate.

`--cuda-cache-mib N` requires CUDA and configures its input cache.
`--max-evaluations N` sets the shared search budget (default 800, minimum 13).
`--rotation-offset-deg X Y Z` and `--translation-offset-m X Y Z` declare the
controlled initial perturbation: defaults `(0.5, 0, 0)` degrees and `(0.5, 0, 0)`
metres. Rotation uses local Euler angles; translation is added in the camera
frame. Search bounds are ±2 degrees and ±1 metre per axis around this initial
pose. Record changed settings before comparison; do not tune on holdout results.

Open `outputs/kitti-six-dof-demo/report.html` locally. Images and links are
relative, so move the entire directory together. Use a new or empty output
directory for every run; existing evidence is preserved.

For a comparison with timing variation, request repeated invocations:

```bash
calibrex demo kitti-six-dof \
  /path/to/2011_09_30_drive_0018_sync /path/to/featdepth-provider.json \
  --repetitions 3 --output-dir outputs/kitti-six-dof-repeat3 --json
```

`--repetitions` defaults to 1 and must be a positive integer. Every invocation
uses the same verified problem bytes, declared initial pose, and prespecified
20/5 split. CPU runs first on odd-numbered repetitions and the candidate runs
first on even ones. Each run verifies and loads inputs and prepares a fresh
projector. The first run is included; there is no separate warm-up, and later
invocations share process-wide kernel caches. These times describe this process,
rather than guaranteeing fully cold or warmed-up latency.

The HTML shows all attempts and successful-run median/minimum/maximum solve
and pipeline times, with success/attempt counts. Failed or partial trials remain
visible and are excluded from the timing summary. The ratio of method medians
appears only if all runs converge with exact numerical parity. Pipeline total
sums load/verification, preparation, solve, and holdout phases; shared demo
setup and report rendering are excluded. Repeated timings on these same frames
are not independent calibration recovery trials or confidence intervals.

Each repetition saves its own standard result. The first pair keeps
`numpy.result.json` and the candidate's usual name; later files use
`numpy.repeat-02.result.json`, `cuda.repeat-02.result.json`, and so on.
All result hashes are bound by the benchmark, and result provenance records
the repetition number, count, order, and shared problem hash. The report's
overlays always show repetition 1, without replacing a missing first result.
The JSON command output includes `timing_summaries` and every individual trial.

## Read the evidence

- Exactly 25 endpoint-inclusive uniform positions are selected from numeric-
  sorted provider frames. If it contains 25, all are used. Every fifth selected
  frame is holdout (20 fit / 5 holdout), fixed before optimization and histogram
  range selection. This is a sample of the supplied provider, not necessarily
  all frames of the drive.
- Both runs receive the same perturbed initial pose. Dataset calibration creates
  this controlled prior and supplies reference differences; it never becomes
  the optimizer's search target. The prior is not an independent calibration.
- Initial/final reference differences and fit/holdout MI are reported separately.
  Convergence describes optimizer termination. Standard-result quality remains
  `warn`: covariance and observability are unknown.
- Exact numerical parity checks the full search trace, final poses, metrics,
  status, and initial/final holdout evaluations, across both backends and every
  repetition. Backend identities and timing remain in individual results.
  Even equal CPU/candidate pairs fail parity if they drift between repetitions.
- Load/verification, preparation, solve, and holdout times are separate. CUDA
  solve time includes lazy compilation. One invocation per backend is
  descriptive. Use repeated invocations to expose variation, without treating
  those repetitions as independent recovery evidence or statistical confidence.
- Three prespecified holdout frames show initial and output pose overlays using
  NumPy visibility semantics. At most 1,200 finite depth pairs are sampled for
  display; optimization uses full scans. Colours use a fixed 0–80 m LiDAR range.
- Learned-depth overlap, depth errors, and synchronization affect the objective.
  FeatDepth training overlap with drive 0018 is recorded as possible. Dataset
  calibration is reference evidence rather than independent metrology.

Schema-valid outputs are `source.problem.json`, `demo.problem.json`,
`numpy.result.json`, the candidate result, `comparison.definition.json`, and
`comparison.benchmark.json`. The benchmark binds results, provider, problem,
preview images, and generator sources by SHA256. Standard results retain
effective settings, train/holdout IDs, backend identity, transfer counters,
input hashes, and the full trace. HTML links to all available artifacts.

Exit code `0` requires every requested run to converge, evaluable holdout, and exact parity.
Code `2` retains partial/failed comparisons and reasons in HTML and JSON. Bad
inputs are rejected before the comparison starts.

## Local public-data evidence

On the original public drive 0018 frames, both backends converged in 229
evaluations with exact numerical parity. Reference rotation difference changed
from 0.5000° to 0.3909°, translation from 0.5000 m to 0.1601 m, and holdout MI
from 1.0857 to 1.2935 nat. The final GTX 1660 Ti invocation took 54.824 s CPU /
6.734 s CUDA for solving, and 60.751 / 15.398 s for the measured pipeline phases.
The earlier invocation's solve-time ratio was 4.58x versus 8.14x for the final
invocation, illustrating timing variation.

With `--repetitions 3` on the same GTX 1660 Ti, all six runs converged with the
same 229-evaluation numerical trace. CPU solve median was 74.306 s (range
51.270–75.150 s); CUDA median was 3.611 s (3.302–8.685 s), a ratio of medians of
20.58x. Measured pipeline medians were 76.865 / 7.652 s (10.04x), excluding shared
setup and report rendering. All invocations are included, with alternating
execution order and no separate warm-up. The variation across these and earlier
observations remains part of the evidence.

[Published schema-valid evidence and limitations](../assets/public_six_dof_demo/README.md)
retain individual timings, repeated summaries, and earlier full numerical traces. This is
one development drive and perturbation, rather than a multi-dataset recovery
evaluation or a portable speed guarantee.

## Compare six initializations and parallelize CPU cases

Use a prespecified signed-axis sweep to check that CPU/candidate parity extends
beyond one initial pose:

```bash
calibrex demo kitti-six-dof-sweep \
  /path/to/2011_09_30_drive_0018_sync /path/to/featdepth-provider.json \
  --output-dir outputs/kitti-six-dof-sweep --json
```

The six cases are x-positive, x-negative, y-positive, y-negative, z-positive,
and z-negative, in that order. Each adds rotation and translation on that axis
to the dataset reference: local Euler rotation ±0.5 degrees and camera-frame
translation ±0.5 m by default. `--rotation-deg` and `--translation-m` set positive
magnitudes within the solver bounds. This is a controlled prior, not an
independently measured initialization. Every case uses the same prespecified
25 frames, 20/5 fit/holdout split, frozen provider, and solver settings.

For CPU-only comparisons, independent cases can run in separate processes:

```bash
calibrex demo kitti-six-dof-sweep \
  /path/to/2011_09_30_drive_0018_sync /path/to/featdepth-provider.json \
  --candidate-backend numba_cpu --workers 2 \
  --output-dir outputs/kitti-six-dof-sweep-cpu --json
```

`--workers` defaults to 1 and must be a positive integer; at most six workers
are created. Each spawned process owns its input arrays, projector, and result
directory. Memory use grows with the number of workers. Use `numpy` as the
candidate for an explicit CPU replay if Numba is unavailable. CUDA comparisons
require `--workers 1` and reject concurrent GPU contexts before loading inputs.
This parallelizes independent cases, not the solver's ordered search decisions.
Individual timing measurements under multiple workers include CPU/memory
contention and are not isolated serial latencies.

`--repetitions`, `--cuda-cache-mib`, and `--max-evaluations` work as in the
single-case demo. Invocations inside each case are serial and alternate method
order across repetitions. CPU/candidate parity is checked within each case and
all its repetitions; different cases can legitimately have different traces
and final poses. No cross-case speed ratio is inferred.

Before optimization, `sweep.plan.definition.json` saves a schema-valid plan
with entries explicitly marked `not executed; frozen plan entry`. It binds the
verified `sweep.source.problem.json`, solver settings, worker count, and fixed
initialization policy. The plan is never rewritten as completed evidence.
`sweep.definition.json` and `sweep.benchmark.json` preserve every completed,
partial, or failed attempt and bind the plan, child comparisons, and original
standard results. Each case retains the usual demo artifacts and HTML overlays
in its own subdirectory. The suite `report.html` links all six case reports.
Successful-only medians do not hide failures: success/attempt counts and
failure reasons remain visible. Input verification errors abort the suite and
preserve already-written evidence.

The JSON output includes elapsed time through all case reports: shared
verification, worker startup, input loading, optimization, holdout, result
serialization, and case HTML rendering. Final suite summary writing is
excluded. These six fixed initializations describe one development drive,
not independent metrology, a full recovery basin, or statistical confidence.
Exit `0` requires all requested attempts to converge with evaluable holdout
and exact parity; exit `2` retains incomplete comparisons.

On public drive 0018, all six signed-axis CPU/CUDA pairs converged with exact
full numerical parity (229–313 evaluations per backend). Y-positive rotation
difference increased from 0.5 to 0.527379 degrees, illustrating why convergence
and reference differences are reported separately. In a separate fixed
13-evaluation NumPy/replay execution check, serial elapsed was 139.843 s versus
74.844 s with two CPU worker processes (one observed 1.87x ratio). All twelve
serial/parallel numerical payloads matched exactly; these capped calibration
attempts remained partial. [Published sweep evidence](../assets/public_six_dof_sweep/README.md)
retains every condition, timings, original hashes, and interpretation limits.

## Parallelize frames inside one CPU estimate

CPU case processes and frame threads handle different work. To compare one
serial NumPy estimate with a NumPy candidate using two persistent frame threads:

```bash
calibrex demo kitti-six-dof \
  /path/to/2011_09_30_drive_0018_sync /path/to/featdepth-provider.json \
  --candidate-backend numpy --candidate-frame-workers 2 --repetitions 3 \
  --output-dir outputs/kitti-six-dof-cpu-frames --json
```

The reference remains serial. The candidate reuses one thread pool for every
search evaluation and both holdout evaluations. Each frame's projection and
histogram run independently; results are consumed in input order, preserving
the serial floating-point mean and search decisions. The comparison checks the
entire numerical trace, final pose, train/holdout metrics, and skipped frames.
Its report labels the candidate's requested thread count, while each standard
result's `run.provenance.configuration` records `frame_workers` and
`effective_frame_workers`. Pool capacity is capped at the larger fit/holdout
frame count. Pools are joined on success and on errors.

For an existing problem, use `calibrex camera-lidar estimate-six-dof
problem.json --projection-backend numpy --frame-workers 2 --output result.json`.
The default is one frame worker. CUDA rejects more than one before input loading.
CPU backends accept frame threads; the public timing evidence measures NumPy.
The `numba_cpu` backend uses a fused compiled histogram path and releases the
GIL so its frame calculations can run concurrently.

The sweep also accepts `--candidate-frame-workers`. With `--workers 2
--candidate-frame-workers 2`, at most two cases run in separate processes and
each candidate uses up to two frame threads; each reference is serial within
its process. Increase these settings deliberately: the active candidates can
use up to process count times frame-thread count, in addition to numerical
library threads. Timing depends on frame size, hardware, and other running work.

On public drive 0018, three runs per method all converged with the same complete
229-evaluation numerical payload. Solve medians were 70.178 s serial and
54.549 s with two frame threads (one observed ratio of 1.29x); ranges were
64.058–72.149 / 42.317–60.098 s. Pipeline phase medians were 73.584 / 57.475 s.
[Published CPU frame evidence](../assets/cpu_frame_parallelism/README.md)
retains every invocation, original hashes, representative full traces, and
timing/accuracy limitations. These repetitions measure one workload's latency.

## Use the fused CPU backend

The optional Numba implementation performs projection, nearest-point selection,
and histogram accumulation directly, avoiding paired feature arrays and repeated
bin searches. Prepared frames copy the verified input features into immutable
snapshots and cache one histogram configuration per frame. Scratch is private
to each thread, grows to the largest image encountered, and clears only pixels
hit by the current scan. All histogram counts, visibility gates, input-order
ties, and float64 arithmetic retain the native objective; no point subsampling
or search-budget change is involved.

```bash
pip install 'calibrex[numba]'
calibrex demo kitti-six-dof \
  /path/to/2011_09_30_drive_0018_sync /path/to/featdepth-provider.json \
  --candidate-backend numba_cpu --candidate-frame-workers 2 --repetitions 3 \
  --output-dir outputs/kitti-six-dof-fused-cpu --json
```

The same backend is available to `estimate-six-dof` and CPU signed-axis sweeps.
Pair-array consumers can still request individual projected samples. Constructor
snapshots preserve explicitly prepared inputs even if the caller later edits its
buffers; unprepared observations use their current values without being retained.
Memory includes the prepared feature copies/bin indices and one scratch image
per active thread. The first compiled call can include JIT compilation/cache
loading. Timing comparisons should retain that call and the backend preparation
phase, and use the same frozen problem, split, prior, and evaluation settings.

The implementation uses [Numba's GIL release](https://numba.readthedocs.io/en/stable/user/jit.html#nogil)
and preserves [NumPy histogram edge conventions](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html#notes).

On public drive 0018, three runs per method all retained the exact complete
229-evaluation numerical payload. Solve medians were 67.078 s for serial NumPy
and 3.624 s for the fused backend with two frame threads (**18.51x**); pipeline
phase medians were 70.886 / 6.500 s (**10.91x**). The first candidate included
compilation from an absent Numba cache and took 5.139 s versus its paired
53.928 s serial solve (**10.49x**). The six previously frozen signed-axis
problems also retained their full CPU numerical payloads (229–313 evaluations).
[Published fused CPU evidence](../assets/fused_cpu_histogram/README.md) retains
all timed invocations, original hashes, full representative traces, six-axis
validation, cold-cache conditions, and timing/accuracy limitations. These are
observations on one i7-9750H workload with other running work; numerical equality
preserves the existing outcome and does not establish calibration accuracy.
