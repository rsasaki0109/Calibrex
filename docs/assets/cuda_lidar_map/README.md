# CUDA LiDAR–LiDAR map registration

The optional `cuda` registration backend moves exact 3D neighborhood queries,
normal covariance/eigensolves, point transforms, point-to-plane residuals,
analytic Jacobians and Huber-weighted normal equations onto the GPU. A captured
SciPy tree supplies the same bounded spatial topology; its construction stays
on CPU. Six-dimensional updates, residual medians and calibration/evidence
policy retain the native definitions. One solve-scoped snapshot is reused for
the training fit, four block jackknife fits, holdout and six known-bad controls.

The measured source commit is `6d6600cc476203ae578e7ac629b576a20818a7a9`.
Both methods use that commit with the same input bytes, settings and sample
identities. CPU uses the existing vectorized NumPy/SciPy spatial implementation.
The optional adapter is isolated from core imports; CPU remains the default.
The saved artifact schema is unchanged. CUDA execution identity is recorded
in the artifact options and includes runtime versions, FP64/query-FMA policy,
compute capability and kernel hash, without a device model label.

## Frozen public workload

The public KITTI raw drive `2011_09_30_drive_0018_sync` supplies Velodyne scans
115, 230 and 345, whose byte hashes are retained. Each becomes two map samples
with generated rigid reference poses. The corresponding target is a copy with
fixed 0.01 m Gaussian noise (seed 20261012). The initial translation is
[0.08, -0.07, 0.05] m. These are noisy identity execution controls rather than
independent two-sensor extrinsic accuracy or a motion-distortion experiment.

The final cohort uses `--half-window 0`: 127,158 / 127,303 / 126,598 original
points per scan, all retained. Six samples contain 762,118 target rows, with
508,362 training and 253,756 holdout rows. No point cap, approximation or lower
precision is introduced. Every third sample block is held out. Normal
neighborhoods, planarity threshold, Huber loss and iteration limits are fixed.

These generated sample blocks reuse three underlying scans and noise arrays;
their train/holdout split is an execution-policy control, not independent
temporal validation. The complete raw scan references and sample hashes remain
visible so that this limitation can be assessed.

The benchmark supports larger raw-scan windows when those original inputs are
available; the published timing claim applies only to this measured cohort.

## Complete-process measurements

| Scope | CPU median (s) | CUDA median (s) | Ratio |
|---|---:|---:|---:|
| Fresh complete process | 16.951654 | 6.646739 | 2.55x |
| Complete map-evaluation API | 13.896949 | 3.800446 | 3.66x |
| All-thread worker CPU | 16.156250 | 5.750000 | 2.81x |

| Pair | Order | CPU wall (s) | CUDA wall (s) |
|---|---|---:|---:|
| 1 | CPU → CUDA | 12.621121 | 17.157697 |
| 2 | CUDA → CPU | 19.158078 | 6.646739 |
| 3 | CPU → CUDA | 16.980075 | 6.605833 |
| 4 | CUDA → CPU | 15.631914 | 6.863701 |
| 5 | CPU → CUDA | 16.951654 | 6.384456 |

Five paired fresh interpreters reverse order every other pair. Both methods
start with separate empty compiler-cache directories; subsequent processes
retain their method's cache. First compilation, imports, raw input reading,
generated sample preparation, GPU snapshot construction, the complete blocked
evaluation, input/result hashes and schema-valid result saves are included in
complete wall time. Parent byte guards, aggregation and publication are outside.
There is no separate warmup or excluded timing. The first CUDA run is slower
than CPU and remains visible. Cache file hashes are recorded before/after every
run. All-thread worker CPU excludes interpreter startup and the final
measurement JSON write; its scope differs from complete wall.

The map-evaluation API includes normal preparation, training optimization,
jackknife, held-out scoring and all six perturbation controls. It excludes raw
input preparation and result saves, which complete wall includes. This is not
an end-to-end ROS-bag odometry benchmark: bag decoding, reference odometry and
scan preprocessing remain CPU operations in the ordinary CLI.

All ten results converge with the same policy `pass`, correspondence counts,
six estimated DoFs, four jackknife fits and all known-bad control decisions.
All complete evaluation fields are compared. Float fields use absolute 1e-9
and relative 1e-6 tolerances; covariance uses absolute 1e-13 and relative 1e-5.
Counts, iterations, policies, decisions and strings match exactly, excluding
only the precisely declared execution identity. GPU eigensolves and reductions
have different floating-point order, so this is numerical tolerance parity.
The benchmark retains full unfiltered evaluation and saved-artifact payloads.

Every CUDA run records 9,147,116 actual spatial-query rows and 66 scalar query
repairs, with 65,299,350 retained input bytes. Roundoff-close neighborhood
boundaries, nearest ties, gate boundaries and ill-conditioned normal geometry
use native repairs. Spatial queries are exact searches with no approximate
epsilon. Prepared normal queries are chunked at 65,536 rows. Retained device
payloads are bounded to 1 GiB; scratch and allocator pools are additional.
Oversized/unsupported inputs and unavailable CUDA report explicit errors.
Normal neighborhoods from 2 through 16 are supported by this optional backend.

These are descriptive observations on a shared host, not portable confidence
estimates or independent sensor-calibration validation. The earlier four
exploratory attempts are retained separately in `earlier-attempts.json`; their
final aggregation failed a required metric-description field, which was repaired
before freezing this final cohort. They are not pooled with the final timings.
Their original tool bytes were not archived; original working/source/input
hashes and complete results remain in those records.

## Validation and provenance

Local validation: 34 relevant tests pass, including 17 real-device CUDA tests;
Ruff and strict mypy over 351 modules pass. All 126 exported schemas remain
byte-identical. Tests cover neighbor identities/distances, small clouds, ties,
inclusive/exclusive boundaries, copied snapshots, moving reference poses,
full solves/covariances, held-out controls, blocked jackknife policy, CPU-only
imports, explicit CLI dispatch and comparator rejection of hidden changes.
All eight CI jobs passed at the measured source commit.

`publication-index.json` binds exact Git source archives, both benchmark
schemas, all ten saved results and complete measurement records, the separate
exploratory ledger and the publication launcher. Original raw-result/report
hashes are retained separately from portable publication hashes. Workspace and
dataset prefixes become placeholders; raw scans and compiled binaries remain
external. The aggregate is recomputed from the full definition.

## Use and reproduce

Install the optional CUDA adapter and select registration explicitly:

```bash
python -m pip install -e ".[cuda]"
calibrex lidar-lidar ros2 path/to/bag \
  --reference-topic /reference/points --target-topic /target/points \
  --initial initial-transform.yaml --registration-backend cuda \
  --dataset-family your-dataset --dataset-license your-dataset-license \
  --output outputs/lidar-map-cuda.json
```

For the frozen public-data comparison, obtain the original KITTI drive scans
under the [official dataset terms](https://www.cvlibs.net/datasets/kitti/raw_data.php)
and use the measured source commit with a new output folder:

```bash
python tools/benchmark_cuda_lidar_map.py \
  --sequence path/to/2011_09_30_drive_0018_sync \
  --half-window 0 --repetitions 5 --output-dir outputs/cuda-lidar-map-comparison
```
