# Resident CUDA geometry for LiDAR-IMU bags

`imu-lidar joint --resident-odometry` keeps decoded scan coordinates, deskewed
returns, voxel centroids and sliding local maps on CUDA across the initial and
gyro-deskewed odometry passes. It uses the existing registration loop, native
small motion solves, stopping policy and numerical repairs. Pose-only spline
point placement also avoids computing unused IMU derivatives.

Install the optional `calibrex[cuda]` dependency and explicitly select the
odometry backend and all three geometry/motion grids. For the public GLIM
Livox bag, an example development run is:

```bash
calibrex imu-lidar joint data/public/glim_versatile/livox.bag \
  --lidar-topic /livox/points --imu-topic /livox/imu --acceleration-unit g \
  --dataset-family glim_versatile --dataset-license "upstream AIST GLIM dataset terms" \
  --max-seconds 6 --max-rounds 4 \
  --odometry-backend cuda --resident-odometry --fit-plane-backend cuda \
  --odometry-voxel-size-m 0.03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/resident-calibration.json
```

These explicit settings define the experiment; the resident flag does not
choose a voxel density or silently enable rounding. CPU execution and ordinary
CUDA odometry remain available with their existing defaults.

## Complete production comparison

Three fresh cyclic triads replay six seconds of the public GLIM Livox bag with
four map/fit rounds, all eight odometry passes, the same explicit settings
above, one BLAS thread and a separate initially empty compiler cache per worker.
Startup, first compilation, decoding, covariance, holdout, all 32 signed
controls, saving and import verification enter whole-process wall time.

| Triad | CPU (s) | Ordinary CUDA (s) | Resident CUDA (s) | CPU / resident | CUDA / resident |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 264.455 | 68.247 | 53.222 | 4.969x | 1.282x |
| 2 | 244.027 | 67.348 | 50.918 | 4.793x | 1.323x |
| 3 | 250.123 | 74.704 | 98.514 | 2.539x | 0.758x |
| Median time | **250.123** | **68.247** | **53.222** | **4.700x** | **1.282x** |

The last row divides median worker times. The cumulative CPU comparison
includes CUDA odometry, the existing CUDA spline plane backend, resident
geometry and pose-only CPU placement. The incremental comparison is 1.282x
against ordinary CUDA. Resident execution was faster in two of three GPU
pairs; the third was slower. Host resources were unreserved, and the cause of
the variation was not identified. These observations do not establish a
consistent latency improvement for every run, other data or other devices.

The [complete bound archive](../assets/resident_lidar_geometry/README.md)
retains all nine workers. Every CPU/GPU decision and evidence gate passes;
all 24 CPU/GPU full-fit comparisons preserve status, iterations and dimensions
and meet the original numerical tolerances. Ordinary/resident CUDA results
and all 12 paired full fits match every floating-point bit after only the
declared execution metadata. All 1,188,000 compared placed point rows also
match by digest. The final verdict remains INCONCLUSIVE.

The [CPU-only auditor](../assets/resident_lidar_geometry/auditor-v2.py.txt)
recomputes both ratios and their variability, verifies all
source/import/kernel/input bindings and checks the complete results and fits.
The measured runtime is frozen at
[5fda3a1f](https://github.com/rsasaki0109/Calibrex/commit/5fda3a1f29a33289828798a04cd4236af9d5b4b7);
the CPU and ordinary CUDA baseline is
[ae5a5582](https://github.com/rsasaki0109/Calibrex/commit/ae5a55822485b4ee9daf2b0195aa35bd4c5b7484).

## Resources and fallback behavior

Each calibration owns its stream and content cache. Repeated decoded scans
reuse an immutable device capture keyed by coordinates and per-point times;
scan time and the current gyro model are evaluated for each pass. The retained
scan cache is limited to 512 MiB, and preprocessing, target capture and query
workspaces have explicit 1 GiB guards. These are component budgets, rather than
a bound on the CUDA allocator's total memory pool.

Voxel centroids sum returns in input order and preserve lexicographic voxel
ordering. Cells containing more than 32 returns or coordinates outside the
packed key domain use native preprocessing. The device spatial grid accepts a
query only with a distance certificate; all remaining queries scan every
target in parallel. An unsupported target domain uses the existing CUDA
spatial tree. Ties, ambiguous normal eigenspaces, gates, stopping decisions and
motion-grid boundaries retain native repairs, including a complete native
registration when required. No query is discarded to meet a work limit.

Arbitrary rotation callbacks use ordinary odometry. The bag front end supplies
its own native gyro model for the resident path. No global backend or callback
is replaced; an exception drains the owned stream and restores the caller's
stream.

Artifacts record the requested flag, actual per-pass identities, kernel hashes,
query/repair counters and `options.resident_geometry_execution` cache and
geometry work. Hardware model names are omitted.

## Reproducing comparisons

The existing whole-process tool accepts `--current-resident-odometry`. It
freezes source trees before launching fresh workers and includes startup,
compilation, decoding, all odometry passes, map/fit rounds, covariance, signed
controls, saving and import verification in wall time.

For an ordinary-CUDA/resident-CUDA comparison, use `--comparison cuda-cuda`,
the same `--prior-fit-plane-backend cuda --current-fit-plane-backend cuda`,
all five explicit odometry settings above, and
`--require-exact-resident-results`. This compares every result and full-fit
floating-point bit and all common work counters after checking and removing
only the declared execution metadata. A CPU/cumulative-CUDA comparison uses
`--comparison cpu-cuda --prior-fit-plane-backend numpy --current-fit-plane-backend cuda`;
its physical, evidence, decision and full-fit numerical gates remain active.

This spent public sequence is development and performance data. It supplies
no independent extrinsic ground truth, and acceleration does not change the
calibration verdict or establish a new accuracy claim.
