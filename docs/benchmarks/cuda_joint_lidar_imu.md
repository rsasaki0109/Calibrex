# CUDA odometry in joint LiDAR-IMU calibration

`calibrex imu-lidar joint` can run initial, gyro-deskewed and feedback-check
odometry on the optional CUDA backend. The B-spline fitter, map association,
covariance and evidence evaluation run on CPU. CPU remains the default.
Each pass records its requested backend, actual code/runtime identity,
GPU queries and systems, normal rows, and native repair counters.

The GPU adapter batches normal estimation for dense overlap. Sparse sources
request only newly hit normals and reuse captured coordinates and matches
when refreshing their robust system. Explicit reads of the complete normal
or planarity arrays still materialize all rows. Close spatial, eigenspace,
stopping and motion-grid decisions retain native repairs.

## Temporal numerical policy

Repeated deskew and calibration couple consecutive scan poses. On this
public clip, registration differences below 1e-9 still changed the angular
initial estimate and subsequent map associations. The first uncanonicalized
joint comparison failed the fixed result tolerances and is retained as a
rejected exploratory attempt.

The successful comparison uses the same explicit policy for CPU and CUDA:
0.03 m odometry voxels, a three-scan local map, a 1 nm coordinate grid,
and returned-motion grids of 1 micrometre translation and 0.1 microradian
rotation-vector components. Rotations are reconstructed as proper SO(3)
matrices. A CUDA result close to a grid boundary is registered with the
native method. All grids default to zero (disabled); selecting CUDA does
not implicitly change density or the numerical policy.

This is a comparison against a CPU reference with the same declared grids.
It is not a comparison against an unquantized trajectory and does not
assert sensor accuracy at those resolutions.

## Complete public-data comparison

The original public GLIM Livox `livox.bag` provides PointCloud2
`/livox/points` with point-time `t`, and `/livox/imu` with acceleration in g.
The first six seconds feed four map rounds with the default return sampling,
30 fit iterations per round, gyro feedback passes, noise updates, holdout and
signed controls. Supplied GT is not used.

Three alternating pairs use fresh processes, one BLAS thread and the same
committed source revision. Each GPU worker starts with an isolated empty
CuPy compilation cache; driver/runtime caches may persist. Whole time includes
imports, decoding, GPU setup and compilation, all odometry, association,
fits and covariance, evidence, hashing, saves and verified imports.

| Scope | CPU median | CUDA odometry median | Speedup |
| --- | ---: | ---: | ---: |
| Complete worker | 211.124 s | 108.970 s | 1.937x |
| Four CPU fits, including covariance | 13.796 s | 15.290 s | 0.902x |

The [complete archive](../assets/cuda_joint_lidar_imu/README.md) binds all
worker results, full per-round trajectories and covariance, executed module
digests, exact sources, benchmark launcher and schemas. The audit also
verifies GPU identity hashes against the archived CUDA kernel strings and
requires actual GPU work in every odometry pass.

All final results remain **INCONCLUSIVE**. The estimated plane map and its
formal covariance do not provide independent extrinsic truth. Speed depends
strongly on density and cold startup. A coarse 40-scan exploratory run took
about 1.9 s on CPU and 22 s on CUDA including compilation; CUDA is opt-in.
These exploratory timings are not pooled with the formal comparison.

## Use and reproduce

Install the optional `calibrex[cuda]` extra for a compatible CUDA runtime.
The explicit dense policy used here is:

```bash
calibrex imu-lidar joint data/public/glim_versatile/livox.bag \
  --lidar-topic /livox/points --imu-topic /livox/imu --acceleration-unit g \
  --dataset-family glim_versatile --dataset-license "upstream AIST GLIM dataset terms" \
  --max-seconds 6 --max-rounds 4 --odometry-backend cuda \
  --odometry-voxel-size-m 0.03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-joint-cuda.json
```

Choose `--odometry-backend cpu` for the matching CPU reference. The Python
API exposes the same fields in `WindowingOptions` and `ScanOdometryOptions`.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cpu-cuda \
  --max-seconds 6 --max-rounds 4 --repeats 3 \
  --odometry-voxel-size-m 0.03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-cuda-joint-comparison
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_joint_lidar_imu
```

Use a new output directory and committed runtime sources. The archive audit
needs ordinary CPU dependencies, neither a bag nor a GPU.
