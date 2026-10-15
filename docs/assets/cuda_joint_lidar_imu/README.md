# Complete public GLIM CPU/CUDA joint calibration

Three alternating fresh-process pairs use the same committed source revision,
45303e2c5470525487720d53f34d7f9ff054f1de, and the original public GLIM Livox recording.
Initial, gyro-deskewed, feedback-check and final-segment odometry use CPU or CUDA;
the same B-spline fitting, map association, covariance and evidence remain on CPU.

| Scope | CPU median | CUDA odometry median | Speedup |
| --- | ---: | ---: | ---: |
| Complete worker | 211.124 s | 108.970 s | 1.937x |
| Four CPU fits, including covariance | 13.796 s | 15.290 s | 0.902x |

| Pair | Method | Whole seconds | Fit seconds |
| --- | --- | ---: | ---: |
| 0 | CPU | 300.679681 | 24.896402 |
| 0 | CUDA odometry | 131.458493 | 14.361856 |
| 1 | CUDA odometry | 108.970129 | 15.289628 |
| 1 | CPU | 211.124359 | 13.796051 |
| 2 | CPU | 203.177950 | 13.613951 |
| 2 | CUDA odometry | 106.148851 | 15.549821 |

## Fixed workload and timing

- First six seconds, four map rounds, default return sampling and 30 fit iterations.
- PointCloud2 `/livox/points`, point-time `t`; IMU `/livox/imu`, acceleration in g.
- Bag SHA-256: `f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Raw data is not redistributed.
- Both methods use 0.03 m odometry voxels, three local-map scans, a 1 nm point
  coordinate grid, and motion grids of 1 micrometre translation and 0.1 microradian
  rotation-vector components. The comparison is against the matching quantized
  CPU reference. All grids default to disabled outside this declared workload.
- Whole time includes imports, reading, first GPU setup/compilation, every
  odometry pass, association, all fits/covariance, noise updates, holdout, signed
  controls, hashes, saves and verified imports. Parent freezing is outside timing.
- Each CUDA worker has an isolated empty CuPy compilation cache. Driver/runtime
  caches may persist. No cold or slow trial is excluded.

## Quality, actual work and limits

All three complete comparisons pass. Maximum differences are
0 deg rotation, 0 m lever arm
and 0 s clock offset. The audit checks all per-round
trajectories, covariance, costs, convergence and complete evidence decisions.
Every odometry pass records actual queries/systems and native repair counts.
Its runtime and kernel identities are bound to the exact archived source strings.
The CPU-only audit checks these bindings without importing CuPy.

All six final results remain **INCONCLUSIVE**. This estimated plane map and its
formal covariance do not establish independent absolute extrinsic accuracy.
This result applies to an explicit dense workload; coarse point clouds can be
slower on CUDA. The B-spline fitter is still on CPU, so its ratio measures timing
variation rather than a GPU fitting implementation. Ratios from separate cohorts
are not multiplied.

The initial uncanonicalized joint pair failed quality and is retained under
[rejected-no-motion-grid](rejected-no-motion-grid/README.md). Its complete results,
measurements, original helper and verified imported source bytes are preserved.
Exploratory wall times are not pooled into the formal cohort.

## Bindings and reproduction

Every formal worker has its complete schema-valid result, all fitted calibration,
trajectory and covariance arrays, original log, input hash and actual import
hashes. Both complete source trees, raw manifests, executed benchmark launcher,
comparison schema and publisher are retained. The publication index binds every
file except itself.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_joint_lidar_imu
python -m tools.benchmark_bspline_lidar_imu --comparison cpu-cuda \
  --max-seconds 6 --max-rounds 4 --repeats 3 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-cuda-joint-comparison
```

Use a new output directory and committed runtime sources. See the
[implementation and CLI](../../benchmarks/cuda_joint_lidar_imu.md).
