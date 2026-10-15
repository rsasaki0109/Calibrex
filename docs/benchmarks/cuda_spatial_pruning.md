# CUDA spatial pruning and bulk tree capture

The optional CUDA map, point-to-point ICP and scan-odometry adapters share
the same captured spatial tree. Queries now use the tight three-dimensional
box of each subtree to skip branches that cannot contain a closer neighbor.
Candidate distance arithmetic, FP64, disabled FMA, distance/index tie order,
normal estimation and all native ambiguity repairs are retained. Equal and
roundoff-close box bounds are searched. This is exact search, without a
reduced neighbor budget or approximate-neighbor option.

Tree capture also reads SciPy's serialized node metadata in bulk on supported
64-bit layouts. Native pointers are ignored. Node dimensions, counts, spans,
child topology, depth, full leaf coverage and split geometry are validated
against the captured point coordinates before any GPU query. Unknown or
inconsistent layouts use the public node traversal. The implementation follows
[SciPy's node declaration](https://github.com/scipy/scipy/blob/v1.17.1/scipy/spatial/ckdtree/src/ckdtree_decl.h);
the compatibility fallback is exercised by corrupt-layout and geometry tests.
No ROS or compiled extension dependency is introduced.

## Complete public-data comparison

| Scope | CPU median | CUDA median | Speedup |
| --- | ---: | ---: | ---: |
| Complete worker | 293.774 s | 118.996 s | 2.469x |
| Four CPU fits, including covariance | 19.980 s | 20.945 s | 0.954x |

All three comparisons pass: maximum rotation difference 0 deg, lever-arm difference 0 m and clock difference 0 s. Per-round covariance, trajectories, costs and all evidence decisions also pass the existing unchanged audit tolerances.

Three alternating fresh-process pairs replay the first six seconds of the
original public GLIM Livox bag, including four map rounds, all initial/gyro
and feedback odometry, fits, covariance, noise updates, holdout, signed controls,
saved results and verified imports. Both variants use the same committed source,
one BLAS thread and the explicit dense numerical policy from the
[earlier CUDA joint comparison](cuda_joint_lidar_imu.md): 0.03 m voxels,
three local-map scans, a 1 nm coordinate grid and returned-motion grids of
1 micrometre translation and 0.1 microradian rotation-vector components.
Each GPU worker starts with an isolated empty CuPy compilation cache;
driver/runtime caches may persist. CPU remains the default and all grids
remain disabled by default.
Host background load is not reserved. These alternating-pair timings are
observations on this host, rather than a controlled cross-hardware estimate.

The [bound archive](../assets/cuda_spatial_pruning/README.md) includes all
complete worker results, per-round covariance and trajectories, original logs,
executed import digests, both exact source trees, launcher and schemas. Its
CPU-only audit checks complete quality, actual GPU work and source/kernel
identities. The CUDA workers also report whether each tree used bulk capture
or the compatibility fallback.

All final results remain **INCONCLUSIVE**. No independent extrinsic truth is
supplied by this clip. The B-spline fitter and evidence evaluation stay on CPU;
their timing variation is not a GPU fitting speedup. The reported whole-worker
ratio belongs to this declared dense workload. Coarse scans and short workloads
can still favor CPU. Ratios from earlier cohorts are not multiplied.

## Reproduce

Use the existing `calibrex[cuda]` extra and the explicit dense CLI settings
from the earlier comparison. The shared CUDA adapters use pruning automatically.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cpu-cuda \
  --max-seconds 6 --max-rounds 4 --repeats 3 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-cuda-spatial-comparison
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_spatial_pruning
```

To compare two committed CUDA implementations directly, use
`--comparison cuda-cuda --baseline-commit <previous-revision>` with the same
explicit settings. Both workers then run CUDA; neither is labeled as a CPU
reference. Use a new output directory and committed runtime sources.

## Direct previous-CUDA check

A separate, source-bound previous/current CUDA pair ran the same complete
six-second workload and numerical policy. Previous CUDA took 154.593 s;
current CUDA took 132.839 s (1.164x).
Both fresh workers started with isolated empty CuPy compilation caches.
All result and full per-round quality checks pass. This single pair is a
preliminary incremental observation, without repeated-trial stability claims;
it is not pooled with the three CPU/CUDA pairs or multiplied by their ratio.
Host background load was not reserved.

[Bound direct comparison](../assets/cuda_spatial_pruning/previous_cuda/README.md).
