# Fused CUDA normal preparation

The CUDA scan-odometry adapter prepares complete dense targets without
deduplicating every row on the GPU. Once all normals are ready, subsequent
deskew rounds reuse the completion state. Sparse sources still compute only
newly hit normals; completing a partially prepared target preserves cached rows.

A dedicated FP64 kernel computes each neighborhood's covariance, its lowest
symmetric eigenspace, planarity and repair flags together. It scales the 3×3
matrix, solves its characteristic roots and chooses the largest row-cross-product
for the normal. This is an independent implementation, without a new library
dependency. [Eigen's direct-solver documentation](https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1SelfAdjointEigenSolver.html)
notes the accuracy tradeoff of direct roots; no Eigen code
is copied into the core.

Native PCA repairs retain spatial-boundary membership, small eigengaps and
classification boundaries. An eigenvector-residual guard catches inaccurate
directions. The classification guard is more conservative near repeated upper
roots. The existing nearest-neighbor policy, FP64 distance arithmetic, disabled
FMA and coordinate/returned-motion settings are retained. No return or neighbor
budget is reduced. Actual kernel calls, normal rows and native repairs are saved.

## Complete previous/current CUDA comparison

| Scope | Previous CUDA median | Fused-normal CUDA median | Speedup |
| --- | ---: | ---: | ---: |
| Complete worker | 116.955 s | 90.017 s | 1.299x |
| Normal preparation within the worker | 32.733 s | 8.693 s | 3.766x |
| Four unchanged CPU fits, including covariance | 20.671 s | 17.056 s | 1.212x |

All three complete comparisons have zero rotation, lever-arm, clock, holdout and decision difference. The existing audit also checks every per-round covariance, trajectory, cost and convergence result.

Three alternating fresh previous/current CUDA pairs replay the first six seconds
of the original public GLIM Livox bag with four map rounds and complete initial,
gyro, feedback and final-segment odometry. The previous source is the
[spatial-pruning implementation](cuda_spatial_pruning.md). Both workers use one
BLAS thread and an isolated initially empty CuPy compilation cache; runtime and
driver caches may persist. Host background load is not reserved.

Both methods use the same explicit dense policy: 0.03 m voxels, three local-map
scans, a 1 nm coordinate grid and returned-motion grids of 1 micrometre translation
and 0.1 microradian rotation-vector components. CPU remains the default and
all grids remain disabled by default. The B-spline fitter, map association,
covariance and evidence evaluation remain on CPU.

Whole time includes imports, decoding, first GPU setup and compilation, every
odometry pass, association, fits/covariance, noise updates, holdout, signed
controls, hashes, saves and verified imports. Normal-preparation time records
outermost preparation calls, including repair and cache-check work, without
adding GPU synchronization. It is a part of whole time; the two ratios are
not multiplied. First compilation belongs to both observed scopes.

The [bound archive](../assets/cuda_normal_preparation/README.md) retains every
complete result, full per-round covariance and trajectory, original log, exact
source archive, import/kernel identity, launcher and schema. A CPU-only audit
checks every comparison and the actual work. All final real-data verdicts remain
**INCONCLUSIVE**: this clip supplies no independent extrinsic truth. The speed
observation belongs to this explicit dense workload, not default coarse scans.

## Reproduce

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cuda-cuda \
  --baseline-commit 3f070af51921f60451c28b192ea815cc3d466025 \
  --max-seconds 6 --max-rounds 4 --repeats 3 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-cuda-normal-comparison
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_normal_preparation
```

Use a new output directory and committed runtime sources. The existing dense
`calibrex imu-lidar joint --odometry-backend cuda` CLI settings apply without
an additional switch. Correctness tests include rotated/scaled scenes,
repeated-root classification boundaries, thin/tied neighborhoods, partial-cache
completion and a target crossing the 65,536-row chunk boundary.
