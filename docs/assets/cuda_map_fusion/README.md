# Fused FP64 CUDA map evaluation

Three alternating previous/current fresh-process pairs with separate initially
empty CuPy caches retain exact complete evaluation results and reduce first-use
time. This measures the public KITTI copied-sensor execution control, not
independent two-sensor accuracy or reference-odometry generation.

| Scope | Previous median | Fused median | Observed speedup |
| --- | ---: | ---: | ---: |
| Complete fresh worker | 17.288 s | 10.043 s | 1.721x |
| Complete calibration/evidence API | 12.450 s | 5.265 s | 2.365x |
| Nested NVRTC compilation | 9.319 s | 2.459 s | 3.789x |
| Compiled programs per worker | 63 | 17 | — |

The API includes target preparation, all fits, covariance, holdout, signed
controls and jackknife. Worker time additionally includes process imports,
raw input reads, hashes, saved artifacts and frozen import verification.
Compilation is a nested scope; component ratios must not be multiplied. These
first-use results are not a claim about the speedup with already warm caches.

The new single CUDA module captures tree compaction, uncertainty flags, native
repair scatters, source-ordered residual packing, Jacobians and robust products.
Existing exact spatial traversal, FP64 distances, covariance/eigensolves, BLAS
reductions and native numerical repairs keep their semantics. There is no
neighbor-count specialization, no gyro device path and no implicit geometry or
motion grid. Sampling, neighbor counts, iterations and evaluation budgets are
unchanged. CPU/default execution remains ROS-independent and imports no CuPy.

All three full evaluation outputs match exactly, including transforms,
covariance, signed controls, holdout, jackknife and decisions. Each worker keeps
six samples, 762,118 map rows and 9,147,116 query rows. The candidate actually
executes 276 fused map kernels over 8,384,998 residual input rows and 3,838,813
normal-system rows. Device construction and final arrays retain the same point
order. Original GPU snapshots and the fused path retain byte-identical residuals,
Hessians and gradients in independent real-device tests.

Input: KITTI raw sequence `2011_09_30_drive_0018_sync`, scans 115, 230 and 345,
copied central scans, fixed 0.01 m target noise and generated rigid poses. Raw
scans are not redistributed. This is an identity execution control, not a
physical two-sensor ground-truth calibration.

No foreign or unknown CUDA compute process appeared in 11
ten-second observations. Sampling does not guarantee continuous GPU exclusivity,
detect all graphics activity or reserve CPU resources. Workers use one
BLAS/OMP/MKL thread. Only the owned cohort would be stopped on interference.
No process IDs, names or hardware models are published.

Baseline: `1f45f4f1b034b2430e7b22c4036912e7c68936d1`. Measured candidate:
`211f11225f4e24d2b9d3da08ec9891e97a7a47fc`. The publication index binds all six workers,
both source trees, launchers, schemas, tests, observer and publisher. Both spatial
and fused arithmetic shader bytes participate in the runtime identity.

```bash
python -m tools.audit_cuda_map_pipeline docs/assets/cuda_map_fusion
```

The [warm check](warm/summary.json) excludes imports and reports 1.514 s → 1.430 s (1.059x) across three calls after each variant's own prime, with zero new compilations and exact complete results. Its fixed order and unreserved resources make it descriptive. The main benefit is first-use compilation reduction.

The [GLIM shared-path validation](joint-validation/summary.json) retains four complete identical fitted trajectories/covariances/costs, holdout and 32 signed controls. It is a quality check only; timings from the earlier baseline are not compared.
