# CUDA spline plane fit comparison

| Scope | NumPy planes | CUDA planes | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 65.884 s | 61.146 s | 1.077x |
| Four fits, including covariance | 10.239 s | 3.506 s | 2.921x |

Two alternating fresh-process pairs replay the original public GLIM Livox bag's
first six seconds, four map rounds, one BLAS thread and unchanged CUDA scan
odometry. Every worker starts with a separate empty CuPy compiler cache.
Initialization, compilation, all odometry passes, association, noise updates,
all fits/covariance, holdout/controls, input hashing, saves and frozen-import
verification remain inside whole time. Fit time includes evaluator construction,
first plane kernel compilation and covariance. No warmup is discarded.

Baseline `c3845d1eb42326e6da11a235b5e6239c33448160`, measured candidate
`f4c8e5662a43f5e5ce24bb4181924c6ca5b3bb68`. Bag SHA-256:
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. The dense odometry policy is 0.03 m voxels, three
local-map scans, 1 nm coordinate quantization, 1 micrometre translation and
0.1 microradian rotation-component quantization. These grids remain off by
default. Plane/IMU data, return budgets, convergence checks and evidence gates
are unchanged. NumPy is still the default fitter plane backend.

The GPU computes FP64 spline plane residuals, analytic local derivatives and
the seven-block control normal band. Each band entry has one owner and a fixed
reduction order; there are no floating-point atomic reductions. The core keeps
IMU residuals/derivatives, clock finite differences, LM acceptance/stopping,
the solve, marginal covariance and all evidence evaluation on CPU.

The bound audit checks every result, all full per-round fits and real kernel
identity/work against the frozen source and reported plane counts. Fit status,
iterations and dimensions must match exactly. Trajectories/calibration/costs
use abs 1e-9 / rel 1e-6, covariance abs 1e-11 / rel 1e-5, alongside the existing
whole-result physical/evidence tolerances. This GPU path permits roundoff;
it does not claim bit equality. The final verdict stays INCONCLUSIVE; the clip
has no independent extrinsic truth. Host load is unreserved. Ratios describe
this workload and these pairs, and are not multiplied across stages/cohorts.

All four full results, measurements/logs, deterministic source capsules,
manifests, launcher/schema, publisher and source-bound correctness run are
retained. Raw bag data and compiler caches are not redistributed.
