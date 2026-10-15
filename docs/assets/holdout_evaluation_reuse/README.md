# Complete fixed-trajectory holdout evaluation comparison

| Scope | Previous | Reused holdout evaluations | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 82.051 s | 83.963 s | 0.977x |
| Signed controls after holdout RMS | 2.073258 s | 0.013284 s | 156.076x |
| Unchanged four CPU fits, including covariance | 12.768 s | 13.567 s | 0.941x |

The complete worker is **2.3% slower** in these observations, despite removing
about 2.06 seconds from signed-control evaluation. Unchanged fit and GPU normal
preparation timings also vary; these two pairs do not isolate the cause of the
whole-worker difference. This implementation is retained for reduced CPU
evidence work. These runs establish no full-calibration speed gain and are not
counted as successful end-to-end acceleration.

Two alternating fresh-process pairs replay the public GLIM Livox bag's first
six seconds with four map rounds. This is a descriptive observation on an
unreserved host. Both variants execute the same adopted CUDA scan-odometry
kernels and CPU fitter. The new work removes repeated CPU evidence evaluation.
Stage and whole ratios have different scopes and are not multiplied.

The previous revision is `e59600c1354dc14f0b5afd06f92d90bf61d56f41` and the current revision
is `1e2940ab833299cb1bfa9bf7288c0ce998e234ab`. The input bag SHA-256 is
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Raw scans and compiler caches are not redistributed.

Each worker uses one BLAS thread and a separate initially empty CuPy cache.
First GPU initialization and compilation belong to whole time. The unchanged
dense policy uses 0.03 m voxels, three local-map scans, an opt-in 1 nm coordinate
grid and returned-motion grids of 1 micrometre translation and 0.1 microradian
rotation-vector components. Grids remain disabled by default.

The signed-control scope starts after holdout RMS, so the current implementation
can reuse its prepared states. It includes both baseline chi-square calls and
all 32 signed scores: 34 score calls in one control evaluation for every worker.
The control timer does not represent cold initialization. Whole time includes
startup/imports, decoding, every odometry pass, association, fits/covariance,
noise updates, all train/holdout preparation, controls, hashes, saves and source
verification. Unchanged fit timing is reported to expose host variation.

All complete physical/evidence fields, stored initial estimates and execution
reports are exactly equal, excluding only provenance timestamps, commands and
source revisions. Every full fit record matches exactly after excluding wall
time, including calibration, trajectory, covariance, costs, dimensions, status
and iterations. All final verdicts remain **INCONCLUSIVE**; the clip supplies
no independent extrinsic truth. Samples, control perturbations and thresholds
are unchanged. Public APIs and result schemas are unchanged.

The index binds all four complete results, original measurements/logs, source
capsules/manifests and frozen launcher/schema/publisher. The CPU-only audit
recomputes quality/equality, timing scopes, score counts and median ratios.

Local validation also found an existing screw-linear synthetic recovery test
that reports `max_iterations` rather than `converged`. It fails with both the
current implementation and frozen baseline sources, including with one BLAS
thread. This is a separate path from the B-spline bag pipeline. The bound
`baseline-validation.json`, log and launcher retain the baseline reproduction
and source-manifest identity. No test, tolerance or iteration limit was relaxed.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/holdout_evaluation_reuse
```

See [implementation and reproduction](../../benchmarks/holdout_evaluation_reuse.md).
