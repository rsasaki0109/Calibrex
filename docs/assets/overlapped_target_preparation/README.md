# Complete comparison of overlapped CUDA target preparation

**Not adopted.** Runtime sources are restored to the previous revision. The
candidate and its 51 passing device/unit tests are retained for a controlled
repeat, rather than enabled as a demonstrated whole-calibration improvement.

| Scope | Sequential preparation | Overlapped preparation | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 97.706 s | 86.766 s | 1.126x |
| Unchanged four CPU fits, including covariance | 16.384 s | 18.928 s | 0.866x |
| Unchanged CUDA normal preparation | 7.259 s | 5.895 s | 1.231x |

| Pair | Sequential complete worker | Overlapped complete worker | Observed ratio |
| --- | ---: | ---: | ---: |
| 1 (sequential first) | 72.916 s | 76.616 s | 0.952x |
| 2 (overlap first) | 122.497 s | 96.917 s | 1.264x |

The paired comparisons change direction: one is slower and the other faster.
Unchanged CPU fit and GPU normal times also vary substantially. The median
ratio is descriptive; these runs establish no stable end-to-end speed gain.
Adoption is deferred pending a repeat under less host contention.

Two alternating fresh-process pairs replay the public GLIM Livox bag's first
six seconds with four map rounds. This is a descriptive observation on an
unreserved host. One worker prepares the local map, captured native tree and
CUDA target while the caller deskews and downsamples the source. Registration
waits for the same prepared target, with all original kernels and arithmetic.
One preparation is outstanding at most, and result snapshots close the pool.
The worker uses the caller's device and a reusable private stream, synchronized
before publishing the target. Rotation callbacks remain on the caller thread.

The previous revision is `88e4a3a1db0e67dc538e3faf73c9a12b570ab3de` and the current revision
is `7fe26771f6fb2a8017be54b9053b112fcaf2efc0`. The input bag SHA-256 is
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. No raw scans or compiler caches are redistributed.

Each worker uses one BLAS thread and a separate initially empty CuPy compilation
cache. Whole timing includes startup/imports, decoding, every odometry pass,
target/source work and worker shutdown, association, fits/covariance, noise
updates, evidence, hashing, saves and source verification. It includes first
GPU initialization and compilation. The unchanged dense policy uses 0.03 m
voxels, three local-map scans, an opt-in 1 nm coordinate grid and motion grids
of 1 micrometre translation and 0.1 microradian rotation-vector components.
Those grids remain disabled by default. CPU odometry remains the default.

All complete physical/evidence fields, initial estimates, execution identities
and GPU work counters match exactly, excluding only provenance timestamps,
commands and source revisions. Every full fit record matches exactly excluding
wall time, including covariance, trajectory, calibration, costs, dimensions,
status and iterations. Both pairs satisfy the stronger exact-output protocol.
The final real-data verdict remains **INCONCLUSIVE**; there is no independent
extrinsic truth. Return budgets, neighborhoods, convergence and controls are
unchanged. No public option, dependency or result-schema change is introduced.

Unchanged fit and normal timings expose host variation, and the complete-worker
ratio is specific to these runs. Ratios from distinct scopes or cohorts are
not multiplied. Overlap can increase simultaneous CPU working memory; the
number of preparations is bounded, rather than process RSS.

The index binds all four full results/measurements/logs, both source capsules,
launcher, schema, publisher and an existing Windows test-failure reproduction.
`prototype-validation` additionally binds the candidate's successful 51-test
device/unit run, exact test files, validation launcher, source manifest and log.
The archive audit checks these bindings without importing an accelerator.
`test_default_cache_dir_respects_xdg` expects `HOME` to redirect `Path.home()`;
it fails with both current and frozen baseline sources locally. The baseline
receipt binds that source manifest, validation log and reproduction launcher.
The test and production cache-path policy are unchanged.

Audit without a GPU or bag:

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/overlapped_target_preparation
```

See [implementation and reproduction](../../benchmarks/overlapped_target_preparation.md).
