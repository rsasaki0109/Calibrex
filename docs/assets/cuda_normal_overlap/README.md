# Asynchronous CUDA normal preparation

| Scope | Synchronous preparation | Asynchronous preparation | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 57.791 s | 51.622 s | 1.120x |

| Repeat | Variant | Complete seconds |
| --- | --- | ---: |
| 0 | prior | 57.812417 |
| 0 | current | 50.961965 |
| 1 | current | 52.282880 |
| 1 | prior | 57.770263 |

Two alternating fresh-process pairs replay six seconds of the original public
GLIM Livox bag, four map rounds, one BLAS thread, CUDA odometry and CUDA fitter
planes. Every worker has its own initially empty CuPy compiler cache. Whole time
includes initialization/compilation, all initial/gyro/feedback/final odometry,
association, fits/covariance, noise updates, holdout, controls, hashing, saves and
frozen-import verification. No warmup or slow worker is discarded.

Baseline `105e1031206ba6c551d08a1b07742b0307e5f523`; candidate
`1f1b7c46a49cdbadbcfa42761dd9676598968d59`; input bag SHA-256
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. The unchanged dense policy uses 3 cm voxels, three
local scans, a 1 nm coordinate grid and returned-motion grids of 1 micrometre /
0.1 microradian. CPU remains the default and all grids remain disabled by default.

One bounded dense normal batch is submitted before CPU source deskew/preprocess.
The target retains GPU temporaries, pinned download buffer and completion event
until the unchanged native repairs finish. There is no extra CPU thread. Larger
targets and sparse source hints retain the existing demand-driven preparation.
Tests cover cross-stream consumption, snapshots, sparse hints, the chunk boundary,
abandoned targets, failed native repairs and retries, and empty sources.

The exact gate permits only run metadata and the declared schedule suffix /
`gpu_async_normal_batches` counter to differ. Every calibration/evidence field,
old work counter and complete fit field including trajectory and covariance
matches exactly. The original physical/evidence checks also pass. This checks
preservation of previous outputs; the clip has no independent extrinsic truth.
The final policy verdict remains INCONCLUSIVE. All 130 public schema/ledger
exports are byte-identical between the two frozen runtime trees.

Host load is unreserved; two pairs do not establish a universal speed ratio.
Ratios are not multiplied across scopes or earlier cohorts. The retained
normal-preparation timer records host time in submit/finish calls, including
native repairs, excludes CPU work between calls and does not measure overlapped
GPU work. Its ratio is not a normal-kernel arithmetic speedup.

All four full results/fits/logs, deterministic source capsules/manifests,
launcher/schema, publisher, schema checker/receipt and source-bound correctness
run are indexed by SHA-256. The audit runs without CUDA. Raw bag data and compiler
caches are not redistributed.
