# Public KITTI signed-axis and CPU process evidence

Five schema-valid benchmark artifacts record a fixed six-case CPU/CUDA sweep
and a separate CPU process execution check on public KITTI raw drive
`2011_09_30_drive_0018_sync/image_02`. All runs use the same 25 original frames,
3,130,279 Velodyne points, 20-fit/5-holdout split, and frozen FeatDepth provider:
`7e3ef953a360a7c01ed0cdc3cf947db97fb0aff5912ced805d80bebe210d3360`.
Hardware is the i7-9750H / GTX 1660 Ti 6 GiB Windows environment used in the
[earlier demonstration](../public_six_dof_demo/README.md): Python 3.12.10,
NumPy 2.3.4, CuPy 14.2.0, driver 596.36, local CUDA 12.8 toolkit.

## Six fixed initializations

`sweep.plan.definition.json` retains the plan written before optimization.
Its trial entries explicitly say `not executed; frozen plan entry`; it is not
completed execution evidence. `sweep.definition.json` / `sweep.benchmark.json`
record all twelve subsequent invocations. The cases add local Euler rotation
and camera-frame translation of ±0.5 degrees / ±0.5 m on x, y, or z to the
dataset reference, in the declared x+/x-/y+/y-/z+/z- order. Search settings
(including the 800-evaluation budget) and frame split are fixed throughout.
Each invocation verifies/loads inputs and prepares a fresh projector.
Histogram ranges are resolved on fit frames under that case's declared initial
pose and remain fixed during its search; holdout is excluded from this step.

All twelve invocations converged. Within every case, CPU/CUDA full traces,
final poses, metrics, and initial/final holdout records match exactly. Cases
can have different traces and output poses and are not compared as replays.

| Case | Evaluations per backend | CPU solve | CUDA solve | Final rotation difference | Final translation difference | Final holdout MI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x-positive | 229 | 90.446 s | 3.085 s | 0.390923° | 0.160078 m | 1.293495 nat |
| x-negative | 313 | 66.425 s | 8.450 s | 0.410874° | 0.104583 m | 1.287191 nat |
| y-positive | 277 | 83.026 s | 3.446 s | 0.527379° | 0.167705 m | 1.294101 nat |
| y-negative | 241 | 52.591 s | 12.053 s | 0.266233° | 0.091001 m | 1.281463 nat |
| z-positive | 265 | 90.288 s | 4.861 s | 0.258640° | 0.087052 m | 1.283678 nat |
| z-negative | 241 | 54.330 s | 4.297 s | 0.285129° | 0.089486 m | 1.281675 nat |

Initial reference differences are 0.5 degrees / 0.5 m for every case. The
y-positive final rotation difference increases to 0.527379 degrees; this
observation remains in the population. Optimizer termination and exact backend
parity do not establish calibration accuracy. Dataset calibration is reference
evidence, FeatDepth training overlap is possible, and covariance/observability
remain unknown with standard-result quality `warn`.

Measured elapsed time through all case reports is 568.615 s. First invocations
are included, with no separate warm-up; process-wide kernel caches persist
between cases. Timings vary substantially, and cross-case summaries mix
different search workloads. No cross-case speed ratio, accuracy acceptance
threshold, statistical confidence, or portable speed guarantee is inferred.

## CPU process execution check

`cpu-process-smoke.definition.json` / `cpu-process-smoke.benchmark.json`
compare the same six-case NumPy/NumPy-replay sweep at a fixed 13-evaluation
budget, once with `--workers 1` and once with `--workers 2`. All twelve standard
numerical result payloads match exactly between serial and parallel execution,
including the full search trace, final poses, metrics, and holdout records.
Every trace also matches the first thirteen evaluations of its corresponding
longer CPU/CUDA run above.

| Execution | Elapsed through all case reports |
| --- | ---: |
| Serial | 139.842971 s |
| Two spawned CPU worker processes | 74.843543 s |

The observed ratio is 1.87x for this one execution pair. Elapsed includes shared
verification, worker startup, input verification/loading, solve, holdout,
serialization, and individual HTML reports; suite summary writing is excluded.
There is no separate warm-up or dedicated isolation from background load.
Parallel runs include CPU/memory contention and each process owns its arrays
and projectors. This does not establish scaling for longer converged searches
or additional worker counts.

All calibration attempts stop at `max_evaluations=13` and remain partial.
The two smoke benchmark `success` statuses mean completed execution and exact
serial/parallel result parity, with `partial_calibration_trials=12` explicitly
reported for each method. They do not label calibration convergence. The
original child/suite comparisons preserve those twelve partial attempts as
failed trials, with stopping reasons and all available results.

## Provenance and reproduction

Raw images, scans, depth arrays, weights, standard results, and generated HTML
remain external. Original per-case result hashes and child comparison hashes
are preserved in the suite artifacts. CPU smoke output hashes bind the original
external suite benchmark files, and its source-artifact map binds the original
CLI summaries and local manual aggregation script. That aggregation verifies
every numerical payload and capped trace prefix before publishing these files.

Published paths are normalized to `${DATASET_ROOT}` / `${WORKSPACE_ROOT}`.
Numerical values, timings, configuration hashes, and external input/output
hashes are unchanged. The published suite binds the normalized published plan
bytes and also preserves the original local plan hash. Trial notes retain the
original plan hash used during execution. Source problems, provider, child
comparisons, and original results refer to external files rather than new
normalized copies.

The full CPU/CUDA measurement used the development worktree based on `46516ec`;
its generator source hashes identify code retained in commit `84390d2`. The CPU
process measurements used `84390d2` with the subsequent mobile text-wrapping
fix. The GPU overview was regenerated with that presentation fix, retaining its
original HTML and all original JSON locally. Desktop/mobile overview rendering,
expanded limitations, all six case reports, and all 54 images/SVGs were checked.

Follow the [sweep commands](../../tutorials/public_six_dof_demo.md#compare-six-initializations-and-parallelize-cpu-cases).
For the CPU execution check, use `--candidate-backend numpy --max-evaluations 13`
and fresh output directories, first `--workers 1`, then `--workers 2`.
Exit `2` is expected for these partial comparisons; keep every attempt and
compare their numerical payloads before interpreting elapsed times.
