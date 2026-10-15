# Further odometry acceleration candidates

Ten candidate strategies were screened after the optional CUDA spline fitter.
All remain experimental: the strongest complete target-construction/registration
pilot is 1.158x, and no whole-bag candidate speedup is established. Production
runtime remains identical to `3b006a4be3babcf2b0e5c537993105b7d1c68b46`.

| Candidate | Observed ratio | Timed scope |
| --- | ---: | --- |
| Stable GPU voxel reductions | 1.053x raw / 0.821x processed | Eight scans of each kind |
| Fixed k=11 spatial queries | 1.014x | Eight target constructions and registrations |
| Unrolled neighbor insertion | 1.045x | Same complete registration scope |
| Leaf-packed reference coordinates | 0.995x | Same complete registration scope |
| GPU radix tree and lazy native repairs | 1.158x | Same complete registration scope |
| Tree-ordered normal queries | 0.994x | Same complete registration scope |
| Morton-ordered source queries | 0.980x | Same complete registration scope |
| CPU-tree-ordered source queries | 0.944x | Same complete registration scope |
| Guarded batches of four GPU updates | 0.969x | Same complete registration scope |
| Guarded batches of eight GPU updates | 1.019x | Same complete registration scope |

The [complete archive](../assets/odometry_candidate_screen/README.md) retains
five separate cohorts, all 744 observations, all source/launcher snapshots and
input identities. Four cohorts use five rotated-order repeats over eight
captured public GLIM registration inputs. Each variant receives one explicit
first-case compilation warmup outside timing. The voxel pilot retains all seven
rounds; its warm medians use rounds one through six, with the 2.935 second first
candidate round retained separately. Module setup is outside that pilot's scope.
These screens do not measure fresh-process startup or complete bag calibration.
Their baselines differ, and their ratios are not pooled or multiplied. Host
CPU/GPU activity is unreserved; one BLAS thread is used.

The voxel launcher asserts identical coordinate bits and point counts, but only
counts and timings are retained for CPU audit. Other cohorts retain every
registration result and work counter. Layout, tree and ordering variants keep
all numeric registration fields equal; batched updates use abs 1e-9 / rel 1e-6
for transforms, RMSE and the Hessian eigenvalue. Decisions, iterations and
correspondence counts match. The batched screen exercises no full-path fallback,
so it does not establish fallback/boundary correctness or production readiness.

The GPU radix pilot illustrates the need to measure the complete operation:
median construction time falls from approximately 0.248 to 0.092 seconds over
eight targets, while subsequent registration gets slower. The complete ratio
is only 1.158x. Query layout, order and device-side update batches also fail to
establish a useful combined improvement on these fixed inputs.

## Current pipeline diagnosis

A separate source-bound cProfile run replays six seconds of the original public
bag, using the adopted CUDA spline fitter and four map rounds. It identifies
work spread across target construction, preprocessing/local maps, registration,
normal preparation and gyro rotations. These scopes overlap. The profile has
instrumentation and fixture-capture overhead, and is not compared with ordinary
whole-worker benchmark times. The clip provides no independent extrinsic truth.

The next substantial improvement needs a pipeline design that reduces combined
preparation and registration cost. Existing complete-bag physical/evidence
checks remain the acceptance gate; small component ratios do not establish
that benefit.

## Incremental local-map experiments

A subsequent screen reuses world-coordinate preparation across consecutive
local maps and tries deterministic GPU hash aggregation. The strongest warm
local-map preparation pilot improves by 2.737x, while complete fresh-worker
calibration improves only 1.010x. No candidate is adopted.

| Candidate | Observed ratio | Scope |
| --- | ---: | --- |
| CPU retained coordinates and voxel keys | 1.483x | Local-map preparation |
| CPU retained unique groups and remapping | 0.776x | Local-map preparation |
| Deterministic GPU voxel hash | 2.266x | Warm local-map preparation |
| GPU hash with retained scan cache | 2.737x | Warm local-map preparation |
| GPU hash/cache, complete calibration | 1.010x | Fresh process, all startup and output work |

The [incremental-map archive](../assets/incremental_local_map_screen/README.md)
retains 2,478 micro observations over 59 consecutive public-data map windows,
four complete worker results and fits, source capsules, launchers and input
identities. The micro launchers assert coordinate bit equality and retain each
output hash and point count. Coordinate arrays are identified but not included,
so the CPU audit checks their saved identities rather than recomputing geometry.
GPU micro medians exclude round zero; module setup is outside that timed scope.

Two alternating fresh-process pairs include separate empty CuPy caches and all
candidate compilation. Whole times are 74.201 versus 76.410 seconds in the first
pair, and 75.067 versus 71.348 seconds in the second. The median is 74.634 versus
73.879 seconds. Complete calibration/evidence and fit fields match exactly,
allowing only run metadata and fit duration. Each candidate worker executes 448
map calls over 25,427,273 input rows with zero native repairs. One BLAS thread is
used and host CPU/GPU activity is unreserved. Neither a robust whole-worker gain
nor unexercised boundary/fallback correctness is established by this screen.

```bash
python -m tools.audit_incremental_local_map_screen docs/assets/incremental_local_map_screen
```

```bash
python -m tools.audit_odometry_candidate_screen docs/assets/odometry_candidate_screen
```

The CPU-only audit checks source capsules, launchers, input/protocol bindings,
all observations, registration fields/work, timing medians and declared scopes.

## Traversal, scheduling and point placement

Eight further strategies retain [492 observations in four separate cohorts](../assets/odometry_scheduling_screen/README.md).
None is adopted, and these component screens do not establish a whole-worker
speedup. The runtime remains unchanged.

| Candidate | Ratio | Timed scope |
| --- | ---: | --- |
| Parent-linked stackless traversal | 1.000x | Target construction and registration |
| Heap-path stackless traversal | 0.986x | Target construction and registration |
| Demand-driven target normals | 0.854x | Target construction and registration |
| Initial-query normal prefetch | 0.827x | Target construction and registration |
| Transposed bounds reductions | 0.909x | Bounds calculation and validation |
| Separate coordinate bounds reductions | 0.973x | Bounds calculation and validation |
| Pose-only point placement | 1.090x | Point placement |
| Pose-only placement with control-edge reuse | 1.423x | Point placement |

Stackless traversal reduces recorded kernel local storage from 288 to 24 bytes
per thread in fused registration, yet the complete operation does not improve.
This is a compiled local-frame attribute, not total GPU-memory usage. Lazy
normals prepare 43.8% fewer rows, but additional queries, compaction and system
refreshes outweigh that saving. Both are examples of lower resource/work counts
failing to establish a useful speedup.

Registration outputs match exactly, and traversal retains the original work
counts. Bounds/placement launchers assert output bit equality and save hashes
and dimensions; coordinate payloads are not redistributed. The archive binds
all protocols, launchers, generated traversal kernels and input identities to
the shared source capsule. Its CPU-only auditor checks saved records without
executing experimental code or importing optional runtimes. One BLAS thread is
used on an unreserved host. No cold-start or whole-calibration claim is made.

```bash
python -m tools.audit_odometry_scheduling_screen docs/assets/odometry_scheduling_screen
```
Frozen diagnostic scripts have repository-relative input/output paths and are
archived as text. They introduce no production API or optional dependency.
