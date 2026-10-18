# LiDAR preparation and registration follow-up

The best component candidate reached **1.214x**, but one complete fresh-process
comparison was **53.425 → 51.930 s (1.029x)**. No candidate is adopted, and the
runtime remains the code measured in `917748b10d5eff9bb077b332a018eafdb33e9fcd`.
These are additional spent-data development screens, following the earlier
[rejected execution candidates](lidar_execution_screen.md).

## Separate component cohorts

| Candidate | Prior median, eight targets | Candidate median | Ratio |
| --- | ---: | ---: | ---: |
| Captured CPU tree, lightweight certified broad search | 0.4293 s | 0.4176 s | 1.028x |
| GPU radix construction and broad search | 0.4988 s | 0.4311 s | 1.157x |
| Resident radix/broad tree, replacing the resident grid | 0.3653 s | 0.3009 s | 1.214x |
| Direct eight-point radix leaves, without compaction | 0.3650 s | 0.4835 s | 0.755x |
| Cooperative GPU execution of all ICP iterations | 0.5017 s | 0.5089 s | 0.986x |

Each cohort has six alternating repeats over the same eight fixed public GLIM
registration captures. All 480 timed registrations and ten first-case warmups
are retained. Timings include target construction, registration and work-counter
collection; compilation of the first-case warmup is outside these component
times. CPU/GPU activity is unreserved and BLAS uses one thread. Baselines differ
between cohorts: the first two use ordinary captured-tree CUDA registration;
the next two use resident-grid registration. The last returns to the ordinary
captured tree. Medians and ratios are never pooled or multiplied.

Broad search uses FP32 candidate distances and a conservative cast/rounding
margin, then evaluates selected distances in the original FP64 arithmetic.
Uncertain cutoffs repeat the complete original FP64 query. This reduces the
per-candidate interval work of the previous rejected search. The radix variant
builds spatial topology on GPU. Keeping its target on device removes the full
target download; native repairs retain sparse neighborhood transfers and lazy
native-tree access. Direct block leaves remove construction compaction, but
registration gets slower enough to outweigh that saving.

The cooperative prototype runs query/system construction, reductions, a small
FP64 pivoted solve and pose updates in one cooperative-grid launch per
registration. The host receives the packed final state. Ambiguous gates,
stopping conditions or motion-grid boundaries retain a complete ordinary-CUDA
fallback. The fixed captures exercise no such fallback, so this screen provides
no boundary/fallback correctness validation for a production backend.

All registration result floats match bit for bit for the four tree cohorts.
Cooperative execution changes solve/reduction order; its protocol fixes the
existing CPU/CUDA registration tolerances before execution: relative `1e-6`,
absolute `1e-9` for pose, RMSE and Hessian eigenvalue, with exact convergence,
iteration and correspondence decisions. It passes those gates rather than a
bit-parity gate. All original outputs and actual work counters remain available.

The four tree protocols record an empty cache before CuPy import. The
cooperative protocol's original directory-existence flag is false. These are
warm component screens, without a cold-start claim. Prototype source, generated
kernels and dependencies are bound separately from inherited base-runtime
identity strings. None is exposed as a production backend.

## Complete pipeline check

Only the best resident-tree component advanced to this check. Both fresh workers
replay six seconds of the public GLIM Livox bag with existing resident geometry,
CUDA spline planes, 0.03 m odometry voxels, three local-map scans, a 1 nm geometry
grid, a 1 micrometre translation grid and a 0.1 microradian rotation-vector grid.
Each starts with an independent empty compiler cache and disabled driver cache.
Existing IMU coverage, all eight odometry passes, four map/fit rounds, full
covariance, held-out scoring and all 32 signed controls remain in scope.

Whole time includes process startup, imports, first compilation, decoding,
calibration/evidence, saves and source/input verification. Prior runs first,
candidate second. All eight full fits and complete physical/evidence artifacts
match every float bit after declared run metadata and spatial-implementation
counters. The candidate validator changes only the required spatial-build
counter; the CPU auditor checks that source transformation without executing it.

The observed **1.029x** whole-worker ratio is too small to justify adopting this
design. One pair on an unreserved host does not establish a stable latency gain.
The pair is retained without rerunning to replace its observation. The final
calibration verdict remains INCONCLUSIVE, without independent extrinsic truth
or a new accuracy/SOTA claim.

## Separate full-pipeline diagnosis

A source-bound cProfile run of the unchanged resident pipeline completes all
four fits and evidence. Its nested host markers report:

| Marker | Calls | Instrumented time |
| --- | ---: | ---: |
| Covered odometry additions | 456 | 31.29 s |
| Registration | 896 | 19.17 s |
| Geometry preprocessing | 1,352 | 7.06 s |
| Target capture | 448 | 1.79 s |
| Gyro point rotation | 392 | 0.45 s |
| Map association | 4 | 1.97 s |

Markers overlap and must not be summed. Instrumentation, compilation and host
scheduling change these timings; they are not compared with the fresh-worker
pair. Approximately 4,800 reduced-system host reads appear in the profile.
Their durations include waiting for GPU computation, rather than isolating
transfer time. Reducing those reads alone did not produce a component gain in
the cooperative screen above. The path-bearing binary profile stays private;
sanitized text, provenance, results and full-fit records are archived separately
from the timing cohorts.

## Replay

The [bound follow-up archive](../assets/lidar_pipeline_followup/README.md)
preserves all original observations, launchers, generated code, logs and helper
snapshots. Existing capsules reconstruct all 396 runtime modules; coordinate
capture metadata is shared with the earlier archive. Raw point captures and
compiler caches are not redistributed. CPU replay checks stored coordinate
identities and numerical receipts; it does not regenerate geometry or execute
experimental text.

```bash
python -m tools.audit_lidar_pipeline_followup docs/assets/lidar_pipeline_followup --check
```
