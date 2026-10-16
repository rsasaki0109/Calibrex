# Ordered batches for Doppler velocity estimation

Complete scan-wise Radar velocity estimation improves from **13.412055 to
4.976802 seconds (2.694915x)** on the generated public-data execution control
below. Complete solver calls improve from **11.682495 to 3.150524 seconds
(3.708111x)**. This is automatic CPU acceleration through the existing
`RadarEgoVelocitySolver.solve` API.

## Change

Previously every IRLS iteration evaluated each Doppler residual, Huber weight
and normal-equation contribution through Python loops. Scans with at least
64 ordinary finite float rows now prepare fit-local NumPy arrays and evaluate
these expressions together. Normal equations retain the scalar products and
the original left-to-right additions: `np.add.accumulate` carries each prefix
into the next chunk instead of reassociating the sum. Chunks contain at most
4,096 rows, with twelve accumulation columns and one carried prefix (393,312
bytes). Prepared directions, targets, weights and residuals use linear memory.

Normalization, the 3x3 pivoted solve, eigenvalue diagnostics, convergence test,
inlier selection and final Python RMSE reduction retain their original code.
Small scans and unusual numeric objects or non-finite/extreme input rows use
the scalar path. Each call prepares its own arrays; edits to observations or
options are observed on the next call. No new option or dependency is added;
NumPy is already a core dependency. Public result fields and schemas retain
their existing definitions.

## Workload and complete measurements

The [original public KITTI velocity control](radar_reference_indexing.md)
uses all 2,762 OXTS samples from `2011_09_30_drive_0018_sync` to generate 2,758
interior Radar-frame velocities. This comparison uses every velocity to
generate a 256-return Doppler scan: fixed noncoplanar directions, deterministic
0.015 m/s sinusoidal noise, a 4 m/s cosine outlier term on every seventeenth
return, and unequal weights from 1.0 to 1.4. There are **706,048 returns**.

**These Doppler returns and Radar velocities are generated controls.** They
are not measured Radar, independent accuracy evidence, or an acceleration
measurement of raw LiDAR registration or a complete extrinsic pipeline.
No accuracy holdout or leaderboard claim is opened or changed.

Three alternating pairs run in fresh processes with frozen 380-module source
trees and real Git identities. One BLAS thread is used on an unreserved host.
Whole-worker time includes startup, input loading, generation of every return,
all solves, full normal-system capture, compression, serialization and saving.
API time sums complete solves, including normalization, preparation, all IRLS
iterations, diagnostics and capture. Downloads, source freezing and external
audit are outside the timed workers. All runs are retained.

| Pair (order) | Previous whole s | Batches whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (previous, batches) | 13.412055 | 4.976802 | 2.695x |
| 2 (batches, previous) | 11.929565 | 4.804866 | 2.483x |
| 3 (previous, batches) | 13.783308 | 6.377092 | 2.161x |

Reported ratios divide the two three-run medians. The observed benefit depends
on scan size, geometry, iteration count and host load. It does not establish
the same ratio for other Radar captures or the surrounding calibration stages.

## Parity, validation and reproduction

All six workers retain exactly equal full per-scan results and every initial
and weighted IRLS normal system, RHS and solved velocity. Each executes
**17,658 IRLS iterations**. Default iteration limits, Huber threshold, noise,
rank, convergence and inlier policies are unchanged. Numeric comparisons use
exact bytes, with only declared source identities differing in provenance.

Tests compare every residual, robust weight, normal system and complete solve
against the preserved scalar path. They cover the 64-row threshold, 4,096-row
chunk boundary, signed zeros, degenerate geometry, nonpositive-weight filtering,
changed inputs/options and unusual input fallback. Windows/Python 3.12 and
Linux/Python 3.10 pass the same parity and recovery tests. Archive tests reject
rehashed work reductions, false timings, missing histories, changed IDs,
unfrozen imports and unrelated runtime changes. Strict typing and the public
schema registry checks remain in force.

The [complete archive](../assets/radar_doppler_ordered_batches/README.md)
contains every full result and binary history, all six measurements/logs,
the exact prepared source control, measured launcher, source capsules/manifests,
tool schema, post-measurement auditor, publication code and validation evidence.
The original velocity control retains its source-file identities and data
license. Runtime baseline is `36446a4fabdbfe8d406ea1908677357f4f027c0f`;
measured runtime is `782d4cfdb314985e5991a0d816b24a137f5c75ae`. Only
`solvers/radar_ego_velocity_solver.py` changes across those runtime manifests.

```bash
python -m tools.benchmark_radar_doppler_batches \
  --baseline-commit 36446a4fabdbfe8d406ea1908677357f4f027c0f \
  --output outputs/radar-doppler-reproduction --repetitions 3
python -m tools.benchmark_radar_doppler_batches \
  --audit-only docs/assets/radar_doppler_ordered_batches
```

Use committed runtime sources and a new output directory. The audit needs no
Radar recording, ROS installation, GPU or download.
