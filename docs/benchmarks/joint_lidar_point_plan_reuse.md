# Guarded preparation reuse for joint LiDAR calibration

Compared with [ordered point batches](joint_lidar_point_batches.md), the complete
public Livox joint pipeline improves from **23.324646 to
9.522338 seconds (2.449466x)**. Complete
`run_calibration` calls improve from **20.801696 to 7.212691
seconds (2.884041x)**. This additional CPU acceleration reuses
point/plane preparation during one joint solve. The array geometry and numerical
Jacobian arithmetic retain their previous implementation.

## Change and lifetime

The previous version validated every point's factory callback, numeric geometry,
initial frames and weights and rebuilt the same read-only arrays for every
linearization, candidate evaluation and holdout probe. For this complete workload
that meant 106 preparations. The new version builds one training plan and one
holdout plan and reuses them for the other 104 evaluations. The measured workers
record both preparation-call and actual-build counts; the offline auditor checks
106 calls and 106 previous builds versus two current builds in every run.

The native adapter creates its factor graph internally and passes a private,
immutable factor tuple to the optimizer. These factors are not returned to a
caller or handed to a user callback. After the complete ordinary-input and
factory-purity checks, the cached plan takes a private snapshot of initial
frames as well as point/plane arrays. Stored quaternion bits are copied without
another normalization. Fixed native geometry therefore needs no recursive
point inspection on the other 104 requests. This optimization is restricted to
that private ownership marker and is not a new public option.

General optimizer callers still receive identity/code/closure checks on every
reuse: factor order, weights, names, whitening, factory descriptor/owner,
point/plane tuples, normals and both initial frames and their stored vectors.
Identity checks distinguish signed zero and avoid user-defined equality.
Their changed inputs rebuild preparation or take the scalar fallback. Current
parameter values and dimensions and the global method/decoder eligibility are
checked for both paths on every request. Global method, decoder or parameter eligibility failures clear retained
preparation before scalar fallback. Ineligible graphs are never
negatively cached. Mutable or custom initial-frame leaves are prepared afresh.

Preparation belongs to one synchronous `BackendNeutralJointOptimizer.solve`
invocation and is released on return or error. Nested solves and separate
threads use separate contexts. Calls made outside a solver invocation retain fresh
preparation. At most two plans and 65,536 combined factor snapshots are retained;
eviction removes a complete cached plan and recomputes it when needed. These are
retention bounds, not total process memory bounds. The existing 4,096-row geometry
chunks and dense Jacobian allocation are unchanged. No factor, input point or
iteration is discarded to meet a bound.

Family diagnostics and the native adapter's final metric evaluation also receive
the existing bounded transform scratch scope. They still invoke every original
scalar callback in its original order and retain the original Python reduction.
Public signatures, configs, schemas, solver choices, convergence rules and
dependencies are unchanged. Custom callbacks and unsupported inputs retain the
previous conservative behavior.

## Complete public workload

The official [Livox Horizon pair](https://github.com/Livox-SDK/Livox_automatic_calibration)
contains 23,921 source and 23,666 target points according to its raw PCD headers.
Both reader caps are disabled. Every run retains 16,884 point-plane observations,
40 joint iterations, 41 complete 13,605 x 12 Jacobians, 65 residual passes and
24 known-bad probes. Training has 13,599 point factors and a six-residual train-only
pose prior; holdout has 3,285 points. Inspection, map/plane preparation, the
independent fixed-baseline solve and report saving remain in the timed pipeline.

This existing development capture remains **degenerate with rank 9** in both
versions. Status, optimizer history, diagnostics, probe responses and final
outputs are preserved. This is execution/parity evidence and does not establish
independent calibration accuracy. No SOTA cohort or leaderboard claim is opened.

## Complete measurements

Two alternating pairs run in fresh processes using frozen 382-module source trees
and actual Git identities. One BLAS thread is used on an unreserved host.
Whole-worker time includes cold startup, data loading, the complete pipeline,
array-hash capture, compression and saving. API time covers the complete
`run_calibration` call, including its report/artifact saving and trace capture.
Downloads, source freezing and external auditing are outside the timed workers.
All four runs are retained.

| Pair (order) | Previous whole s | Reuse whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (prior, current) | 22.468484 | 8.620691 | 2.606x |
| 2 (current, prior) | 24.180807 | 10.423985 | 2.320x |

Reported speedups divide the two-run medians. Results describe this existing
capture and host; graph size, geometry and host load affect the gain. This is a
comparison with the immediately preceding optimized version. Earlier published
ratios are not multiplied into this measurement.

## Parity and validation

All four full schema-valid results match after removing only run ID, dates and
the three independently verified source-identity fields. All 106 ordered event
states, factor-ID hashes and complete residual/Jacobian little-endian float64
array hashes match. The auditor also verifies work counts, full dimensions,
final values and every known-bad parameter state. The exact measured launcher
and public auditor have identical bytes.

Tests check full residual/Jacobian bytes, complete dense/Schur LM results,
synthetic extrinsic recovery and existing fallback behavior. New tests cover
actual build reduction, owned-versus-guarded full-solve parity, exact frame
snapshot bits and scope release, changed point/plane geometry and weights, initial-frame
translations/quaternions, changed callbacks and closure cells, descriptors,
ownership, whitening, factor order, dynamic parameter values/dimensions,
conservative limits, whole-plan eviction and nested/concurrent/error cleanup.
Windows/Python 3.12 and Linux/Python 3.10 run the same checks. Archive mutation
tests reject rehashed work reductions, false build counts, changed numeric
history, IDs/order/probe states, timings and source identities. All 129 public
schemas and their ledger remain byte-identical across both measured runtimes.

The [complete archive](../assets/joint_lidar_point_plan_reuse/README.md) retains
four full results, traces, measurements and logs, exact config/launcher/schemas,
both source capsules, upstream input identities and validation/publication code.
Raw PCDs remain external inputs; the historical upstream manifest is preserved
verbatim, while protocol counts come from the actual PCD headers.

Baseline: `c517dca9b98c6bc092a184707406db4f0c56a908`.
Measured runtime: `dfd9df76515fc36b3b252e48a6c58d605b636b93`.
Only four existing runtime modules differ; no runtime module is added or removed.

```bash
python -m tools.benchmark_joint_point_plan_reuse \
  --baseline-commit c517dca9b98c6bc092a184707406db4f0c56a908 \
  --output outputs/joint-plan-reuse-reproduction --repetitions 2
python -m tools.benchmark_joint_point_plan_reuse \
  --audit-only docs/assets/joint_lidar_point_plan_reuse
```

Use committed runtime sources and a new output directory. Offline archive audit
requires no PCD download, ROS installation or GPU.
