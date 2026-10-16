# Ordered point batches for joint LiDAR calibration

Compared with [scoped transform reuse](joint_lidar_transform_reuse.md), the
complete public Livox joint pipeline improves from **248.161254 to
22.305351 seconds (11.125638x)**. Complete
`run_calibration` calls improve from **245.586553 to 20.052062
seconds (12.247446x)**. This additional CPU acceleration is
automatic for eligible built-in point-to-plane joint graphs.

## Change and eligibility

The previous version reused composed transforms but still rotated each point
and formed every numerical-Jacobian entry through individual Python calls.
This version prepares immutable point/plane/normal/weight arrays per evaluation
pass and groups points sharing the same two tangent blocks and initial frames.
It evaluates their geometry and central differences together, retaining the
two ordered Hamilton products, both scalar quaternion normalizations, zero
terms, scalar dot-product addition order and Python `math.sqrt` weights.
It uses no rotation-matrix substitution, reassociated reduction or analytic
Jacobian. The dense normal solve, damping, robust loss and convergence policy
retain their existing code.

Factory-owned point and prior callbacks are identified by their function,
code and closure identities. Wrapped or arbitrary callbacks, factor subclasses,
modified residual methods, unusual/non-finite/extreme numeric objects and point
whitening use the previous scalar path. Priors retain their scalar residual and
whitening evaluation. All original factor IDs, ordering, grouped holdout and
diagnostics are preserved. Public signatures, schemas and options are unchanged;
NumPy is already required.

The prepared path accepts 64 through 65,536 point factors and up to 128 initial
frame/block groups. Linearization also requires at most 64 total parameter
dimensions and ordinary finite positive finite-difference steps. Geometry
temporaries cover at most 4,096 rows; prepared arrays are read-only and local to
one pass. These bounds do not describe total process memory or replace the
existing dense Jacobian allocation. Graphs outside these bounds evaluate every
point through the previous path. No input point or optimizer iteration is
removed to meet a bound.

## Complete public workload

The [official public Livox Horizon pair](https://github.com/Livox-SDK/Livox_automatic_calibration)
contains 23,921 source and 23,666 target points according to its raw PCD headers.
Both reader caps are disabled. The comparison retains the full-point
configuration's 1 m voxel, 1.5 m gate, 40-iteration ceiling and convergence
tolerance, using `native_joint_slac` and resolved data/output paths.

Every run retains **16,884 point-plane observations**, **40 joint iterations**,
**41 complete 13,605 x 12 Jacobians**, **65 residual passes**, and **24 known-bad
probes**. There are 13,599 training point factors plus the six-residual train-only
pose prior, and 3,285 holdout factors. Inspection, map/plane preparation, the
fixed-baseline solve and report saving also remain in the complete pipeline.

The existing development capture remains **degenerate with rank 9** in both
versions. Status, full diagnostics and final outputs are preserved. This
execution comparison does not establish independent calibration accuracy.
No SOTA cohort or leaderboard claim is opened.


## Complete measurements

Two alternating pairs run in fresh processes with frozen 381-module baseline
and 382-module current source trees and actual Git identities. One BLAS thread
is used on an unreserved host. Whole-worker time includes cold startup, data
loading, the complete pipeline, array-hash capture, compression and saving. API
time covers the complete `run_calibration` call, including its own report/artifact
saving and array-hash capture. Downloads, source freezing and external audit
are outside the timed workers. All four runs are retained.

| Pair (order) | Previous whole s | Batches whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (prior, current) | 262.691688 | 22.749495 | 11.547x |
| 2 (current, prior) | 233.630820 | 21.861206 | 10.687x |

Reported ratios divide the two-run medians. The result describes this existing
capture and host; other geometry, graph sizes and host load can change the gain.
The comparison is against the previous optimized version, not a fresh timing
of earlier unoptimized revisions.

## Parity and validation

All four full schema-valid calibration results match exactly after removing
only run ID, wall-clock dates and the three independently verified source
identity fields. Every complete residual and Jacobian has the same SHA-256 of
its little-endian float64 array bytes. All 106 evaluation events, parameter
states, ordered factor IDs, final values and probe states are checked. Full
optimizer histories and diagnostic values are retained inside each result.

Tests compare actual residual/Jacobian bytes with the previous scalar path at
the eligibility threshold and across 4,096-row chunk boundaries. They cover
distinct initial frames, signed zeros, outliers, unequal weights, fixed blocks,
dense/Schur solvers, extra prior blocks, modified callbacks/methods, whitening
fallback, changed values/weights and conservative input bounds. Observable
synthetic graphs still recover the original extrinsic. Windows/Python 3.12 and
Linux/Python 3.10 pass the same tests. Archive mutation tests reject rehashed
work reductions, missing evaluations, changed array hashes, factor/probe state
changes, false timings and incorrect source identities. All 129 public schema
exports and their ledger remain byte-identical across the measured runtimes.

The [complete archive](../assets/joint_lidar_point_batches/README.md) retains all
four full results, traces, measurements and logs, both source capsules, the
exact launcher/config/schemas, input identities and validation/publication code.
Raw PCDs are obtained separately from their official source; the archive does
not redistribute them or infer a data license from the code license.

Baseline: `f5c52fc018f298627598920db2225042fe8ce133`.
Measured runtime: `8461408bbbe90531fb3c299dc0035cd0ba20ca30`.
Only the two existing graph modules and new private batching module differ
across the runtime source manifests. The measured launcher and public auditor
have identical bytes.

```bash
python -m tools.benchmark_joint_point_batches \
  --baseline-commit f5c52fc018f298627598920db2225042fe8ce133 \
  --output outputs/joint-point-batches-reproduction --repetitions 2
python -m tools.benchmark_joint_point_batches \
  --audit-only docs/assets/joint_lidar_point_batches
```

Use committed runtime sources and a new output directory. Offline archive audit
requires no PCD download, ROS installation or GPU.
