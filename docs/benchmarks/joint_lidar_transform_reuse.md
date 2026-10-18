# Scoped transform reuse for joint LiDAR calibration

The complete public Livox joint SLAC pipeline improves from **800.253310
to 231.253153 seconds (3.460508x)**. Complete
`run_calibration` calls improve from **796.891655 to 228.921617
seconds (3.481068x)**. This is automatic CPU acceleration of
the existing native point-to-plane joint factors.

## Change

Each point factor previously decoded both six-dimensional tangent blocks and
composed their initial transforms on every residual evaluation. Thousands of
points share those same transforms during each central-difference Jacobian
column. The graph now provides a fresh, bounded scratch cache for each
linearization or residual pass. Built-in point-to-plane factors reuse only the
composed immutable transform. Every point rotation, scalar residual, whitening,
finite difference and optimizer decision still executes in its original order.

Keys contain the exact float bytes of both initial frames and both tangent
blocks, including signed zero. At most 128 transforms are retained per scope;
this is an entry bound, not a process-memory claim. Each scope resets on return
or exception and isolates nested and concurrent evaluations. Standalone factors,
custom SE3 subclasses and unusual/non-finite numeric inputs keep the scalar
path. Generic user callbacks still run on every evaluation. Public signatures,
result fields, options and schemas retain their existing definitions. No new
dependency is added.

## Complete public workload

The two PCDs come from the [official public Livox Horizon example](https://github.com/Livox-SDK/Livox_automatic_calibration). Their raw headers
contain **23,921 source and 23,666 target points**. Both reader point caps are
disabled. The existing [full-point configuration](https://github.com/rsasaki0109/Calibrex/blob/c0fbf5e2ddbebd589fab9e11b235e9f7ab84d5e6/examples/public_datasets/livox_horizon_horizon_pcd_sample/native_full_points_config.yaml)
uses the `native_joint_slac` backend for this execution comparison, with its
existing 1 m voxel, 1.5 m correspondence gate, 40-iteration ceiling and
convergence tolerance. Only backend and resolved data/output paths differ from
that configuration.

Every run processes **16,884 point-plane observations**, **40 joint LM
iterations**, **41 complete Jacobians**, and **65 residual passes**, including
the candidate objectives, spatial holdout and **24 known-bad probes**. There
are 13,599 training point factors plus the six-residual train-only pose prior,
and 3,285 holdout factors. Each complete Jacobian has shape **13,605 x 12**.
Inspection, plane preparation, fixed-baseline solving and report saving also
remain inside the complete pipeline.

This existing development capture remains **degenerate, with rank 9**, in both
versions. Its status, diagnostics and final output are preserved. These are timing
and parity measurements, not independent calibration-accuracy evidence. No SOTA
cohort or leaderboard claim is opened.

## Measurements

Two alternating pairs use fresh processes and frozen source trees with actual
Git identities. One BLAS thread is used on an unreserved host. Whole-worker time
includes cold startup, input loading, the complete pipeline, trace capture,
compression and saving. API time covers the entire `run_calibration` call,
including array-digest capture and the pipeline's own artifact saving. Downloads,
source freezing and external audit are outside the timed workers. All four
runs are retained.

| Pair (order) | Previous whole s | Reuse whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (prior, current) | 717.827525 | 244.669707 | 2.934x |
| 2 (current, prior) | 882.679095 | 217.836599 | 4.052x |

Reported ratios divide the two-run medians. This small comparison establishes
the observed benefit for this capture and host; other graphs and surrounding
stages can have different costs.

## Parity and reproduction

All four full schema-valid calibration results match exactly after removing
only run ID, wall-clock dates and verified source identity. The full parameter
and evaluation trace also matches: every complete residual and Jacobian array
has the same SHA-256 digest of its little-endian float64 bytes. Event counts,
shapes, ordered factor IDs, final states and all probe states are checked.
Full optimizer histories and diagnostics are retained inside each result.

The measured launcher is preserved verbatim. The post-measurement auditor
additionally verifies both duplicate Git labels in run and solid-state metadata
before excluding all three recorded source identities. The repair record binds
the original launcher, final auditor and unchanged measured-file hashes; no
calibration value, input, trace, timing or source capsule was edited.

Unit tests compare actual residual/Jacobian bytes and complete LM results with
the scalar path, including observable geometry, outliers, unequal weights,
whitening, dense/Schur solvers and fixed blocks. Tests also cover signed zero,
distinct initial frames, bounded eviction, changed values, nested/concurrent
scopes, errors, direct calls and unusual-input fallback. Windows/Python 3.12 and
Linux/Python 3.10 pass the same tests. Archive mutation tests reject rehashed
work reductions, altered array hashes, missing events, changed factor/probe
states, unfrozen imports and false reported timings.

The [complete archive](../assets/joint_lidar_transform_reuse/README.md) contains
all full results, all event traces, measurements/logs, frozen source capsules,
the exact launcher/config/tool schema, input identities and validation evidence.
Raw PCDs are obtained separately from their original public source; the archive
does not redistribute them or imply a data license from the code license.
The baseline is `c0fbf5e2ddbebd589fab9e11b235e9f7ab84d5e6` and the measured runtime is
`8e0a7fe645b07bc10badc641e9902e35362c8925`. Only the three declared graph modules differ.

```bash
python -m tools.benchmark_joint_lidar_transform_reuse \
  --baseline-commit c0fbf5e2ddbebd589fab9e11b235e9f7ab84d5e6 \
  --output outputs/joint-lidar-transform-reproduction --repetitions 2
python -m tools.benchmark_joint_lidar_transform_reuse \
  --audit-only docs/assets/joint_lidar_transform_reuse
```

Use committed runtime sources and a new output directory. Offline archive audit
requires no PCD download, ROS installation or GPU.
