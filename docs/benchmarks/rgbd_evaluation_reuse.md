# Evaluation reuse for RGB-D joint calibration

On the existing public TUM fr1/xyz development sequence, the complete native
RGB-D joint pipeline improves from **806.338037 to
242.364507 seconds (3.326964x)**. Complete
`run_calibration` calls improve from **803.657115 to
239.525472 seconds (3.355205x)**, and the combined
24 optimizer calls from **613.376353 to
62.428832 seconds (9.825209x)**.
These are CPU measurements. No GPU backend or dependency is introduced.

## Change and lifetime

The previous depth and XYZ residual factories repeated a Python callback,
parameter dictionary construction, transform composition, point transformation
and small NumPy derivative allocations for every point and finite difference.
The new path groups factory-owned geometry into read-only arrays and evaluates
complete residual vectors and central-difference columns. It covers scale/bias
depth, trilinear depth, centered trilinear depth, full-XYZ point/plane and
two-sided XYZ pair factors.

Lattice corrections deliberately retain the original Python weighted sums,
including their summation behavior on each supported Python version. They are
not replaced with a BLAS dot product or a different reduction. Transform arrays
use the scalar core's two Hamilton products, and plane residuals retain the
same subtraction, multiplication and addition order. Priors, whitening, shape
terms, smoothness and rigid gauges run their original callbacks. Numerical
Jacobians, robust loss, convergence, Schur steps and probes are unchanged.

Each prepared graph has at most two calibrated point-array states per group.
An unchanged immutable lattice tuple reuses its points during pose/extrinsic
perturbations. Identity keys retain parameter tuples, preventing identity reuse
and preserving signed zero. The complete plan and these states belong to one
linearization or residual pass and are discarded afterward. Preparation is
fresh on the next call, so replaced nominal depths, points, frames and weights
are inspected again.

Eligibility requires genuine factory callbacks, their original code and
closure cells, sealed binding metadata, ordinary finite parameter tuples and
immutable numeric geometry. Scalar regularizers must close over plain numeric
data. Wrapped, replaced, custom or otherwise unsupported callbacks and inputs
retain the previous scalar path. The original callback of a custom graph is
never silently skipped. Scalar RGB-D factories also use the existing bounded
128-entry evaluation-local transform scratch; it is released on return/error
and isolated for nested evaluations and threads.

At most 65,536 data factors and 128 geometry groups use the array path; larger
or unsupported graphs fall back without dropping work. Rotation and residual
kernels use 4,096-row chunks. Point-array states have the complete prepared
rows, with two states per group and two endpoints for pair factors. These are
bounds on additional retained geometry, not process memory; the original
dense Jacobian allocation remains. Public signatures, config/result schemas,
ROS independence and dependencies are unchanged.

## Complete public workload

The unchanged [TUM configuration](https://github.com/rsasaki0109/Calibrex/blob/da6f57833fe00cd896b0534bad8723f3c138014e/examples/public_datasets/tum_rgbd_freiburg1_xyz/native_joint_slac_config.yaml)
selects three disjoint-map/query windows beginning at 60, 180 and 300. Each has
three map frames and eight query frames. The primary window retains 2,466 data
correspondences and eight train-only pose priors, a 56-dimensional joint system,
nine LM iterations and the original grouped holdout and signed probes.

Every timed worker completes all 24 optimizer calls plus scalar and full-XYZ
ablations, local-rotation updates, two-sided pair factors, cross-window
transfers, reassociation and support/probe diagnostics, fixed/rematched
curvature, multistart search and report saving. It records
168 complete Jacobian calls and 1624
residual passes. No frame, point cap, iteration limit or diagnostic is reduced.

The public sequence was already used for development. Existing mounting/depth
reference failures and identifiability limitations remain in the complete
results. Faster execution does not establish improved calibration accuracy or
open an independent SOTA claim. The upstream manifest and all 798 depth-PNG
fingerprints, four text-file fingerprints and sizes are retained with the
comparison; dataset images are not republished.

## Measurements

Two alternating pairs use fresh processes and actual Git identities, with one
BLAS thread on an unreserved host. The order is reuse/previous, then
previous/reuse. Whole-worker time includes startup, imports, source-identity
checks, the complete pipeline, trace capture, serialization, compression and
saving. API time covers the entire `run_calibration` call and its own saved
reports. Optimizer time sums the 24 original `solve` calls, including their
numeric-array tracing. Downloading, input fingerprinting, source freezing and
external audits are outside timed workers. All four runs are retained.

| Pair (order) | Previous whole s | Reuse whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (current, prior) | 845.921963 | 254.685042 | 3.321x |
| 2 (prior, current) | 766.754111 | 230.043972 | 3.333x |

Reported ratios divide two-run medians. They describe this capture and host;
graph size, geometry and host load affect the gain. No earlier speedup is
multiplied into these measurements.

## Parity and validation

All four full schema-valid calibration results match exactly after removing
only run ID, wall-clock dates and the independently verified Git identity
fields. Every recorded residual/Jacobian byte digest, factor-order digest,
parameter-state digest and all 24 optimizer-result digests match exactly.
Reference verdicts, warnings, histories, ranks, whitening and known-bad probes
are included in the full comparison. Source manifests and imported-module
hashes verify all 382 previous and 383 current modules. The three source changes
are `graph/joint_factors.py`, `graph/joint_optimization.py` and the new private
`graph/_rgbd_batches.py`. Measured array-plan requests and accepted plans are
retained in every measurement and independently checked by the auditor.

The focused regression suite passes 248 tests on Windows and 211 graph tests
on Linux, including 100 RGB-D cases covering all five families, dense/Schur
optimization, whitening, robust loss, changing parameters, signed zero, mixed-family row order, fixed-pose columns,
ineligible inputs, mutation, chunking, eviction and scope cleanup. The archive tests
exercise offline re-auditing and reject rehashed changes to workload, inputs,
results, numeric histories, timings and source identity. Strict typing and
Ruff checks pass. All 129 public schemas and their export ledger remain byte
identical between the measured runtime trees.

## Reproduce and inspect

```console
python tools/download_public_dataset.py tum_rgbd_freiburg1_xyz
python tools/benchmark_rgbd_transform_reuse.py --baseline-commit da6f57833fe00cd896b0534bad8723f3c138014e --output outputs/rgbd-evaluation-replay --repetitions 2
python tools/benchmark_rgbd_transform_reuse.py --audit-only docs/assets/rgbd_evaluation_reuse
```

The benchmark requires the pinned public input bytes and unchanged example
configuration. The offline auditor needs no dataset download and recomputes
parity and medians from the published records. It verifies source capsules,
config/options, all three windows, complete work counts, dimensions, source
identity, input fingerprints, trace/result bindings and indexed file bytes.

- [Summary](../assets/rgbd_evaluation_reuse/summary.json)
- [Protocol](../assets/rgbd_evaluation_reuse/protocol.json)
- [All attempts](../assets/rgbd_evaluation_reuse/attempts.json)
- [Complete byte inventory](../assets/rgbd_evaluation_reuse/index.json)
- [Public input fingerprints](../assets/rgbd_evaluation_reuse/inputs.json)
- [Schema export parity](../assets/rgbd_evaluation_reuse/schema-parity.json)
- [Previous runtime](https://github.com/rsasaki0109/Calibrex/commit/da6f57833fe00cd896b0534bad8723f3c138014e)
- [Measured reuse runtime](https://github.com/rsasaki0109/Calibrex/commit/15de8f45df9580e6ba1ab8dfacf2215867e50d6a)
