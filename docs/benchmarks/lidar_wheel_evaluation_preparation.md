# Preparing LiDAR–wheel residual inputs once per fit

Complete prepared-input LiDAR–wheel evaluation falls from **2.389020 to
0.491396 seconds (4.861703x)**. The whole fresh comparison worker falls from
**5.353725 to 3.397488 seconds (1.575789x)**. The larger ratio applies to the
calibration API; imports and measurement instrumentation dominate the remaining
process time. These are ratios of three-run medians, with every default fit,
holdout, uncertainty estimate and known-bad control retained.

## Input and scope

All **2,762 original KITTI OXTS poses** of `2011_09_30_drive_0018_sync` generate
a sensor trajectory through a declared body-to-sensor transform: XYZ Euler
angles `[1, -2, 3]` degrees and translation `[0.8, -0.3, 0.9]` m. The unchanged
two-frame motion construction yields **2,760 intervals in 29 ten-second
blocks**. The common original time origin is removed from both sensor and proxy
timestamps, so finite-difference clock steps are resolvable. Samples and motion
intervals are not thinned. All original `vf` (column 8) and `wu` (column 22)
values supply the wheel proxy. These are the same columns used by the existing
KITTI wheel-proxy adapter; no synthetic wheel measurements are added.

KITTI has **no wheel odometry**. This is a generated sensor-motion execution
control against an OXTS proxy, without bag reading or raw LiDAR registration.
It measures prepared calibration/evaluation performance and numerical parity,
without independent accuracy evidence or a GPU claim. The unchanged policy
verdict is **warn**: the 20 ms clock control is not detected, and roll remains
unobservable. The reported clock value is not independently validated.

Every third block is held out, retaining **1,824 training and 936 holdout
motions** before the unchanged 2 m/s gate. There are **nine complete solves**:
one main fit and all eight default block-jackknife fits. Every solve retains
all **301 yaw-clock scan candidates** and the original six-parameter robust
least-squares tolerances, bounds and 200-evaluation maximum. The holdout base,
three rotation controls, speed-scale control, clock control and two residual
medians remain in the evaluation.

## Change and work

Each fit prepares the already gated motion velocity/rate arrays and interval
midpoints once. An eight-state LRU reuses wheel interpolation and its velocity
noise model for exact binary clock states. Signed zero and every numerical
derivative step remain distinct. Rotation, matrix products and all six residual
rows per motion are recomputed in their original order for every request.
There is no residual cache or cache shared between fits. Derived wheel classes
retain their per-request interpolation callbacks.

A conservative **64 MiB** bound accounts for prepared arrays, all eight clock
states and working memory at 1,024 bytes per motion. Larger inputs and empty
sets retain the original public residual path. Public signatures, defaults,
noise models and core schemas are unchanged.

| Work per complete evaluation | Prior | Prepared |
| --- | ---: | ---: |
| Main plus jackknife solves | 9 | 9 |
| Residual requests, including holdout checks | 411 | 411 |
| Least-squares results | 9 | 9 |
| Motion selection calls | 420 | 17 |
| Motions visited by selection | 646,689 | 22,968 |
| Wheel interpolation calls, including scans | 3,120 | 2,836 |
| Wheel sample queries, including scans | 4,829,055 | 4,389,516 |

The audit independently reconstructs every fit's original and gated motion
indices, the full block split, all clock scan candidates and the eight-state
LRU from the complete parameter history. The selected control exercises 13–15
distinct clock states per fit. The main fit retains all 1,743 gated training
motions; the other intervals remain represented by the original holdout and
gating policy.

## Measurement and validation

Three fresh frozen pairs alternate execution order with one BLAS thread on an
unreserved host. Whole-worker time includes imports, actual Git identity
checks, input loading/construction, complete evaluation and trace
instrumentation, official artifact saving, trace compression, imported-source
hashes and measurement validation. The calibration API boundary starts before
prepared input loading and ends after the official result is saved. Control
generation, source freezing, raw-input compression and parent audits are
outside both boundaries.

| Pair (execution order) | Prior whole s | Prepared whole s |
| --- | ---: | ---: |
| 1 (prior, prepared) | 5.215874 | 3.444593 |
| 2 (prepared, prior) | 5.353725 | 3.397488 |
| 3 (prior, prepared) | 6.049091 | 3.342003 |

Baseline: `88d6fba240ed38a744bccd023111303f84b7be5b`.
Measured source: `479d15b79bf4258bfc8f94175ed63b77d6eccd0a`.
Only `calibrex/solvers/lidar_wheel_solver.py` differs in the complete core
source manifests. All six official artifacts, every requested residual and
each optimizer result match exactly, including full residual/Jacobian byte
digests, all parameter values, active masks, termination fields, costs and
evaluation counters. Only creation time and independently checked actual
source Git identities are excluded from equality. Input and protocol identities
stay in the comparison.

The earlier absolute-Unix-time cohort is excluded because the finite clock
steps were smaller than timestamp spacing. It exercised only two clock states
per fit. Its timings and byte hashes are retained in
`qa/cohort-selection.json`; both cohorts have identical core source manifests.
The relative-time cohort above is the sole selected speed claim. No measured
runtime was changed after that cohort.

Regression tests cover every-bit residual parity with all six parameters,
exact and signed-zero clock keys, eviction, changed inputs between fits,
memory fallback and derived-wheel callbacks. A complete default evaluation
also compares every optimizer result and analytic covariance with the scalar
path. Benchmark and archive tests reject altered histories after rebound
hashes, omitted controls, unresolved clocks, changed selection/Jacobian sizes,
wrong actual Git IDs, timing/counter errors, raw-input mutations and changed
generator source beyond CRLF/LF encoding. See the byte-indexed
[archive and validation receipts](../assets/lidar_wheel_evaluation_preparation/README.md).

All **59 distinct targeted Windows cases** pass: three existing solver/CLI,
five preparation, 20 benchmark/replay, 16 archive and 15 registry cases.
The 44 solver/CLI, preparation, benchmark and archive cases also pass on Linux
with different NumPy/SciPy versions. Strict mypy checks all 378 core modules;
Ruff, strict MkDocs and unchanged exports of all 129 public schemas plus their
ledger pass. The initial Windows archive launch's missing temporary parent and
the corrected passing run are retained with their logs.

Original packets, source files, proxy samples, times and recorded numerical
payloads remain exact. Only offline motion regeneration permits platform math
roundoff: `rtol=0`, `2e-7` m/s for velocity and `1e-12` rad/s for angular rate.
These tolerances never apply to recorded results, residuals or Jacobians.

KITTI data and generated derivatives retain **CC BY-NC-SA 3.0**, attributed to
Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun:
[KITTI](https://www.cvlibs.net/datasets/kitti/),
[data license](https://creativecommons.org/licenses/by-nc-sa/3.0/).
Source code retains Apache-2.0. This control is outside accuracy claims and
leaderboard cohorts.

## Reproduce

```bash
python -m tools.benchmark_lidar_wheel_preparation \
  --audit-only docs/assets/lidar_wheel_evaluation_preparation
```

For remeasurement, extract only regular files from `raw-oxts.tar.gz` into a
fresh directory and rename `download-provenance.json` to `provenance.json`:

```bash
python tools/benchmark_lidar_wheel_preparation.py \
  --baseline-commit 88d6fba240ed38a744bccd023111303f84b7be5b \
  --problem docs/assets/lidar_wheel_evaluation_preparation/problem.json \
  --oxts-dir outputs/wheel-remeasurement-oxts \
  --output-dir outputs/wheel-remeasurement --repetitions 3
```

The pinned generator documents the complete transform and common time origin.
To generate a fresh control from the original packets, run
`tools/prepare_lidar_wheel_control.py --oxts-dir PATH --output NEW.json`.
