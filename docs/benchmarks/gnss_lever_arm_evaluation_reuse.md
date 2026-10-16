# Reusing and batching GNSS–LiDAR lever-arm evaluation

The complete prepared-input GNSS–LiDAR evaluation falls from **13.871298 to
4.896686 seconds (2.832793x)**. The phase from prepared input loading through
official result saving falls from **11.028193 to 2.085993 seconds (5.286783x)**.
These are ratios of three-run medians on a public-trajectory execution control,
with unchanged default fitting, uncertainty, holdout and known-bad controls.

## Scope and input

The reference is all **2,762 original KITTI OXTS poses** from
`2011_09_30_drive_0018_sync`. The control declares a body-to-LiDAR transform,
an antenna lever arm, a 12 ms GNSS clock shift and seeded GNSS/odometry noise.
It retains the full natural drive in **29 non-overlapping ten-second windows**,
using the segmenter's default minimum of 20 poses. Every third window is held
out: **19 training windows and ten holdout windows**. No window, pair, robust
iteration, optimizer evaluation, jackknife fit or policy control is removed.

The GNSS measurements and odometry are **generated** from this public trajectory.
This measures prepared-input calibration/evaluation performance and numerical
parity. It does not measure bag reading, raw LiDAR registration, GPU execution
or independent calibration accuracy. Segment scan counts describe the declared
generated pose epochs. The unchanged result remains **inconclusive** for the
lever arm on this mostly planar drive; the observable clock component and the
unobservable translation axes retain the same evidence and policy decisions.

## Implementation

Each solve owns prepared GNSS displacement pairs, noise terms and ordered array
indices. An exact binary-clock LRU holds 64 interpolation states and a second
LRU holds 256 normalized data residuals. Parameter values and signed zeros are
never rounded. Returned cached arrays are read-only; each IRLS pass still
recomputes robust weights, variance and anchors. Each jackknife solve owns its
own context.

For ordered, disjoint windows, one SciPy SLERP and three NumPy interpolation
calls handle all the epochs of a trial clock state. Every query remains inside
its original window. An out-of-window observability difference follows the
original per-window path and preserves its error behavior. Overlapping,
unordered or ambiguous windows retain per-window interpolation. Independent
3x3 Procrustes SVDs are batched; each window's covariance and final matrix
products keep their original shapes and reduction order. The offset-dependent
pose-noise blend and lever-dependent displacement noise remain in the model.

A conservative **64 MiB** estimate covers both LRUs, prepared arrays, the
combined interpolator and working arrays. Larger inputs retain the original
scalar evaluator. Public APIs, defaults and all core schemas are unchanged.

| Work per complete evaluation | Prior | Reused/batched |
| --- | ---: | ---: |
| Main plus jackknife solves | 9 | 9 |
| Residual requests, including held-out checks | 1,542 | 1,542 |
| Least-squares results | 36 | 36 |
| Per-window interpolation calls | 26,029 | 138 |
| Combined interpolation calls | 0 | 600 |
| Interpolated epochs, including repeated states | 2,368,419 | 935,169 |
| Procrustes SVD dispatches | 26,000 | 1,258 |
| Window alignments evaluated after residual reuse | 26,000 | 19,532 |

The audit independently replays both LRUs from every recorded parameter state,
then checks interpolation batches, every queried epoch and SVD dispatches
against the observed counters. Repeated mathematical work is reused; the
requested residual sets, optimizer budgets and evaluation evidence are identical.

## Measurements and validation

Each variant ran in three fresh frozen processes, alternating execution order,
with one BLAS thread. The host was not reserved. Whole-process timing includes
imports, actual Git checks, input construction, complete evaluation and trace
instrumentation, artifact saving, trace compression and imported-source hashes.
Control generation, freezing, raw-input compression and external audits are
untimed.

| Pair (execution order) | Prior whole s | Reused whole s |
| --- | ---: | ---: |
| 1 (prior, reused) | 13.770146 | 4.896686 |
| 2 (reused, prior) | 14.099818 | 4.875520 |
| 3 (prior, reused) | 13.871298 | 4.954276 |

Baseline: `01127d3a8e6f830d99786d183dbef6a311e50c14`.
Measured source: `1c91f5a5bec5f6324be5ca10f652bf863bc04b02`.
Only `gnss_lever_arm_solver.py` differs in the complete core source manifests.
All six complete public results, every residual parameter/digest and every
least-squares result match exactly, including residual/Jacobian digests,
termination messages, costs and evaluation counters. Comparison excludes only
creation time and the independently verified actual source Git identities.
Input and protocol identities remain compared.

Regression coverage includes clock modes, noise terms, unequal windows,
rank-deficient/reflected alignments, cache bounds and signed zeros, mutable
inputs between solves, memory fallback, exact endpoints, interval gaps and
out-of-window differences. Complete block evaluation and every optimizer result
are compared with the original evaluator. Archive mutation checks cover
histories beyond rebound hashes, counters, epochs, timing, raw pins and actual
Git IDs. Validation logs and the two repaired endpoint fixture attempts are
retained in the [portable archive](../assets/gnss_lever_arm_evaluation_reuse/README.md).
All **76 distinct Windows cases** pass: 13 existing solver/evaluation, 18 reuse,
18 benchmark/replay, 12 archive and 15 registry cases. The 36 reuse/benchmark
cases and the complete offline archive audit also pass on Linux with a
different NumPy/SciPy version. Strict mypy checks all 378 core modules; Ruff,
strict MkDocs and unchanged exports of all 129 public schemas plus their
ledger also pass.

Raw packet and source bytes are hash-exact. Only offline geometry regeneration
allows platform math differences: `rtol=0`, `2e-8` m translation and `1e-13`
rotation-matrix elements. Epoch timestamps, window IDs and all recorded
result/residual/Jacobian comparisons remain exact.

KITTI data and its generated control retain **CC BY-NC-SA 3.0** attribution to
Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun; see the
[KITTI dataset](https://www.cvlibs.net/datasets/kitti/) and
[data license](https://creativecommons.org/licenses/by-nc-sa/3.0/).
Code retains Apache-2.0. This performance control is outside accuracy claims
and leaderboard cohorts.

## Reproduce

Audit offline from the repository root:

```bash
python -m tools.benchmark_gnss_lever_arm_reuse \
  --audit-only docs/assets/gnss_lever_arm_evaluation_reuse
```

For remeasurement, extract only regular files of `raw-oxts.tar.gz` into a fresh
directory and rename `download-provenance.json` to `provenance.json`. Then:

```bash
python tools/benchmark_gnss_lever_arm_reuse.py \
  --baseline-commit 01127d3a8e6f830d99786d183dbef6a311e50c14 \
  --problem docs/assets/gnss_lever_arm_evaluation_reuse/problem.json \
  --oxts-dir outputs/gnss-remeasurement-oxts \
  --output-dir outputs/gnss-remeasurement --repetitions 3
```

The pinned `control-generator.py.txt` documents every transform, cut and noise
draw. To generate a fresh equivalent control from the original packets, run
`tools/prepare_gnss_lever_arm_control.py --oxts-dir PATH --output NEW.json`.
