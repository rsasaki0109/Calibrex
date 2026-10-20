# Native prior-information whitening development

This correctness and development record supports the
[reliability study](../../research/continuous_time_reliability.md#2-complete-state-derivatives).
It uses existing spent synthetic fixtures, with no execution-speed,
independent accuracy or general convergence claim.

## Correct information quadratic

For the existing zero-centered prior, `delta = Log(anchor^-1 * knot)`, the
declared objective is `delta.T @ information @ delta`. Whitened residuals
`W @ delta` and Jacobians therefore require `W.T @ W = information`.
The previous lower Cholesky and eigen fallback used the opposite orientation;
the fallback also inserted a positive `1e-9` floor into every null direction.

The corrected factor equilibrates by the positive diagonal, then uses
`diag(sqrt(eigenvalues)) @ eigenvectors.T` and restores the coordinate scales.
It retains positive weak information and exact zero-coordinate information,
without adding a floor. Matrices must be finite, real and 6x6, with a
nonnegative diagonal and exact zero rows/columns at zero diagonal entries.
Symmetry and spectral checks use 60 machine epsilons times the respective
normalized norm. Correlations beyond one by more than that roundoff allowance
are rejected. Only tiny negative normalized spectral roundoff is projected.
Negative diagonal entries are rejected, including roundoff-sized negatives;
implicit Schur repairs are outside this input contract.

Of 79 maintained regression checks, 60 fail against the preceding source.
They cover the declared quadratic, independent objective gradients, both
actual assembly paths, null and weak information, nonidentity poses, mixed
coordinate units, global scales from `1e-200` to `1e200`, malformed/nonfinite/
complex inputs and invalid weak directions hidden by a much larger diagonal.
Null-space energy is checked at the Gram matrix's machine-precision scale;
the positive spectral square root itself has square-root-precision uncertainty
near rank deficiency.

## Frozen sources and complete records

The production numerical source is
[`ac7f90b6f584d23928a4ff56ed6ee1661e5f14b5`](https://github.com/rsasaki0109/Calibrex/commit/ac7f90b6f584d23928a4ff56ed6ee1661e5f14b5).
The historical production control is source
[`8861ed65ebbcffb13a182ac4cd549db88ef763f2`](https://github.com/rsasaki0109/Calibrex/commit/8861ed65ebbcffb13a182ac4cd549db88ef763f2),
whose tree is identical to the main merge immediately before this change.
Only `continuous_time_marginalization.py` differs among the 401 Python source
modules. The [manifest](source-manifest.json), [receipt](receipt.json),
[receipt schema](receipt.schema.json) and [original producer](generator.py.txt)
bind the source, runtime, commands and original bytes.

The [complete archive](complete-development.tar.gz) retains all 49 declared
files: historical production and the earlier private candidate, their inputs,
all six historical fitter results and schemas, original producers/helpers,
current inputs, all three current fitter results, direct and recorded API
results, schemas, retained normal inputs and source manifests. Nothing from
the historical two-attempt screen is discarded. The earlier private candidate
is labelled historical and is separate from the current production result.
Every archive member is indexed by size and SHA-256 in the receipt.

OpenBLAS/MKL/OMP limits were one; concurrent host activity was allowed.
The receipt records the installed Python, NumPy, SciPy and Pydantic versions.
The Log finite differences, prior centering, solver limits and quality gates
are unchanged. Artifact schemas and historical evidence are retained.

## Same-input window comparison

The first fixture is the existing `test_sliding_window_matches_batch_overlap`
case, seed 7: six knots, 90 plane measurements, windows of four and three
knots, one overlap knot and the existing 80-iteration setting. All three
problem/options pairs are exactly equal to the historical production records.
The prior-free batch and first-window results are also exactly equal.

| Actual fitter call | Historical production iterations | Corrected production iterations |
| --- | --- | --- |
| Batch | 7 | 7 |
| First window, no prior | 5 | 5 |
| Second window, correlated prior | 26 | 7 |

| Development metric | Historical production | Corrected production |
| --- | --- | --- |
| Existing unit quality gate | pass | pass |
| Maximum batch/sliding overlap translation (m) | 0.00363866510 | 0.00116506884 |
| Maximum batch/sliding overlap rotation (rad) | 0.00523790851 | 0.00143682083 |
| Maximum physical trajectory translation error (m) | 0.02449377651 | 0.02449377651 |
| Maximum trajectory rotation error (rad) | 0.02301490181 | 0.02073100880 |
| Second-window declared objective | 0.000145432367 | 0.000134759966 |

The recorder passes through the maintained production fitter without changing
its numerical functions. A separate, uninstrumented public API call produces
an exactly equal complete result; both calls are retained. The current
aggregate sliding result copies some batch statistics, so the table uses the
actual per-window fit and independently recomputed objective. Iteration counts
are not latency measurements, and this fixture has no holdout or signed control.

## Native holdout and known-bad control

The separate [complete native result](native-seed-20260820.json), seed 20260820,
passes the existing quality policy with 63 training and 27 holdout measurements.
Batch holdout RMSE is 0.0140859652 m and sliding holdout RMSE is 0.0215409233 m.
The existing known-bad skipped-marginalization/shifted-overlap control increases
RMSE by 0.0223417821 m and is detected under the unchanged gate. Batch and
sliding fit labels are `converged`. This supplementary fixture does not replace
the nonzero correlated-information comparison above.

These are development cases. Total maximum translation error is unchanged in
the same-input case. The native `converged` label can still mean a small damped
step, and does not establish undamped stationarity. The packing helpers still
drop nonzero marginalized gradients and cross-knot blocks; that approximation
is separate from whitening. Clock boundaries and convergence remain audit work
before the fixed-budget research comparison.

## Reproduce

At the pinned numerical source, set `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `PYTHONDONTWRITEBYTECODE=1`.
The native command is:

```bash
calibrex trajectory recover-sliding-window --seed 20260820 --output replay-native.json --json
```

For the complete current comparison, copy the original producer to a fresh
`outputs/prior-replay/run.py`, keep a copy of the complete archive, and run:

```bash
python outputs/prior-replay/run.py --historical-archive complete-development.tar.gz --output-dir replay/prior-information
```

The archive supplies all historical inputs directly, without extracting or
rerunning the earlier candidate. The output directory must be fresh and inside
the checkout; no result is overwritten. Timestamps, output paths and platform
floating-point behavior can differ on replay. Historical replay requires its
pinned source and the archived original helper files at the working paths
declared by its producer.

## Integrity

| Artifact | SHA-256 |
| --- | --- |
| complete-development.tar.gz | `0d75dbd3d009b0c63fec818fe71ad8aa000b2c1c7ae8d016dcd64bfdebfe717a` |
| native-seed-20260820.json | `b358e7cf3b32ae6790d5bf4b969bdd0ed2ff1d0bb252bb8fde9c9fea1c615541` |
| source-manifest.json | `8c0c0f3743892b71479ea1381cbce212b288d8e08cc28a294ed1caf44236ed25` |
| generator.py.txt | `3648e386331069158d2aa9742bd327162b9aa4ad6dc4e77c8699c7349c16f6e3` |

Maintained tests validate the complete member index, every current snapshot,
both historical attempts, native schema/control, exact input equality,
unaffected phase results, transparent recording and original source bindings.
