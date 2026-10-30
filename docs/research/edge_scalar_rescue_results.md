# Image-only scalar rescue recovers negative-bias truth starts

Frozen at `0b0fa0bb` before evaluation on 2026-10-30. See
[protocol](edge_scalar_rescue_protocol.md),
[plan](../assets/edge_scalar_rescue_20261030/plan.json),
[all 63 results](../assets/edge_scalar_rescue_20261030/evaluation.json) and
[schema](../assets/edge_scalar_rescue_20261030/evaluation.schema.json).

Only the residual/sensitivity/joint-fit imports change from the frozen scalar
rescue algorithm. Its function AST is identical. The image-only sampler, joint
optimizer, objective, thresholds and support/rank gates remain unchanged. All
63 baseline fits rerun exactly equal to their archived records; all returned
poses are exactly the baseline poses. No oracle scene information is sampled.

| Sphere controls | Image-only sampler | Sampler + scalar rescue |
| --- | ---: | ---: |
| Perturbed pose+bias recovery | 32/36 | 32/36 |
| Truth-start pose+bias recovery | 10/18 | 16/18 |
| Median absolute bias error, all 54 | .002071 m | .001974 m |
| Baseline joint numerical termination | 54/54 | 54/54 |

Six gains and zero regressions relative to the image-only baseline. All six
negative-bias truth starts are recovered; this includes the two staggered cases
that regressed when moving from bilinear to gated sampling. Their corrected bias
errors are .001494–.005005 m. All nine parallel-plane controls remain rejected
as degenerate and never enter rescue.

Eight residual triggers lead to eight adopted corrections. Two are near_foreground
positive-bias positive-seed cases: cost and bias error improve, but error remains
.148018/.159259 m, so **both still fail**. Four other failures are unchanged:
original positive-bias positive seeds (.164649/.164893 m error) and original
positive-bias truth starts (.050904/.051026 m error). Do not round the latter into
success or adjust the .05 m gate. Pose-only accuracy passes for all 54 sphere
cases; numerical termination does not certify accurate bias estimation.

Actual optimizer work is 8–215 calls, including the baseline finite differences,
probes and scalar trials, below the unchanged 800 limit. Diagnostics are accounted
separately as in the baseline. Runtime including diagnostics is 21.47 seconds;
this is not a controlled speed comparison. Adopted corrections strictly reduce
the measured cost; all fallbacks retain baseline results. The nested baseline_fit
and rescue audit preserve both stages. Inherited status/nfev/njev describe the
joint stage; adopted results have staged joint/scalar termination, not demonstrated
joint stationarity after the bias update.

Ten focused tests, repository Ruff and strict documentation build pass. Checks
also verify schema, all frozen source hashes, the 63-case denominator, exact
baseline reproduction, unchanged poses, nonincreasing cost and hard call budget.

This closes the specific negative-bias truth-start regression on these synthetic
controls. It does not establish production readiness or SOTA; six sphere failures
remain and the data are spent. Core/defaults are unchanged. Next prioritize a
pre-registered transfer check on the existing real-data development recordings,
with the fixed image-only sampler/rescue and a comparable bilinear metric baseline.
Report all drives and both perturbation signs, truth-start drift, rejection and
runtime. Real depth has no known injected-bias truth, so do not reuse synthetic
bias-recovery claims there. Do not tune these six residual synthetic failures or
consume fresh holdouts before freezing the real-data comparison.
