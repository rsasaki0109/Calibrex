# Scalar rescue preserves baseline recovery

Protocol frozen at `cb8c265c` before evaluation on 2026-10-30. See
[protocol](scalar_rescue_protocol.md), [plan](../assets/scalar_rescue_20261030/plan.json),
[complete results](../assets/scalar_rescue_20261030/evaluation.json) and
[schema](../assets/scalar_rescue_20261030/evaluation.schema.json).

| Control | Joint baseline | Fixed-pose scalar rescue |
| --- | ---: | ---: |
| Perturbed pose+bias recovery | 33/36 | 33/36 |
| Truth-start pose+bias recovery | 11/18 | 17/18 |
| Numerically terminated baseline joint fits | 50/54 | 50/54 |

All 54 baseline fits reran exactly equal to their archived records. Every output
pose is exactly the baseline pose. Six residual probes triggered scalar search;
all six candidates were adopted. These are the six negative-bias truth starts,
but neither bias labels nor seed labels enter the decision. Six recovery gains,
zero regressions. Corrected bias errors range .000218–.006862 m. All adopted
costs decrease; all other solutions retain the baseline. Actual optimizer work,
including joint finite differences, probes and scalar evaluations, is 11–639
calls per case, below the fixed 800 limit. Runtime including diagnostics was
17.51 seconds on this environment; this is not a controlled speed comparison.

The remaining four failures are unchanged: near_foreground positive-bias positive
seed for both populations, and staggered all-return zero-bias positive seed and
truth start. They retain the original joint iteration-limit failure. Scalar
termination cannot certify convergence of an unsuccessful pose optimization.

An adopted result combines the original joint termination with a successful
fixed-pose scalar optimization. It does **not** establish joint stationarity at
the corrected bias. Result status/nfev/njev refer to the original joint stage;
`baseline_fit` preserves it, `rescue` records the scalar stage, and
`actual_residual_calls` counts both. Support is unchanged because pose is fixed;
robust sensitivity is recomputed before adoption. Cost descent alone is not an
accuracy certificate.

This supports a limited bias rescue mechanism, not production adoption or SOTA:
the sampler still knows the synthetic scene. Nine focused tests pass, including
fallback on unaccepted baseline, insufficient budget, no improvement and
candidate degeneracy. Report/schema, all frozen hashes, exact baseline equality,
pose preservation, budgets and cost monotonicity were checked.

Next prioritize the known deployability gap: freeze an image-only depth-sampling
comparison against the existing bilinear baseline. Keep this oracle result as an
upper-bound diagnostic; do not spend further iterations tuning these four oracle
failures. Include all bias signs, both seed directions, truth starts, occlusion
populations and planar degeneracy controls before any real-data adoption claim.
Separate the sampling comparison from scalar rescue to attribute changes clearly.
