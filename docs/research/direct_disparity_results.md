# Direct disparity improves bias controls but not pose recovery

Frozen `c577773e`, evaluated 2026-10-30. See [protocol](direct_disparity_protocol.md),
[plan](../assets/direct_disparity_20261030/plan.json),
[171 results](../assets/direct_disparity_20261030/evaluation.json) and
[schema](../assets/direct_disparity_20261030/evaluation.schema.json).
Baseline is the archived fixed-metric method from the preceding uncertainty study.

| Condition | Start | Fixed metric joint recovery | Direct disparity |
| --- | --- | ---: | ---: |
| Clean | Perturbed | 12/36 | 12/36 |
| Clean | Reference | 6/18 | 6/18 |
| Noise seed 730 | Perturbed | 0/36 | 0/36 |
| Noise seed 730 | Reference | 0/18 | 12/18 |
| Noise seed 731 | Perturbed | 0/36 | 0/36 |
| Noise seed 731 | Reference | 0/18 | 12/18 |

All clean candidate cases retain the baseline's pass/fail classification: no gains
or regressions. There are 24 noisy reference-start gains and no recovery regressions
across all sphere cases. These controls do not count as recovery from perturbation.
All 72 noisy perturbed candidate fits fail numerical convergence/support and none
passes pose-only recovery. Noisy reference controls converge in 15/18 and 18/18;
clean controls converge in 54/54. All nine parallel-plane controls are rejected.

Candidate work is 8–800 actual residual calls per fit; total solve time including
diagnostics is 377.32 seconds. Budgets are unchanged; separate diagnostic passes
and serialization are excluded. No caps or accuracy gates were relaxed.

The candidate bilinearly samples disparity and compares it with 120/(z+bias),
using .2 residual units per disparity pixel before the existing asymmetric robust
loss. This avoids measured-depth-squared weighting, but changes interpolation
space together with the measurement residual. It is not an isolated sampler test
or an exact Gaussian likelihood after saturation. Noise with a metric offset
added after inversion also differs from constant-variance observed disparity.
Improved reference-start bias recovery does not prove a globally useful objective
or establish why the perturbed fits fail.

Five focused tests, repository Ruff and strict documentation build pass. Validation
checks schema, frozen hashes, the full 171-case count, every candidate depth hash
against its paired archived baseline, budgets and all plane rejections. Core and
production defaults remain unchanged. No real data or fresh holdouts were used.

Do not adopt or transfer this candidate to real data yet. The depth-weighting
candidate's clean regression is avoided, but pose recovery under noise remains
unresolved. Before another residual variant, freeze an offline objective landscape
comparison at reference and failed fitted poses, with bias controlled, on these
same paired noise realizations. Determine whether reference is actually preferred
before choosing a different search method. Do not merely raise iteration caps or
count inaccurate/nonconverged fits as successes. No SOTA claim.
