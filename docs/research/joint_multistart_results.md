# Fixed-budget multistart results

[Protocol](joint_multistart_protocol.md) frozen at `b221d707`. All 15 baseline
fits and 75 restart fits completed. Baseline result objects exactly reproduce
the preceding known-geometry study.

| Condition | Seed | Selected start | Baseline deg / m | Five-start deg / m | Selected converged | Used calls baseline / five-start |
| --- | ---: | ---: | --- | --- | --- | --- |
| exact_z_visible | +1 | 3 | 1.0459 / 0.2160 | 0.3172 / 0.0559 | False | 229 / 800 |
| exact_z_visible | -1 | 2 | 0.8972 / 0.1700 | 0.5651 / 0.1500 | False | 253 / 800 |
| exact_z_visible | +0 | 0 | 0.0349 / 0.0063 | 0.0349 / 0.0063 | True | 121 / 758 |
| exact_z_all | +1 | 3 | 1.0717 / 0.2349 | 0.3661 / 0.0637 | False | 193 / 800 |
| exact_z_all | -1 | 1 | 1.3748 / 0.2359 | 0.7842 / 0.1381 | False | 253 / 800 |
| exact_z_all | +0 | 3 | 0.0271 / 0.0000 | 0.8646 / 0.1420 | False | 109 / 734 |
| exact_range_visible | +1 | 3 | 0.4725 / 0.0718 | 0.2526 / 0.0515 | False | 337 / 800 |
| exact_range_visible | -1 | 2 | 0.8184 / 0.2767 | 0.6125 / 0.0750 | False | 265 / 800 |
| exact_range_visible | +0 | 0 | 0.2500 / 0.0000 | 0.2500 / 0.0000 | True | 73 / 713 |
| biased_z_visible | +1 | 3 | 1.0459 / 0.2160 | 0.3172 / 0.0559 | False | 229 / 800 |
| biased_z_visible | -1 | 2 | 0.8972 / 0.1700 | 0.5651 / 0.1500 | False | 253 / 800 |
| biased_z_visible | +0 | 0 | 0.0349 / 0.0063 | 0.0349 / 0.0063 | True | 121 / 758 |
| holes_z_visible | +1 | 3 | 1.1449 / 0.2278 | 0.3594 / 0.0718 | False | 193 / 800 |
| holes_z_visible | -1 | 2 | 1.2791 / 0.2236 | 0.6131 / 0.1118 | False | 193 / 800 |
| holes_z_visible | +0 | 0 | 0.0884 / 0.0153 | 0.0884 / 0.0153 | True | 133 / 746 |

## Outcome

**Accepted perturbed recovery falls from 1/10 to 0/10.** All ten selected multistart
runs hit their 160-call limit. Ignoring convergence, five poses satisfy the geometric
thresholds versus one baseline pose; these five are explicitly not accepted.
Every perturbed five-start trial consumes 800 calls while baseline trials terminate
in 193–337. This is equal maximum budget, not equal used work. Measured total times
for the ten perturbed trials are 5.23 s baseline versus 17.71 s multistart on this
CPU. Reference-start controls take 1.20 s versus 7.69 s in total.

The selected MI improves versus baseline in 7/10 perturbations and worsens in 3/10:
the original start also receives only 160 calls, so it cannot reproduce its fully
converged baseline. This allocation neither preserves the best baseline result
nor gives the promising alternative starts enough finishing budget. The observed
geometric improvement is a useful search signal, not validation of the algorithm.

Reference-start controls pass 5/5 baseline versus 4/5 multistart. In the all-return
condition, a higher-MI alternative is selected with 0.8646 degrees / 0.1420 m error
rather than the near-truth baseline (0.0271 degrees / 0 m). The selected run is
unfinished and correctly rejected. Its MI already exceeds the converged baseline
by 0.00364, illustrating why higher MI alone cannot certify true calibration.
The other four reference-start outputs match their baseline poses.

## Decision and next test

Do not promote this equal-split multistart strategy or run it on real data as an
improvement. Keep all attempts and the failed convergence gates. A next controlled
experiment should preserve the fully converged baseline, then spend only the
remaining declared budget on exploration/refinement with explicit continuation
state. Budget must count the baseline and every exploratory call. A better objective
cannot by itself override independent pose/visibility validation; the all-return
reference-start drift remains a required rejection control. Freeze any allocation
rule before evaluation, and do not tune start positions against these results.

[Full evidence](../assets/joint_multistart_20261029/evaluation.json) includes every
start, trace, selection and runtime; [schema](../assets/joint_multistart_20261029/evaluation.schema.json)
records the report envelope. Seven focused unit tests, repository Ruff, strict docs
build, source-hash validation and report/budget/selection checks pass. Neither the
core solver nor production defaults changed; no SOTA or real-data improvement claim.
