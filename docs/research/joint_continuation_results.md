# Baseline-preserving continuation results

[Protocol](joint_continuation_protocol.md) frozen at `420daa35`. All 15 cases
completed, with all 15 baselines exactly reproducing the prior control. The four
probe locations are unchanged; only allocation/continuation and selection changed.

| Condition | Seed | Baseline deg / m | Returned deg / m | Pass | Selected explorer | Total calls |
| --- | ---: | --- | --- | --- | --- | ---: |
| exact_z_visible | +1 | 1.0459 / 0.2160 | 0.3358 / 0.0559 | True | 2 | 569 |
| exact_z_visible | -1 | 0.8972 / 0.1700 | 0.8972 / 0.1700 | False | baseline | 581 |
| exact_z_visible | +0 | 0.0349 / 0.0063 | 0.0349 / 0.0063 | True | baseline | 413 |
| exact_z_all | +1 | 1.0717 / 0.2349 | 0.3677 / 0.0673 | True | 2 | 509 |
| exact_z_all | -1 | 1.3748 / 0.2359 | 0.7891 / 0.1381 | False | 0 | 557 |
| exact_z_all | +0 | 0.0271 / 0.0000 | 0.8560 / 0.1427 | False | 0 | 365 |
| exact_range_visible | +1 | 0.4725 / 0.0718 | 0.2509 / 0.0395 | True | 2 | 689 |
| exact_range_visible | -1 | 0.8184 / 0.2767 | 0.5731 / 0.0875 | False | 1 | 665 |
| exact_range_visible | +0 | 0.2500 / 0.0000 | 0.2500 / 0.0000 | True | baseline | 353 |
| biased_z_visible | +1 | 1.0459 / 0.2160 | 0.3358 / 0.0559 | True | 2 | 569 |
| biased_z_visible | -1 | 0.8972 / 0.1700 | 0.8972 / 0.1700 | False | baseline | 581 |
| biased_z_visible | +0 | 0.0349 / 0.0063 | 0.0349 / 0.0063 | True | baseline | 413 |
| holes_z_visible | +1 | 1.1449 / 0.2278 | 0.3528 / 0.0718 | True | 2 | 497 |
| holes_z_visible | -1 | 1.2791 / 0.2236 | 0.5914 / 0.1134 | False | 0 | 569 |
| holes_z_visible | +0 | 0.0884 / 0.0153 | 0.0884 / 0.0153 | True | baseline | 377 |

## Recovery and limits

Perturbed recovery improves **1/10 → 5/10**, including convergence. All five
positive-direction trials now pass; all five negative-direction trials still fail.
The exact optical-z visible case improves from 0/2 to 1/2. These are dependent
conditions on one scene, and the biased-depth variant repeats the affine-invariant
exact-z objective; do not present five passes as five independent scenes.

Every refined explorer converges. Total baseline-plus-exploration work is
353–689 calls across all 15 cases, below the 800 cap. Unused calls are not spent
once the chosen explorer converges. Baseline time totals 6.51 s, additional search
10.34 s, combined 16.85 s on this CPU. Earlier equal-split budget exhaustion is
resolved by preserving state and allocating finishing work, not relaxing gates.
Six cases return the exact baseline; nine replace it with a converged,
strictly higher-MI solution. The existing baseline recovery is retained. This
preserves objective quality and convergence, not necessarily physical accuracy.

Reference-start controls worsen **5/5 → 4/5**. With all LiDAR returns, including
camera occlusions, the returned converged solution drifts **0.8560 degrees and
0.1427 m** despite higher MI. This is now a converged false solution, not a budget
artifact. The other four reference-start controls return the baseline unchanged.
The oracle-visible condition's small drift does not establish robustness to real
occlusion. Wider search can exploit a misaligned measurement objective.

## Decision

Keep this as an experimental research adapter. Do not promote or rerun real data
as though the optimizer were now reliable. The declared reference-drift control
failed. Search allocation yielded a real improvement in this frozen diagnostic,
but objective-only candidate selection still permits harmful updates.

Next pre-register a geometric validation of returned poses on a separate LiDAR
sample population in this analytic scene. Compare visible-only and all-return
conditions and keep the reference-start drift case. The validation must operate
without reference extrinsics, retain occluded/unsupported counts, and be tested
against both correct and wrong candidates; fitting and validation points must be
disjoint. Do not simply choose whichever objective reproduces known reference
scores. Synthetic validation still cannot certify real-data calibration.

[Full evidence](../assets/joint_continuation_20261029/evaluation.json) and
[schema](../assets/joint_continuation_20261029/evaluation.schema.json) retain baseline,
all exploratory states, step sizes, traces, refinement/selection and runtime.
Nine focused tests, repository Ruff, strict documentation build, frozen-hash
checks, baseline equality, budget accounting and selection validation pass.
No core solver or production defaults changed. No SOTA claim.
