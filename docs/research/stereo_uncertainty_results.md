# First-order depth uncertainty normalization fails recovery controls

Frozen at `2eb1336d` before evaluation on 2026-10-30. See
[protocol](stereo_uncertainty_protocol.md),
[plan](../assets/stereo_uncertainty_20261030/plan.json),
[all 342 fits](../assets/stereo_uncertainty_20261030/evaluation.json) and
[schema](../assets/stereo_uncertainty_20261030/evaluation.schema.json).

**Reject this normalization candidate.** It does not recover noisy perturbed
poses and substantially regresses the clean controls. This is a test of one
specific measured-depth, first-order uncertainty surrogate, not a rejection of
range-dependent stereo uncertainty as a physical phenomenon.

| Noise condition | Start | Fixed joint pose+bias recovery | Normalized recovery |
| --- | --- | ---: | ---: |
| Clean | Perturbed | 12/36 | 1/36 |
| Clean | Reference | 6/18 | 0/18 |
| .5 px, seed 730 | Perturbed | 0/36 | 0/36 |
| .5 px, seed 730 | Reference | 0/18 | 3/18 |
| .5 px, seed 731 | Perturbed | 0/36 | 0/36 |
| .5 px, seed 731 | Reference | 0/18 | 5/18 |

On clean sphere cases there is one gain and **18 regressions**, across perturbed
and reference starts. All 63 clean fixed-baseline outputs reproduce the archived
joint_pose_bias records exactly. Paired methods use byte-identical generated depth
arrays, whose hashes are retained in every row. Thus the clean regression is not
explained by a changed baseline, geometry or noise generation.

Noisy perturbed pose-only recovery is also 0/72 for both methods. Normalization
improves noisy reference-start correction in eight cases, but reference starts
are controls, not successful recovery from perturbation. Across noisy reference
starts, median bias error changes from .5000 m for the fixed method to .05646 m
(seed 730) and .05648 m (731) for the candidate, still beyond the .05 m gate in
many cases. On clean perturbed starts, median bias error worsens .05729→.07060 m.
Do not replace joint accuracy with favorable bias-only statistics.

## Convergence and degeneracy

| Condition/start | Fixed convergence/support | Normalized convergence/support |
| --- | ---: | ---: |
| Clean perturbed | 36/36 | 35/36 |
| Clean reference | 18/18 | 18/18 |
| Seed 730 perturbed | 25/36 | 0/36 |
| Seed 730 reference | 17/18 | 15/18 |
| Seed 731 perturbed | 25/36 | 2/36 |
| Seed 731 reference | 18/18 | 18/18 |

Both methods flag all nine clean parallel-plane controls as degenerate and reject
them. Each method uses 8–800 actual optimizer calls; all caps hold, including the
finite-difference evaluations. Total solve time including diagnostics was 136.36 s
for fixed and 198.29 s for normalized; report serialization is excluded. This is
not a controlled performance benchmark. All failed/nonconverged cases remain.

## Interpretation and next decision

The candidate uses measured depth squared to approximate the variance induced by
.5 px disparity noise, with a .1 m floor. It changes the effective metric robust
loss and positive-occlusion scales as well as downweighting far measurements.
Weights depend on projected/interpolated measurements. Therefore it is not an
exact Gaussian disparity likelihood, and its behavior cannot be attributed to a
single isolated change in covariance. The candidate assumes .5 px on clean data
too; the clean controls explicitly test robustness to that mismatch.

The test confirms that simply normalizing metric residuals is insufficient under
this protocol. It does not identify weighting, interpolation, asymmetric loss or
nonlinear optimization as the sole cause, and it does not test correlated stereo
errors, bad matches or a calibrated real sensor noise model. No real recordings
were used to fit scales and no new real data were consumed.

Seven focused tests, repository Ruff and strict documentation build pass. Checks
validate report/schema, all frozen source hashes, 342-case completion, paired depth
hashes, budgets and exact 63-case clean baseline reproduction. Core/defaults and
the real-data MI baseline remain unchanged; no adoption or SOTA claim.

Do not retune the floor or assumed sigma on these outcomes. Before another real
transfer, the next distinct hypothesis is a direct disparity-domain measurement
model, avoiding depth inversion's nonlinear error and measured-depth weighting.
Specify it independently and freeze the same clean/noisy/occlusion/bias controls
before running; retain separate numerical convergence and accuracy reporting.
These results provide no evidence that the proposed next model already works.
