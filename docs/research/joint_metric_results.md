# Direct metric-depth fitting results

[Protocol](joint_metric_protocol.md) frozen at `d6651dd8`. All 15 MI and 15 metric
fits completed. Every recomputed MI result exactly matches the original analytic
control. All metric fits satisfy numerical convergence and support retention.

| Condition | Seed | MI deg / m | MI pass | Metric deg / m | Metric pass | Residual calls | Holdout pass |
| --- | ---: | --- | --- | --- | --- | ---: | --- |
| exact_z_visible | +1 | 1.0459 / 0.2160 | False | 0.1415 / 0.0363 | True | 127 | True |
| exact_z_visible | -1 | 0.8972 / 0.1700 | False | 0.1433 / 0.0366 | True | 111 | True |
| exact_z_visible | +0 | 0.0349 / 0.0063 | True | 0.1414 / 0.0363 | True | 142 | True |
| exact_z_all | +1 | 1.0717 / 0.2349 | False | 0.7519 / 0.1054 | False | 121 | False |
| exact_z_all | -1 | 1.3748 / 0.2359 | False | 0.7518 / 0.1055 | False | 145 | False |
| exact_z_all | +0 | 0.0271 / 0.0000 | True | 0.7523 / 0.1054 | False | 142 | False |
| exact_range_visible | +1 | 0.4725 / 0.0718 | True | 0.1433 / 0.0366 | True | 139 | True |
| exact_range_visible | -1 | 0.8184 / 0.2767 | False | 0.1433 / 0.0366 | True | 134 | True |
| exact_range_visible | +0 | 0.2500 / 0.0000 | True | 0.1414 / 0.0363 | True | 142 | True |
| biased_z_visible | +1 | 1.0459 / 0.2160 | False | 0.3831 / 0.2222 | False | 202 | False |
| biased_z_visible | -1 | 0.8972 / 0.1700 | False | 0.3843 / 0.2224 | False | 144 | False |
| biased_z_visible | +0 | 0.0349 / 0.0063 | True | 0.3843 / 0.2224 | False | 158 | False |
| holes_z_visible | +1 | 1.1449 / 0.2278 | False | 0.1091 / 0.0329 | True | 132 | True |
| holes_z_visible | -1 | 1.2791 / 0.2236 | False | 0.1099 / 0.0331 | True | 157 | True |
| holes_z_visible | +0 | 0.0884 / 0.0153 | True | 0.1025 / 0.0318 | True | 127 | True |

## Interpretation

Perturbed recovery improves **MI 1/10 → metric 6/10**, versus 5/10 for the separately
recorded continuation experiment. Exact visible optical z, correctly converted
radial depth and missing-rectangle conditions each recover **2/2**. Both signed
initializations reach similar solutions, unlike continuation's positive-only
successes. Visible exact-depth errors remain about 0.14 degrees / 3.6 cm: analytic
point intersections and pixel-interpolated depth are not identical near silhouettes,
and no claim of an unbiased/global optimum is established.

The all-return camera-occlusion condition recovers **0/2**, with approximately
0.752-degree rotation error. The +0.25 m depth bias condition recovers **0/2**, with
approximately 0.222 m translation error. Reference-start controls fall from MI's
5/5 to **3/5**: those same two conditions drift outside the pose gates even from
truth. All six of these failed metric outputs nevertheless converge numerically
and retain support. Robust loss alone does not solve visibility mismatch or
measurement bias, and numerical success remains unsafe as an accuracy criterion.

The frozen held-out consistency checks pass all nine accurate metric outputs and
reject all six inaccurate ones in this run. This descriptive agreement does not
repair the validator's previously demonstrated false accepts on other candidates.
It must not be promoted or treated as independent ground truth. The same scene
and camera-depth generation underpin fitting and diagnostic samples.

Actual residual evaluations (including numerical Jacobians) are **111–202**, below
the hard 800 ceiling. No retries, timeouts, budget failures or support failures.
Total measured fitting time for 15 cases: MI **6.39 s**, metric **4.12 s** on this
CPU. Different objectives, interpolation, optimizers and effective caps mean this
is not an equal-work solver benchmark or an isolated objective ablation. The
metric adapter also computes support counts; held-out checks are outside its timed
solve block. Scene/conditions are dependent development controls, not independent
real-world successes.

## Decision

Retain the metric estimator as experimental. No production promotion or SOTA
claim: the declared occlusion and reference-drift controls still fail. Preserve
both successes and the six converged inaccurate results. Do not soften the
0.20 m gate for the 0.222 m bias outcome or prune points using reference visibility.

The next investigation should explicitly separate unobserved/occluded geometry
from camera-depth bias, with predeclared controls on additional analytic scenes.
A physically motivated visibility model must be tested without oracle extrinsics;
a bias parameter must be checked for confounding with extrinsic translation.
Do not add both mechanisms at once or tune them against held-out acceptance labels.
First establish whether an occlusion-aware metric objective preserves visible-only
recovery while avoiding the all-return truth-start drift, then address bias.

[Full results](../assets/joint_metric_20261029/evaluation.json) and
[schema](../assets/joint_metric_20261029/evaluation.schema.json) retain every status,
pose, support count, residual-call count, runtime and held-out diagnostic. Seven
focused tests, repository Ruff, strict documentation build, report/source-hash
validation and exact baseline reproduction pass. Core code/defaults are unchanged.
