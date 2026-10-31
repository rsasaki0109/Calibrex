# Disjoint geometric validation results

[Protocol](joint_geometric_validation_protocol.md) frozen at `3249838a`. All
40 candidate checks (120 frame checks) completed without rerunning calibration.
All validation clouds are disjoint from fitting points and retain camera occlusions.

| Condition | Candidate | Geometry pass | Pose gate pass | Worst frame median m | Worst frame p90 m |
| --- | --- | --- | --- | ---: | ---: |
| exact_z_visible | reference_control | True | True | 0.0039 | 0.0211 |
| exact_z_visible | wrong_control | False | False | 0.5000 | 11.6051 |
| exact_z_visible | baseline_1 | False | False | 0.0817 | 0.2105 |
| exact_z_visible | continuation_1 | True | True | 0.0329 | 0.1085 |
| exact_z_visible | baseline_-1 | False | False | 0.0615 | 0.1823 |
| exact_z_visible | continuation_-1 | False | False | 0.0615 | 0.1823 |
| exact_z_visible | baseline_0 | True | True | 0.0047 | 0.0229 |
| exact_z_visible | continuation_0 | True | True | 0.0047 | 0.0229 |
| exact_z_all | reference_control | True | True | 0.0039 | 0.0211 |
| exact_z_all | wrong_control | False | False | 0.5000 | 11.6051 |
| exact_z_all | baseline_1 | False | False | 0.1209 | 0.2622 |
| exact_z_all | continuation_1 | True | True | 0.0371 | 0.1200 |
| exact_z_all | baseline_-1 | False | False | 0.0692 | 0.2331 |
| exact_z_all | continuation_-1 | False | False | 0.0594 | 0.1849 |
| exact_z_all | baseline_0 | True | True | 0.0045 | 0.0213 |
| exact_z_all | continuation_0 | True | False | 0.0475 | 0.1546 |
| exact_range_visible | reference_control | True | True | 0.0039 | 0.0211 |
| exact_range_visible | wrong_control | False | False | 0.5000 | 11.6051 |
| exact_range_visible | baseline_1 | True | True | 0.0307 | 0.0929 |
| exact_range_visible | continuation_1 | True | True | 0.0168 | 0.0480 |
| exact_range_visible | baseline_-1 | False | False | 0.2006 | 0.2966 |
| exact_range_visible | continuation_-1 | True | False | 0.0358 | 0.1020 |
| exact_range_visible | baseline_0 | True | True | 0.0052 | 0.0300 |
| exact_range_visible | continuation_0 | True | True | 0.0052 | 0.0300 |
| biased_z_visible | reference_control | False | True | 0.2502 | 0.2613 |
| biased_z_visible | wrong_control | False | False | 0.3361 | 11.7582 |
| biased_z_visible | baseline_1 | False | False | 0.3250 | 0.4600 |
| biased_z_visible | continuation_1 | False | True | 0.2269 | 0.2789 |
| biased_z_visible | baseline_-1 | False | False | 0.2630 | 0.3935 |
| biased_z_visible | continuation_-1 | False | False | 0.2630 | 0.3935 |
| biased_z_visible | baseline_0 | False | True | 0.2502 | 0.2630 |
| biased_z_visible | continuation_0 | False | True | 0.2502 | 0.2630 |
| holes_z_visible | reference_control | True | True | 0.0039 | 0.0207 |
| holes_z_visible | wrong_control | False | False | 0.5383 | 11.7145 |
| holes_z_visible | baseline_1 | False | False | 0.0763 | 0.1985 |
| holes_z_visible | continuation_1 | True | True | 0.0283 | 0.0696 |
| holes_z_visible | baseline_-1 | False | False | 0.0695 | 0.2014 |
| holes_z_visible | continuation_-1 | False | False | 0.0562 | 0.1558 |
| holes_z_visible | baseline_0 | True | True | 0.0078 | 0.0262 |
| holes_z_visible | continuation_0 | True | True | 0.0078 | 0.0262 |

## The proposed gate fails its key control

The all-return continuation from the true pose, already known to drift by
0.8560 degrees / 0.1427 m, **passes** the new validation. Its worst-frame absolute
median is about 0.048 m and p90 0.155 m. The exact-range negative-seed continuation
also passes despite 0.5731-degree rotation error (median 0.036 m, p90 0.102 m).
Disjoint points and small aggregate depth residuals therefore do not establish
full-pose accuracy even in this noiseless analytic scene.

Across the 40 checks, 20 meet the pose gates and 20 do not. The validator accepts
16 correct poses and **2 incorrect poses**, rejects 18 incorrect poses and
**4 correct poses**. All four correct-pose rejections belong to the +0.25 m biased
camera-depth condition, including its true-pose control. Thus a failed consistency
check cannot distinguish extrinsic error from measurement bias. Every deliberately
large wrong-pose control is rejected, but that easy success does not rescue the
nearby false accepts. These counts are descriptive, not independent statistical
rates: candidates repeat across initializations and conditions in one shared scene.

The same depth maps and scene geometry underpin optimization and validation.
Disjoint sampling prevents point reuse but not shared systematic errors or weak
constraints. Depth median/p90 statistics are not rotation/translation uncertainty
bounds. Behind/in-front residual counts remain in the report; they are diagnostic
signs of disagreement, not proven occlusion labels. No residuals were silently
removed to obtain passing scores.

## Decision

Do not promote this validator, relax pose gates or tune its .05/.20 m thresholds
to reject the two observed failures. In particular, do not claim the continuation
optimizer is safe because it passed a geometric holdout. The declared drift check
failed. No real-data rerun is warranted under that claim.

The next substantive algorithmic test should compare directly fitting metric
geometry against maximizing MI, on the same original fitting samples and fixed
initial poses, with these holdout samples reserved for diagnostics. Keep depth
bias and camera occlusion conditions; they expose different failure modes. Freeze
a bounded robust metric estimator and its budget before testing, retain the MI
baseline and full-pose labels, and do not let held-out thresholds select fitting
parameters. This is a change of calibration objective, not another acceptance
threshold variant. Any synthetic improvement still needs broader independent
geometry and real-world evidence before promotion.

[Evidence](../assets/joint_geometric_validation_20261029/evaluation.json) and
[schema](../assets/joint_geometric_validation_20261029/evaluation.schema.json)
include all support counts and metrics. Tests cover depth units, deliberate metric
error, missing support and spatial disjointness. Six focused tests, repository
Ruff, strict docs build, frozen-hash and report/population checks pass. Production
code/defaults and previous experimental verdicts remain unchanged. No SOTA claim.
