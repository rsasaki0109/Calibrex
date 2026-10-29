# Depth-bias response and local confounding protocol

Implement stages A/B of the [Japanese review](development_brief_ja.md), without
adding a fitted bias parameter. Freeze the successful one-sided visibility treatment
from [occlusion controls](occlusion_pose_results.md); do not tune saturation or gates.

Use all three existing sphere arrangements, both visible-only and all-return fitting
populations, camera optical-depth offsets 0,+.25,-.25 m, both original perturbations
and separate exact-reference initialization: **36 perturbed fits + 18 truth-start
controls**. Only camera depth changes; points, scene, intrinsics and seeds remain
identical. These are dependent synthetic development conditions, not real data.

Run unchanged six-DoF solver/options: ≤100 nfev and hard ≤800 actual residual calls,
same convergence/support and <.5 deg/<.20 m gates. No fitted depth-bias term yet.
Record every output, signed xyz translation error, status/support, actual calls,
time and drift. Reproduce all 18 zero-bias fits from the previous experiment.
Truth is used only for scene generation, initial perturbations and final labels,
not the estimator or ambiguity diagnostic.

At each returned pose, independently diagnose local ambiguity using raw supported
metric residuals, NOT robust/saturated cost. Compute six central-difference pose
columns with .001 degree/.001 m steps; use the exact intersection of supported
points across the central and 12 perturbed evaluations. Bias correction b would
add b to the residual, so its derivative is +1. Scale columns by 1 degree for
rotation, .1 m for translation and .1 m for bias. Project the bias column onto the
six pose columns by least squares (rcond=1e-10). Report the unexplained norm fraction,
the corresponding pose perturbation and all seven scaled singular values.

Flag fraction <.05 as a provisional local near-confounding diagnostic, not an
acceptance gate or global identifiability proof. Require >=100 common points for
availability. Count the 13 additional full-batch diagnostic passes separately;
they do not enter optimizer budget or fitting time. Include all supported points,
no oracle or residual-based outlier filtering in this diagnostic. Pixel/depth
boundaries, visibility and robust weighting limit its interpretation. Finite local
rank cannot establish unbiased or globally unique estimation. A frontoparallel-plane
unit control must expose the known bias/axial-translation degeneracy.

Do not add simultaneous bias fitting based only on a low training residual. Summarize
positive/negative offsets, zero-bias preservation and true-start drift separately;
then decide whether a separately frozen nuisance-parameter study is justified.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/evaluate_depth_bias.py docs/assets/depth_bias_response_20261029/plan.json \
  docs/assets/depth_bias_response_20261029/evaluation.json
```
