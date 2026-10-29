# Analytic first-surface sampling diagnostic

Test the [joint bias offset](joint_pose_bias_results.md) without retuning solver,
saturation, bias limits, seeds or success gates. Use the same 54 sphere-scene cases:
3 layouts × visible/all-return inputs × 0,+.25,-.25 m measurement bias × ± seeds
and truth start. Keep 36 perturbation trials and 18 truth-start checks separate.
Parallel-plane degeneracy was checked in the preceding experiment; it is not
rerun or counted in this sampler comparison. Archived bilinear results and source
are hash-pinned, not rerun or included in new execution timing.

Change only depth sampling. At each candidate-transformed LiDAR point, form the
continuous camera ray with z component one. Raycast the first surface of the known
analytic scene from the corresponding camera origin, then add the fixed measurement
offset. Compute LiDAR z minus this depth. Keep the original valid image domain
(u in [0,width-1), v in [0,height-1), z>.1), unsupported penalty, point population,
positive saturation and soft-L1. No point is filtered with true visibility by the
sampler; all-return points behind the first camera surface retain large residuals.
The favorable visible-only population is unchanged from preceding inputs.

The oracle sampler explicitly knows scene surfaces and camera origins and builds
the biased measurement function using the injected offset. The optimizer sees only
its residuals, never a target correction or true extrinsic. This separates interpolation
from measurement geometry in a synthetic control, not a deployable estimator,
independent dataset or evidence of real-data calibration accuracy. Do not mistake
knowing how to generate biased measurements for estimating bias without measurements.

Seven-parameter fitting remains zero-initialized correction, ±.5 m bias bounds,
original pose bounds/scales, ≤100 nfev / hard ≤800 actual calls, same support and
raw/effective-robust sensitivity diagnostics. Pose recovery and joint recovery
(bias error <.05 m and locally identifiable) use unchanged gates. Include every
case, no retries or post-score settings changes. Record actual bias error even
when a coarse joint gate passes. The analytic solver copy must have identical
function definitions to the previous solver except its sampler import.

Decide from visible-only and all-return separately: whether the systematic offset
shrinks, whether pose remains accurate and whether any residual occlusion bias or
local degeneracy remains. The finite-difference rank check may change when image
silhouette interpolation is removed; report it, do not relax rank gates. If results
improve, investigate implementable discontinuity-aware sampling next rather than
promoting an oracle. If they do not, retain asymmetric-objective mismatch as a live
hypothesis instead of blaming bilinear interpolation by assumption.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/evaluate_analytic_sampling.py docs/assets/analytic_sampling_20261029/plan.json \
  docs/assets/analytic_sampling_20261029/evaluation.json
```
