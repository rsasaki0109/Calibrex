# One-sided visibility-motivated metric residual

Compare the frozen [symmetric metric fit](joint_metric_results.md) against a single
asymmetric modification. A camera records the first visible surface: a LiDAR hit
behind that surface can be occluded, whereas a hit in front contradicts observed
free space. This motivates a one-sided bounded influence approximation, not a
complete probabilistic visibility model or proof of occlusion.

Let r = transformed LiDAR z minus bilinear camera z. For supported r>0 replace
r with r/sqrt(1+(r/.2)^2); otherwise retain r. The residual saturates toward .2 m
behind the surface but preserves local unit slope. Keep every unsupported point's
1 m penalty unchanged. Apply the same outer soft-L1(.1 m), TRF, bounds, scales,
tolerances, 100 nfev/800 actual-call cap and support-retention checks as baseline.
Do not remove points or use true pose to identify occlusions within the estimator.
No depth-bias parameter is added. Record large positive/negative residual counts,
but do not label every positive residual as a true occlusion.

Potential failure: reducing behind-surface influence can favor moving geometry
behind the measured surface or produce one-sided pose bias. Require exact-visible
recovery and true-start drift checks rather than only reporting all-return gains.
Thresholds, saturation and scene layouts are fixed before optimization results.

Use three scene layouts: original five spheres, a nearer foreground arrangement,
and a staggered arrangement (all constants in tools/occlusion_geometry.py). Same
slanted background, cameras/extrinsics and three views. New layouts are additional
synthetic controls sharing generator/surface family, not independent real sensors.
For each layout use exact-z visible-only (oracle population as favorable control)
and exact-z all-return observations (no camera visibility filtering). Both methods
receive the identical population in each condition. Two original signed seeds plus
truth start: **12 perturbation cases + 6 drift controls**, two methods, 36 fits.
No outcomes omitted, retries or altered pose gates. Fitting receives no reference
extrinsics. Truth labels outcomes only after solving.

Compare converged recovery (<.5 deg/<.20 m), status, support, runtime/call budgets,
and all truth-start drift. Reproduce the six original-scene symmetric outputs.
No held-out thresholds select candidates or parameters. Failure in either control
blocks production promotion; success is still only a synthetic diagnostic. Existing
bias failure remains unresolved, not assumed cured by this experiment.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/evaluate_occlusion_pose.py docs/assets/occlusion_pose_20261029/plan.json \
  docs/assets/occlusion_pose_20261029/evaluation.json
```
