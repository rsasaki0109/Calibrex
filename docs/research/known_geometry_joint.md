# Independent known-geometry joint calibration control

## Frozen protocol

This diagnostic follows the [joint stereo comparison](joint_learned_stereo.md).
The existing six-DoF unit test backprojects one synthetic range image and asserts
objective improvement, not full-pose recovery. This experiment independently
raycasts camera and LiDAR against five spheres and a slanted background plane.
The scene has curved surfaces and depth discontinuities. Camera depth is optical
z at integer pixel centers; LiDAR samples independent fractional angular rays
from a displaced origin. No camera-depth backprojection creates the LiDAR cloud.

Use three views at fixed translated camera origins, 320×192 camera, 80×48 LiDAR
rays, nonidentity extrinsics (xyz angles 2,-3,1 degrees; translation .2,-.1,.3 m).
Scene constants and all generating/solver code are pinned in the
[plan](../assets/known_geometry_joint_20261029/plan.json). Analytic ray-intersection,
visibility, transform and perturbation-isolation tests pass before calibration.
No real sensor data, model inference or historical scores enter the generator.

Five conditions, each using the same joint three-observation solver defaults:

- Exact optical z and oracle camera-visible LiDAR returns.
- Exact optical z and all first-hit LiDAR returns, including camera occlusions.
- Exact camera radial range with the same visible LiDAR points (representation control).
- Optical z plus constant 0.25 m bias with visible LiDAR points.
- Exact optical z with a fixed 90×100 missing rectangle, same visible LiDAR points.

The oracle visibility check raycasts from the camera center toward each LiDAR
hit in continuous coordinates; it does not use the solver projector or a tuned
pixel-depth threshold. The visible-only condition is deliberately favorable and
not a deployable visibility estimator. Pixel discretization still remains.

For each condition use both signed xyz rotation perturbations (1,-.5,.75 degrees)
and translation perturbations (.15,-.1,.1 m), plus exact-reference initialization:
**15 fits**, with perturbed recovery reported over **10** and reference-start drift
over **5** separately. Reference-start trials do not inflate perturbation recovery.
Use convergence plus the unchanged <0.5-degree/<0.20-m gates. Record ground-truth
objective under the same histogram ranges, complete traces, all pose errors,
point/depth counts and failures. Do not tune geometry, seeds, bins or gates after
seeing outcomes. Synthetic success is not real-data success; failure in one
scene does not prove the cause of errors on KITTI.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/known_geometry_joint.py \
  docs/assets/known_geometry_joint_20261029/plan.json \
  docs/assets/known_geometry_joint_20261029/evaluation.json
```
