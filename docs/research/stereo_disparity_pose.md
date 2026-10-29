# Joint pose fitting with pixel-scaled inverse-depth residuals

## Frozen development protocol

Implement a new experimental six-DoF estimator in the ROS-independent core.
Residual is `fx * baseline * (1 / stereo_z - 1 / transformed_lidar_z)` in pixel
units, divided by 0.5 px. It is an **equivalent-disparity/inverse-depth residual**,
not the exact raw horizontal disparity for non-horizontal stereo baselines.
This formulation naturally gives smaller metre-error weight at large depth,
without a reference-derived distance cutoff. It replaces noisy point-plane
normals with bilinear inverse-depth sampling and jointly fits rotation/translation.

Freeze code, options and the 12-case plan before testing on real inputs. Use all
six spent recordings (0005,0002,0009,0011,0013,0015), both original perturbations,
and the exact depth/cloud/intrinsic inputs used by each archived joint D2D run.
The first four drives retain historical depth conversion; the last two use its
corrected version. Each paired comparison uses identical inputs; results across
these groups do not isolate frontend differences. Do not silently regenerate
historical depth or use D2D output as the new method's starting point.

- Select an initial nearest-visible-point population, with four finite depth
  neighbors, local equivalent-disparity range <=2 px and initial residual <=8 px.
  Cap at 3,000 points/frame by deterministic point-index priority; require 128.
- Hold these LiDAR point identities fixed throughout the solve. Sample inverse
  depth bilinearly at the current pose. Missing support receives a fixed 16-unit
  residual rather than disappearing from the objective. This penalty and validity
  boundaries are discontinuous and can still produce local traps.
- Optimize left-multiplied rotation-vector increments and additive translation
  with soft-L1 least squares, 150 evaluations, per-axis bounds ±3 deg and ±0.5 m,
  and `ftol=xtol=gtol=1e-8`. Publish the exact options and optimizer diagnostics.
- Reject unsuccessful termination, proximity to a bound, final support below 128
  or 90% of its initial population, and robustly weighted Jacobian condition
  number >1e6. Parameters are scaled by the search bounds for conditioning.
  This is a local numerical check, not global observability or accuracy assurance.
- Preserve every raw candidate. Rejected fits return the initializer exactly.
  Score reference rotation/translation only after the solver finishes. Compare
  accepted-stage recovery against the unchanged 0.5 deg / 0.20 m joint gates.

Synthetic tests recover all six components from independent known planes,
reject an unobservable single plane and reject absent depth. No production
registration/default is changed, no fresh evaluation data is used, and no
post-result adjustment or favorable-case rerun is planned. The implementation is
a development candidate, not a SOTA claim.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/evaluate_disparity_pose.py \
  docs/assets/disparity_pose_20261029/plan.json /tmp/disparity-pose.json
```
