# Stereo surface registration with existing tangent-plane ICP

## Frozen experiment

Use the existing Chen–Medioni discrete tangent-plane ICP implementation rather
than extending the image-support penalty. Correspondences are refreshed 3D
nearest neighbors with mutual checks, a 0.5 m gate and 80% point-plane trimming.
Target normals come from local PCA; rank/condition checks and a spatial 20% source
holdout are retained. Multi-start and effective-curvature diagnostics are disabled
for bounded runtime. The holdout is a spatial subset of the same frame, not
independent sensor truth or a separate recording.

Before real-data execution, five controlled three-plane tests recover <0.05 deg
and <5 mm pose error and <5 mm holdout plane RMSE: complete surfaces, target holes,
partial source overlap with distant unmatched points, their combination, and a
foreground patch replacing a background patch only in the target. These are
geometric point-cloud controls, not a realistic stereo noise/synchronization
simulation. They validate recovery with missing/conflicting geometry, beyond
continuity alone.

Freeze code/options/protocol before real execution. Use the **middle archived
frame** for each of six spent recordings and both original initial perturbations:
12 ICP fits and 12 newly matched single-frame D2D controls. Both receive the same
raw frame depth, cloud, intrinsics and initial pose. Do not compare success rates
as though the historical three-frame D2D run used the same observations.

Backproject valid 0.1–80 m stereo depth into original-left camera coordinates.
Select LiDAR points in the initial camera field of view and depth range, without
using reference calibration. For ICP only, deterministically retain the first
point in 0.2 m voxels then uniformly cap at 6,000 points per cloud. D2D retains
its existing preprocessing/defaults. No target-normal filling, support-field
penalty or cross-frame point association is introduced.

ICP retains its 50-iteration default and 1e-7 update tolerance. Report all raw
candidates, status, source/target sizes, train/holdout plane residuals, tangent
rank/condition and signed probes. Treat `converged` as numerical acceptance only;
score the unchanged <0.5 deg and <0.20 m gates after both fits. Non-converged ICP
returns the original pose while preserving the raw candidate when available.
Represent nonfinite diagnostic evidence explicitly as strings, not JSON NaN.

No defaults are promoted, no fresh evaluation inputs read, and no accuracy-driven
reruns or parameter changes are allowed. Compare all twelve cases, including
failed geometry or support, without counting a rejected candidate as recovery.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/evaluate_stereo_surface_registration.py \
  docs/assets/stereo_surface_registration_20261029/plan.json /tmp/surface-icp.json
```
