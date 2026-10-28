# Camera–LiDAR: local boundary tangent ablation

This post hoc development experiment tests whether actual local depth-boundary
geometry improves correspondence over the scalar edge objective. It uses all
fourteen previously spent, reacquired inputs from KITTI raw 0005 and 0002.
No fresh evaluation data, acceptance thresholds or default settings change.

## Fixed hypothesis and method

The earlier azimuth-direction proxy did not recover physical boundary direction.
Here each prepared LiDAR depth-discontinuity point receives a local PCA line
estimate: at most eight neighbors within 0.75 m, at least five neighbors,
largest covariance eigenvalue above 1e-12, and second/largest eigenvalue ratio
at most 0.2. The principal eigenvector is the tangent. Unsupported neighborhoods
receive zero vectors and retain the scalar response. All settings were fixed
before the batch, without examining held-out scores or reference errors.

Rotate the image gradient tensor by 90 degrees and reuse the existing projected
direction objective. A projected 3D boundary tangent should be perpendicular to
the image gradient, independent of tangent sign. The penalty attenuates the
nonnegative image response and retains background penalties and fixed support.
Image gradients and tangent estimates are fixed preprocessing; pose search uses
training frames only. Held-out frames score the final poses.

This is an estimated local line, not a guaranteed physical boundary: prepared
points can mix surfaces, sparse boundaries may have no support, and image
gradients are local rather than propagated to distant pixels. Candidate-based
point filtering and the original image-gradient confidence remain unchanged.

The source was frozen at `e289bc3e` in an isolated worktree. Both arms use
training-only initialization and multi-start nuisance fitting. The previously
rejected control-seeded baseline restarts are disabled. The original arm must
reproduce the earlier training-only report within absolute 1e-9 on every input
before the changed arm runs. All failures and all folds are retained.

Run from the frozen source with:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. python -m tools.kitti_camera_lidar_baseline_experiment \
  /path/to/reacquired/capture --output /path/to/new/output --boundary-tangent
```

## Interpretation

Detected controls measure local displacement distinguishability, not calibration
accuracy. Always inspect active-axis training-fold reference disagreements as
well as counts. Forty-two correlated axis controls from two drives do not form
forty-two independent evaluations. Runtime is a single sequential run, not a
repeated benchmark. Reference calibration is not ground truth.
