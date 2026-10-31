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

The estimator was frozen at `e289bc3e`; the full batch uses `de09e3e7`
in an isolated worktree with unchanged estimator code. Both arms use
training-only initialization and multi-start nuisance fitting. The previously
rejected control-seeded baseline restarts are disabled. The original arm must
reproduce the earlier training-only report within absolute 1e-9 on every input
before the changed arm runs. All failures and all folds are retained.

Run from the frozen source with:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. python -m tools.kitti_camera_lidar_boundary_batch \
  /path/to/reacquired/capture --output /path/to/new/output
```

## Results

All fourteen original-arm numerical parity checks pass within absolute 1e-9;
all paired inputs complete. No condition was excluded.

| Diagnostic | Scalar baseline | Boundary tangent |
| --- | ---: | ---: |
| Detected controls, 0005 | 19/21 | 19/21 |
| Detected controls, 0002 | 0/21 | 0/21 |
| Total detected controls | 19/42 | 19/42 |
| Worst active-axis fold reference disagreement among detected controls | 0.350 deg | 6.547 deg |
| Target 0002 pitch +3 control t | 0.77397 | 1.29804 |
| Target pitch +3 detected | No | No |
| Sum of profile wall times (concurrent workers) | 177.1 s | 870.3 s |

Directions are supported for 35.38% of prepared points across all input copies.
The worst detected case is 0005 roll +3, yaw-axis control, fold index 4,
with +6.547 deg active-axis reference disagreement. The unchanged total hides
both gained and lost detections; the summary retains every changed flag.
The target pitch t statistic increases but remains below the unchanged 2.5
threshold. Its small fitted pitch error does not establish observability.

## Decision

Do not promote this objective. A local tangent cue alone does not resolve the
wrong-correspondence problem: a detected control can still accompany a
training-fold pose far from the reference. Adding a direction penalty changes
the optimization landscape and can create another convincing, incorrect basin.
This experiment does not isolate whether local tangent errors, image-gradient
support, or cross-surface matches dominate; those mechanisms were not separately
tested. Do not tune these constants or thresholds against the same results.

The [complete paired report](../assets/boundary_tangent_20261028/report.json),
[summary](../assets/boundary_tangent_20261028/summary.json), and
[hash manifest](../assets/boundary_tangent_20261028/manifest.json) retain all
conditions, per-fold poses, support counts and the interrupted sequential run.

## Interpretation

Detected controls measure local displacement distinguishability, not calibration
accuracy. Always inspect active-axis training-fold reference disagreements as
well as counts. Forty-two correlated axis controls from two drives do not form
forty-two independent evaluations. The batch uses four isolated worker processes, so timings include CPU
contention and are not a performance benchmark. An initial sequential run was
interrupted to change scheduling; its partial results are retained separately,
and the complete batch re-runs every input with unchanged algorithm settings. Reference calibration is not ground truth.
