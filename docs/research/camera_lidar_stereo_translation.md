# Camera–LiDAR: metric stereo translation refinement

This spent-data experiment tests a different geometric objective for the
translation regression seen in the [four-drive joint study](camera_lidar_expanded_joint.md).
It does not tune MI thresholds or revisit frame selection. Freeze the same eight
joint D2D outputs, three observations per case and existing success criteria
before fitting. The [plan](../assets/stereo_translation_20261029/plan.json) pins
source hashes, exact baseline identities and the full case inventory.

## Method and assumptions

`stereo_translation_refiner.py` is an opt-in, ROS-independent experimental solver;
it is not registered as a production default. Hold D2D's rotation fixed exactly
and fit only translation. Back-project metric stereo **optical-axis z** into 3D,
form local surface normals using neighboring points, and minimize point-to-plane
residuals between projected LiDAR returns and those surfaces. This compares
geometry in the camera frame, not camera z with LiDAR range.

The frozen [options](../assets/stereo_translation_20261029/options.json) specify:

- Center and four-neighbor positive finite depth, with a 5% local depth-jump mask.
- Nearest-point z buffering; at most 2,000 uniformly indexed matches per frame.
- Stereo disparity uncertainty 0.5 px, propagated via `z²/(fx*baseline)` into
  normal-direction uncertainty, combined with a 0.02 m floor.
- A depth association gate of `max(1 m, 3*sigma_depth)` and soft-L1 fitting.
- At least 64 matches in **every** frame, weighted-design condition number at most
  1,000, translation bounds ±0.5 m about D2D, and at most 15 outer iterations.
- Accept only a converged step below 0.1 mm with final support/rank checks and no
  bound hit. Otherwise return the D2D transform unchanged and retain the raw
  candidate, rejection reason and trace.

These are modeling choices frozen before real-data fitting, not independently
validated uncertainty or covariance estimates. Stereo depth can be biased and
surface normals are noisy; held-out calibration evidence is absent. The rank
check detects degenerate fitted geometry, not global calibration observability.
Fixed erroneous rotation can still be absorbed into translation; retaining
rotation isolates this failure mode rather than proving it absent.

Synthetic tests recover translation under a nonidentity fixed rotation from
three independent planes, reject a single-plane ambiguity, reject missing depth,
and retain the initializer when the bounded solution hits a limit. The real-data
fit receives only D2D's output, clouds, stereo depth, intrinsics and stereo
baseline. The scorer reads the reference **after** the fitting subprocess exits.

## Frozen scoring

All eight cases from 0005, 0002, 0009 and 0011 remain scheduled. Use four affinity
CPUs, one OpenBLAS thread, and a 180-second timeout per case. No outcome-dependent
reruns or parameter changes. Primary diagnostics are translation disagreement
change against D2D, counts improved/worsened/unchanged, and accepted/rejected fits.
Retain raw candidate errors even when the published experimental output falls
back. Separately report accepted-refinement recovery and whole-pipeline recovery
including the explicit unchanged-D2D fallback; fallback success is not success
of the new stage.

The strict joint event remains `<0.5 deg / <0.20 m`. Rotation stays unchanged, so
this stage cannot repair the two D2D initializers already failing rotation.
Do not compare an eight-case assisted-input result to a monocular SOTA result.
Four spent drives, two seed directions, uncertain stereo/motion models and
outstanding 55-drive / learned-baseline / independent-family / fresh-audit
requirements remain limitations.
