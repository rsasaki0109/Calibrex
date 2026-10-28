# Camera–LiDAR assisted-input development matrix

This plan freezes a 12-case execution before the matrix is run. It is spent-data
development, not fresh evaluation, and covers only 2 of the required 55 raw
drives. The previous single-frame result is not used to choose successful frames.

The [machine-readable plan](../assets/assisted_matrix_20261028/plan.json) and
[schema](../assets/assisted_matrix_20261028/plan.schema.json) pin source hashes,
archive identities, frame inventory and perturbations:

- KITTI 0005 (154 frames): indices 38, 76, 114.
- KITTI 0002 (376 frames): indices 93, 187, 281.
- Selection is `floor((N-1)*q/4)` for `q=1,2,3`, without image inspection.
- Each receives xyz Euler `[1,-1,0.5]` deg / translation `[0.1,-0.05,0.05]` m
  and their sign-negated versions. Rotation is left-multiplied onto the reference;
  translation is added in camera coordinates, as in the earlier diagnostic.

Each case uses the unchanged stereo/OXTS frontend and default bounded six-DoF
D2D solver from the [aligned stereo execution](camera_lidar_aligned_stereo_d2d.md).
Koide uses its unchanged official NID BFGS optimizer on the identical float32
cloud, with the same seed. Koide's preprocessing is bypassed for the shared
motion model. D2D additionally consumes stereo depth. This is an assisted-input
comparison with unequal sensor use, not monocular superiority evidence.

Cases run sequentially. Native execution is limited to four affinity CPUs with
one OpenBLAS thread; Koide retains its four-CPU/4-GB container and default internal
threading. Both get 180 seconds per case. The native timer includes its process;
Koide's inherited timeout also includes Xvfb startup. No speed-comparison claim
is planned. There are no outcome-dependent parameter changes or retries.

Report both rotation geodesic disagreement and translation norm against the
reference, plus the strict `<0.5 deg / <0.20 m` recovery event. D2D must also
converge, and both methods must complete with a finite rigid transform. Failed
preparation, timeout, invalid output or interrupted case remains in the fixed
12-case denominator. Missing cases are never dropped. Report per-drive outcomes
and descriptive median/p95 errors with valid-output counts; do not treat these
correlated frame/seed cases as independent recordings for confidence intervals.

A successful plan execution is not a SOTA verdict. Even complete success here
would leave 55-drive development coverage, independent-family evidence,
monocular depth-provider exposure, learned baselines and fresh preregistration
outstanding. No fresh evaluation sensor payload is authorized by this plan.
