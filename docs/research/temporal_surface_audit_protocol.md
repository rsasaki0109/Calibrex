# Independent temporal measurement readiness audit

Pre-registered2026-10-30. Audit adjacent pairs of the three archived frames on each
of six spent drives:12 pairs. Do not fit temporal depth masks or extrinsic parameters.
Pin images, camera/OXTS timestamps, selected OXTS packets and available calibration
paths. Record camera intervals and same-index camera/IMU timestamp offsets; these
are availability/timing evidence, not validated motion interpolation.

Available official calibration is IMU→Velodyne→camera. That chain consumes the
reference LiDAR-camera extrinsic being evaluated, so it is not an independent
metric camera-motion source for a selection rule. Record absence of configured
direct camera/IMU or metric-camera-trajectory artifacts in this local input set.
OXTS IMU motion is not silently re-labeled as camera motion. No true extrinsics
enter the image-pair algorithm, even if reference files exist on disk.

Image-only support test: ORB2000, Hamming cross-check, deterministic seed0/thread1,
fixed essential-matrix RANSAC .999/1px using original camera intrinsics, recoverPose
cheirality. Record matches, epipolar and cheirality counts, and occupied4x4 cells
in each image. Necessary directional-support screen: >=30 cheirality inliers and
>=6 cells in both. This is a prespecified engineering screen, not a claim that
ORB matches measure true overlapping surface fraction or accurate camera motion.
Recovered monocular translation has unknown metric scale. E failures/missing
features remain explicit zero support. Do not drop or retune any pair.

If image support is inadequate or independent metric motion is missing, report
that as a temporal-depth-check blocker. An independently validated stereo/visual
odometry module would be new work with its own frozen tests; current SGBM depth
is under evaluation and cannot be treated as independent metric motion truth.
No downloads, new recordings, selection using target extrinsics, ICP rerun or SOTA.
