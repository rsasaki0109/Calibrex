# Current sparse frames do not support an independent metric temporal check

Frozen `be471af7`, audited2026-10-30. See
[protocol](temporal_surface_audit_protocol.md),
[plan and available calibration inventory](../assets/temporal_surface_audit_20261030/plan.json),
[all12 image pairs](../assets/temporal_surface_audit_20261030/evaluation.json),
[schema](../assets/temporal_surface_audit_20261030/evaluation.schema.json) and
[adjacent-frame local availability](../assets/temporal_surface_audit_20261030/adjacent-availability.json).

Do not build a temporal depth filter on the current sparse triplets. Only1/12
adjacent archived image pairs meets the fixed image-only directional-support
screen. Independent metric camera motion is absent from the declared local input
set: OXTS provides IMU poses, and the available IMU-to-camera chain would consume
the target camera/LiDAR extrinsic. Monocular essential-matrix translation has
unknown scale. Neither is silently presented as independent metric camera motion.

The image intervals are3.728–11.590 seconds, much longer than adjacent capture
frames. Same-index camera/OXTS timestamp differences reach12.238ms; these are
availability/timing statistics, not validated interpolation or synchronization.
All selected OXTS packets exist. No independent direct camera/IMU calibration or
metric camera-trajectory artifact was configured in the audited directories;
this is a local inventory finding, not a claim that no such evidence can exist.

| Drive | Frame pair | Interval(s) | Cross-check matches | Essential inliers | Cheirality inliers | Image cells | Screen |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2011_09_26_0005 | frame-38→frame-76 | 3.926 | 583 | 32 | 24 | [4, 6] | False |
| 2011_09_26_0005 | frame-76→frame-114 | 3.925 | 540 | 30 | 16 | [8, 3] | False |
| 2011_09_26_0009 | frame-111→frame-223 | 11.590 | 555 | 26 | 15 | [8, 7] | False |
| 2011_09_26_0009 | frame-223→frame-334 | 11.496 | 568 | 27 | 20 | [7, 5] | False |
| 2011_09_26_0011 | frame-58→frame-116 | 6.007 | 605 | 33 | 19 | [4, 6] | False |
| 2011_09_26_0011 | frame-116→frame-174 | 6.007 | 641 | 189 | 179 | [10, 10] | True |
| 2011_09_26_0013 | frame-35→frame-71 | 3.728 | 516 | 27 | 12 | [2, 4] | False |
| 2011_09_26_0013 | frame-71→frame-107 | 3.728 | 594 | 25 | 9 | [2, 5] | False |
| 2011_09_26_0015 | frame-74→frame-148 | 7.666 | 518 | 48 | 38 | [4, 6] | False |
| 2011_09_26_0015 | frame-148→frame-222 | 7.666 | 519 | 47 | 28 | [3, 4] | False |
| 2011_09_28_0002 | frame-93→frame-187 | 9.746 | 1070 | 884 | 0 | [0, 0] | False |
| 2011_09_28_0002 | frame-187→frame-281 | 9.746 | 1354 | 1267 | 0 | [0, 0] | False |

The screen requires30 cheirality-supported correspondences spanning6/16 cells
in both images. Passing0011/frame116→174 is not a validated camera trajectory or
recovered calibration. The0002 pairs have884/1267 essential inliers but zero
cheirality support, illustrating that many image matches alone do not establish
metric motion; this could reflect low parallax or degeneracy, not absence of
shared scene content. Likewise failed ORB support does not prove geometric overlap
is zero. These are fixed-algorithm readiness measures, not ground-truth overlap.

A file-existence inventory of ±1 capture neighbors around all18 selected frames
finds0/36 complete local stereo pairs. No neighboring image contents were read or
new data downloaded by this audit. The existing data therefore cannot be silently
reinterpreted as short-baseline temporal coverage.

Three focused tests, repository Ruff and strict documentation build pass. Checks
validate report/schema, all12 pairs and all frozen input/source hashes. Tests check
blank-image zero support, camera-geometry validity and analytic parameter parity.
No reference camera/LiDAR extrinsic enters ORB/motion estimation; available
calibration files are inventoried, not composed into a selection transform. No
extrinsic fits, temporal masks, ICP reruns or accuracy verdicts change.

## Concrete next prerequisite

Before another estimator, freeze a bounded short-baseline acquisition/preparation
plan for ±1 stereo neighbors within these same six already-spent recordings.
Pin archive identity, exact member names/timestamps, byte limits and provenance
before fetching; do not broaden to fresh recordings or test holdouts. Then audit
image support again on those short intervals and independently validate the metric
motion source. Stereo visual odometry may provide metric scale from camera baseline,
but shares stereo errors with the measurements under test and cannot be called
independent ground truth. Any temporal gate must disclose that coupling and pass
controlled-motion tests first. Do not bridge the current limitation using target
extrinsics, arbitrary motion scale, long-gap interpolation or selected good pairs.
