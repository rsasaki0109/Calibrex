# KITTI Camera-LiDAR Check Audit

The first audited claim for the `camera-lidar` pair. It is about `calibrex check
--pairs camera-lidar`: a targetless check that starts at the deployed calibration and judges,
axis by axis, whether its rotation is right. The audit runs the check on KITTI raw drives it
has never read, against KITTI's own calibration and against that calibration with the camera
turned 1 or 3 deg.

It is a **check** claim, not a calibration-accuracy claim. It does not compare the check
with targetless calibration methods that start from large errors, such as the learned
methods tracked in the [camera-LiDAR research plan](../research/camera_lidar_sota_plan.md),
and no external camera-LiDAR baseline is integrated. The reference is KITTI's target-based
calibration, itself a calibration with error, not ground truth.

## Pre-registration

**Status: pre-registered; no evaluation drive has been run.** The claim, drives, runs,
scoring and thresholds are in
[`kitti_camera_lidar_preregistration.yaml`](kitti_camera_lidar_preregistration.yaml), frozen
at code commit `71a115b`.

**Development.** The method was changed on 13 KITTI 2011_09_26 drives until KITTI's
calibration passed on all of them; those drives are spent. The account is in
[Camera-LiDAR check: thirteen KITTI drives](camera_lidar_check.md#thirteen-kitti-drives-2026-10-14).

**Evaluation drives.** Every 2011_09_26 drive that no Calibrex camera-LiDAR method has read
and that has at least 125 frames (12.5 s, as long as the shortest development drive): 0011,
0013, 0018, 0019, 0023, 0029, 0032, 0036, 0056, 0061, 0064, 0070, 0084, 0086, 0087, 0091,
0093, 0095, 0096, 0101, 0104, 0106 and 0117. They were chosen from the archives' file
lists only; each archive's size and ETag, and the calibration archive's SHA-256, are
pinned.

**Runs.** Each drive is converted with `calibrex convert kitti-raw --camera image_02
--velodyne-time azimuth` and checked seven times with the defaults: once with KITTI's
calibration (the *baseline*) and once with the camera turned 3 or 1 deg about each of its
own axes. A *trial* is one turn about an axis the check judged on that drive's baseline.

**Gates.** "Judged" is the check's own rule: estimated, reported std at most 0.5 deg, its
2 deg known-bad control detected on held-out frames.

| Requirement | Rule | Development (13 drives) |
| --- | --- | --- |
| `no-failed-drives` | no drive fails to download, convert or run | 0 failed |
| `vendor-calibration-not-flagged` | no baseline verdict is `warn` or `fail` | 0 of 13 |
| `judged-axis-accuracy` | every judged baseline axis within 0.75 deg of KITTI | 0.35 deg |
| `judged-axis-coverage` | at least half of the baseline axes judged (coverage) | 36 of 39 |
| `turned-judged-axis-accuracy` | every axis judged in a turned run within 1.5 deg of KITTI | 0.89 deg |
| `turn-3deg-flagged` | at most 10 percent of the 3 deg trials not flagged on the turned axis | 0 of 36 |
| `turn-3deg-never-judged-right` | no 3 deg trial judges the turned axis `pass` | 0 of 36 |
| `turn-1deg-flagged` | at most 25 percent of the 1 deg trials not flagged | 0 of 36 |
| `scored-drives-coverage` | at least 15 of the 23 drives scored (coverage) | 13 of 13 |

A failed requirement refutes the claim, except the two coverage requirements, which leave
it incomplete. The error thresholds are 1.5 times the worst development value, rounded up
to the next 0.25 deg; the rates and counts are set in the pre-registration's
`threshold_rationale`. The development scores are in
[`docs/assets/kitti_camera_lidar/development/scores.json`](../assets/kitti_camera_lidar/development/scores.json).

**What it will not show.**

* Translation: the check never judges it.
* Other rigs: every drive shares one rig, one day and one calibration.
* Starts beyond 3 deg, or turns about more than one axis at once.
* Axes the check does not judge: they count only against coverage. A turned run that leaves
  the turned axis unjudged is a miss in the flagged rate, not a wrong verdict.

## Results

Not run yet.
