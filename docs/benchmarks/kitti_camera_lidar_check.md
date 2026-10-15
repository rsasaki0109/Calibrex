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

**Status: refuted, 5 of 9 gates** ([results](#results-refuted-5-of-9-gates)). The claim,
drives, runs, scoring and thresholds are in
[`kitti_camera_lidar_preregistration.yaml`](kitti_camera_lidar_preregistration.yaml), frozen
at code commit `71a115b`. Its successor, with two estimator rules changed and drives of
KITTI's other four days, is [v2](kitti_camera_lidar_v2_check.md).

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

## Results: refuted, 5 of 9 gates

**The runs.** All 23 drives ran once, from the merged pre-registration (`ecf7d68`, whose
`src/` and `tools/` equal the frozen `71a115b`). Every archive matched its pinned size and
ETag, the calibration archive its SHA-256, and all 23 conversions and 161 checks exited 0 on
their first attempt. The frozen scorer and builder scored them, and `calibrex sota audit`
records the same verdict
([protocol](../assets/kitti_camera_lidar/evaluation/kitti_camera_lidar_sota_protocol.yaml),
[audit](../assets/kitti_camera_lidar/evaluation/kitti_camera_lidar_sota_audit.yaml),
[scores](../assets/kitti_camera_lidar/evaluation/scores.json); every artifact and log is under
[`evaluation/runs/`](../assets/kitti_camera_lidar/evaluation/runs/evaluate_record.json)).

| Requirement | Observed | Threshold | Gate |
| --- | --- | --- | --- |
| `no-failed-drives` | 0 failed | 0 | achieved |
| `vendor-calibration-not-flagged` | 3 of 23 flagged (0029 and 0056 `fail`, 0032 `warn`) | 0 | **contradicted** |
| `judged-axis-accuracy` | 9.60 deg (0029 yaw) | 0.75 deg | **contradicted** |
| `judged-axis-coverage` | 61 of 69 axes judged (0.88) | at least 0.5 | achieved |
| `turned-judged-axis-accuracy` | 9.66 deg (0029 yaw, 3 deg roll run) | 1.5 deg | **contradicted** |
| `turn-3deg-flagged` | 4 of 61 not flagged (0.066) | at most 0.1 | achieved |
| `turn-3deg-never-judged-right` | 1 (0056 roll) | 0 | **contradicted** |
| `turn-1deg-flagged` | 1 of 61 not flagged (0.016) | at most 0.25 | achieved |
| `scored-drives-coverage` | 23 of 23 scored | at least 15 | achieved |

Estimate minus KITTI for the baseline (deg; "-" not judged), then the check's verdict on the
turned axis and its reading (deg; `*` the axis is not judged on the drive's baseline):

| Drive | Baseline | roll | pitch | yaw | roll +3 / +1 | pitch +3 / +1 | yaw +3 / +1 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 0011 | pass | +0.20 | -0.10 | -0.00 | fail 2.75 / warn 0.80 | fail 3.05 / fail 1.10 | fail 2.95 / warn 1.00 |
| 0013 | pass | +0.10 | -0.10 | -0.20 | fail 2.85 / warn 0.85 | fail 3.10 / fail 1.10 | unjudged / fail 1.20 |
| 0018 | pass | +0.25 | -0.10 | +0.25 | fail 2.60 / warn 0.70 | fail 3.00 / fail 1.05 | fail 2.65 / warn 0.65 |
| 0019 | pass | - | -0.10 | +0.00 | unjudged / unjudged * | fail 3.05 / fail 1.05 | fail 3.00 / fail 1.05 |
| 0023 | pass | +0.20 | -0.10 | -0.05 | fail 2.75 / warn 0.80 | fail 3.10 / fail 1.10 | fail 3.05 / fail 1.05 |
| 0029 | fail | - | -0.10 | +9.60 | unjudged / unjudged * | fail 3.10 / fail 1.15 | unjudged / fail 1.05 |
| 0032 | warn | - | -0.05 | -0.80 | fail 3.50 / unjudged * | fail 3.00 / warn 1.00 | fail 3.85 / fail 1.85 |
| 0036 | pass | - | -0.15 | +0.10 | unjudged / unjudged * | fail 3.15 / fail 1.15 | fail 2.90 / warn 0.90 |
| 0056 | fail | +2.75 | +0.15 | - | pass 0.25 / fail -1.75 | unjudged / unjudged | unjudged / unjudged * |
| 0061 | pass | +0.05 | -0.00 | +0.05 | fail 2.85 / warn 0.90 | fail 3.00 / warn 1.00 | fail 2.95 / warn 0.95 |
| 0064 | pass | +0.15 | -0.05 | -0.10 | fail 2.80 / warn 0.85 | fail 3.00 / warn 1.00 | fail 3.05 / fail 1.05 |
| 0070 | pass | - | -0.05 | - | unjudged / unjudged * | fail 3.00 / warn 1.00 | unjudged / unjudged * |
| 0084 | pass | +0.15 | -0.10 | +0.05 | fail 2.75 / warn 0.85 | fail 3.10 / fail 1.10 | fail 2.95 / warn 0.90 |
| 0086 | pass | +0.05 | -0.20 | -0.30 | fail 2.85 / warn 0.95 | fail 3.15 / fail 1.20 | fail 2.90 / fail 1.10 |
| 0087 | pass | +0.10 | -0.05 | +0.05 | fail 2.80 / warn 0.90 | fail 3.00 / fail 1.05 | fail 3.00 / warn 0.95 |
| 0091 | pass | +0.20 | -0.15 | -0.20 | fail 2.75 / warn 0.80 | fail 3.10 / fail 1.15 | fail 3.20 / fail 1.20 |
| 0093 | pass | -0.10 | -0.20 | -0.45 | fail 3.00 / fail 1.05 | fail 3.15 / fail 1.20 | fail 3.45 / fail 1.50 |
| 0095 | pass | +0.10 | -0.10 | +0.05 | fail 3.10 / warn 0.85 | fail 3.05 / fail 1.05 | fail 2.95 / warn 1.00 |
| 0096 | pass | +0.10 | -0.05 | +0.00 | fail 2.85 / warn 0.85 | fail 3.05 / fail 1.05 | fail 3.05 / warn 1.00 |
| 0101 | pass | - | -0.30 | -0.35 | unjudged / unjudged * | fail 3.25 / fail 1.30 | fail 3.35 / fail 1.35 |
| 0104 | pass | +0.05 | -0.10 | +0.05 | fail 2.85 / warn 0.90 | fail 3.00 / fail 1.05 | fail 2.95 / warn 0.95 |
| 0106 | pass | +0.10 | -0.10 | -0.10 | fail 2.85 / warn 0.90 | fail 3.05 / fail 1.10 | fail 3.00 / fail 1.10 |
| 0117 | pass | +0.15 | -0.05 | -0.10 | fail 2.80 / warn 0.85 | fail 3.05 / fail 1.05 | fail 3.10 / fail 1.10 |

**Twenty drives behave as the development drives did.** On them KITTI's calibration
passes, every judged axis is within 0.45 deg of it (0093 yaw), every axis judged in a
turned run within 0.50 deg, and every turn about a judged axis is flagged except one 3 deg
yaw turn on 0013 that its turned run leaves unjudged (the check passes it with yaw
unchecked).

**Three drives do not**, all through axes the check judged with held-out controls just
above their threshold (`t` from 2.09 to 2.33 against 2):

* **0029**: KITTI's calibration fails on yaw, judged 9.60 deg off. The estimate is the
  optimum of the search started at the profile maxima (-2.5, 0, +4.0 deg), which observes
  as many rotation axes (2) as the candidate's and has a higher objective. Its 3 deg roll run
  judges yaw 9.66 deg off, and its 3 deg yaw turn is left unjudged and passed.
* **0056**: KITTI's calibration fails on roll, judged 2.75 deg off, with yaw at the -12 deg
  search bound (not judged). The estimate comes from the search started at the yaw profile's
  maximum (-8 deg), which observes 2 axes against 0 from the candidate. The 3 deg roll turn
  stays at the same optimum: it reads 0.25 deg and is judged right.
* **0032**: KITTI's calibration warns on yaw, judged 0.80 deg off from the candidate's own
  optimum. Five of its six turned runs judge yaw 0.80 to 0.85 deg off too; the sixth leaves
  it unjudged.

**What this shows.** The check judges correctly where it was judging correctly in
development, but an evaluation drive can still hold a wrong optimum that passes the
check's own controls: KITTI's calibration was flagged on 3 of 23 unseen drives, against 0
of 13 development drives after the changes made on them. The search starts that fixed the
development drives' turned candidates also produced two of the three false alarms (0029 and
0056).

**Post-hoc observations** (after the verdict; they change nothing above):

* All 12 judged axes more than 0.75 deg off KITTI, in any run, come from these three drives
  and have control statistics from 2.09 to 2.33. A rule judging only axes with `t >= 2.5`
  would have left all 12 unjudged, and also 41 of the 404 axes judged within 0.75 deg.
* 10 of the 12 come from an optimum of an alternative start that replaced the candidate's
  (0029 and 0056): on 0029's baseline by tying its observed count and winning on the
  objective, in the other runs by observing more axes than a candidate optimum that observed
  none or one. Every one on 0056 has yaw at the -12 deg search bound.
* A next round would need drives camera-lidar has not read: every 2011_09_26 drive with 125
  frames or more is now spent; KITTI's other four days are not, and each has its own
  calibration.
