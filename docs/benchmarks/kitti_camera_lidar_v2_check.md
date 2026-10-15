# KITTI Camera-LiDAR Check Audit v2

The second audited claim for the `camera-lidar` pair, after
[v1](kitti_camera_lidar_check.md) was refuted. Like v1, it is about `calibrex check
--pairs camera-lidar`: a targetless check that starts at the deployed calibration and judges,
axis by axis, whether its rotation is right. v2 runs the check on KITTI raw drives of four
days it has never read, each against its own day's calibration and against that calibration
with the camera turned 1 or 3 deg.

It is a **check** claim, not a calibration-accuracy claim. It does not compare the check
with targetless calibration methods that start from large errors, such as the learned
methods tracked in the [camera-LiDAR research plan](../research/camera_lidar_sota_plan.md),
and no external camera-LiDAR baseline is integrated. The reference is each day's KITTI
target-based calibration, itself a calibration with error, not ground truth.

## Pre-registration

**Status: pre-registered; the evaluation has not run.** The claim, drives, runs, scoring and
thresholds are in
[`kitti_camera_lidar_v2_preregistration.yaml`](kitti_camera_lidar_v2_preregistration.yaml),
frozen at code commit `f0afa94`.

**What changed since v1.** v1's runs showed two rules letting wrong axes through; both
changed after the audit
([the account](camera_lidar_check.md#after-the-kitti-audit-a-stricter-control-and-far-optima-2026-10-14)):

* a held-out control now needs `t` 2.5 (was 2);
* an optimum found from the profile maxima replaces the candidate's only when it observes
  more rotation axes, at least two, and none at the search bound.

**Development.** 37 drives: all 36 KITTI 2011_09_26 drives (v1's development and
evaluation drives, spent) and 2011_09_30_0018, which earlier camera-LiDAR research read. They
were run with the frozen code through this audit's own harness. KITTI's calibration passes
on all 37, and every judged axis is within 0.45 deg of it (0.89 deg in turned runs). 89 of
111 axes are judged; 86 of 89 3 deg turns and 89 of 89 1 deg turns are flagged, and none is
judged right. 2011_09_30_0018 is from another day, whose calibration is 0.97 deg from
2011_09_26's: the check passes that day's calibration there and flags all six turns. The
account, with v1's rules side by side, is in
[Camera-LiDAR check: after the KITTI audit](camera_lidar_check.md#after-the-kitti-audit-a-stricter-control-and-far-optima-2026-10-14).

**Evaluation drives.** Every drive of 2011_09_28, 2011_09_29, 2011_09_30 and 2011_10_03 that
no Calibrex camera-LiDAR method has read and that has at least 125 frames (v1's rule):

| Day | Drives | Frames |
| --- | --- | --- |
| 2011_09_28 | 0002, 0016, 0021, 0039, 0043 | 145 to 376 |
| 2011_09_29 | 0004, 0026, 0071 | 158 to 1059 |
| 2011_09_30 | 0016, 0020, 0027, 0028, 0033, 0034 | 279 to 5177 |
| 2011_10_03 | 0027, 0034, 0042, 0047 | 837 to 4663 |

2011_09_30_0018 is excluded: earlier camera-LiDAR research read it, and it is a development
drive here. The drives were chosen from the archives' file lists only. Each archive's size
and ETag, and each day's calibration archive SHA-256, are pinned. Each day's calibration
differs from 2011_09_26's by 0.77 to 1.01 deg, mostly in roll, so every day is checked
against a calibration of its own.

**Runs.** Each drive's first 600 frames of `image_02`, the Velodyne and OXTS are fetched by
HTTP range (`If-Match` on the pinned ETag; every member's CRC-32 checked). The check reads
at most 48 scans 1 s apart from the start, so 60 s holds everything it reads. Each drive is
converted with `calibrex convert kitti-raw --camera image_02 --velodyne-time azimuth` and
checked seven times with the defaults: once with the day's calibration (the *baseline*) and
once with the camera turned 3 or 1 deg about each of its own axes. A *trial* is one turn
about an axis the check judged on that drive's baseline.

**Gates.** "Judged" is the check's own rule: estimated, reported std at most 0.5 deg, its
2 deg known-bad control detected on held-out frames (`t` at least 2.5).

| Requirement | Rule | Development (37 drives) |
| --- | --- | --- |
| `no-failed-drives` | no drive fails to fetch, convert or run | 0 failed |
| `vendor-calibration-not-flagged` | no baseline verdict is `warn` or `fail` | 0 of 37 |
| `judged-axis-accuracy` | every judged baseline axis within 0.75 deg of the day's calibration | 0.45 deg |
| `judged-axis-coverage` | at least half of the baseline axes judged (coverage) | 89 of 111 |
| `turned-judged-axis-accuracy` | every axis judged in a turned run within 1.5 deg of the day's calibration | 0.89 deg |
| `turn-3deg-flagged` | at most 10 percent of the 3 deg trials not flagged on the turned axis | 3 of 89 |
| `turn-3deg-never-judged-right` | no 3 deg trial judges the turned axis `pass` | 0 of 89 |
| `turn-1deg-flagged` | at most 25 percent of the 1 deg trials not flagged | 0 of 89 |
| `scored-drives-coverage` | at least 12 of the 18 drives scored (coverage) | 37 of 37 |

A failed requirement refutes the claim, except the two coverage requirements, which leave
it incomplete. The thresholds follow v1's rules from the development values and come out as
v1's, except the scored-drives floor (the pre-registration's `threshold_rationale`). The development scores are in
[`docs/assets/kitti_camera_lidar_v2/development/scores.json`](../assets/kitti_camera_lidar_v2/development/scores.json).

**What it will not show.**

* Translation: the check never judges it.
* Other cars: every drive is one car's, over four days.
* Starts beyond 3 deg, or turns about more than one axis at once.
* Axes the check does not judge: they count only against coverage. A turned run that leaves
  the turned axis unjudged is a miss in the flagged rate, not a wrong verdict.
* Whether a day's calibration is right: the reference is KITTI's, and a check that disagrees
  with it counts as a flag.

**Known risks.** A day whose calibration is off by more than the check's tolerance on a
judged axis fails the claim even if the check is right to flag it; only one development
drive (2011_09_30_0018) comes from an evaluation day. The stricter control judges fewer axes
(86 of 108 on the 2011_09_26 drives, against 97 under v1's rules), and a drive that stands
for most of its first 48 s, or runs fast on a highway (2011_10_03_0042), may judge little;
the coverage gates allow that.
