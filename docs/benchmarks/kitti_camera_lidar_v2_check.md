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

**Status: refuted, 8 of 9 gates** ([results](#results-refuted-8-of-9-gates)). The claim,
drives, runs, scoring and thresholds are in
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

## Results: refuted, 8 of 9 gates

**The runs.** All 18 drives ran once from the merged pre-registration (`0dd7c1d`, whose
`src/` and `tools/` equal the frozen `f0afa94`). Every archive matched its pinned size and
ETag, and every day's calibration archive its SHA-256.

* **One fetch was retried.** The fetch of 2011_10_03_0027 failed on a dropped connection
  (curl exit 18) before any check ran. By the rerun policy the driver ran again with the
  same command and fetched it, and both attempts are in the record.
* **Everything else exited 0:** all 18 conversions and 126 checks.

The frozen scorer and builder scored the runs, and `calibrex sota audit` records the same
verdict
([protocol](../assets/kitti_camera_lidar_v2/evaluation/kitti_camera_lidar_v2_sota_protocol.yaml),
[audit](../assets/kitti_camera_lidar_v2/evaluation/kitti_camera_lidar_v2_sota_audit.yaml),
[scores](../assets/kitti_camera_lidar_v2/evaluation/scores.json); every artifact and log is
under [`evaluation/runs/`](../assets/kitti_camera_lidar_v2/evaluation/runs/evaluate_record.json)).

| Requirement | Observed | Threshold | Gate |
| --- | --- | --- | --- |
| `no-failed-drives` | 0 failed | 0 | achieved |
| `vendor-calibration-not-flagged` | 0 of 15 flagged | 0 | achieved |
| `judged-axis-accuracy` | 0.45 deg (2011_09_28_0043 yaw) | 0.75 deg | achieved |
| `judged-axis-coverage` | 36 of 45 axes judged (0.80) | at least 0.5 | achieved |
| `turned-judged-axis-accuracy` | 1.503 deg (2011_09_28_0002 roll, 3 deg yaw run) | 1.5 deg | **contradicted** |
| `turn-3deg-flagged` | 0 of 36 not flagged | at most 0.1 | achieved |
| `turn-3deg-never-judged-right` | 0 | 0 | achieved |
| `turn-1deg-flagged` | 0 of 36 not flagged | at most 0.25 | achieved |
| `scored-drives-coverage` | 15 of 18 scored | at least 12 | achieved |

Estimate minus the day's calibration for the baseline (deg; "-" not judged), then the
check's verdict on the turned axis and its reading (deg; `*` the axis is not judged on the
drive's baseline, so its turns are not counted):

| Drive | Baseline | roll | pitch | yaw | roll +3 / +1 | pitch +3 / +1 | yaw +3 / +1 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 2011_09_28_0002 | pass | - | +0.05 | - | fail 4.45 / unjudged * | fail 3.05 / fail 1.10 | unjudged / unjudged * |
| 2011_09_28_0016 | not usable | | | | | | |
| 2011_09_28_0021 | not usable | | | | | | |
| 2011_09_28_0039 | pass | +0.00 | -0.10 | -0.05 | fail 2.90 / warn 0.95 | fail 3.00 / fail 1.10 | fail 3.05 / fail 1.05 |
| 2011_09_28_0043 | pass | +0.35 | -0.00 | +0.45 | fail 2.50 / warn 0.65 | fail 2.90 / warn 1.00 | fail 2.55 / warn 0.55 |
| 2011_09_29_0004 | pass | +0.35 | +0.05 | - | fail 2.55 / warn 0.50 | fail 2.90 / warn 0.95 | unjudged / unjudged * |
| 2011_09_29_0026 | not usable | | | | | | |
| 2011_09_29_0071 | pass | +0.20 | -0.05 | +0.00 | fail 2.75 / warn 0.80 | fail 2.95 / warn 1.00 | fail 3.00 / fail 1.00 |
| 2011_09_30_0016 | pass | +0.00 | +0.00 | - | fail 2.90 / warn 0.95 | fail 2.90 / warn 1.00 | unjudged / unjudged * |
| 2011_09_30_0020 | pass | +0.25 | -0.10 | -0.05 | fail 2.75 / warn 0.80 | fail 3.05 / fail 1.10 | fail 3.05 / fail 1.10 |
| 2011_09_30_0027 | pass | +0.15 | +0.00 | -0.10 | fail 2.75 / warn 0.85 | fail 2.90 / warn 0.95 | fail 3.10 / fail 1.10 |
| 2011_09_30_0028 | pass | +0.20 | -0.00 | -0.05 | fail 2.70 / warn 0.75 | fail 2.95 / warn 1.00 | fail 3.15 / fail 1.10 |
| 2011_09_30_0033 | pass | - | +0.00 | - | unjudged / unjudged * | fail 3.00 / warn 1.00 | unjudged / unjudged * |
| 2011_09_30_0034 | pass | +0.15 | +0.00 | +0.05 | fail 2.75 / warn 0.80 | fail 2.95 / warn 1.00 | fail 2.95 / warn 0.95 |
| 2011_10_03_0027 | pass | +0.15 | +0.00 | -0.05 | fail 2.80 / warn 0.85 | fail 2.95 / warn 1.00 | fail 3.05 / fail 1.05 |
| 2011_10_03_0034 | pass | - | -0.05 | +0.00 | unjudged / unjudged * | fail 3.00 / fail 1.05 | fail 2.95 / warn 1.00 |
| 2011_10_03_0042 | pass | - | -0.10 | - | unjudged / unjudged * | fail 3.00 / fail 1.10 | unjudged / unjudged * |
| 2011_10_03_0047 | pass | +0.20 | +0.00 | -0.35 | fail 2.75 / warn 0.80 | fail 2.95 / warn 0.95 | fail 3.35 / fail 1.35 |

**Every day's calibration passes.** On all 15 scored drives, over the four days, the check
passes that day's calibration (v1 flagged KITTI's on 3 of 23). Every judged axis is within
0.45 deg of it, and every turn about a judged axis is flagged, 36 of 36 at 3 deg and 36 of
36 at 1 deg; none is judged right.

**Three drives are not usable.** On 2011_09_28_0016, 2011_09_28_0021 and 2011_09_29_0026
the rig stands or creeps through most of the first 48 s. The repeated-view guard keeps 1, 7
and 5 frames where 8 are needed, so the baseline estimator refuses (`inconclusive`). The
rules count them against coverage only.

**One drive refutes the claim: 2011_09_28_0002.** Its baseline judges pitch only: roll's
held-out control is at `t` -0.29, yaw's at 0.46.

* In four of its six turned runs (roll +3, pitch +3, yaw +3 and pitch +1) the search
  started at the profile maxima reaches an optimum with roll 1.45 to 1.50 deg off the
  day's calibration.
* That optimum observes roll and pitch against the candidate optimum's pitch alone, so by
  the v2 rule it replaces it.
* Roll's control there is at `t` 2.58 to 2.65, just above the new 2.5 bound.
* Three of these axes fall within the 1.5 deg gate (1.45, 1.48 and 1.49 deg). In the 3 deg
  yaw run, roll is 1.503 deg off and the gate is contradicted.

That account is post hoc: it explains the verdict and does not change it. It is v1's
failure mode one step later: a far optimum judged with a control just above its bound.
v2's rules removed the false alarms on KITTI's calibration (0 of 15). They did not keep such
an optimum out of the turned runs.

