# INS-LiDAR per-axis audit preparation

This page records preparation of the `ins-lidar` audit using KITTI and Boreas.
This page records development and audit preparation. The separate
[pre-registration](../benchmarks/ins_lidar_rotation_v1_preregistration.yaml)
fixes the proposed evaluation contract and must be merged before evaluation.
The completed [first audit](../benchmarks/ins_lidar_rotation_v1.md) is now
**refuted, 6/8 gates**. This page preserves the development and selection record;
the three supported sensor pairs on the [leaderboard](../benchmarks/sota_leaderboard.md)
remain unchanged. The frozen registration and original development evidence
have not been revised after evaluation.

## Proposed claim scope

Evaluate the rotation axes the native trajectory hand-eye method actually
observes, their differences to each dataset's reference calibration, the
fraction of rotation axes observed, and held-out known-bad controls. Count
failed recordings and missing results in the coverage denominator. A prior
must not count as observed data, and the proposed method uses no priors.

The clock offset and reference scale remain nuisance parameters. The available
dataset calibration alone does not establish an absolute clock-error target,
so neither clock accuracy nor a full extrinsic is part of this first proposal.
Dataset calibration is a reference estimate, not independent metrology. Passing
this scoped audit would not establish superiority over every published method.

The intended coverage is **two dataset families and two independent platforms**:
KITTI and Boreas. Different dates or recordings of the same platform do not
count as additional independent rigs.

## Metadata access, 2026-10-22

The [Boreas devkit](https://github.com/utiasASRL/pyboreas) provides original
Velodyne scans and an Applanix GNSS/INS reference. The
[data specification](https://github.com/utiasASRL/pyboreas/blob/master/DATA_REFERENCE.md)
describes the frames and reference calibration; the
[AWS registry](https://github.com/awslabs/open-data-registry/blob/main/datasets/boreas.yaml)
lists the public bucket and CC BY 4.0 data license.

`tools/boreas_ins_lidar_metadata.py` queries only S3 file listings. It cannot
GET sensor, calibration or trajectory objects. It fixes candidates by recording
names and the first 600 numeric-stem LiDAR filenames, retaining candidates with
missing inputs rather than silently replacing them.

The [metadata receipt](../assets/ins_lidar_sota_readiness/boreas_metadata.json),
its [JSON schema](../assets/ins_lidar_sota_readiness/boreas_metadata.schema.json),
and ten raw listing responses preserve the actual names, sizes, ETags and
response hashes. The listing bodies total 807,963 bytes. No sensor, trajectory
or calibration object bodies were downloaded by that inventory. ETags are object
identity pins, not SHA-256 digests of the file contents.

| Recording | Proposed role at inventory time | First 600 LiDAR scans | Required Applanix reference |
| --- | --- | ---: | --- |
| `boreas-2020-11-26-13-58` | development | 3,039,592,056 bytes | present |
| `boreas-2020-12-01-13-26` | evaluation candidate, not read | 3,017,816,496 bytes | present |
| `boreas-2020-12-04-14-00` | evaluation candidate, not read | 3,034,396,584 bytes | **missing** `gps_post_process.csv` |

These are candidates, not a frozen evaluation set. The third recording is not
ready for this adapter. Any further candidate selection must be explained by
metadata eligibility before reading its trajectory or scans, never by scores.
The receipt deliberately leaves the exposure audit marked incomplete.

## Development exposure

KITTI `2011_09_26` drives `0005`, `0009`, `0014`, `0015` and `0022` have already
been read by INS-LiDAR check/estimate work. The
[real-data estimate report](../benchmarks/estimate_real_data.md) includes the
`0015` round trip; the original handoff omitted it.

KITTI `2011_09_30_drive_0018_sync` is also spent for INS-LiDAR development.
The trajectory hand-eye performance controls from
[PR #248](https://github.com/rsasaki0109/Calibrex/pull/248) used its OXTS reference
with generated sensor motions. Those were performance controls, not independent
LiDAR calibration evidence, but their INS reference has been seen and must not
be presented as unseen evaluation data.

Boreas `boreas-2020-11-26-13-58` is development-only. Its first 64 KiB of the INS
CSV and 401-byte calibration were inspected for input-format validation. The
subsequent acquisition recipe fixes all first 600 metadata-listed scans, the
first 8 MiB of the INS CSV and the reference calibration, with an additional
raw-input budget below 4 GiB. Each object is checked against its declared size
and ETag and receives a SHA-256 receipt. Failed acquisition attempts are kept.
The CSV prefix is clipped only to its last complete newline, with a separate
derived-file digest. No evaluation-candidate objects are fetched by this recipe.

The `2011_09_30_0018` generated-motion control must not supply independent
accuracy, coverage or detection evidence for the proposed audit. KITTI and
Boreas development observations must be distinguished from synthetic controls.

## Implemented preparation

- `calibrex.data.boreas_ins_lidar` reads the independent Applanix-frame
  `gps_post_process.csv`, six-field Velodyne records, and the separate reference
  extrinsic. Sensor-pose CSVs are refused as the independent trajectory.
- INS CSV timestamps are seconds; scan filenames are UTC microseconds. Passive
  rotations follow the official ordered `Rx(-roll) Ry(-pitch) Rz(-heading)`
  convention. Noncommuting rotation and timestamp controls test both points.
- `calibrex.evaluation.boreas_ins_lidar` reuses the native KITTI hand-eye solver,
  defaults, held-out blocking, jackknife, controls and artifact schema. Scan
  registration receives LiDAR alone, with no INS or reference-extrinsic prior.
  The present baseline treats scans as rigid snapshots, without per-point deskew.
- `tools/score_ins_lidar_sota.py` scores only data-estimated rotation axes. It
  retains missing runs in the denominator, checks finite reference evidence and
  held-out controls, and reports each family's completion and axis coverage.
  It does not assign SOTA verdicts or invent an absolute clock-error metric.

The development calibration ran from a separate checkout pinned to
`a795fb659cc13f12a803642677be01f7fadb012c`. The complete acquisition receipt was
verified before any scan entered the method. All 600 scans and 599 consecutive
registrations were retained; all registrations converged. The
[native artifact](../assets/ins_lidar_sota_readiness/development/development.yaml)
and [run report](../assets/ins_lidar_sota_readiness/development/report.json)
record two observed rotation axes, maximum reference difference
**0.19184132984568794 deg**, and **0/2 missed 1 deg controls**.
The overall artifact remains inconclusive for unobserved quantities; these
development numbers do not establish a full calibration or a SOTA result.

Four raw-object transfers disconnected during the initial acquisition. The
[failed receipt](../assets/ins_lidar_sota_readiness/development/acquisition-failed.json),
[exact-object recovery](../assets/ins_lidar_sota_readiness/development/recovery.json),
and [completed receipt](../assets/ins_lidar_sota_readiness/development/acquisition-recovered.json)
are retained. The calibration itself executed once. No method has read the
evaluation candidates during this stage.

The proposed error ceiling is `ceil(1.5 * 0.19184132984568794 / 0.25) * 0.25`,
or **0.5 deg**. We choose a **0.5 coverage floor in each family**, stricter than
the handoff's development-derived rounded floor of 0.25. We also require zero
control misses in the finite declared evaluation, a stricter integrity gate
than its binomial allowance. Two development controls cannot establish a
population detection rate or a future 95% sensitivity guarantee.

The fixed cases are KITTI `2011_09_29_0071`, `2011_09_30_0020`,
`2011_09_30_0027` and Boreas `boreas-2020-12-01-13-26`, first 600 scans each.
The [KITTI inventory](../assets/ins_lidar_sota_readiness/kitti_metadata.json)
selects the first three eligible records from the existing eighteen-archive
metadata pool by order and frame counts. Camera-LiDAR v2 previously read their
Velodyne and OXTS streams; this cross-pair exposure is disclosed. KITTI odometry
aliases 06 and 07 map to the selected 0020 and 0027 raw drives; alias 05 maps to
the excluded, spent 0018. The mapping was checked against the official devkit
readme; its archive URL and digest are in the registration.

## Completed evaluation

Code was frozen at `5cc11d43e313bec6d29f3de5e549493eee594352` and the binding
registration committed separately, then merged in PR #301 before sensor
acquisition. Evaluation ran once from merged `main` at `e2ac3e20`, with no method
reruns or acquisition retries. The 16 GiB raw-data cap and 20 GiB free-disk floor
were preserved.

All four commands completed. Eight axes were estimated, with maximum reference
difference 0.285729 deg and all eight observed controls detected. KITTI 0071
provided no observable rotation, so completeness and empty-control integrity
fail. The [result page](../benchmarks/ins_lidar_rotation_v1.md) publishes all cases
and the immutable 6/8 refuted verdict. All four windows are now spent for
INS-LiDAR evaluation; they are available for development, not a second claim.
