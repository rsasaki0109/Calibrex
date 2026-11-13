# INS-LiDAR rotation audit v2: refuted, 7 of 8 gates

The second pre-registered INS-LiDAR rotation audit is **refuted**, on coverage. On the
seven KITTI recordings the method estimated 9 of 21 rotation axes, so 57 % of the axes were
unobserved against a ceiling of 50 %. Every other gate held in both dataset families:

- no method failure on any of the ten recordings;
- every one of the 17 estimated axes within 0.373 deg of its reference (gate 0.5 deg);
- every one of the 17 held-out 1 deg controls on an estimated axis detected.

The standing of `ins-lidar` stays **refuted**. Dataset calibration is a reference
estimate, not independent metrology. No translation, full-extrinsic or absolute
clock-accuracy claim is made.

## What v2 asked

[v1](ins_lidar_rotation_v1.md) was refuted at 6 of 8, and not on accuracy: one KITTI
recording had no observable rotation axis, and v1 counted that as a failed recording and as
a missed control. v2 gates what the method judges and how much it judges:

- agreement and controls on the axes the method reports as estimated;
- coverage separately: at most half of the axes unobserved, on average, per dataset family;
- a recording with no estimated axis counts in coverage and is not a failure;
- a method failure (a crash, a timeout, no convergence) is still a failure.

No numeric gate is looser than v1's.

- [Pre-registration](ins_lidar_rotation_v2_preregistration.yaml), merged in
  [PR #379](https://github.com/rsasaki0109/Calibrex/pull/379) before any evaluation
  sensor data was read. Frozen code: `8751c3f6c3d7165041e898c8c6eb237d225cdcee`.
- Tools: `tools/ins_lidar_sota_v2_evaluate.py`, `tools/build_ins_lidar_sota_v2_audit.py`.
- Evaluation set, chosen from metadata only: the seven eligible KITTI raw recordings no
  INS-LiDAR method had read, and the first three complete Boreas recordings after v1's.
  The KITTI recordings' first 600 frames had been read by camera-lidar audits (disclosed
  in the plan).

## Result

| Gate | KITTI raw (7 recordings) | Boreas (3 recordings) |
| --- | --- | --- |
| completion: method failures | 0, achieved | 0, achieved |
| agreement: worst estimated axis, at most 0.5 deg | 0.373 deg, achieved | 0.178 deg, achieved |
| coverage: mean unobserved fraction, at most 0.5 | **0.571, contradicted** | 0.111, achieved |
| controls: missed on estimated axes | 0 of 9, achieved | 0 of 8, achieved |

Per recording (difference to the reference in degrees; `-` is an axis the method reported
as unobservable, with its reported std in parentheses):

| Recording | roll | pitch | yaw | estimated |
| --- | --- | --- | --- | --- |
| KITTI 2011_09_30_0028 | - (0.142) | -0.099 | -0.373 | 2/3 |
| KITTI 2011_09_30_0033 | - (0.201) | - (0.121) | - (0.126) | 0/3 |
| KITTI 2011_09_30_0034 | - (0.112) | -0.035 | -0.094 | 2/3 |
| KITTI 2011_10_03_0027 | +0.149 | +0.103 | - (0.227) | 2/3 |
| KITTI 2011_10_03_0034 | -0.082 | +0.171 | - (0.105) | 2/3 |
| KITTI 2011_10_03_0042 | - (3.339) | -0.172 | - (0.129) | 1/3 |
| KITTI 2011_10_03_0047 | - (2.354) | - (0.544) | - (0.290) | 0/3 |
| Boreas 2020-12-18-13-44 | -0.057 | -0.050 | -0.004 | 3/3 |
| Boreas 2021-01-15-12-17 | -0.030 | - (0.113) | +0.081 | 2/3 |
| Boreas 2021-01-19-15-08 | -0.010 | -0.099 | +0.178 | 3/3 |

An axis is estimated when its reported std is at most 0.1 deg.

What the table shows:

- **The method does not overclaim.** Where it reported an axis as estimated, the axis
  agreed with the reference and its control was detected, 17 times of 17.
- **On KITTI's first 600 scans it often cannot judge.** Two recordings gave no axis, and
  no KITTI recording gave all three.
  - Several unobserved axes miss the 0.1 deg ceiling narrowly: 0.105, 0.112, 0.121,
    0.126, 0.129 and 0.142 deg.
  - Their differences to the reference are small too, but the method did not claim them
    and they are not scored.
- **Boreas is observed well:** 8 of 9 axes.

The plan named this risk before the run: development had 4 of 9 KITTI axes unobserved
against the same ceiling, and 2011_10_03_0042 is a highway drive. The gate was not relaxed
for it, before or after.

## How the run went, and one amendment

The frozen runner ran once from merged `main` on 2026-11-13.

- Nine recordings were acquired and fitted in the first pass (08:54-09:22 UTC), each
  method attempt once.
- **2011_09_30_0028 was not acquired in that pass.** Both registered transport attempts of
  its 21 GB archive ended in a connection reset, a minute apart. By the registration the
  case was then missing and the audit incomplete.
- **Amendment 1** ([file](ins_lidar_rotation_v2_amendment_1.yaml),
  [PR #380](https://github.com/rsasaki0109/Calibrex/pull/380)) allowed up to three further
  acquisition rounds for that case, at least ten minutes apart. It changed nothing else.
  - It was written after the run and **before any result was read**: only attempt
    statuses had been looked at, and the run's files were committed unread first.
  - It is a change to a plan after registration, made by the repository owner's decision.
    A reader who does not accept it should read this audit as **incomplete** for KITTI.
  - The case was acquired in the first amended round (about 12:14 UTC) and fitted once.
- The exhausted fetch record is kept in
  `docs/assets/ins_lidar_sota_v2/evaluation/transport/`.

The amendment did not change the verdict's direction in the audit's favour: with the
tenth case the audit is refuted, and without it the audit would have been incomplete.

## Records

- Audit: [`ins_lidar_sota_audit.yaml`](../assets/ins_lidar_sota_v2/evaluation/audit/ins_lidar_sota_audit.yaml)
  and [`scores.json`](../assets/ins_lidar_sota_v2/evaluation/audit/scores.json).
- Run manifest and method artifacts:
  [`run.json`](../assets/ins_lidar_sota_v2/evaluation/runs/run.json) and one YAML and log
  per recording beside it.
- Acquisition receipts (input digests; the raw inputs were deleted after each fit):
  `docs/assets/ins_lidar_sota_v2/evaluation/inputs/`.
- Development on v1's four recordings:
  `docs/assets/ins_lidar_sota_v2/development/`.

## What follows

- `ins-lidar` stays refuted. The supported-pair count does not change.
- The finding is about coverage, not accuracy: on 60 s of KITTI driving, the default
  method judges under half of the rotation axes.
- All ten recordings are now spent for `ins-lidar`. KITTI's eligible pool is used up.
- A further audit would need longer windows or other data, registered anew. It is not
  planned.
