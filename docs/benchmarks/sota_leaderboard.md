# SOTA Leaderboard

Calibrex may call a calibration method state of the art only for a claim that a
frozen audit marks `supported`. This page lists every audited claim, one
standing per sensor pair, and the target pairs that have no audited claim yet.

## Current standings

As of 2026-10-15, **three pairs have supported SOTA claims**:

- **`imu-lidar`** (Livox MID360, against LI-Init):
  - rotation and clock offset;
  - the full extrinsic, second round. The first full-extrinsic round was
    refuted and stays listed.
- **`lidar-vehicle`** (KITTI): the motion-only LiDAR-to-vehicle rotation,
  against KITTI's calib_imu_to_velo, on eight unseen drives.
- **`camera-imu`** (Hilti 2022, the two forward cameras, against Kalibr): honest
  per-axis verdicts, third round
  ([v3](hilti_camera_imu.md#pre-registered-audit-v3-supported-7-of-7-gates)).
  - On five unseen recordings, every axis the method judged was within 0.53 deg
    (gate 1.0) and every clock offset within 0.47 ms (gate 0.75). It judged 26 of
    30 axes.
  - The claim gates only the axes the method judges. v1
    ([refuted](hilti_camera_imu.md#pre-registered-audit-refuted)) and v2
    ([refuted](hilti_camera_sota.md#camera-imu-v2-refuted-5-of-7-gates)) required
    every axis constrained; both stay listed.

See [MID360 IMU-LiDAR Calibration](mid360_imu_lidar.md),
[KITTI LiDAR-Vehicle Rotation](kitti_lidar_vehicle.md#pre-registered-audit-supported)
and [Hilti 2022 Camera-IMU Rotation](hilti_camera_imu.md) for the scope of each claim
and what it does not claim.

Three pairs have only refuted audits:

- **`lidar-lidar`**: 2 of 4 gates
  ([NTU VIRAL LiDAR-LiDAR](ntu_viral_lidar_lidar.md#pre-registered-audit-refuted)).
- **`camera-camera`**: 4 of 6 gates
  ([Hilti camera SOTA audits](hilti_camera_sota.md#camera-camera-refuted-4-of-6-gates)).
  - Accuracy against Kalibr held: every scored pair was within 0.33 deg.
  - A 1 deg turn of a camera was flagged in only 70 percent of the trials and named
    in 48 percent.
- **`camera-lidar`**: two refuted audits of `calibrex check`'s per-axis verdicts.
  - **v2: 8 of 9 gates**
    ([KITTI camera-LiDAR check audit v2](kitti_camera_lidar_v2_check.md#results-refuted-8-of-9-gates)).
    - On 15 scored drives of KITTI's other four days, each checked against its own day's
      calibration, the check never flagged it. Every judged axis was within 0.45 deg,
      and all 36 3 deg and 36 1 deg turns about judged axes were flagged.
    - On one drive, in a turned run, it judged an axis 1.503 deg off against a 1.5 deg
      gate.
  - **v1: 5 of 9 gates**
    ([KITTI camera-LiDAR check audit](kitti_camera_lidar_check.md#results-refuted-5-of-9-gates)).
    - On 3 of 23 unseen 2011_09_26 drives the check flagged KITTI's calibration, judging
      an axis up to 9.6 deg off.

Every other target pair has a native method but no audited claim yet.

Work to add an `ins-lidar` claim is tracked in the
[cross-dataset audit plan](../research/ins_lidar_sota_plan.md): metadata-only
Boreas inventory, an independent INS input adapter and failure-aware scoring
are prepared. Development and pre-registration must finish before evaluation;
this preparation does not change the standings below.

| Pair | Standing | Supported | Refuted | Incomplete | Target |
| --- | --- | ---: | ---: | ---: | --- |
| `camera` | no_claim | 0 | 0 | 0 | yes |
| `camera-camera` | refuted | 0 | 1 | 0 | yes |
| `camera-imu` | supported | 1 | 2 | 0 | yes |
| `camera-lidar` | refuted | 0 | 2 | 0 | yes |
| `gnss-imu` | no_claim | 0 | 0 | 0 | yes |
| `imu-lidar` | supported | 2 | 1 | 0 | yes |
| `imu-vehicle` | no_claim | 0 | 0 | 0 | yes |
| `ins-lidar` | no_claim | 0 | 0 | 0 | yes |
| `lidar-lidar` | refuted | 0 | 1 | 0 | yes |
| `lidar-vehicle` | supported | 1 | 0 | 0 | yes |
| `lidar-wheel_odometry` | no_claim | 0 | 0 | 0 | yes |

| Pair | Category | Quantities | Verdict | Gates (achieved/total) | Protocol |
| --- | --- | --- | --- | --- | --- |
| `camera-camera` | targetless_camera_camera | rotation | refuted | 4/6 | `hilti-camera-camera-v1` |
| `camera-imu` | targetless_camera_imu | rotation, time_offset | refuted | 6/7 | `hilti-camera-imu-v1` |
| `camera-imu` | targetless_camera_imu | rotation, time_offset | refuted | 5/7 | `hilti-camera-imu-v2` |
| `camera-imu` | targetless_camera_imu | rotation, time_offset | supported | 7/7 | `hilti-camera-imu-v3` |
| `camera-lidar` | targetless_camera_lidar_check | rotation | refuted | 5/9 | `kitti-camera-lidar-check-v1` |
| `camera-lidar` | targetless_camera_lidar_check | rotation | refuted | 8/9 | `kitti-camera-lidar-check-v2` |
| `imu-lidar` | targetless_imu_lidar | rotation, translation, time_offset | refuted | 0/8 | `mid360-imu-lidar-extrinsic-vs-li-init-v1` |
| `imu-lidar` | targetless_imu_lidar | rotation, translation, time_offset | supported | 4/4 | `mid360-imu-lidar-extrinsic-vs-li-init-v2` |
| `imu-lidar` | targetless_imu_lidar | rotation, time_offset | supported | 4/4 | `mid360-imu-lidar-rotation-vs-li-init-v1` |
| `lidar-lidar` | targetless_lidar_lidar | rotation, translation | refuted | 2/4 | `ntu-lidar-lidar-v1` |
| `lidar-vehicle` | targetless_lidar_vehicle | rotation | supported | 3/3 | `kitti-lidar-vehicle-v1` |

The machine-readable version is
[`sota_leaderboard.json`](../assets/sota_leaderboard.json)
(`slac.sota_leaderboard/v0.1`).

## What a standing means

| Standing | Meaning |
| --- | --- |
| `supported` | At least one audited claim for this pair meets every frozen required gate and its dataset-family and independent-rig coverage. |
| `refuted` | No claim is supported, and at least one required gate of an audited claim is contradicted by its evidence. |
| `incomplete` | Claims exist, but evidence or coverage is missing, so nothing may be claimed. |
| `no_claim` | A target pair without any audited claim. |

The counts for each pair are kept separate, so a refuted claim is never hidden
behind a supported one. The leaderboard copies verdicts from the audits and
never computes its own.

## Auditing a claim

A claim is scoped by its modalities, the quantities it estimates, and a
method category. For example, with illustrative thresholds:

```yaml
schema_version: slac.sota_audit_protocol/v0.1
protocol_id: kitti-ins-lidar-hand-eye-v1
scope:
  modalities: [ins, lidar]
  quantities: [rotation, translation]
  category: trajectory_hand_eye
claim_text: >-
  Native INS-LiDAR hand-eye calibration matches the KITTI vendor calibration
  within 0.2 deg and 5 cm (p95) across two sequences.
minimum_dataset_families: 2
minimum_independent_rigs: 1
requirements:
  - requirement_id: kitti-rotation-p95
    phase: phase2
    description: p95 rotation error to the vendor calibration
    dataset_family: kitti_raw
    independent_rig: true
    rig_id: kitti-2011-09-26
    evidence_path: kitti-ins-lidar-benchmark.yaml
    evidence_sha256: <SHA-256 of the benchmark artifact>
    method_id: calibrex_native_hand_eye
    metric: rotation_error_deg
    statistic: metric_p95
    comparison: less_equal
    threshold: 0.2
provenance:
  generator: manual
  generator_version: "1"
```

Each piece of evidence is a digest-pinned `slac.benchmark/v0.1` artifact whose
provenance declares `data_verified: true`. Freeze the protocol before looking
at the results, then run:

```bash
calibrex sota audit protocol.yaml --output audit.yaml
```

The command exits with 0 only for a `supported` verdict.
`calibrex camera-lidar audit-sota` keeps accepting Camera-LiDAR protocols, and
both commands evaluate claims with the same engine.

## Regenerating this page

```bash
calibrex sota leaderboard AUDIT... \
  --target-pair camera-lidar --target-pair lidar-lidar \
  --target-pair imu-lidar --target-pair camera-imu --target-pair camera \
  --target-pair camera-camera \
  --target-pair gnss-imu --target-pair ins-lidar \
  --target-pair lidar-vehicle --target-pair imu-vehicle \
  --target-pair lidar-wheel_odometry \
  --output docs/assets/sota_leaderboard.json --markdown standings.md
```

Paste `standings.md` into the table above.
