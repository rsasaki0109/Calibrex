# SOTA Leaderboard

Calibrex may call a calibration method state of the art only for a claim that a
frozen audit marks `supported`. This page lists every audited claim, one
standing per sensor pair, and the target pairs that have no audited claim yet.

## Current standings

As of 2026-09-30, **one pair has a supported SOTA claim: `imu-lidar`.**
The claim covers the rotation and clock offset of the Livox MID360 built-in
IMU, against LI-Init, on two recordings from two dataset families.

A second, pre-registered `imu-lidar` claim is **refuted**. That claim added
the lever arm to the comparison, on two unseen recordings. Calibrex did not
return a passing lever-arm estimate on either recording: x stayed
unobservable. The refutation is kept next to the supported claim, not hidden
by it. See
[MID360 IMU-LiDAR Calibration](mid360_imu_lidar.md#extrinsic-audit-against-li-init-refuted)
for both claims, their scope, and what they do not claim. Every other target
pair is `no_claim`.

| Pair | Standing | Supported | Refuted | Incomplete | Target |
| --- | --- | ---: | ---: | ---: | --- |
| `camera` | no_claim | 0 | 0 | 0 | yes |
| `camera-imu` | no_claim | 0 | 0 | 0 | yes |
| `camera-lidar` | no_claim | 0 | 0 | 0 | yes |
| `gnss-imu` | no_claim | 0 | 0 | 0 | yes |
| `imu-lidar` | supported | 1 | 1 | 0 | yes |
| `imu-vehicle` | no_claim | 0 | 0 | 0 | yes |
| `ins-lidar` | no_claim | 0 | 0 | 0 | yes |
| `lidar-lidar` | no_claim | 0 | 0 | 0 | yes |
| `lidar-vehicle` | no_claim | 0 | 0 | 0 | yes |
| `lidar-wheel_odometry` | no_claim | 0 | 0 | 0 | yes |

| Pair | Category | Quantities | Verdict | Gates (achieved/total) | Protocol |
| --- | --- | --- | --- | --- | --- |
| `imu-lidar` | targetless_imu_lidar | rotation, translation, time_offset | refuted | 0/8 | `mid360-imu-lidar-extrinsic-vs-li-init-v1` |
| `imu-lidar` | targetless_imu_lidar | rotation, time_offset | supported | 4/4 | `mid360-imu-lidar-rotation-vs-li-init-v1` |

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
  --target-pair gnss-imu --target-pair ins-lidar \
  --target-pair lidar-vehicle --target-pair imu-vehicle \
  --target-pair lidar-wheel_odometry \
  --output docs/assets/sota_leaderboard.json --markdown standings.md
```

Paste `standings.md` into the table above.
