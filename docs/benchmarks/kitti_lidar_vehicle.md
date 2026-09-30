# KITTI LiDAR-Vehicle Rotation

Calibrex's first sensor-to-vehicle calibration. It estimates the rotation of
`T_vehicle_velodyne` from the vehicle's own motion, with no target and no
other sensor. The vehicle frame is x forward, y left, and z up.

```bash
calibrex lidar-vehicle kitti 2011_09_26_drive_0005_sync 2011_09_26_drive_0009_sync ... \
  --output lidar_vehicle.yaml
```

## Method

A road vehicle moves along its forward axis and turns about its vertical
axis (`calibrex.solvers.vehicle_frame_solver`). In the vehicle frame, the
LiDAR's velocity from its odometry therefore has:

- no vertical component;
- a lateral component equal only to the yaw rate times the LiDAR's distance
  ahead of the axle it turns about (a nuisance lever);
- an angular velocity along z, up to zero-mean pitch and roll rates.

Robust least squares over 0.2 s intervals gives `R_vehicle_velodyne`. Each
constraint pins different axes:

- **yaw** comes from the lateral velocity;
- **pitch** comes from the vertical velocity and the turning axis;
- **roll** comes only from the tilt of the turning axis, so it needs turns.

Velocities are chords expressed in each interval's midpoint orientation. In
the start orientation they would lean into turns by half the turned angle.

Evidence:

- 10-second blocks, with every third block held out;
- an 8-group block jackknife;
- a 1 deg rotation about each vehicle axis must raise the held-out
  chi-square by at least 9.

Two references are compared after the fit:

- **KITTI's `calib_imu_to_velo`.** This gives the OXTS frame, which need not
  be aligned with the vehicle.
- **A vehicle frame derived independently from the OXTS unit's own
  velocities and angular rates.** It uses the same solver applied to the
  INS, composed with `calib_imu_to_velo`.

## Development result (2011_09_26 drives 0005, 0009, 0014, 0015, 0022)

Drives 0027, 0028, 0035, 0039, 0046, 0051, 0057, and 0059 are held back
for a pre-registered audit.

Verdict: `inconclusive`. Pitch and yaw are estimated; roll is not (jackknife
std 0.42 deg: these drives turn too little).

| Axis | Estimate | Reported std (analytic / jackknife) | 1 deg held-out control | vs. KITTI calib (OXTS frame) | vs. OXTS-motion vehicle frame |
| --- | ---: | --- | ---: | ---: | ---: |
| roll | 0.22 deg | 0.42 (0.07 / 0.42) deg, unobservable | Δχ² 47 | +1.07 deg | -0.04 deg |
| pitch | 0.62 deg | 0.04 (0.01 / 0.04) deg | Δχ² 986 | +0.50 deg | +0.06 deg |
| yaw | -0.26 deg | 0.05 (0.01 / 0.05) deg | Δχ² 816 | -0.31 deg | -0.15 deg |

- **Pitch and yaw agree with the independent vehicle frame.** That frame is
  derived from the OXTS velocities, and the agreement is within 0.06 and
  0.15 deg: 1.5 and 2.7 reported std. On the held-out blocks, that reference
  costs Δχ² 17, against 272 for KITTI's OXTS frame.
- **KITTI's `calib_imu_to_velo` is not the vehicle frame.** The OXTS unit sits
  about 1 deg in roll and 0.5 deg in pitch off the motion-defined vehicle
  frame.
- **Correction to #82.** The first version of this page compared against an
  OXTS-motion frame built from KITTI's `vf, vl, vu` and `wf, wl, wu`. Those
  are forward/left/up components in a *level* frame that follows the heading,
  not the body frame: level = `Ry(pitch) Rx(roll)` body, checked to
  3e-4 rad/s on the rates. The resulting 0.5 deg pitch disagreement was
  attributed to HDL-64 odometry drift. That attribution was wrong; it came
  from the reference. The OXTS signals are now rotated into the body frame
  with each packet's roll and pitch.

[Artifact](../assets/kitti_lidar_vehicle/dev_2011_09_26.yaml).

## Pre-registered audit: supported

The claim, scoring, drives, and thresholds were committed in
[`kitti_lidar_vehicle_preregistration.yaml`](kitti_lidar_vehicle_preregistration.yaml)
(commit `705639b`), before lidar-vehicle ran on any evaluation drive. The
evaluation drives are 0027, 0028, 0035, 0039, 0046, 0051, 0057, and 0059.
Scoring runs `tools/score_kitti_lidar_vehicle.py`, and the audit is built by
`tools/build_kitti_lidar_vehicle_audit.py`.

**Verdict: `supported`, 3/3 gates**
([protocol](../assets/kitti_lidar_vehicle_sota_protocol.yaml),
[result](../assets/kitti_lidar_vehicle_sota_audit.yaml)).

| Gate | Observed | Threshold |
| --- | --- | --- |
| Calibrex estimates pitch and yaw on the pooled evaluation drives | yes (roll unobservable) | required |
| Paired improvement over KITTI's calib_imu_to_velo on held-out blocks, 95 % CI low | 0.051 (mean 0.080, CI high 0.110), better in 9 of 9 blocks | ≥ 0 |
| Mean per-drive pitch/yaw agreement with the OXTS-motion vehicle frame | 0.214 deg (KITTI's own rotation: 0.476 deg) | ≤ 0.5 deg |

- **Held-out score.** The mean held-out block score is 0.793 for Calibrex
  and 0.873 for KITTI's rotation, whose lever is also fitted on the train
  blocks.
- **Per-drive agreement** ranges from 0.01 deg (0057) to 0.39 deg (0039).
- **One held-out block was dropped for both methods.** In 0057 block 1 the
  car stood still, so neither method had a moving interval to score. The
  pre-registration did not cover this case; the builder drops such blocks
  for every method alike, and the benchmark lists them.

What the claim does not say:

- It covers pitch and yaw only; roll stays unobservable on these drives.
- The vehicle frame is the one the non-holonomic model defines.
- The only external comparison is the common practice of taking the INS
  frame as the vehicle frame.

## INS-vehicle (`calibrex imu-vehicle kitti`)

The same solver, applied to the OXTS unit's own body-frame velocity and
angular rates, estimates `R_vehicle_imu`:

```bash
calibrex imu-vehicle kitti <drives> --lidar-vehicle lidar_vehicle.yaml --output imu_vehicle.yaml
```

On the same five drives the verdict is `inconclusive`: roll 1.04 ± 0.46 deg,
pitch 0.47 ± 0.13 deg, and yaw -0.22 ± 0.15 deg. All three std are over
the 0.1 deg bound, because the 10 Hz INS velocities are noisier than the
LiDAR odometry chords. Every 1 deg held-out control is detected (Δχ² 40-917).

The **closure** composes the LiDAR-vehicle estimate with KITTI's
`calib_imu_to_velo`. It agrees with this INS-vehicle estimate within
(-0.03, -0.03, +0.09) deg: two sensors, one vehicle frame.
[Artifact](../assets/kitti_lidar_vehicle/imu_vehicle_dev_2011_09_26.yaml).

## Limitations

- The vehicle frame assumes no side slip and no vertical velocity.
- It depends on the vertical accuracy of the sensor's velocity.
- Roll needs turns.
- The translation of `T_vehicle_sensor` is not estimated.

No SOTA claim is made for lidar-vehicle.
