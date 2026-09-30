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

| Axis | Estimate | Reported std (analytic / jackknife) | 1 deg held-out control | vs. KITTI calib | vs. OXTS-motion vehicle frame |
| --- | ---: | --- | ---: | ---: | ---: |
| roll | 0.22 deg | 0.42 (0.07 / 0.42) deg, unobservable | Δχ² 47 | +1.07 deg | +0.96 deg |
| pitch | 0.62 deg | 0.04 (0.01 / 0.04) deg | Δχ² 986 | +0.50 deg | +0.57 deg |
| yaw | -0.26 deg | 0.05 (0.01 / 0.05) deg | Δχ² 816 | -0.31 deg | -0.15 deg |

- **Motions.** 1998 motions. The median normalized residual is 0.35 on
  train and 0.39 held out. The nuisance lever is 0.65 m.
- **References on the held-out blocks.** Both references cost Δχ² 245-272.
- **Yaw** is 0.15 deg (2.8 std) from the OXTS-motion vehicle frame, and
  0.31 deg from KITTI's OXTS frame. The OXTS unit itself sits 0.3-0.8 deg
  in yaw off its own direction of motion.
- **Pitch is about 0.5 deg from both references, and the likely cause is
  the odometry, not the rotation.** The INS-LiDAR hand-eye on the same rig
  ([KITTI INS-LiDAR](kitti_ins_lidar.md)) matches KITTI's pitch within
  0.06 deg, so the LiDAR-to-OXTS rotation is consistent. The disagreement
  is between the direction of the LiDAR odometry's translation and that of
  the OXTS velocity. KITTI's HDL-64E is known to have vertical-angle
  calibration errors that make LiDAR odometry drift vertically, which would
  bias exactly this axis. This is not verified.

[Artifact](../assets/kitti_lidar_vehicle/dev_2011_09_26.yaml).

## Limitations

- The vehicle frame assumes no side slip and no vertical velocity, so it
  depends on the sensor odometry's vertical accuracy.
- Roll needs turns.
- The translation of `T_vehicle_sensor` is not estimated.

No SOTA claim is made for lidar-vehicle.
