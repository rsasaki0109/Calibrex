# MID360 IMU-LiDAR Rotation

This page reports Calibrex's native calibration of the Livox MID360 built-in
IMU against LiDAR odometry. The run estimates the rotation of `T_lidar_imu`,
the clock offset (`t_imu = t_lidar + dt`), and the gyro bias. The reference is
the MID360 design value: IMU axes aligned with the LiDAR. It is used only for
this comparison, and it is not a measurement.

Two recordings from two dataset families are used:

| Recording | Motion | Artifact |
| --- | --- | --- |
| [Zenodo 14841855](https://zenodo.org/records/14841855) "Driving SLAM Test with Livox MID360" | vehicle, 277 s | [`mid360_imu_lidar_rotation_driving_slam.yaml`](../assets/mid360_imu_lidar_rotation_driving_slam.yaml) |
| [RTK-SLAM](https://rtk-slam-dataset.github.io/) `stadtgarten_seq2` | hand-held, 15 min | [`mid360_imu_lidar_rotation_rtk_slam_seq2.yaml`](../assets/mid360_imu_lidar_rotation_rtk_slam_seq2.yaml) |

## Results

Both runs are `inconclusive`: **no rotation axis is calibrated to the 0.1 deg
policy threshold yet.**

| Quantity | Driving (vehicle) | RTK-SLAM seq2 (hand-held) |
| --- | --- | --- |
| roll | -0.48 ± 0.11 deg, control detected | -0.49 ± 0.17 deg, control detected |
| pitch | -0.04 ± 0.34 deg, control detected | +0.54 ± 0.19 deg, control detected |
| yaw | +0.47 ± 0.28 deg, **1 deg control not detected** | -1.40 ± 0.18 deg, control detected |
| clock offset | -0.5 ± 1.7 ms (estimated) | +10.0 ± 0.9 ms (estimated) |
| gyro bias | estimated, all axes | estimated, all axes (x: 0.016 rad/s) |
| held-out / training median rate residual | 0.0055 / 0.0055 rad/s | 0.080 / 0.032 rad/s |

Values are relative to the design reference. The reported std is the larger
of the analytic std and an 8-group window jackknife.

### Reading the results

- **The vehicle recording cannot determine yaw.** It rotates almost only
  about the vertical axis, so a 1 deg yaw shift is invisible to held-out
  windows (Δχ² 1.5). The artifact reports yaw as unobservable instead of
  trusting its value.
- **The hand-held recording excites all axes**, and every 1 deg control is
  detected. Even so, its held-out rate residual is 2.5 times the training
  residual, and its yaw sits 1.4 deg from the design value. We read this as a
  limit of the method rather than of the sensor. The fit compares mean rates
  (the LiDAR rotation over one period divided by its duration) with gyro
  means. When fast hand-held motion changes the rotation axis within a
  period, the two differ by a coning term. That term is second order in the
  rotation per period, but systematic.
- **The next step is on-manifold gyro preintegration.** The fit would then
  compare integrated gyro rotation increments with LiDAR rotation increments
  directly, instead of comparing mean rates. Until that lands, neither
  recording supports a rotation claim.

## Method

1. **Motion-compensated odometry.**
   - Each sweep is deskewed from per-point capture times under a
     constant-velocity model, re-registered with the refined motion, and
     matched against the union of the last five scans.
   - On RTK-SLAM, deskewing cut the one-second distance error against RTK
     from 1.94 cm to 1.53 cm (p90 5.89 cm to 3.95 cm).
2. **Rate alignment.** For each pair of consecutive scans, the LiDAR mean rate
   equals `R_lidar_imu (mean_gyro[t + dt] - b)`. Gyro interval means come
   from a cumulative trapezoid integral. The rotation is initialized by
   Procrustes, which handles any mounting, then refined with Huber IRLS.
3. **Held-out evidence.**
   - Every third 10-second window is held out.
   - Each rotation axis is shifted by 1 deg and the clock offset by 10 ms,
     and held-out windows must detect each shift (Δχ² ≥ 9).
   - A `pass` requires roll, pitch, and yaw to be estimated.

Recording conventions are explicit per stream profile:

| Profile | Point topic | Point time | Acceleration |
| --- | --- | --- | --- |
| `rtk-slam` | `/livox/points` | `offset_time`, seconds after the header | m/s² |
| `livox-ros-driver2` | `/livox/lidar` | `timestamp`, absolute nanoseconds | g |

```bash
calibrex imu-lidar livox BAG_DIR --profile livox-ros-driver2 \
  --dataset-family zenodo_driving_slam_mid360 --dataset-license "CC BY 4.0" \
  --output driving.yaml
```

## Limitations

- The design reference is not a measurement.
- The translation of `T_lidar_imu` is not estimated. It needs the
  accelerometer and a full LiDAR-inertial model.
- The two recordings come from two different MID360 units, so their
  rotations need not agree.
