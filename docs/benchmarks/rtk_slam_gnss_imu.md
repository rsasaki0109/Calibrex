# RTK-SLAM GNSS-IMU Lever Arm

The GNSS antenna position in the MID360 IMU frame. It is **composed** from
two pieces of evidence, each pinned by its SHA-256:

- the [GNSS-LiDAR lever arm](rtk_slam_gnss_lidar.md);
- the [IMU-LiDAR rotation and translation](mid360_imu_lidar.md).

The formula is `p_imu = R_lidar_imu^T (p_lidar - t_lidar_imu)`, with the std
propagated from the inputs:

```bash
calibrex gnss-imu compose \
  --gnss-lidar rtk_slam_gnss_lidar_stadtgarten_seq2.yaml \
  --imu-lidar-rotation mid360_imu_lidar_rotation_rtk_slam_seq2.yaml \
  --imu-lidar-translation mid360_imu_lidar_translation_rtk_slam_seq2.yaml \
  --reference 0.023 -0.023 0.090 --output gnss_imu.yaml
```

An axis is `estimated` only when two conditions hold:

- its propagated std is at most 2 cm;
- no input axis it depends on is unobservable.

The composition never upgrades an input's verdict. The reference is the
dataset's CAD value (`calib.yaml` `gnss_antenna_phase_center`, from the IMU
origin).

| Recording | x | y | z | Verdict |
| --- | --- | --- | --- | --- |
| stadtgarten_seq2 | 30.3 ± 11.6 mm (CAD +7.3) | -26.5 ± 7.2 mm (CAD -3.5) | 80.8 ± 40.4 mm, unobservable (CAD -9.2) | inconclusive |
| stadtgarten_seq1 | 26.8 ± 13.0 mm, unobservable (CAD +3.8) | -31.2 ± 9.0 mm (CAD -8.2) | 113.0 ± 13.3 mm, unobservable (CAD +23.0) | inconclusive |

- **Where the unobservable axes come from.** On seq2, z inherits the
  GNSS-LiDAR z std of 4 cm. On seq1, x inherits the IMU-LiDAR translation's
  unobservable x, and z inherits the GNSS-LiDAR's unobservable z.
- **Agreement with CAD.** Where estimated, the axes agree with the CAD value
  within 1 cm.
- **It is a composition, not a direct fit.** The std assumes the inputs'
  errors are independent, and a direct GNSS-IMU fit is not implemented.

Artifacts: [`docs/assets/rtk_slam_gnss_imu/`](https://github.com/rsasaki0109/Calibrex/tree/main/docs/assets/rtk_slam_gnss_imu).

No SOTA claim is made for gnss-imu.
