# Hilti 2022 Camera-IMU Rotation

Calibrex's first targetless camera-IMU calibration. It estimates the rotation
of `T_cam_imu`, the clock offset (`t_imu = t_cam + dt`), and the gyro bias,
by aligning camera rotations from tracked image features with the gyro. No
calibration target is needed.

```bash
pip install "calibrex[opencv]"
calibrex camera-imu rotation exp21_ros2 \
  --camchain calibration_files/calib_3_cam0-1-camchain-imucam.yaml --camera cam0 \
  --imu-topic /alphasense/imu \
  --dataset-family hilti2022 --dataset-license "Hilti SLAM Challenge 2022 terms" \
  --output cam0.yaml
```

The camchain supplies the camera's intrinsics. Its `T_cam_imu` and
`timeshift_cam_imu` are used only as a comparison. They come from Hilti's
target-based Kalibr calibration, which is a reference, not ground truth.

## Method

1. **Camera rotations.**
   - Shi-Tomasi features are tracked with pyramidal Lucas-Kanade across every
     4th frame (10 Hz), and undistorted with the camera model.
   - The frame-to-frame rotation comes from the essential matrix (5-point
     RANSAC with cheirality). A pure-rotation fit (bearing Kabsch in RANSAC)
     is the fallback, and it replaces a degenerate essential solution.
   - The essential matrix is preferred because walking translation produces
     image flow that a pure rotation absorbs as a rotation of about
     t / depth. On exp21, a configuration that used pure rotation for about
     90 % of the pairs was 0.3-0.5 deg further from Kalibr.
   - A failed pair ends the track segment; zero motion is never assumed.
2. **Evidence.** The same as for [MID360 IMU-LiDAR](mid360_imu_lidar.md):
   - every third 10-second window is held out;
   - an 8-group window jackknife;
   - 1 deg and 10 ms known-bad controls that held-out windows must detect.
3. **Rotation uncertainty and reference differences** are small rotations
   about the camera axes, so they stay meaningful for cameras mounted near
   an Euler singularity. The Alphasense cameras are rotated by about 90 deg
   from the IMU. The reported roll, pitch, and yaw values are xyz Euler
   angles.

## Development results

Only exp21 and exp07 were used, for development. exp01-exp04 are held back,
untouched, for a pre-registered audit.

| Camera | Recording | Difference to Kalibr about camera x / y / z (deg) | Reported std (deg) | dt (ms), Kalibr | Controls detected | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| cam0 | exp21 | +0.01 / -0.29 / -0.03 | 0.12 / 0.15 / 0.12 | 1.76, 1.91 | 4/4 | inconclusive |
| cam1 | exp21 | -0.11 / -0.40 / -0.16 | 0.08 / 0.12 / 0.11 | 1.75, 1.90 | 4/4 | inconclusive |
| cam2 | exp21 | -0.28 / -0.24 / +0.20 | 0.08 / 0.27 / 0.14 | 1.61, 1.94 | 4/4 | inconclusive |
| cam3 | exp21 | -0.18 / -0.50 / +0.00 | 0.12 / 0.12 / 0.15 | 1.85, 1.81 | 4/4 | inconclusive |
| cam4 | exp21 | -0.42 / -0.67 / -0.19 | 0.25 / 0.24 / 0.31 | 1.51, 1.71 | 4/4 | inconclusive |
| cam0 | exp07 | -0.82 / -0.24 / -0.65 | 0.76 / 0.51 / 0.88 | 0.95, 1.91 | 3/4 | inconclusive |
| cam1 | exp07 | -0.32 / -0.08 / -0.24 | 0.44 / 0.23 / 0.30 | 1.41, 1.90 | 3/4 | inconclusive |
| cam2 | exp07 | -0.53 / -0.62 / +0.65 | 0.43 / 0.41 / 0.54 | 1.78, 1.94 | 4/4 | inconclusive |
| cam3 | exp07 | -0.28 / -1.02 / -1.74 | 1.88 / 1.49 / 2.15 | 2.17, 1.81 | 3/4 | inconclusive |
| cam4 | exp07 | -0.49 / -0.49 / -0.06 | 0.97 / 0.09 / 2.10 | 1.39, 1.71 | 2/4 | fail |

Artifacts: [`docs/assets/hilti2022_camera_imu/`](https://github.com/rsasaki0109/Calibrex/tree/main/docs/assets/hilti2022_camera_imu).

**Findings**

- **The clock offset agrees with Kalibr to within 0.35 ms** on every exp21
  camera.
- **Rotation on exp21 is 0.0-0.7 deg from Kalibr**, but no camera passes: the
  reported std (0.08-0.3 deg) exceeds the 0.1 deg observability bound
  inherited from the IMU-LiDAR evaluation.
- **There is a systematic bias about the camera y axis.** It is negative for
  all five cameras (-0.24 to -0.67 deg on exp21), even though they face
  different directions. The cause is not identified; candidates are residual
  translation leakage into the rotation fits, and the Kalibr reference
  itself.
- **Tracking fails in the exp07 corridor.** 40-60 % of the frame pairs fail,
  and the std grows to 0.4-2 deg; one camera fails its held-out check.

No SOTA claim is made for camera-IMU. The y-axis bias has to be explained
first, and then an audit pre-registered on exp01-exp04 against Kalibr and a
targetless external baseline.
