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
   - Images are contrast-equalized (CLAHE). Up to 600 Shi-Tomasi features
     (quality 0.001) are tracked with pyramidal Lucas-Kanade across every
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
untouched, for a pre-registered audit. The artifacts in
[`docs/assets/hilti2022_camera_imu/`](https://github.com/rsasaki0109/Calibrex/tree/main/docs/assets/hilti2022_camera_imu)
come from the current version: CLAHE, and 600 features at quality 0.001.

| Camera | Recording | Difference to Kalibr about camera x / y / z (deg) | Reported std (deg) | dt (ms), Kalibr | Controls detected | Failed pairs |
| --- | --- | --- | --- | --- | --- | ---: |
| cam0 | exp21 | -0.10 / -0.35 / +0.09 | 0.13 / 0.08 / 0.21 | 1.76, 1.91 | 4/4 | 0 / 1527 |
| cam1 | exp21 | -0.00 / -0.15 / +0.09 | 0.12 / 0.19 / 0.18 | 1.98, 1.90 | 4/4 | 0 / 1527 |
| cam2 | exp21 | -0.12 / -0.22 / +0.12 | 0.14 / 0.16 / 0.23 | 1.70, 1.94 | 4/4 | 0 / 1527 |
| cam3 | exp21 | -0.62 / -0.32 / +0.08 | 0.14 / 0.10 / 0.30 | 1.56, 1.81 | 3/4 | 2 / 1527 |
| cam4 | exp21 | -0.43 / -0.80 / -0.31 | 0.15 / 0.12 / 0.23 | 2.03, 1.71 | 2/4 | 5 / 1527 |
| cam0 | exp07 | -0.23 / -0.20 / -0.20 | 0.12 / 0.24 / 0.19 | 2.06, 1.91 | 4/4 | 16 / 1321 |
| cam1 | exp07 | -0.16 / -0.05 / -0.32 | 0.10 / 0.18 / 0.18 | 1.90, 1.90 | 4/4 | 17 / 1321 |
| cam2 | exp07 | -0.74 / -1.67 / +0.44 | 0.57 / 0.62 / 0.41 | 1.88, 1.94 | 3/4 | 9 / 1321 |
| cam3 | exp07 | -0.40 / -0.79 / +0.38 | 0.45 / 0.57 / 0.64 | 2.65, 1.81 | 4/4 | 42 / 1321 |
| cam4 | exp07 | -1.14 / +0.25 / +0.72 | 0.35 / 0.60 / 0.70 | 1.77, 1.71 | 3/4 | 137 / 1321 |

Every verdict is `inconclusive`: the reported std exceeds the 0.1 deg bound
inherited from the IMU-LiDAR evaluation.

**Contrast equalization: what changed.** The first version tracked without
CLAHE, using 400 features at quality 0.01.

- **Failed frame pairs dropped sharply.** On the exp07 corridor, cam0 went
  from 577 to 16 of 1321, and cam4 from 532 to 137.
- **The forward cameras on exp07 improved.** Their differences shrank from
  up to 0.8 deg to 0.3 deg, and their std from 0.4-0.9 deg to 0.1-0.24 deg.
- **The side-looking cameras did not improve, and two got worse.**
  - On exp07, cam2-cam4 are still 0.4-1.7 deg from Kalibr.
  - On exp21, cam3 roll moved from -0.18 to -0.62 deg, and cam4 detects only
    2 of 4 controls.
  - Near walls and large parallax are the likely cause, but this is not
    verified.

**The clock offset** is within 0.35 ms of Kalibr on every camera except
cam3 on exp07 (0.84 ms).

## A rig-level offset about the IMU vertical axis

`tools/analyze_camera_imu_consistency.py` expresses each camera's
difference to Kalibr about the IMU axes. It also computes camera-to-camera
rotations, in which any common IMU-frame offset cancels.

| exp21 | First version | Current version |
| --- | --- | --- |
| Common IMU-frame offset (deg) | (-0.14, +0.07, -0.41) | (+0.05, +0.10, -0.35) |
| Per-camera residual after removing it (deg) | 0.17-0.40 | 0.21-0.69 |

**What the evidence shows**

- **A rotation of about -0.4 deg about the IMU vertical axis is shared by
  every camera.** The cameras face forward (cam0, cam1), down (cam2), left
  (cam3), and right (cam4).
- **It is not translation leakage.** The opposite-facing cam3 and cam4 carry
  the same sign, whereas leakage would flip it.
- **It is not coning.** The rotation increments are already preintegrated.
- **Gyro non-orthogonality is not supported.** A full linear fit
  `w_cam = A w_imu` on exp21 (cam0 and cam3) gives the same antisymmetric
  part, about -0.48 deg about IMU z, but inconsistent symmetric parts.

**What remains open**

The offset lies between the gyro frame, which this method measures, and
Kalibr's IMU frame. Kalibr's report lists the IMU model as `calibrated`: the
gyro and the accelerometer are assumed to share one frame. A rotation between
them inside the IMU would appear exactly like this, but the data at hand
cannot distinguish that from an error in the reference.

## Next steps

- Understand the side-camera errors.
- Decide how the audit treats a rig-level IMU-frame offset. The rotation of
  each camera relative to the gyro frame is what this method estimates.
- Pre-register an audit on exp01-exp04 against Kalibr and a targetless
  external baseline.

No SOTA claim is made for camera-IMU.
