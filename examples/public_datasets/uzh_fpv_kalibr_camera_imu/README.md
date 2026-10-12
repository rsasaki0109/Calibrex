# UZH-FPV: a public Kalibr camchain on a separately captured recording

UZH-FPV publishes the Kalibr camera-IMU calibration of each sensor setup, together with
the raw calibration recordings. This example evaluates the **indoor-forward Snapdragon**
camchain on the **outdoor-forward** recording of the same camera, which is a recording
Kalibr did not fit.

The Kalibr camchain, `camchain-imucam-..indoor_forward_calib_snapdragon_imu.yaml`:

- **Fitted on:** `indoor_forward_calib_snapdragon_imu.bag`.
- **Translation `t_cam_imu`:** `[0.0217, -0.0001, -0.0005] m`.
- **`timeshift_cam_imu`:** -16.7 ms.

The evaluation recording is `outdoor_forward_calib_snapdragon_imu.bag`:

- `/snappy_cam/stereo_l`, 640x480 at 30 Hz.
- `/snappy_imu` at 500 Hz.
- An AprilGrid of 5x4 tags (7.5 cm).

**Data and license boundary.** The files are listed with their SHA-256 in `files.yaml`.
They are about 1.9 GB and live in `data/public`, outside the repository. The dataset
page states no license (checked 2026-10-12). Its companion code, `uzh-rpg/uzh_fpv_open`,
is GPL-3.0 and is not used. Cite Delmerico et al., *Are We Ready for Autonomous Drone
Racing? The UZH-FPV Drone Racing Dataset*, ICRA 2019.

## Reproduce

```bash
python3 tools/download_public_dataset.py uzh_fpv_kalibr_camera_imu
D=data/public/uzh_fpv_kalibr_camera_imu
CAMCHAIN="$D/indoor_forward_calib_snapdragon/camchain-imucam-..indoor_forward_calib_snapdragon_imu.yaml"

# Camera poses against the AprilGrid, plus gyro and accelerometer (needs OpenCV;
# pip install opencv-python-headless). Calibrex itself does not depend on OpenCV.
python tools/aprilgrid_camera_imu_recording.py "$D/outdoor_forward_calib_snapdragon_imu.bag" \
    --camchain "$CAMCHAIN" --target "$D/indoor_forward_calib_snapdragon/target.yaml" \
    --image-topic /snappy_cam/stereo_l --imu-topic /snappy_imu --detect-scale 2 \
    --output "$D/outdoor_forward_calib_snapdragon_recording.json"

calibrex external-run import-kalibr "$CAMCHAIN" --output run.json \
    --input-artifact "$D/indoor_forward_calib_snapdragon_imu.bag" \
    --tool-version "Kalibr (UZH-FPV release; version not stated)" \
    --training-isolation-declared \
    --training-isolation-evidence "fitted on indoor_forward_calib_snapdragon_imu.bag"
calibrex external-run evaluate-camera-imu run.json \
    --recording "$D/outdoor_forward_calib_snapdragon_recording.json" \
    --output evidence.json --json
```

The opt-in test `tests/integration/test_uzh_fpv_kalibr_camera_imu.py` runs this check and
the perturbed candidates below.

## Result (2026-10-12)

**Camera poses.** 1227 of 1438 frames were posed, with a median reprojection error of
0.73 px. Frames with fewer than 4 tags or more than 1 px RMS were dropped.

**Translation (accelerometer): PASS.** The data were split into 477 train and 337 holdout
windows of +-0.25 s.

| Check | Result | Gate |
| --- | ---: | ---: |
| Holdout specific-force RMSE | 0.102 m/s^2 (train 0.101) | ≤ 0.3 m/s^2 |
| Lever-arm excitation (weakest axis) | 2.68 s^-2 | ≥ 0.5 s^-2 |
| +-5 cm controls detected | 6/6 (+0.025 to +0.070 m/s^2) | ≥ 80 % |

The train-only fit is plausible:

- The fitted gravity magnitude is 9.93 m/s^2.
- The fitted accelerometer bias is `[0.057, -0.058, -0.011] m/s^2`.

**Known-bad candidates on the same recording.** The published camchain was shifted by
+-5 cm along each camera axis, giving six candidates. Every one fails: holdout RMSE is
0.127 to 0.172 m/s^2, and the opposite control fits better.

**Rotation and time shift (gyroscope): not admissible on this recording.** The check
reports `fail`:

- The holdout angular-rate RMSE is 0.171 rad/s against a 0.05 rad/s budget.
- None of the eight rotation and time controls is detected.

The camera rates come from differencing consecutive AprilGrid poses at 30 Hz. Pose noise
of a few tenths of a degree is therefore amplified to about 0.17 rad/s, so this says
nothing against Kalibr's rotation. One signal does stand out: the +10 ms time control
*lowers* the residual (by 0.007 rad/s). That is consistent with UZH-FPV's own outdoor
Kalibr run, which reports a time shift 8.7 ms later (-8.0 ms) than the indoor one. It is
not a gated finding.

**Also tried: the DAVIS346 setup.** This is the indoor-forward DAVIS camchain on the
outdoor DAVIS recording, with 346x260 images.

- Only 1014 of 2322 frames could be posed, with a 0.51 px median reprojection error.
- The poses are too noisy for either check.
- The holdout specific-force RMSE is 0.339 m/s^2, over the 0.3 m/s^2 budget, so the
  verdict is `fail`.
- The angular-rate RMSE is 0.887 rad/s.

No accuracy claim is made for that pairing.
