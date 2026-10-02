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

## Focal lengths against the gyro (`calibrex camera-imu focal`)

Tracked camera rotations scale with the focal length that the features were
normalized with:

- a rotation about the camera y axis reads as `theta fx_true / fx_used`;
- a rotation about the x axis reads as `theta fy_true / fy_used`;
- a rotation about the optical axis does not depend on the focal length.

With the camera-IMU rotation, clock offset, and gyro bias taken from a
rotation artifact, `calibrex camera-imu focal` regresses the camera's
angular rates on the gyro's, axis by axis (`calibrex.evaluation.camera_focal`).
The fit is robust and gives `fx_est = fx_used k_y` and `fy_est = fy_used k_x`,
with no calibration target. Evidence:

- every third window is held out, and the held-out ratios must agree within
  3 std;
- an 8-group jackknife sets the std;
- the optical-axis ratio `k_z` must be 1 within 3 std, otherwise something
  other than the focal length is scaling the rotations.

A synthetic test tracks with a focal length 3 % too long and recovers the
0.971 ratios on x and y, with 1.000 on z.

| exp21 | Ratio about x / y / z | Std | Held-out ratio | fx, fy estimate (Kalibr) | Verdict |
| --- | --- | --- | --- | --- | --- |
| cam0 | 0.995 / 0.998 / 1.001 | 0.005 / 0.007 / 0.002 | 0.993 / 1.005 / 1.009 | 350.7, 349.8 (351.3, 351.5) px | inconclusive (std over 0.005) |
| cam1 | 0.994 / 1.002 / 1.004 | 0.005 / 0.004 / 0.004 | 0.987 / 0.994 / 0.999 | 353.4, 350.6 (352.6, 352.9) px | **pass** |
| cam2 | 0.977 / 0.988 / 0.999 | 0.010 / 0.005 / 0.004 | 1.004 / 0.987 / 1.000 | 346.4, 342.9 (350.7, 350.9) px | inconclusive |
| cam3 | 0.977 / 1.013 / 0.997 | 0.006 / 0.013 / 0.003 | 1.015 / 0.989 / 1.001 | 357.5, 345.0 (353.0, 353.3) px | fail (held-out disagreement) |
| cam4 | 1.012 / 1.003 / 0.995 | 0.010 / 0.006 / 0.002 | 0.998 / 1.013 / 0.992 | 352.5, 356.1 (351.5, 351.8) px | inconclusive |

- **The forward cameras agree with Kalibr's focal lengths** within 0.5 %
  (about 2 px).
- **The side and down cameras scatter by 1-2 %** and fail or miss the bound,
  in line with their rotation results above.
- **The optical-axis control stays within 0.8 % of 1** on every camera.

Artifacts: `docs/assets/hilti2022_camera_imu/focal_cam*_exp21.yaml`.

## Development results

Only exp21 and exp07 were used, for development. exp01-exp04 are held back,
untouched, for a pre-registered audit. The artifacts in
[`docs/assets/hilti2022_camera_imu/`](https://github.com/rsasaki0109/Calibrex/tree/main/docs/assets/hilti2022_camera_imu)
come from the current version: CLAHE, and 600 features at quality 0.001.

| Camera | Recording | Difference to Kalibr about camera x / y / z (deg) | Analytic std (deg) | Jackknife std (deg) | Verdict | Static windows excluded | Controls detected | dt (ms) |
| --- | --- | --- | --- | --- | --- | ---: | --- | ---: |
| cam0 | exp21 | -0.19 / -0.33 / -0.25 | 0.07 / 0.08 / 0.07 | 0.12 / 0.12 / 0.15 | warn | 2 | 3/4 | 1.70 |
| cam1 | exp21 | -0.15 / -0.33 / -0.18 | 0.06 / 0.07 / 0.06 | 0.11 / 0.10 / 0.18 | pass | 2 | 4/4 | 1.83 |
| cam2 | exp21 | -0.28 / -0.24 / +0.15 | 0.08 / 0.09 / 0.10 | 0.26 / 0.15 / 0.24 | pass | 2 | 4/4 | 1.59 |
| cam3 | exp21 | -0.23 / -0.29 / -0.19 | 0.07 / 0.09 / 0.09 | 0.23 / 0.13 / 0.31 | inconclusive | 1 | 4/4 | 1.81 |
| cam4 | exp21 | -0.18 / -0.60 / -0.02 | 0.09 / 0.12 / 0.11 | 0.18 / 0.16 / 0.21 | pass | 1 | 4/4 | 1.85 |
| cam0 | exp07 | -0.23 / -0.20 / -0.20 | 0.11 / 0.14 / 0.11 | 0.12 / 0.24 / 0.19 | pass | 0 | 4/4 | 2.06 |
| cam1 | exp07 | -0.16 / -0.05 / -0.32 | 0.10 / 0.12 / 0.09 | 0.10 / 0.18 / 0.18 | pass | 0 | 4/4 | 1.90 |
| cam2 | exp07 | -0.74 / -1.67 / +0.44 | 0.13 / 0.14 / 0.20 | 0.57 / 0.62 / 0.41 | inconclusive | 0 | 3/4 | 1.88 |
| cam3 | exp07 | -0.40 / -0.79 / +0.38 | 0.18 / 0.26 / 0.19 | 0.45 / 0.57 / 0.64 | inconclusive | 0 | 4/4 | 2.65 |
| cam4 | exp07 | -1.14 / +0.25 / +0.72 | 0.10 / 0.12 / 0.11 | 0.35 / 0.60 / 0.70 | inconclusive | 0 | 3/4 | 1.77 |

Kalibr's dt is 1.91 / 1.90 / 1.94 / 1.81 / 1.71 ms for cam0-cam4.
These are the artifacts after the camera-specific evidence settings
[below](#camera-specific-evidence-settings); the first version of this
table, with the 0.1 deg bound inherited from IMU-LiDAR and no window
exclusion, had every verdict `inconclusive`.

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
| Common IMU-frame offset (deg) | (-0.14, +0.07, -0.41) | (-0.135, -0.055, -0.34) |
| Per-camera residual after removing it (deg) | 0.17-0.40 | 0.11-0.39 (0.18 / 0.11 / 0.38 / 0.39 / 0.27 for cam0-cam4) |

**What the evidence shows**

- **On exp21 a rotation of about -0.34 deg about the IMU vertical axis is
  shared by every camera.** The cameras face forward (cam0, cam1), down
  (cam2), left (cam3), and right (cam4).
- **The z component is not reproduced across recordings.** The forward
  cameras' common offset is (-0.22, -0.17, -0.33) deg on exp21 and
  (-0.26, -0.20, -0.12) deg on exp07: x and y agree, z does not. Only exp21
  and exp07 are available, and the side cameras on exp07 are not usable, so
  the offset about z is not established as a property of the rig.
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

## Why the reported std exceeds 0.1 deg

Every Hilti verdict is `inconclusive`, and in every case it is the window
jackknife that decides it: the analytic std is 0.05-0.11 deg, but the
jackknife raises the reported std to 0.08-0.30 deg. `tools/diagnose_camera_imu_std.py`
decomposes the gap on exp21.

For a window, the tracked rotation angle is compared with the gyro angle
through the fitted rotation, over that window's intervals. On the two forward
cameras the per-window scale (`sum(obs*exp)/sum(exp^2)`) is not one:

| Camera | Windows | Scale ratio min / median / max | Jackknife (x/y/z deg) | Analytic (x/y/z deg) |
| --- | ---: | --- | --- | --- |
| cam0 | 23 (5 s) | 1.001 / 1.020 / 1.081 | 0.105 / 0.114 / 0.103 | 0.061 / 0.069 / 0.058 |
| cam0 | 12 (10 s) | 1.006 / 1.022 / 1.072 | 0.126 / 0.078 / 0.207 | 0.054 / 0.054 / 0.050 |
| cam0 | 6 (20 s) | 1.012 / 1.025 / 1.038 | 0.101 / 0.156 / 0.168 | 0.062 / 0.066 / 0.058 |
| cam1 | 12 (10 s) | - | 0.121 / 0.195 / 0.176 | 0.054 / 0.054 / 0.048 |

- **The tracked rotation carries a per-window scale of about 1-8 %.** The
  scatter is coherent within a window and differs between windows, so leaving
  a window out moves the fit by a similar amount: the jackknife turns the
  scale scatter into the reported std.
- **It is not a focal-length error or a single global scale.** A global scale
  would bias the fit, not scatter the jackknife; a focal error is what
  `calibrex camera-imu focal` already checks. The scale here varies with the
  window's content, consistent with the documented per-pair essential-matrix
  noise (about 0.07 deg) accumulated differently by each window.
- **Windows with almost no rotation are degenerate.** On exp21 the first and
  last windows turn by under 0.1 deg; they carry no information but still
  enter the solver, and one of them is often the jackknife group that moves
  the fit most.

**Implication** (acted on in the next section). The 0.1 deg bound was inherited from the IMU-LiDAR
evaluation, where the motion comes from LiDAR odometry. For image-tracked
rotations the window-to-window scale scatter is larger, so that bound marks a
well-behaved forward camera `unobservable`. An audit should either compare the
analytic std (which reflects the fit, not the window sampling) or set a bound
appropriate to visual tracking, and should not count near-stationary windows
in the jackknife.

## Camera-specific evidence settings

The diagnosis above led to two changes that apply only to camera-IMU (the
IMU-LiDAR defaults and results are unchanged):

1. **Near-static windows are left out** of the fit, the jackknife, and the
   holdout (`ImuLidarRunOptions.min_window_rotation_deg`, camera default
   1.0 deg, IMU-LiDAR default 0 = keep all). A window's rotation is the sum of
   its per-step angles. On the two forward cameras the windows measured
   0.06-0.13 deg (exp21 first and last windows) against 2.5 deg for the
   smallest moving one, so anything between 0.2 and 2.5 deg separates them;
   1.0 deg is ten times the 0.1 deg std asked of the fit. The count is
   recorded in the artifact (`options.static_windows_excluded`, with
   `static_windows_excluded_reason` and `min_window_rotation_deg`).
2. **The rotation observability bound is 0.3 deg for cameras**
   (`observable_rotation_std_deg`, recorded in the artifact options;
   `--observable-rotation-std-deg` overrides it). It was chosen from the
   diagnosis, before re-running: tracked rotations carry a per-window scale
   scatter of 1-8 % (median 2 %), which the jackknife turns into 0.08-0.30 deg
   on the forward cameras, so a bound below the top of that range marks a
   well-behaved camera unobservable. It is 3x the IMU-LiDAR bound, which comes
   from far more precise LiDAR odometry. No schema changed: both values live
   in the artifact's free-form `options`.

**What changed on exp21 and exp07**

- **Verdicts.** 5 of 10 are now `pass`, 1 `warn`, 4 `inconclusive` (before:
  10 `inconclusive`).
  - On exp07 the flips cam0 and cam1 `inconclusive` to `pass` come from the
    threshold alone: no window there is near-static, so the results are
    otherwise identical.
  - On exp21 one or two windows were excluded per camera (two for cam0-cam2, one for cam3 and cam4). Removing them
    also shifts the held-out assignment, so exclusion and threshold effects
    are not separated there.
- **The jackknife is not fixed by exclusion.** On exp21 cam0 it went from
  0.13 / 0.08 / 0.21 to 0.12 / 0.12 / 0.15 deg: it fell on yaw and rose on
  pitch. The scale scatter between moving windows remains; the new threshold
  accommodates it rather than removes it.
- **cam0 exp21 is a `warn`**: a known-bad 1 deg yaw shift is not detected on
  the held-out windows now that yaw is reported as estimated.
- **Accuracy against Kalibr did not systematically improve.** cam0 yaw moved
  from +0.09 to -0.25 deg; cam4 exp21 passes while its pitch is 0.60 deg from
  Kalibr (std 0.16), a 3.7 sigma disagreement that the verdict does not
  see. A `pass` here means well-constrained and consistent with the held-out
  windows, not agreement with Kalibr, which is itself only a reference (see
  the rig-level offset above).
- **The side and down cameras on exp07 still fail** to constrain the
  rotation (jackknife 0.35-0.7 deg, 0.4-1.7 deg from Kalibr); cam3 on exp21 is
  `inconclusive` on yaw (0.31 deg). The 0.3 deg bound did not rescue them.

The bound is a judgement made from development data on two recordings. It has
been validated on held-out recordings only by the pre-registered audit below,
which refuted the claim on the shortest recording (exp04).

## Pre-registered audit: refuted

The claim, metrics, and thresholds were committed in
[`hilti_camera_imu_preregistration.yaml`](hilti_camera_imu_preregistration.yaml)
(commit `c4d51d2`, SHA-256 `fad67cd1...a3ec1`), before camera-imu ran on any
evaluation recording. The code was frozen at commit `7fd89ee`; `src/` and
`tools/` were identical to it when the audit ran, and the bag hashes matched
the pre-registration. The evaluation recordings are exp01-exp04 and the gated
cameras are cam0 and cam1 (eight units); development was exp21 and exp07 only.
Each unit ran the pre-registered `calibrex camera-imu rotation` command once
(`--frame-stride 4 --min-window-rotation-deg 1.0 --observable-rotation-std-deg 0.3`),
and no run crashed or was repeated. Scoring is `tools/score_hilti_camera_imu.py`
and the audit is built by `tools/build_hilti_camera_imu_audit.py`, both unchanged
([protocol](../assets/hilti_camera_imu_sota_protocol.yaml),
[result](../assets/hilti_camera_imu_sota_audit.yaml),
[scores](../assets/hilti2022_camera_imu_audit/scores.json), and the eight
artifacts beside it).

**Verdict: `refuted`, 6/7 gates.** All eight units are usable and scored.

| Unit | Rotation to Kalibr | Time error | Constrained | Windows | Failed pairs | Policy |
| --- | ---: | ---: | :---: | ---: | ---: | --- |
| cam0 exp01 | 0.552 deg | -0.33 ms | yes | 24 | 0.2 % | pass |
| cam0 exp02 | 0.435 deg | -0.24 ms | yes | 48 | 1.6 % | pass |
| cam0 exp03 | 0.696 deg | +0.13 ms | yes | 36 | 2.4 % | pass |
| cam0 exp04 | 0.497 deg | -0.02 ms | **no** | 14 | 2.7 % | inconclusive |
| cam1 exp01 | 0.377 deg | -0.08 ms | yes | 24 | 0.3 % | pass |
| cam1 exp02 | 0.383 deg | -0.33 ms | yes | 48 | 1.4 % | pass |
| cam1 exp03 | 0.365 deg | +0.04 ms | yes | 35 | 2.0 % | warn |
| cam1 exp04 | 0.512 deg | -0.33 ms | **no** | 15 | 2.6 % | inconclusive |

Cross-recording consistency (largest angle between two recordings' estimates):
cam0 0.347 deg, cam1 0.430 deg. Relative cam0-to-cam1 rotation error:
exp01 0.300, exp02 0.193, exp03 0.375, exp04 0.406 deg.

| Requirement | Observed | Threshold | Result |
| --- | --- | --- | :---: |
| rotation-vs-kalibr (every unit) | max 0.696 deg | <= 1.0 deg | pass |
| time-offset-vs-kalibr (every unit) | max 0.332 ms | <= 0.5 ms | pass |
| constrained (every unit) | 2 of 8 units not constrained | 0 | **fail** |
| no-failed-units | 0 failed | 0 | pass |
| cross-recording-consistency | max 0.430 deg (cam1) | <= 0.75 deg | pass |
| relative-cam0-cam1 | max 0.406 deg (exp04) | <= 0.5 deg | pass |
| scored-recordings-coverage | 4 of 4 recordings | >= 3 | pass |

- **The accuracy and reproducibility gates pass with margin.** The worst
  rotation error against Kalibr is 0.70 deg (development worst 0.45), the worst
  clock offset error is 0.33 ms, the estimates of one camera agree across
  recordings to 0.43 deg, and the Kalibr relative rotation is reproduced to
  0.41 deg. The rig-level offset seen in development did not break any gate.
- **The `constrained` gate fails on exp04, for both cameras.** exp04 is the
  shortest recording (125.8 s), and only 14 (cam0) and 15 (cam1) windows
  survive the static-window exclusion, against 24-48 elsewhere. On cam0 the
  pitch jackknife std is 0.354 deg (bound 0.3, so pitch is `unobservable`); on
  cam1 roll (0.404 deg) and yaw (0.350 deg) are. The three known-bad controls
  were detected in all of them, and the rotation errors of those units are
  0.50 and 0.51 deg, so the data was not wrong; it was not sufficient to
  constrain every axis to the 0.3 deg the claim required. This is what the
  observability bound exists to report.
- **cam1 exp03 passes `constrained` narrowly.** Its pitch control was not
  detected (2 of 3 detected, the minimum) and its policy is `warn`.
- Because the claim said "constrained by the data on all three rotation axes",
  a short recording that cannot do so contradicts it. The pre-registered rule is
  not changed after seeing this, and the audit stays on the leaderboard.

The result is a mixed one. Against Kalibr, the method is accurate on
construction-site recordings, but the claim as written (every unit constrained)
does not hold on the shortest recording. The thresholds were set from two
development recordings, and the 0.3 deg observability bound did not generalise
to the shortest evaluation recording. A future
claim could pre-register a minimum recording length or window count, or report
`constrained` per recording rather than per unit, but that would be a new
claim on new recordings.

Runtime was 1240 s of wall time with two runs in parallel (cam0 and cam1 each
199, 502, 409, and 129 s for exp01-exp04).

The held-out recordings are now spent for camera-IMU. cam2-cam4 were not run on
them.

## Next steps

- Understand the side-camera errors.
- Decide how the audit treats a rig-level IMU-frame offset. The rotation of
  each camera relative to the gyro frame is what this method estimates.
- Decide how a short recording is judged: the pre-registered audit on
  exp01-exp04 was refuted by the `constrained` gate on the 126 s exp04 alone.
  A new claim needs new recordings (exp01-exp04 are spent) and, ideally, a
  targetless external baseline.

No SOTA claim is made for camera-IMU: its pre-registered audit is refuted.
