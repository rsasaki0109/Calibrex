# Hilti 2022 Camera-IMU Rotation

Calibrex's first targetless camera-IMU calibration. It estimates the rotation
of `T_cam_imu`, the clock offset (`t_imu = t_cam + dt`), and the gyro bias,
by aligning camera rotations from tracked image features with the gyro. No
calibration target is needed.

```bash
python -m pip install "calibrex[opencv] @ https://github.com/rsasaki0109/Calibrex/releases/download/v0.6.4/calibrex-0.6.4-py3-none-any.whl"
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

### TUM-VI room1 (development) and a pre-registered audit

TUM-VI is a hand-held rig with two 190 deg fisheye cameras (equidistant, 512 x 512) and a
BMI160 IMU, calibrated with Kalibr. `calibrex check --tf camchain.yaml --pairs
camera-imu,camera-focal` on room1 (spent), with Kalibr's camchain and with its focal
lengths turned 3 % long or short:

| Candidate | cam0 | cam1 |
| --- | --- | --- |
| Kalibr | pass: fx scale 0.9996, fy 1.0113 | inconclusive: the estimator failed its own held-out check |
| focal 3 % long (true scale 0.971) | fail: fx 0.977 fail; fy 0.993 within tolerance | inconclusive: optical-axis control 0.989 |
| focal 3 % short (true scale 1.031) | fail: fx 1.031 fail | inconclusive: held-out check |

- **fx tracks the injected error.**
- **fy does not.** It reads about 1 % high at Kalibr's value and does not separate the
  3 % errors cleanly.
- **cam1 is often inconclusive.**
- **What is claimed.** No accuracy claim is made. Check's honesty on the correct focal
  lengths and its detection of 3 % errors are
  [pre-registered](tumvi_camera_focal_preregistration.yaml) on room2–room6.

[Development records](../assets/tumvi_camera_focal/development/room1_kalibr.json).

### Result: camera-focal check honesty and detection supported

The [pre-registered](tumvi_camera_focal_preregistration.yaml) audit ran once on 2026-11-07,
from a worktree at the pre-registration commit (a48efac3), whose `src/` and `tools/` equal
`frozen_code_commit` (b9eee429). Each case is one sequence, one camera and one candidate;
the cell is camera-focal's verdict, with the fx (and fy) scale where judged.

| Sequence | Kalibr cam0 | Kalibr cam1 | 3 % long cam0 | 3 % long cam1 | 3 % short cam0 | 3 % short cam1 |
| --- | --- | --- | --- | --- | --- | --- |
| room2 | pass (fx 0.9996) | inconclusive | inconclusive | inconclusive | warn (1.017, 1.028) | inconclusive |
| room3 | pass (1.002, 1.002) | inconclusive | skipped | inconclusive | inconclusive | inconclusive |
| room4 | pass (fx 1.000) | pass (1.002, 1.004) | inconclusive | inconclusive | **fail** (1.032, 1.039) | **fail** (1.028, 1.034) |
| room5 | pass (1.000, 0.996) | inconclusive | inconclusive | **fail** (fx 0.975) | **fail** (fx 1.033) | **fail** (1.030, 1.031) |
| room6 | pass (fx 0.995) | inconclusive | **fail** (fx 0.982) | warn (fx 0.988) | inconclusive | warn (fx 1.015) |

A 3 % long focal length reads as a scale of 0.971; a 3 % short one as 1.031.

- **Honesty: supported.**
  - Check never failed Kalibr's focal lengths.
  - It judged a focal length on 6 of 10 cases (gate 25 %).
- **Detection: supported.**
  - Check passed none of the 20 cases with a 3 %-wrong focal length.
  - It failed 6 (30 %, gate 25 %) and warned on 3 more.
- **The detection is partial.**
  - Half of the perturbed cases came out inconclusive or skipped.
  - In several the estimated scale moved by only half the injected error (1.015–1.017,
    0.982–0.988).
- **cam1 is often inconclusive.** Its estimator failed its own held-out or optical-axis
  checks on 4 of 5 sequences even with Kalibr's values.
- **No accuracy is claimed.** fx agrees with Kalibr within 0.5 % where judged; fy is
  noisier.

What this means for a user:

- On a fisheye camera like TUM-VI's, `calibrex check` does not flag a correct focal
  length.
- It does not pass one that is 3 % off.
- About half the time it says inconclusive rather than fail.

[Run records](../assets/tumvi_camera_focal/evaluation/summary.json);
[audit record](../assets/tumvi_camera_focal/audit/tumvi-camera-focal-check-v1_sota_audit.yaml).
The five sequences are spent.

### Rays beyond 90 deg on a fisheye: an incidence cap, tried and not adopted

The tracker turns pixels into perspective rays (tan of the incidence angle). On TUM-VI's
190 deg lenses the image corners lie about 114 deg off the optical axis, and a ray
beyond 90 deg flips sign. A cap that drops features beyond a given incidence angle was
tried on room1 (spent) only, with the same three candidates:

| Cap | Kalibr: judged / fails | 3 %-wrong: fail / pass (of 4) | cam0 fx scale at Kalibr's value |
| --- | --- | --- | --- |
| none (current) | 1 of 2 / 0 | 2 / 0 | 0.9996 |
| 80 deg | 2 of 2 / 0 (cam0 warn, fy 1.009) | 1 / 0 | 1.0023 |
| 65 deg | 1 of 2 / 0 | 3 / 0 | 1.0024 (inconclusive) |
| 50 deg | 1 of 2 / 0 | 0 / 0 | 1.0074 |

- **No cap is better on every count.** Detection goes 2, 1, 3, 0 of 4 as the cap
  tightens.
- **The scale drifts from 1 as the cap tightens.** The wide-angle rays are not what makes
  cam1 inconclusive.
- **fy is judged less often** with any cap.
- **Not merged.** The default is unchanged. The code is on branch
  `research/fisheye-focal`.

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

### Rig and motion parts on `exp01` and `exp21`

The all-camera `calibrex check` names the IMU on `exp21`: one turn of its candidate,
(+0.32, +0.19, +0.41) deg, explains every `camera-imu` error
([tutorial](../tutorials/calibrex_check.md#which-frame-the-errors-point-to)).
`tools/hilti_camera_imu_offset.py` asks which part of that turn belongs to the rig and
which to the motion.

**How it works.**

- For each camera, the tool tracks the images as the check does. It then fits the gyro
  on three sets of moving windows:
  - all of them;
  - the two halves in time, which see different motion;
  - the even and odd windows, which see the same motion, interleaved.
- It pools the five cameras per split into one rig offset.
- **Convention.** `w` is about the IMU axes, with `E = C exp(w)`. Here `C` is Kalibr's
  `R_cam_imu` and `E` the estimate. The attribution's turn of the IMU is `-w`.
- **The z axis is vertical in use.** The mean specific force on `exp21` is
  (0.67, −0.01, −9.66) m/s², and the median tilt is 6 deg.
- **Data and windows.** The head bags hold 81 s. The fits use 4 s windows, 17 to 19 of
  them. The check's 10 s windows give 7 to 9 windows: the same picture, with rougher
  stds.

| deg (IMU x / y / z) | `exp01` | `exp21` |
| --- | --- | --- |
| Rig offset, all windows | −0.08 ± 0.04 / −0.02 ± 0.05 / **−0.39 ± 0.07** | −0.23 ± 0.08 / −0.23 ± 0.05 / **−0.42 ± 0.04** |
| Cameras' χ² about it (4 dof) | 6.9 / 5.7 / 4.3 | 3.3 / 3.0 / 4.4 |
| First − second half (z-score) | −0.22 (−2.7) / +0.28 (+3.2) / −0.33 (−2.1) | −0.18 (−1.3) / +0.10 (+0.7) / +0.09 (+0.9) |
| Even − odd (z-score) | +0.03 (+0.4) / −0.03 (−0.4) / +0.03 (+0.2) | +0.01 (+0.1) / +0.01 (+0.1) / +0.03 (+0.3) |
| Rig offset with the check's 10 s windows | −0.04 / −0.06 / −0.31 | −0.16 / −0.21 / −0.43 |
| z of the rotation part of a full linear fit, per camera | −0.01 to −0.55 | −0.33 to −0.63 |

The pooled stds treat the cameras as independent. They share the gyro and the motion,
so the stds are optimistic; the χ² shows whether the cameras agree.

**What the evidence shows**

- **About IMU z, every camera reads the same turn on both recordings: −0.39 and
  −0.42 deg.**
  - The cameras face forward, down, left and right. A camera-side cause would not give
    one turn about the IMU's axes.
  - This is a rotation between the gyro frame, which `camera-imu` measures, and Kalibr's
    IMU frame.
  - `exp07`'s forward cameras read −0.12 deg about z (above), so two of three
    recordings agree on it.
- **It is not the gyro's scale or non-orthogonality.**
  - A full linear fit `w_cam = A w_gyro` keeps the turn in the rotation part of `A`.
  - The symmetric parts differ from camera to camera, as found before.
- **The horizontal components move with the motion.**
  - They differ between the recordings: −0.23 / −0.23 against −0.08 / −0.02 deg.
  - On `exp01` they differ between the halves by up to 0.3 deg (z-scores 2.7 and 3.2).
    The z component there also moves (−0.58 against −0.25 deg).
  - Interleaved windows agree everywhere. The stds describe the noise within one kind
    of motion; another part of the recording moves the reading by more. This is the
    same pattern as the camera pairs.
- **What the check names on `exp21` therefore has two parts.**
  - The z part, about 0.4 deg, recurs on `exp01` with every camera.
  - The x and y part, about 0.2 deg, belongs to `exp21`'s motion.
  - Together they make the 0.5 deg `camera-imu` errors. At the 0.5 deg floor, they put
    cam1's pitch at `warn` (0.51 deg).

**What remains open**

The data cannot tell whether the gyro frame or Kalibr's IMU frame is off. Kalibr's
`calibrated` IMU model assumes the gyro and accelerometer share one frame. A rotation
between them, or a Kalibr error about z, would both look like this.

No verdict rule changes. The motion-dependent part is the open question for
`camera-imu` stds, as for the camera pairs. Results are in
`docs/assets/hilti2022_camera_imu/offset_{exp01,exp21}_w{4,10}.json`.

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

## Pre-registered audit v3: supported (7 of 7 gates)

v3 was committed in
[`hilti_camera_imu_v3_preregistration.yaml`](hilti_camera_imu_v3_preregistration.yaml)
(commit `adb2981`, SHA-256 `e32a24e0...eaeece`) before camera-IMU ran on any evaluation
recording. The code was frozen at `ea5ff40`; `src/` and `tools/` were identical to it for
every run.

**What v3 claims.** v1's estimator and command, scored by the method's own per-axis
verdicts.

- **Judged axis.** The artifact reports the axis `estimated` (std at most 0.3 deg) and
  its 1 deg known-bad control is detected on held-out windows. `calibrex check` applies
  the same rule before it judges an axis.
- **Gates.**
  - Every judged axis is within 1.0 deg of Kalibr.
  - Every judged clock offset is within 0.75 ms.
  - At least half of the axes are judged.
  - A judged axis spans at most 1.0 deg across recordings.
  - The cam0-cam1 rotation is within 0.75 deg per axis judged for both cameras.
- **Thresholds.** 1.5 times the worst of the 22 development units (every unit
  camera-IMU had run on), rounded up to 0.25.

**How it ran.**

- **Recordings.** exp05, exp06, exp11, exp14 and exp18 (74 to 152 s): every Hilti 2022
  recording on which no camera-IMU estimate had been made, except exp09 and exp10, which
  do not fit the disk. The camera-camera audit had read their first 5 GB; it made no
  camera-IMU estimate.
- **Execution.** `tools/hilti_camera_sota_evaluate.py camera-imu-v3` downloaded each bag
  in full, checked its SHA-256 against the pinned one (all five matched), ran cam0 and
  cam1, and deleted it. All ten runs exited 0 on their only attempt. The whole
  evaluation took 13 minutes.
- **Records.** Every artifact, log and download record is under
  [`docs/assets/hilti2022_camera_imu_v3/`](https://github.com/rsasaki0109/Calibrex/tree/main/docs/assets/hilti2022_camera_imu_v3),
  with the scores, the six benchmarks, the protocol and the audit.

| Requirement | Observed | Threshold | Result |
| --- | --- | --- | :---: |
| judged-axis-accuracy | max 0.528 deg (cam0 exp14 pitch) | <= 1.0 deg | pass |
| judged-time-offset | max 0.470 ms (cam1 exp05) | <= 0.75 ms | pass |
| judged-axis-coverage | 26 of 30 axes (87 %) | >= 50 % | pass |
| no-failed-units | 0 failed | 0 | pass |
| judged-axis-consistency | max 0.408 deg (cam1 yaw) | <= 1.0 deg | pass |
| judged-axis-relative | max 0.444 deg (exp14 yaw) | <= 0.75 deg | pass |
| scored-recordings-coverage | 5 of 5 recordings | >= 3 | pass |

Errors to Kalibr in deg, about the camera's axes. A bold value is an axis the method did
not judge; the reason follows it.

| Unit | Windows | Roll | Pitch | Yaw | Clock offset | Policy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cam0 exp05 | 14 | -0.007 | -0.232 | +0.057 | -0.10 ms | pass |
| cam0 exp06 | 16 | +0.036 | -0.325 | +0.102 | -0.16 ms | pass |
| cam0 exp11 | 16 | -0.124 | -0.389 | **+0.074** (std 0.445) | -0.41 ms | inconclusive |
| cam0 exp14 | 8 | -0.011 | -0.528 | +0.182 | -0.10 ms | pass |
| cam0 exp18 | 11 | -0.057 | **-0.869** (std 0.338) | +0.057 | +0.22 ms | inconclusive |
| cam1 exp05 | 14 | -0.076 | -0.243 | +0.146 | -0.47 ms | pass |
| cam1 exp06 | 18 | -0.113 | -0.349 | -0.060 | -0.11 ms | pass |
| cam1 exp11 | 17 | +0.064 | **-0.458** (control missed) | -0.038 | -0.41 ms | warn |
| cam1 exp14 | 8 | -0.183 | -0.212 | -0.262 | -0.03 ms | pass |
| cam1 exp18 | 12 | -0.334 | **-0.682** (std 0.815) | -0.168 | -0.40 ms | inconclusive |

**What the result says.**

- **The axes the method declined to judge include the two largest errors.**
  - cam0 exp18 pitch (-0.87 deg) and cam1 exp18 pitch (-0.68 deg) are both reported
    unobservable.
  - Every judged axis is within 0.53 deg.
  - In development too, the largest error (cam0 exp15 pitch, -0.77 deg) was an axis
    whose control was missed.
- **The short recordings held.** exp14 (74 s) gave exactly 8 windows for each camera, the
  least a usable unit may have. It judged all six axes, with stds up to 0.28 deg. Its
  cam0 pitch, -0.53 deg, is the largest judged error.
- **Pitch is negative on every unit, judged or not (-0.21 to -0.87 deg), as in
  development.** For the forward cameras, pitch is about the vertical axis. That fits the
  rig-level offset about the IMU's vertical axis
  ([above](#rig-and-motion-parts-on-exp01-and-exp21)), plus a motion-dependent part.
- **v3 is supported because it claims less than v1** (post-hoc, with v1's unchanged
  scorer; [scores](../assets/hilti2022_camera_imu_v3/posthoc_v1_rules/scores.json)).
  Under v1's rules these same ten runs are refuted on two gates:
  - `constrained` fails on three units (cam0 exp11, cam0 exp18, cam1 exp18);
  - the whole relative rotation on exp14 is 0.570 deg, over 0.5 deg.

**What the claim does not cover.**

- It covers the two forward cameras of one rig, on one dataset family. The reference is
  Hilti's Kalibr calibration, not ground truth.
- It gates only the axes the method judges; 4 of the 30 were not judged.
- Its clock-offset and relative bounds (0.75 ms, 0.75 deg per axis) are looser than v1's
  (0.5 ms, 0.5 deg for the whole rotation). They come from the larger development set,
  by the rule stated before the runs.

## TUM-VI: camera-imu check on a second dataset family

The Hilti audits above are accuracy audits of the estimator. On TUM-VI (two 190 deg
fisheye cameras and a BMI160 IMU, calibrated with Kalibr) the question is what
`calibrex check --pairs camera-imu` says about a correct and a wrong T_cam_imu.

Development, all on spent sequences:

- **Kalibr's camchain, room1–room6, both cameras (12 cases): all pass.** Every case is
  judged on roll, pitch and yaw. The largest error is 0.39 deg (room6 cam1 roll); the
  tolerance is 0.5–0.67 deg. These runs came from the camera-focal audit, which runs
  camera-imu with it.
- **T_cam_imu turned 2 deg about the camera's x, y or z axis, room1 (6 cases): all
  fail, on the turned axis.** The errors read 1.90–2.15 deg.
- **The estimate does not depend on the candidate.** A turned candidate's error is the
  turn plus the estimate's difference from Kalibr: room1 cam0 roll reads 0.01 deg for
  Kalibr and 2.01 deg turned.

Honesty and detection on the unseen corridor1–corridor5 are
[pre-registered](tumvi_camera_imu_preregistration.yaml).
[Development records](../assets/tumvi_camera_imu/development/room1_rx2.json).

### Result: camera-imu check honesty supported, detection refuted

The [pre-registered](tumvi_camera_imu_preregistration.yaml) audit ran once on 2026-11-07,
from a worktree at the pre-registration commit (fa426478), whose `src/` and `tools/` equal
`frozen_code_commit` (db74694a). Each case is one sequence, one camera and one candidate;
the cell is camera-imu's verdict, with the turned axis's error in deg (for Kalibr, the
largest judged error).

| Sequence | Camera | Kalibr | turned about x (roll) | about y (pitch) | about z (yaw) |
| --- | --- | --- | --- | --- | --- |
| corridor1 | cam0 | pass (max 0.10 yaw) | fail (roll 2.06) | fail (pitch 2.08) | fail (yaw 1.90) |
| corridor1 | cam1 | pass (max 0.18 pitch) | fail (roll 1.90) | fail (pitch 2.18) | fail (yaw 2.02) |
| corridor2 | cam0 | pass (max 0.29 roll) | fail (roll 1.71) | fail (pitch 1.86) | fail (yaw 2.04) |
| corridor2 | cam1 | pass (max 0.27 roll), pitch unchecked | fail (roll 1.73) | **pass** (pitch unchecked) | fail (yaw 2.11) |
| corridor3 | cam0 | pass (max 0.31 roll) | fail (roll 1.69) | fail (pitch 2.17) | fail (yaw 2.00) |
| corridor3 | cam1 | pass (max 0.37 roll) | fail (roll 1.63) | fail (pitch 2.14) | fail (yaw 1.99) |
| corridor4 | cam0 | pass (max 0.42 roll), pitch unchecked | fail (roll 1.58) | **pass** (pitch unchecked) | fail (yaw 1.89) |
| corridor4 | cam1 | pass (max 0.44 roll), pitch unchecked | fail (roll 1.56) | **pass** (pitch unchecked) | fail (yaw 1.95) |
| corridor5 | cam0 | pass (max 0.30 roll) | fail (roll 1.70) | fail (pitch 1.81) | fail (yaw 1.94) |
| corridor5 | cam1 | pass (max 0.30 roll), pitch unchecked | fail (roll 1.70) | **pass** (pitch unchecked) | fail (yaw 1.92) |

- **Honesty: supported.**
  - Check never failed Kalibr's T_cam_imu.
  - It judged rotation axes on 10 of 10 cases (gate 50 %), all three axes on 6.
- **Detection: refuted.**
  - Check passed 4 of the 30 turned cases (gate: none). All four are the pitch turn on a
    case where pitch was not judged:
    - corridor2 cam1 and corridor4 cam0: the estimator's 1 deg known-bad control on
      pitch was not detected on held-out windows;
    - corridor4 cam1 and corridor5 cam1: pitch was not constrained by the data.
  - Check printed `pass (partial: roll, yaw only)` and a partial-coverage warning on each.
  - The other 26 (87 %, gate 50 %) failed on the turned axis. Every turned axis that
    was judged failed (26 of 26), at 1.56–2.18 deg.
- **The estimate reads roll 0.27–0.44 deg below Kalibr's** on every corridor case on both
  cameras. This is within the 0.5 deg tolerance, and room6 read the same (0.33, 0.39).
  No accuracy is claimed.

What this means for a user:

- `calibrex check` did not flag a correct camera-IMU rotation on these fisheye recordings.
- A 2 deg error on an axis it judges is flagged on that axis.
- A `pass` with unchecked axes vouches only for the judged axes. Check says so in the
  verdict (`partial`) and in a warning, but with `--fail-on fail` (the default) it still
  exits 0. A 2 deg error on an unchecked axis passes.

[Run records](../assets/tumvi_camera_imu/evaluation/summary.json);
[audit record](../assets/tumvi_camera_imu/audit/tumvi-camera-imu-check-v1_sota_audit.yaml).
The five sequences are spent.

### v2: the same check with `--fail-on partial`, pre-registered

`calibrex check --fail-on partial` exits 1 on a pass that left axes unchecked. v2 asks what a
user running with it gets. A case is *flagged* when its camera-imu pair would make the run
exit 1.

On the spent rooms and corridors, scored this way:

- Kalibr's camchain was never failed. It passed with full coverage on 18 of 22 cases; the
  other 4 left pitch unchecked and would exit 1.
- No turned case would exit 0. 32 of 36 failed on the turned axis.

Honesty (no fail; full passes on at least 50 %) and detection (no turned case unflagged;
at least 50 % failed on the turned axis) are
[pre-registered](tumvi_camera_imu_v2_preregistration.yaml) on the five smallest unseen
sequences: slides1–slides3, magistrale3 and magistrale5. v1 stays refuted.

### Result v2: supported, `--fail-on partial` flags every turned case

The [pre-registered](tumvi_camera_imu_v2_preregistration.yaml) audit ran once on
2026-11-07, from a worktree at the pre-registration commit (7899e471), whose `src/` and
`tools/` equal `frozen_code_commit` (317339c1). The four candidates of a sequence shared
one estimator cache. The cell is camera-imu's verdict, with the turned axis's error in deg
(for Kalibr, the largest judged error). *partial* marks a pass that left an axis
unchecked; it exits 1 under `--fail-on partial`.

| Sequence | Camera | Kalibr | turned about x (roll) | about y (pitch) | about z (yaw) |
| --- | --- | --- | --- | --- | --- |
| slides2 | cam0 | pass, partial: pitch unchecked (max 0.22 yaw) | fail (roll 1.92) | **pass, partial: pitch unchecked** | fail (yaw 1.78) |
| slides2 | cam1 | pass, partial: pitch unchecked (max 0.17 roll) | fail (roll 1.83) | **pass, partial: pitch unchecked** | fail (yaw 1.85) |
| slides1 | cam0 | pass (max 0.17 roll) | fail (roll 2.17) | fail (pitch 2.01) | fail (yaw 1.90) |
| slides1 | cam1 | pass (max 0.27 pitch) | fail (roll 1.90) | fail (pitch 2.27) | fail (yaw 1.94) |
| slides3 | cam0 | pass (max 0.08 roll) | fail (roll 1.92) | fail (pitch 1.99) | fail (yaw 2.01) |
| slides3 | cam1 | pass (max 0.19 roll) | fail (roll 1.81) | fail (pitch 2.06) | fail (yaw 1.88) |
| magistrale5 | cam0 | pass (max 0.20 pitch) | fail (roll 2.02) | fail (pitch 1.80) | fail (yaw 1.93) |
| magistrale5 | cam1 | pass (max 0.17 pitch) | fail (roll 1.96) | fail (pitch 2.17) | fail (yaw 1.90) |
| magistrale3 | cam0 | pass (max 0.20 pitch) | fail (roll 1.94) | fail (pitch 1.80) | fail (yaw 1.84) |
| magistrale3 | cam1 | pass (max 0.12 roll) | fail (roll 1.88) | fail (pitch 2.10) | fail (yaw 2.01) |

- **Honesty: supported.**
  - Check never failed Kalibr's T_cam_imu.
  - It passed with full coverage on 8 of 10 cases (gate 50 %).
  - Both slides2 cameras left pitch unchecked; under `--fail-on partial` those two runs
    exit 1 although the calibration is right.
- **Detection: supported.**
  - Under `--fail-on partial` no turned case exits 0 (gate: none). The slides2 pitch turns
    passed with pitch unchecked, as in v1, and are now flagged.
  - 28 of 30 failed on the turned axis (gate 50 %), at 1.78–2.27 deg.
- **The same pattern as v1.** Every judged turned axis failed; the misses are an axis the
  recording could not judge.

What this means for a user:

- With `--fail-on partial`, a 2 deg camera-IMU rotation error did not get through on these
  recordings.
- The price is a non-zero exit on a correct calibration when the recording cannot judge an
  axis (2 of 10 here). Re-record with more rotation about that axis.

[Run records](../assets/tumvi_camera_imu_v2/evaluation/summary.json);
[audit record](../assets/tumvi_camera_imu_v2/audit/tumvi-camera-imu-check-v2_sota_audit.yaml).
The five sequences are spent.

## Next steps

- Understand the side-camera errors.
- Decide how the audit treats a rig-level IMU-frame offset. The rotation of
  each camera relative to the gyro frame is what this method estimates.
- Decide how observability is judged. The pre-registered audit on exp01-exp04
  was refuted by the `constrained` gate on the 126 s exp04 alone. Its successor,
  v2, kept every gate and required recordings of at least 200 s; it was refuted
  too ([v2 audit](hilti_camera_sota.md#camera-imu-v2-refuted-5-of-7-gates)).
  - Yaw was not constrained on two units of 260 s and 321 s (cam0 exp15, cam1
    exp23p1), so length is not what decides it.
  - cam0's estimates spread 0.81 deg across the five recordings, over the 0.75 deg
    consistency bound.
  - A new claim needs new recordings (exp01-exp04 and the five v2 recordings are
    spent) and, ideally, a targetless external baseline.
- **v3 is supported**
  ([above](#pre-registered-audit-v3-supported-7-of-7-gates)). It gates only the axes the
  method judges, and requires at least half of them judged.
  - A stronger claim, every axis constrained as v1 asked, would need recordings with
    more excitation about the camera's pitch axis.
  - Hilti 2022 has only exp09 and exp10 left for camera-IMU, too large for this
    environment.

camera-IMU has one supported claim, v3, within the scope stated in it. v1 and v2 stay
refuted and listed.
