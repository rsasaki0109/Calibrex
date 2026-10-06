# camera-lidar in `calibrex check`: Hilti 2022 and KITTI development data

`calibrex check`, `estimate` and `drift` judge the **rotation** of `T_camera_lidar` from a
bag's images and point clouds, without a target. This page records how the method was chosen,
what it did against the reference calibrations of two public datasets, and where it does not
work. It is **development evidence, not a claim**: the pair stays `no_claim` on the
[leaderboard](sota_leaderboard.md) and the thresholds below were set while looking at these
recordings (no frozen audit, no held-out data scored).

## Which method, and why

The repository has several camera-LiDAR methods ([development roadmap](../development_roadmap.md)).
Only some run on a bag of images and point clouds without a target:

| Method | Needs | Why not (or why) |
| --- | --- | --- |
| Planar board, Horn point-only, point+plane | a calibration board and an external feature extractor | not targetless |
| Continuous-time, probabilistic refiner, correspondence-based | trajectories or 2D-3D correspondences from an external provider | not a bag-only method |
| Koide-style direct visual-LiDAR | an external executable (ROS, PCL, GTSAM) | an adapter, not native |
| Pandey mutual information (native) | image luminance and LiDAR reflectivity | real A2D2 run FAILs (inputs already camera-registered) |
| Levinson-Thrun edges (native, online) | image edges and LiDAR depth discontinuities | two-pair A2D2 run WARN / INCONCLUSIVE |

The last two are the targetless candidates. Neither had a real-data result that supports a
verdict, so both were tried on the development recordings: the objective is profiled along each
rotation axis around the vendor / reference calibration (a correct objective peaks there).
Rotation grid -6 to +6 deg:

| Data | Pandey MI argmax (roll, pitch, yaw) | Edge objective argmax (roll, pitch, yaw) |
| --- | --- | --- |
| Hilti exp21 `cam0` + Hesai (58 frames) | 0, 0, 0 deg | 0, 0, 0 deg |
| KITTI 0005 `image_02` + HDL-64 (39 frames) | -2, -0.5, +1 deg | 0, 0, 0 deg |

MI on KITTI peaks 1-2 deg away from the vendor calibration, so its optimum would be a biased
yardstick; the edge objective peaks at the reference on both. `calibrex check` therefore uses
the edge objective of Levinson and Thrun (RSS 2013), re-implemented in
`calibrex.evaluation.camera_lidar_edge` as an offline pooled estimator with a jackknife std and
held-out known-bad controls (no GPL code; numpy and scipy only). Translation profiles are flat
within +-5 cm on both datasets (edge objective argmax at -3, -1, +3 cm on KITTI and +5, -1, +5 cm
on Hilti), which is why translation is not judged.

Things tried and dropped: cross-beam (horizontal) depth edges (the profile no longer peaked at the
reference), a 2 deg-grid start search (it found a spurious roll ridge), per-block controls at 1
deg (too noisy).

## Setup

* **Hilti 2022** (`exp21_ros2` development; `exp07_ros2` development; `exp01`-`exp04` used only to
  validate the tool: their camera-IMU claims are spent and no camera-LiDAR claim is made):
  `/alphasense/cam0/image_raw` (720x540, equidistant) + `/hesai/pandar` (PandarXT-32, with
  `ring`/`timestamp`). Reference: `calibration_files/pandar_from_cam0.yaml`, inverted to
  `T_cam0_lidar` and given as a `slac.check_frames` file; intrinsics from the Kalibr camchain
  (`calib_3_cam0-1-camchain-imucam.yaml`). `--max-duration-s 120`, so 48 frames 2.5 s apart.
* **KITTI raw**, development drives only: `2011_09_26_drive_0005_sync` (154 frames, 15 s) and
  `drive_0009_sync` (443, 45 s). Drives 0027-0059 were not touched. Converted with
  `calibrex convert kitti-raw DRIVE --camera image_02`, which writes the rectified `image_02`, its
  `CameraInfo` and `imu_link -> cam2_optical` composed from `calib_velo_to_cam.txt` and
  `calib_cam_to_cam.txt`, so the **vendor calibration is the candidate**. Frames are 1 s apart.
* Perturbations turn the candidate by 1, 3 or 5 deg about each camera axis (`T_camera_lidar' = R
  T_camera_lidar`, `R` about camera x, y, z) or shift it 5 cm along camera x; verdict thresholds are
  the defaults (`3 sigma`, 0.5 deg floor).

```bash
calibrex check BAG --pairs camera-lidar --camera alphasense/cam0/image_raw --max-duration-s 120 \
  --tf calib_3_cam0-1-camchain-imucam.yaml --tf hilti_ref.yaml --cache-dir CACHE --output check.yaml
calibrex check kitti_0009_bag --pairs camera-lidar --cache-dir CACHE --output check.yaml
```

## The estimate against the reference

Where the axis is observed on the recording, the estimate agrees with the reference to 0.35 deg or
better. Entries are estimate minus reference (deg) with the jackknife std; "-" is an axis whose
held-out control was not detected (not judged).

| Recording | Frames / edge points | roll | pitch | yaw |
| --- | --- | --- | --- | --- |
| Hilti exp21 | 48 / 80 307 | -0.10 (0.25) | +0.10 (0.08) | -0.10 (0.05) |
| Hilti exp07 (corridor) | 46 / 28 188 | +0.35 (0.03) | - | - |
| Hilti exp01 | 47 / 90 619 | - | +0.10 (0.11) | - |
| Hilti exp02 | 37 / 46 580 | - | +0.00 (0.14) | - |
| Hilti exp03 | 24 / 12 828 | - | +0.20 (0.35) | - |
| Hilti exp04 | 34 / 51 983 | - | +0.10 (0.03) | - |
| KITTI 0005 (15 s) | 16 / 15 121 | -0.00 (0.26) | -0.10 (0.06) | - |
| KITTI 0009 (45 s) | 45 / 41 347 | +0.20 (0.04) | -0.05 (0.11) | -0.00 (0.04) |

The jackknife std does not include systematic effects: exp07 roll is 0.35 deg off with a std of
0.03 deg. The 0.5 deg verdict floor covers that. Translation was never judged; with translation
searched in a development run its estimate was off by 3-11 cm (KITTI 0005) and wandered to the
search bound on Hilti.

## Verdicts

**The reference is never failed.** The unmodified reference passes on all eight runs above (no
`warn`, no `fail`), on the axes that are judged (`partial` coverage is stated in every row).

**Perturbed candidates**, defaults, `check` verdict and the error of the judged axis against its
tolerance (`error / tolerance` in deg). "unchecked" means the axis was not judged on that recording.

| Perturbation | Hilti exp21 | KITTI 0005 | KITTI 0009 |
| --- | --- | --- | --- |
| roll +1 deg | warn (1.1 / 0.74) | warn (0.95 / 0.69) | warn (0.8 / 0.5) |
| roll +3 deg | fail (3.1 / 0.74) | fail (2.9 / 0.88) | fail (2.75 / 0.5) |
| roll +5 deg | fail (5.05 / 0.78) | fail (5.0 / 0.5) | fail (4.7 / 0.5) |
| pitch +1 deg | warn (0.85 / 0.5) | fail (1.05 / 0.5) | fail (1.05 / 0.5) |
| pitch +3 deg | fail (2.85 / 0.5) | fail (3.0 / 0.5) | fail (3.0 / 0.5) |
| pitch +5 deg | fail (4.85 / 0.5) | **pass**, pitch unchecked | inconclusive |
| yaw +1 deg | fail (1.05 / 0.5) | **pass** (1.0 / 1.38, std too large to resolve) | fail (1.0 / 0.5) |
| yaw +3 deg | fail (3.2 / 0.5) | fail (3.1 / 1.14) | fail (3.0 / 0.5) |
| yaw +5 deg | fail (5.15 / 0.5) | **pass**, yaw unchecked | fail (5.0 / 0.5) |
| x +5 cm | **pass**, translation unchecked | **pass**, translation unchecked | **pass**, translation unchecked |

Of 27 rotation perturbations, 23 are flagged (`warn` or `fail`), 1 is `inconclusive` and 3 pass
with the perturbed axis listed as unchecked or too coarse to resolve (KITTI 0005, 15 frames, pitch
+5 and yaw +1 / +5). A 5 cm translation error is **not detected on any recording**: translation
is not judged. A `pass` here always says `partial` and names the unchecked axes; the command
also prints "partial coverage: unchecked axes were not judged".

## Round trip and drift (Hilti)

* `calibrex estimate exp21 --pairs camera-lidar --tf camchain --tf prior.yaml`, with the prior
  the reference turned 3 deg about camera y: the exported `frames.yaml` holds the estimated
  rotation and the prior's translation, marked `NOT MEASURED x, y, z`. Its rotation is
  (0.01, 0.15, -0.25) deg from the reference (the prior's was (0, 3, 0)).
* `calibrex check <other recording> --tf frames.yaml` on exp01, exp02, exp04, exp07: `pass` on each
  judged axis (exp01 roll 0.05, pitch 0.00, yaw 0.1 deg against 0.5; exp02 pitch 0.1 / 0.5;
  exp04 pitch 0.0 / 0.5; exp07 roll 0.15 / 0.94). Which axes are judged depends on the start, so
  exp01 judges three axes here and one with the reference.
* `calibrex drift exp21 exp01 exp02 exp04 --pairs camera-lidar`: `stable`; pitch agrees to 0.21 deg
  (-127.01, -127.17, -127.22, -127.17 deg), the other axes were observed in one bag only
  (`inconclusive`). The rotation of an optical frame to a LiDAR frame is near a half turn, where a
  rotation vector flips sign; `drift` now re-expresses such vectors in the chart nearest the first
  bag's (without it the same four bags read as `drift` with a 254 deg pitch change).

## Runtime

Shared 8-core machine, first run (cold estimator cache), bag on a USB disk: Hilti exp21 20-41 s (the
bag read is most of it, 48 frames), the other Hilti recordings 51-101 s (cold disk), KITTI 18-38 s.
A repeat with the same candidate is a cache hit (0 s); `estimate` of a candidate already checked
took 6 s. The estimate is not independent of the candidate, so another candidate on the same bag
recomputes (about 20-40 s of compute after the bag read).

## Limitations

* Rotation only; translation is reported unchecked on every run.
* On a recording with little structure an axis is not judged: exp07 (a corridor) and exp02-exp04
  judge a single axis. The verdict says which.
* A start more than about 3 deg away can leave the search in a spurious optimum and the axis
  unchecked (KITTI 0005 pitch +5, yaw +5; KITTI 0009 pitch +5 left nothing judged): the pair then
  reads `pass (partial)` or `inconclusive`, never a confident `pass`.
* The jackknife std ignores systematic effects, so a std of 0.03 deg does not mean 0.03 deg
  accuracy (see exp07); the 0.5 deg floor is the resolution.
* Intrinsics, a non-optical camera frame, motion during a sweep and a solid-state LiDAR are not
  handled; the thresholds (2 deg control, 0.5 deg std bound, t >= 2) come from these dev
  recordings only.
