# camera-lidar in `calibrex check`: Hilti 2022 and KITTI development data

`calibrex check`, `estimate` and `drift` judge the **rotation** of `T_camera_lidar` from a
bag's images and point clouds, without a target. This page records how the method was chosen,
what it did against the reference calibrations of two public datasets, and where it does not
work. It is **development evidence, not a claim**: the pair stays `no_claim` on the
[leaderboard](sota_leaderboard.md) and the thresholds below were set while looking at these
recordings (no frozen audit, no held-out data scored). A claim built on them was audited on
23 unread KITTI drives and **refuted** (5 of 9 gates): KITTI's calibration was flagged on 3
of them ([KITTI camera-LiDAR check audit](kitti_camera_lidar_check.md)).

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
  `drive_0009_sync` (443, 45 s). Drives 0027-0059 were not touched until the 13-drive run
  below. Converted with
  `calibrex convert kitti-raw DRIVE --camera image_02`, which writes the rectified `image_02`, its
  `CameraInfo` and `imu_link -> cam2_optical` composed from `calib_velo_to_cam.txt` and
  `calib_cam_to_cam.txt`, so the **vendor calibration is the candidate**. Frames are 1 s apart
  (until 2026-10-12; both drives are now short recordings, see below).
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

## Wider start and a solid-state LiDAR (2026-10-11)

Two changes to `calibrex.evaluation.camera_lidar_edge`, with the verdict rule unchanged (2 deg
control, 0.5 deg std bound, t >= 2):

* **Per-axis profile start.** The pooled objective is profiled along each rotation axis over
  +-8 deg (0.5 deg steps, the other axes at the candidate) and the rotation search also runs from
  the per-axis maxima. Both optima get their jackknife and held-out controls, and the profile start
  is kept only when more rotation DoFs pass the record rule there (or as many with a higher
  objective); the pair's reasons then name it. A plain "higher objective wins" rule failed on Hilti
  exp07: the corridor has a ridge (pitch +4.0, yaw +3.55 deg) with a higher pooled objective and no
  detected control, which turned the reference's judged roll into `inconclusive`.
* **Clouds without beams.** When no beam structure is found (a Livox-style non-repetitive scan),
  near-side depth edges are taken among the angular neighbours of each return (8 nearest unit
  directions, within four times the median spacing) instead of along beams, and a jump must also
  reach 10 % of the range so that a surface seen at a grazing angle does not read as an edge.

**Development recordings, re-run** (`--tf` candidates as above; Hilti exp21 and exp07 from the
Hugging Face release of the 2022 challenge, reference composed from
`calib_3_cam0-1-camchain-imucam.yaml` and `lidar_calibration.yaml`):

| Perturbation | before | after |
| --- | --- | --- |
| KITTI 0005 pitch +5 deg | `pass`, pitch unchecked (search stopped 2 deg short at a local maximum) | **`fail` (4.95 / 0.5)** |
| KITTI 0009 pitch +5 deg | `inconclusive` | **`fail` (4.95 / 0.5)** |
| KITTI 0005 yaw +5 deg | `pass`, yaw unchecked | `pass`, yaw unchecked: the estimate is right (-5.1 deg correction) but its jackknife std is 0.52 deg, over the 0.5 deg bound (the reference's yaw is unchecked for the same reason, std 0.51 deg on 16 frames) |
| every other KITTI 0005 / 0009 case (+1, +3, +5 deg about each axis) | as in the table above | identical |
| Hilti exp21, all ten cases | as in the table above | identical |
| unmodified reference, KITTI 0005 / 0009 / Hilti exp21 / exp07 | `pass` | `pass`, the same judged axes |

So two of the three +5 deg cases that were left unchecked are now judged and flagged; KITTI 0005
yaw +5 deg is not, because 15 s of that drive does not resolve yaw to 0.5 deg at any start.

**Solid-state (Livox Avia).** The public AIST LiDAR-camera calibration dataset
([Zenodo 7779880](https://zenodo.org/records/7779880), CC BY 4.0, `livox.tar.gz`: five static
rosbag2 recordings of a Livox Avia and an STC-MCS500POE camera) has the camera and the LiDAR
stamped on separate clocks, so the bags were rewritten with the bag receive time as header stamp
(`tools/restamp_rosbag2_headers.py`; valid only because the rig is static). The candidate is the
`T_lidar_camera` that direct_visual_lidar_calibration estimated on this dataset (its example
`calib.json`, a reference from another tool, not ground truth). Before this change every Livox
frame was dropped ("no beam structure", `inconclusive`).

| Recording | Frames / edge points | Verdict | roll | pitch | yaw |
| --- | --- | --- | --- | --- | --- |
| 13_46_54 | 8 / 19 355 | `pass` | -0.2 / 0.5 | -0.0 / 0.5 | -0.1 / 0.5 |
| 13_42_46 | 10 / 27 296 | `fail` | -0.1 / 0.5 | +0.35 / 0.5 | **+1.2 / 0.5** |
| 13_44_10, 13_44_54, 13_46_10 | 5-6 | `inconclusive` (fewer than 8 frames pair within 0.06 s) | - | - | - |

Translation is unchecked on every run. The rig is static, so the frames of one recording see the
same view: the jackknife std then measures image noise, not the scene, and a disagreement such as
yaw on 13_42_46 is a disagreement with the other tool's estimate that this data cannot settle.

```bash
python tools/restamp_rosbag2_headers.py livox/rosbag2_2023_03_09-13_46_54 restamped_13_46_54 \
  --topic /image --topic /camera_info --topic /livox/points
calibrex check restamped_13_46_54 --pairs camera-lidar \
  --tf examples/public_datasets/aist_lidar_camera_livox/koide_reference_frames.yaml
```

## Image in the camera body frame (2026-10-12)

`sensor_msgs/Image` is defined in the camera's optical frame (z forward, x right, y down). A
driver or URDF that stamps the image with the camera's REP 103 body frame (x forward, z up) puts
every LiDAR point outside the view, so the check could only say that no frame was usable. When
most dropped frames fail that way, the check now reads the first 16 synchronized frames once
more with the optical rotation applied. If that makes at least half of them usable (and more
than the frames as given), the reason names the body frame together with the fix. Nothing is
changed or assumed: the verdict stays `inconclusive`.

* KITTI 0005 with `cam2_optical` replaced by the body frame it would have (a `--tf` file with
  the reference rotation composed with the REP 103 rotation): `inconclusive`, "the camera frame
  'cam2_optical' looks like a REP 103 body frame ... read as a body frame, 16 of 16 frames put
  LiDAR edges in the image".
* The fix it names, an optical child of that frame (`rotation_quat_xyzw [0.5, -0.5, 0.5, -0.5]`)
  in a second `--tf` file plus `--frame-map /camera/image_raw=cam2_true_optical`: `pass` with
  the reference's deltas (roll 0.1 / 0.5, pitch 0.1 / 0.5, yaw 0.1 / 1.35 deg). The image header frame takes precedence over a `topic_frames` entry, so the
  topic has to be remapped with `--frame-map`.
* An inverted candidate (`T_lidar_camera` given for `T_camera_lidar`) on the same recording
  still puts edges in the image, the estimator runs and judges no axis (`no_judgeable_axes`), so
  an inverse reading was not added: it would not have fired on the one recording tried.

## Rig motion between the scan and the image (2026-10-12)

A LiDAR point and the image it is compared with are not measured at the same time. KITTI's
camera is triggered by its LiDAR (10.5 ms apart), but an unsynchronized camera can fire up to
half a frame away, and a spinning LiDAR measures each point at its own time during the sweep. On
a moving rig the LiDAR has turned and moved in between. `calibrex check` now moves every point
to the image time with the rig's velocity (constant over the interval):

* **Source.** An odometry or twist topic (rotation and translation; an INS before wheel
  odometry) or else an IMU (gyro, rotation only), whose frame is connected to the LiDAR in the
  frame tree; the twist is carried to the LiDAR origin through that static transform. Without
  one the scans are used as recorded, as before. `--camera-lidar-motion none` turns it off.
* **Time.** Each point by its own measurement time when the cloud has a per-point time field,
  else the whole scan by its reference time (so the rolling sweep is compensated only with the
  field). KITTI's converted clouds have no field.
* **Record.** The evidence gets an optional `motion_compensation` block (source topic, kind,
  frames moved, median and max rotation and translation applied); the check notes name it.

**Synchronized KITTI** (the drives of the benchmark above, `/oxts/odometry`, defaults, 48
frames): the applied move is small (0005: median 0.09 deg and 4.4 cm; 0009: 0.02 deg and 9.3 cm)
and the verdicts move in one direction only:

| Case | before | moved to the image time |
| --- | --- | --- |
| 0005 reference | pass, yaw 0.1 / 1.35 | pass, yaw 0.1 / 0.5 |
| 0005 yaw +1 deg | **pass** (0.9 / 1.35) | warn (0.9 / 0.5) |
| 0009 pitch +1 deg | warn (1.0 / 0.5) | fail (1.05 / 0.5) |

Every other case of the KITTI table (roll, pitch, yaw +-1/3/5 deg, x +5 cm, on both drives) keeps
its verdict; KITTI 0005 now flags all nine rotation errors. The AIST Livox recordings have no
velocity topic connected to the LiDAR, so their verdicts are unchanged (checked, all five).

**Unsynchronized camera (emulated).** `tools/camera_lidar_motion_stress.py` pairs every KITTI
scan with the image one frame later (+1: 114 ms after the scan) or earlier (-1: 93 ms before), a
real image and a real scan recorded about 0.1 s apart on the moving car (0005: 4.2 m/s and
8 deg/s median; 0009: 8.9 m/s). It runs the edge alignment on the same frames as recorded and
moved, and judges both with the check's rule. `error / tolerance` of the perturbed axis:

| Case | 0005 +1 | 0005 -1 | 0009 +1 | 0009 -1 |
| --- | --- | --- | --- | --- |
| reference, as recorded | pass, pitch unchecked | pass, pitch unchecked | pass, pitch unchecked | pass, pitch unchecked |
| reference, moved | pass, all three judged | pass, pitch and yaw unchecked | pass, all three judged | pass, pitch unchecked |
| roll +3, as recorded | fail (3.25 / 0.5) | fail (3.25 / 0.5) | fail (2.9 / 0.72) | fail (2.8 / 0.5) |
| roll +3, moved | fail (3.3 / 0.5) | fail (3.0 / 0.5) | fail (2.85 / 0.5) | fail (2.8 / 0.5) |
| pitch +3, as recorded | **pass**, pitch unchecked | **pass**, pitch unchecked | **pass**, pitch unchecked | **pass**, pitch unchecked |
| pitch +3, moved | fail (3.2 / 0.5) | **pass**, pitch unchecked | fail (3.1 / 0.5) | fail (2.95 / 0.5) |
| yaw +3, as recorded | fail (3.15 / 0.5) | fail (3.1 / 0.74) | fail (3.0 / 0.66) | fail (3.05 / 0.5) |
| yaw +3, moved | fail (3.0 / 0.5) | **pass**, yaw unchecked | fail (2.9 / 0.5) | fail (3.1 / 0.5) |
| yaw +1, as recorded | fail (1.15 / 0.5) | warn (1.15 / 0.74) | warn (1.0 / 0.77) | fail (1.05 / 0.5) |
| yaw +1, moved | fail (1.0 / 0.5) | **pass**, yaw unchecked | warn (0.9 / 0.5) | fail (1.1 / 0.5) |

As recorded, the 0.1 s of motion leaves pitch unjudged on every pairing and all four pitch +3 deg
errors pass. Moved, 13 of the 16 errors are flagged (12 as recorded), the references still pass,
and on the +1 pairings all three axes are judged. **0005 -1 is worse moved:** pitch and yaw are
unchecked and yaw +1 and +3 deg, flagged as recorded, now pass with yaw unchecked. The bias is
not the cause: profiling the objective at the reference, the moved frames put the pitch peak
back where the synchronized frames have it on every pairing (0005: -0.1 deg synchronized, +0.3
deg as recorded at +-1, -0.2 / 0.0 deg moved; 0009: 0.0, +0.1 / +0.2, 0.0 / 0.0 deg). What
changed is the held-out yaw control on 0005 -1, which sits at the t >= 2 line (t = 2.3 as
recorded, 1.6 moved). Two drives and four pairings are a small sample; whether the translation
part of the move (0.39 m median at -1) disturbs the depth edges near the car is not resolved.

## Rolling sweep on a handheld rig: Hilti exp21 (2026-10-12)

The rolling sweep (each point moved by its own measurement time) needs a cloud with a per-point
time field on a moving rig with a camera. Hilti 2022 `exp21_outside_building` is one: handheld,
the PandarXT-32 cloud carries `timestamp`, and the Alphasense IMU is connected to the LiDAR
through the calibration files, so `calibrex check` picks the gyro (rotation only) by itself.

* **Data.** The first 6 GiB of the 12 GB bag ([Hugging Face
  `Hilti-Research/hilti-slam-challenge-2022`](https://huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2022),
  CC BY-NC-SA 3.0), fetched with an HTTP range request and cut after its last complete record
  with `tools/trim_partial_rosbag1.py`; the ROS 1 bag is read directly. That head holds about
  81 s of frames, so the 48 frames are spread over it (short-recording selection).
* **Candidate.** Kalibr's `calib_3_cam0-1-camchain-imucam.yaml` (cam0 to IMU) and the platform's
  `lidar_calibration.yaml` (IMU to PandarXT-32), both given with `--tf`. Perturbations replace
  the second with a `slac.check_frames` file whose PandarXT-32 is turned about the camera axes.
* **Applied move.** Median 0.14 deg at the scan's reference time (max 0.39 deg), and each point
  by its own time within the sweep.

```bash
curl -L -r 0-6442450943 -o exp21_head.bag \
  https://huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2022/resolve/main/rosbags/exp21_outside_building.bag
python tools/trim_partial_rosbag1.py exp21_head.bag
calibrex check exp21_head.bag --pairs camera-lidar --camera alphasense/cam0/image_raw \
  --max-duration-s 120 --tf calib_3_cam0-1-camchain-imucam.yaml --tf lidar_calibration.yaml \
  [--camera-lidar-motion none] --output check.yaml
```

**Ablation at the reference.** The objective is the edge-alignment score at the candidate (higher
means LiDAR edges land on image edges better); "per-point time" off means the cloud's time field
is ignored (scans paired at the header stamp and moved as a whole):

| Per-point time | Motion | Verdict | Judged axes, `error / tolerance` (deg) | Objective at the reference |
| --- | --- | --- | --- | --- |
| used | none | pass | roll 0.15 / 0.5, pitch 0.25 / 0.5, yaw 0.25 / 0.5 | 0.360 |
| used | gyro | pass | roll 0.1 / 0.5, pitch 0.2 / 0.5, yaw 0.15 / 0.63 | **0.387** |
| ignored | none | pass | pitch 0.15 / 0.5 (roll, yaw unchecked) | 0.356 |
| ignored | gyro (whole scan) | pass | pitch 0.2 / 0.5 (roll, yaw unchecked) | 0.379 |

Moving the points raises the objective by about 6 % even scan by scan, and by 2 % more when each
point moves by its own time; pairing at the visible points' time is what lets roll and yaw be
judged at all.

**Perturbed candidates** (per-point time used, `error / tolerance` of the perturbed axis):

| Perturbation | as recorded | moved with the gyro |
| --- | --- | --- |
| roll +1 deg | warn (0.75 / 0.5) | warn (0.9 / 0.5) |
| roll +3 deg | **inconclusive**, no axis judged | fail (2.85 / 0.5) |
| pitch +1 deg | warn (0.75 / 0.5) | warn (0.8 / 0.5) |
| pitch +3 deg | fail (2.75 / 0.5) | fail (2.8 / 0.5) |
| yaw +1 deg | fail (1.2 / 0.54) | warn (1.15 / 0.63) |
| yaw +3 deg | fail (3.2 / 0.5) | fail (3.15 / 0.61) |

Moved, all six errors are flagged (five as recorded: roll +3 left every axis unjudged), the
measured error is closer to the applied one in all five cases judged both ways (mean miss 0.16
deg against 0.23 deg),
and the objective at the candidate is 6-7 % higher in every case. The cost is a slightly larger
yaw std (tolerance 0.5 to 0.61-0.74 deg), which turned yaw +1 from fail into warn. One handheld
recording, rotation-only motion (the gyro has no translation); exp07 and the other Hilti
recordings were not re-run.

## The known-bad control near its threshold (2026-10-12)

Two results above hinge on the held-out control: KITTI 0005 with the earlier image (moved) leaves
yaw unjudged, and the Hilti yaw tolerance widens when the points are moved. The control fits the
other five blocks, scores the held-out block at that fit and at the fit turned 2 deg each way,
and needs a one-sided t of at least 2 over the six blocks in both directions. Whether that rule
or the data is the limit was studied on the emulated KITTI pairings; **the rule is unchanged**.
(After the KITTI audit the bound was raised to 2.5:
[a stricter control](#after-the-kitti-audit-a-stricter-control-and-far-optima-2026-10-14).)

**Per-block drops** (objective at the fit minus objective at the fit turned by -2 deg, x 1e-3):

| Case | Drops of the six held-out blocks | t |
| --- | --- | --- |
| 0005 -1, moved, yaw | 3.3, 2.3, 19.9, 28.7, 0.5, **-3.6** | 1.6 |
| 0009 -1, moved, pitch | 4.7, 1.8, **-1.6**, 38.5, 5.4, 66.0 | 1.7 |
| 0005 -1, as recorded, pitch | 51.5, 9.6, **-13.1**, 7.5, 3.2, 1.3 | 1.1 |

The misses are not noise around a clear effect. In each case a held-out block prefers the
turned fit, and a few blocks carry most of the drop: parts of the drive do not constrain that
axis in that direction.

**Levers tried** (lowest t of the two directions, roll / pitch / yaw, emulated pairings):

| Case | default: 6 blocks, 48 frames | 8 blocks, 48 | 6 blocks, 96 | 12 blocks, 96 |
| --- | --- | --- | --- | --- |
| 0005 -1 as recorded | 3.6 / 1.1 / 2.3 | 4.6 / 1.6 / 2.3 | 3.3 / 1.1 / 2.1 | 3.7 / 1.3 / 2.1 |
| 0005 -1 moved | 3.1 / 1.2 / 1.6 | 4.1 / 1.2 / 1.4 | 3.1 / 1.4 / 1.7 | 4.0 / 1.6 / 1.7 |
| 0005 +1 as recorded | 3.6 / 1.4 / 2.6 | 3.6 / 1.7 / 2.1 | 3.5 / 0.9 / 2.1 | 3.3 / 0.9 / 2.5 |
| 0005 +1 moved | 4.1 / 2.4 / 3.7 | 4.6 / 2.5 / 1.8 | 4.3 / 2.4 / 3.6 | 6.1 / 3.1 / 3.3 |
| 0009 -1 as recorded | 3.0 / 1.3 / 3.3 | 3.2 / 1.4 / 3.4 | 2.7 / 1.3 / 4.1 | 3.1 / 1.8 / 3.2 |
| 0009 -1 moved | 4.1 / 1.7 / 4.9 | 3.4 / 1.9 / 4.6 | 4.0 / 1.9 / 4.7 | 4.1 / 2.5 / 5.5 |
| 0009 +1 as recorded | 2.1 / 0.1 / 2.8 | 2.8 / 0.3 / 2.3 | 2.7 / 0.5 / 3.3 | 2.7 / 1.0 / 2.9 |
| 0009 +1 moved | 4.3 / 3.0 / 5.0 | 4.4 / 3.7 / 4.0 | 4.0 / 3.5 / 4.1 | 4.7 / 4.4 / 4.9 |

* **Twice the frames** (same six blocks) leaves t where it was: the spread is between parts of
  the recording, not within a block.
* **Eight blocks** moves t both ways (0005 +1 moved yaw falls from 3.7 to 1.8): near the line,
  the verdict depends on where the blocks are cut.
* **Twelve blocks of 96 frames** raises t modestly and judges one more axis (0009 -1 moved
  pitch, 2.5). 0005 -1 moved yaw stays at 1.7, and the as-recorded pitch, which is biased,
  stays unjudged. The jackknife costs about four times as much (12 fits on 88 frames against 6
  on 40), so it is not the default.
* **A rank test** (Wilcoxon signed-rank, both directions at p <= 0.05) agrees with the t rule
  on 39 of 42 axis decisions (KITTI pairings and Hilti exp21). It judges one more axis and two
  fewer, so it is no improvement.

The Hilti yaw tolerance (0.5 to 0.61-0.74 deg when moved) is the jackknife std of six fits:
0.13 deg as recorded and 0.21 deg moved. With five degrees of freedom an estimated std scatters
by about 30 %, and the fits move in 0.05 deg steps, so that difference is within the
statistic's own spread. Twelve blocks would narrow it, at the same fourfold cost.

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

## Short recordings (2026-10-12)

`calibrex check` takes at most 48 scans, at least 1 s apart (2.5 s with `--max-duration-s 120`).
A recording shorter than that leaves most of the frame budget unused. KITTI 0005, at 15 s, got
16 frames, and its yaw std (0.51-0.52 deg) stayed just over the 0.5 deg bound.

Such a recording now spreads up to 48 scans, at least 0.25 s apart, evenly over its length. The
scans are grouped into 6 contiguous jackknife blocks as before, so frames close in time stay in
the same block. A long recording gets exactly the frames it got before.

Denser frames add evidence only when the view changes. If the median mean absolute difference
of consecutive denser images is under 0.05 of full scale, the regular selection is kept. The
static AIST Livox recordings measure 0.0035 to 0.011 and keep it; the moving KITTI drives
measure 0.14 to 0.21. The evidence artifact's notes say which selection was used. The verdict
rule is unchanged: 2 deg control, 0.5 deg std bound, t >= 2.

KITTI 0005 now gets 48 frames and KITTI 0009 gets 48 instead of 45. Every cell is the verdict
and the judged axis's `error / tolerance` in deg:

| Perturbation | KITTI 0005, before (16 frames) | KITTI 0005, now (48) | KITTI 0009, now (48) |
| --- | --- | --- | --- |
| unmodified reference | `pass`, yaw unchecked | `pass`, roll, pitch and yaw judged | `pass`, roll, pitch and yaw judged |
| roll +1 / +3 / +5 deg | warn / fail / fail | **fail** (1.1 / 0.5) / fail (3.05 / 0.5) / fail (5.0 / 0.5) | warn (0.8 / 0.5) / fail (2.75 / 0.5) / fail (4.75 / 0.5) |
| pitch +1 / +3 / +5 deg | fail / fail / fail | fail (1.05 / 0.5) / fail (3.05 / 0.5) / fail (5.0 / 0.5) | warn (1.0 / 0.5) / fail (3.0 / 0.5) / fail (4.9 / 0.5) |
| yaw +1 deg | `pass` (1.0 / 1.38) | `pass` (0.9 / 1.35) | warn (1.0 / 0.5) |
| yaw +3 deg | fail (3.1 / 1.14) | fail (2.9 / 0.51) | fail (3.0 / 0.5) |
| yaw +5 deg | `pass`, yaw unchecked | **fail (4.9 / 1.14)** | fail (5.0 / 0.5) |
| x +5 cm | `pass`, translation unchecked | `pass`, translation unchecked | `pass`, translation unchecked |

On KITTI 0005 the yaw tolerance stays wide (1.1 to 1.5 deg, three jackknife sigmas), so a 1 deg
yaw error still passes. All three +5 deg cases left unchecked by the first benchmark are now
judged and flagged: KITTI 0005 pitch and yaw, and KITTI 0009 pitch.

The AIST Livox verdicts are identical to the table above, because the static guard keeps their
frames. The Hilti bags are no longer on disk and were **not** re-run. A Hilti recording
shorter than 117.5 s under `--max-duration-s 120` would now take the short-recording selection.
That might apply to exp03 (24 frames) or exp04 (34), whose short counts may also come from
dropped pairs.

## A calibration-target session: sensor_calib_tas's demo bag (2026-10-13)

[sensor_calib_tas](https://github.com/UniBwTAS/sensor_calib_tas) (UniBw TAS, BSD-3-Clause)
calibrates cameras, LiDARs, radars and GNSS together against a custom target. Its demo
recording (`demo/calibrate_mucar3`, rosbag2, 1.09 GB, Git LFS SHA-256 `6c61dfe4…693a`)
has two surround cameras, a VLS-128, radar detections and the vehicle's `/tf_static`.
`calibrex check` plans two `camera-lidar` pairs and one `camera-camera` pair on it. None
can be judged:

* **camera-lidar (front, right): `inconclusive`.** The recording is 4.9 s long, from a
  parked vehicle, with the target moved in front of it. The short-recording selection
  finds 5 frames that differ enough in view; the estimator needs at least 8.
* **camera-camera (front, right): skipped, `missing_topic`.** The two cameras are not
  synchronised (about 24 ms apart), and the bag has no IMU to fit their timing.
* **The radar detections** (`radar_msgs/DetectionRecord`) have no role in the check.

A session like this is what the target-based tool needs. The targetless check needs
driving. The run found one bug, fixed since: the front camera publishes
`image_rect_color` (already undistorted) next to a CameraInfo whose `D` is the raw
image's (k1 = −0.27). The check now reads a rectified stream's intrinsics from `P`
without distortion, the ROS image_pipeline convention.

The demo bag holds no result, so it has no `*_calibrated` frame on `/tf_static`. Once
a session has been solved and recorded,
`calibrex external-run import-sensor-calib-tas` turns its result into a `--tf` file.
The check then judges that target calibration on a driving bag. The synthetic test
(`tests/unit/test_import_sensor_calib_tas.py`) runs this end to end:

* **Without the import.** A driving bag whose deployed LiDAR rotation is 3 deg off
  fails.
* **With it.** The same bag passes with the imported result, and the override is
  listed in the report.

## Thirteen KITTI drives (2026-10-14)

Before a pre-registered audit, the check ran on 13 KITTI 2011_09_26 drives: 0005, 0009,
0014, 0015, 0022, and the eight drives 0027 to 0059 that it had not read before. On each
drive it judged KITTI's own calibration and six turned candidates: the camera turned 1 and 3
deg about each of its own axes (`T_camera_lidar' = R T_camera_lidar`). The harness is
`tools/kitti_camera_lidar_sota_evaluate.py`, which downloads, converts and checks one drive
at a time; the defaults are unchanged (48 scans, 2 deg control, 0.5 deg std bound, `t >= 2`).
These 13 drives are now spent: everything below was decided while looking at them.

**The first run (the code of `c83ab6c`) failed KITTI's calibration on 5 of the 13
drives.** The estimate settled 2 to 11 deg away from it and passed its own held-out
controls there (estimate minus KITTI, deg):

| Drive | roll | pitch | yaw |
| --- | ---: | ---: | ---: |
| 0015 | +2.55 | - | - |
| 0027 | -2.05 | - | - |
| 0028 | - | -3.50 | -11.45 |
| 0051 | - | +3.40 | - |
| 0057 | +7.70 | - | +0.60 |

Four changes followed, each tested on these drives:

* **The objective pulled towards texture.** The edge response is spilled outward from every
  edge (Levinson and Thrun's Eq. 1), so a textured region (foliage, a facade, gravel) is
  high all over. Moving the LiDAR's depth edges into such a region raises the objective
  whether or not they meet an image edge. On 0028 the yaw profile rose all the way to the
  11 deg the estimate reached. The response now has its local mean (a Gaussian of 20 px of
  the downscaled image, `edge_background_sigma_px`) subtracted, so a point scores for
  lying on an edge rather than in a busy region. 0028's profiles then peak at KITTI's
  calibration; 0005's stay as sharp.
* **A standing rig repeats one view.** 0057 stands 29 s at a crossing: 28 of its 37
  selected frames show the same scan with only passing traffic changed (image change 0.002
  to 0.034, against 0.06 to 0.18 while driving). They add weight but no view, and make the
  jackknife blocks and controls copies of each other. A spinning LiDAR's frame is now
  dropped when its image differs from the last kept frame's by less than 0.05 of full
  scale (`repeated_view_change`). A solid-state LiDAR keeps such frames: its pattern fills
  in. 0057 keeps 9 frames and judges pitch only; 0051 drops 11 and 0009 3.
* **The rolling sweep.** A KITTI scan takes about 0.1 s; at 36 km/h the car moves a metre
  in it. `calibrex convert kitti-raw --velodyne-time azimuth` now adds a per-point `time`
  field modelled from each point's azimuth and the sweep's start and end stamps (the usual
  KITTI deskewing model: clockwise, forward at the header stamp). The check then pairs the
  image with the time its visible points were measured and moves each point to the image
  time with the OXTS velocity. The default stays `none`.
* **A turned candidate could end on a ridge.** With the first three changes every drive
  passed KITTI's calibration, and every 1 deg turn about a judged axis was flagged, but 5
  of 36 3 deg turns were not. From 3 deg away, the search from the candidate, and from all
  the per-axis profile maxima together, ran along a ridge (on 0015, roll +2.5 with yaw -4)
  to an optimum with fewer axes observed. On 0015 the roll turn then read 0.45 deg and was
  judged right. In the four cases examined (0015 roll and pitch, 0028 pitch and yaw), the
  profile maximum of the turned axis alone was at the turn, -3.0 deg, and the search from
  it found the right optimum with all three axes observed. The search now also starts from
  each axis's maximum on its own; an optimum replaces the candidate's by the existing rule
  (more rotation DoFs observed, or as many with a higher objective).

With all four (commit `5d72a6c`, `--velodyne-time azimuth`), every drive passes KITTI's
calibration, and every turn about an axis judged on the drive is flagged on that axis and
on no other. Estimate minus KITTI for the baseline (deg; "-" not judged), then the check's
verdict on the turned axis and its reading (deg; `*` the axis is not judged on the drive's
baseline):

| Drive | Baseline | roll | pitch | yaw | roll +3 / +1 | pitch +3 / +1 | yaw +3 / +1 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 0005 | pass | -0.10 | -0.10 | +0.10 | fail 3.00 / fail 1.05 | fail 3.05 / fail 1.10 | fail 2.90 / warn 0.95 |
| 0009 | pass | +0.15 | -0.05 | +0.05 | fail 2.80 / warn 0.80 | fail 3.00 / fail 1.05 | fail 2.95 / warn 0.95 |
| 0014 | pass | - | -0.05 | -0.05 | unjudged / unjudged * | fail 3.00 / fail 1.05 | fail 3.05 / fail 1.05 |
| 0015 | pass | +0.10 | -0.05 | -0.05 | fail 2.85 / warn 0.90 | fail 3.00 / fail 1.05 | fail 3.05 / fail 1.05 |
| 0022 | pass | +0.05 | -0.15 | -0.15 | fail 2.85 / warn 0.90 | fail 3.10 / fail 1.15 | fail 3.10 / fail 1.05 |
| 0027 | pass | +0.05 | -0.25 | -0.20 | fail 3.00 / warn 0.90 | fail 3.20 / fail 1.20 | fail 3.20 / fail 1.20 |
| 0028 | pass | +0.10 | -0.25 | -0.10 | fail 2.75 / warn 0.90 | fail 3.20 / fail 1.25 | fail 3.10 / fail 1.20 |
| 0035 | pass | +0.10 | -0.20 | -0.00 | fail 2.85 / warn 0.90 | fail 3.15 / fail 1.15 | fail 3.00 / warn 1.00 |
| 0039 | pass | +0.15 | -0.05 | -0.00 | fail 2.80 / warn 0.85 | fail 3.00 / warn 1.00 | fail 3.00 / warn 1.00 |
| 0046 | pass | -0.10 | -0.20 | -0.00 | fail 2.90 / fail 1.00 | fail 3.10 / fail 1.15 | fail 3.05 / warn 1.00 |
| 0051 | pass | +0.20 | -0.05 | -0.05 | fail 2.80 / warn 0.80 | fail 3.00 / fail 1.05 | fail 3.00 / warn 1.00 |
| 0057 | pass | - | +0.35 | - | unjudged / unjudged * | fail 2.10 / warn 0.60 | unjudged / unjudged * |
| 0059 | pass | +0.10 | -0.05 | +0.15 | fail 2.85 / warn 0.90 | fail 3.00 / warn 1.00 | fail 2.85 / warn 0.85 |

* **KITTI's calibration:** 13 of 13 pass; 36 of 39 axes judged; every judged axis within
  0.35 deg of KITTI (0057 pitch; 0.25 deg or less on the other drives).
* **3 deg turns:** 36 of 36 flagged, read within 0.25 deg of 3 except on 0057 (2.10).
* **1 deg turns:** 36 of 36 flagged (`warn` or `fail`), read within 0.25 deg of 1 except on
  0057 (0.60).
* **Every axis judged in a turned run** is within 0.89 deg of KITTI's calibration (0057).
* **Unjudged:** 0014 roll and 0057 roll and yaw on the baseline (their controls are not
  detected), so their turns are not counted. The turned runs leave them unjudged too, except
  0057's 3 deg pitch run, which judges roll 0.81 deg off KITTI.

**The per-point time matters on two drives.** The same code without it
(`--velodyne-time none`) also passes KITTI's calibration on all 13 drives (36 of 39 axes
judged, all within 0.40 deg) and flags 35 of 36 turns of each size. Its turned runs, though,
judge axes up to 7.4 deg off KITTI's calibration (0057: roll, in the 3 deg pitch run) and up
to 2.2 deg on 0014, whose 1 deg yaw turn reads -1.00 and 3 deg yaw turn 1.10; on 0057 a 1 deg
pitch turn is judged right. With the per-point time the worst is 0.89 deg. The audit therefore
converts with `--velodyne-time azimuth`, a choice made on these drives.

Each check took 35 to 140 s, with a drive's seven checks running at once on 4 cores. The
scores are in
[`docs/assets/kitti_camera_lidar/development/`](../assets/kitti_camera_lidar/development/scores.json),
with those without the per-point time (`velodyne_time_none/`) and before the per-axis starts
(`before_per_axis_starts/`, commit `6ea6110`). The claim built on them was pre-registered
and audited on 23 KITTI drives the check had never read, and refuted: on 20 drives the check
behaved as here, on three it flagged KITTI's calibration
([KITTI camera-LiDAR check audit](kitti_camera_lidar_check.md#results-refuted-5-of-9-gates)).

## After the KITTI audit: a stricter control and far optima (2026-10-14)

The pre-registered audit on 23 KITTI drives the check had not read was refuted
([results](kitti_camera_lidar_check.md#results-refuted-5-of-9-gates)): on three drives it
flagged KITTI's calibration. Reading those runs afterwards (post hoc, so those 23 drives are
spent too) showed two rules letting wrong axes through:

* **Controls just over the line.** Every axis judged more than 0.75 deg off KITTI's
  calibration had its held-out control at `t` 2.09 to 2.33, against a bound of 2.
* **Far optima.** 10 of those 12 axes came from the search started at the profile maxima.
  Its optimum replaced the candidate's when it observed more rotation DoFs, or as many with
  a higher objective. On 0029 an optimum that tied the candidate's count won on the
  objective with yaw 9.6 deg off; on 0056 one with yaw at the -12 deg search bound replaced
  a candidate optimum that observed nothing.

Two rules changed (commit `3a05cf8`):

* **`min_control_t` is 2.5.** A DoF is observed only when its held-out control reaches
  `t` 2.5 in both directions.
* **A far optimum replaces the current one only when** it observes strictly more rotation
  DoFs, at least two (`start_scan_min_observed`), and none of its rotation DoFs is at the
  search bound. A tie no longer goes to the higher objective.

**How they were chosen.** The baseline estimator's inputs (the selected frames after
pairing and motion compensation) were saved for all 36 spent 2011_09_26 drives and
re-estimated in memory under candidate rules, with KITTI's calibration and the six turned
candidates. The turned candidates reused the baseline's frames, whereas a real run pairs
and moves the points with its own candidate, so these are approximations of runs. On the
three failing drives each change was needed for one of them: the bound for 0056, the end
of ties for 0029, and `t` 2.5 for 0032; "at least two" then removed a roll judged 6.6 deg
off in a turned run of 0056 (`t` 2.63). With all four parts in place, no drive flagged
KITTI's calibration, the worst judged axis was 0.45 deg off it (0.89 deg in turned runs),
86 of 108 axes were judged, and 83 of 86 3 deg and 86 of 86 1 deg turns were flagged.

**Real runs.** With both changes, the check ran again on all 36 2011_09_26 drives and on
2011_09_30_0018, through the v2 audit's harness (`tools/kitti_camera_lidar_v2_evaluate.py`:
each drive's first 600 frames, `--velodyne-time azimuth`, the six turns). 2011_09_30_0018 is
from another day, with a calibration of its own that differs from 2011_09_26's by 0.97 deg
(mostly roll); earlier camera-LiDAR research read it. Every conversion and check exited 0.
The v1 rows are the runs of the thirteen-drive section and of the audit:

| Rule | Drives flagging KITTI | Worst judged axis, baseline / turned (deg) | Axes judged | 3 deg turns flagged | 3 deg judged right | 1 deg turns flagged |
| --- | --- | --- | --- | --- | --- | --- |
| v1 (`t >= 2`), 36 drives of 2011_09_26 | 3 (0029, 0032, 0056) | 9.60 / 9.66 | 97 of 108 | 93 of 97 | 1 | 96 of 97 |
| v2, the same 36 drives | 0 | 0.45 / 0.89 | 86 of 108 | 83 of 86 | 0 | 86 of 86 |
| v2, 2011_09_30_0018 | 0 | 0.25 / 0.35 | 3 of 3 | 3 of 3 | 0 | 3 of 3 |

Every drive whose result changed (estimate minus KITTI on the judged axes, deg):

| Drive | v1 baseline | v2 baseline | v2 turns flagged, 3 deg and 1 deg |
| --- | --- | --- | --- |
| 0013 | pass: roll +0.10, pitch -0.10, yaw -0.20 | pass: pitch -0.10 | 1 of 1 and 1 of 1 |
| 0015 | pass: roll +0.10, pitch -0.05, yaw -0.05 | pass: pitch -0.05, yaw -0.05 | 2 of 2 and 2 of 2 |
| 0029 | **fail**: pitch -0.10, yaw +9.60 | pass: pitch -0.15, yaw -0.15 | 1 of 2 and 2 of 2 |
| 0032 | **warn**: pitch -0.05, yaw -0.80 | pass: pitch -0.05 | 1 of 1 and 1 of 1 |
| 0046 | pass: roll -0.10, pitch -0.20, yaw -0.00 | pass: roll -0.10, pitch -0.20 | 2 of 2 and 2 of 2 |
| 0051 | pass: roll +0.20, pitch -0.05, yaw -0.05 | pass: roll +0.20, yaw -0.05 | 2 of 2 and 2 of 2 |
| 0056 | **fail**: roll +2.75, pitch +0.15 | inconclusive: none judged | none counted |
| 0057 | pass: pitch +0.35 | pass: pitch +0.10 | 1 of 1 and 1 of 1 |
| 0070 | pass: pitch -0.05 | pass: pitch -0.05 | 0 of 1 and 1 of 1 |
| 0091 | pass: roll +0.20, pitch -0.15, yaw -0.20 | pass: pitch -0.15 | 0 of 1 and 1 of 1 |
| 0096 | pass: roll +0.10, pitch -0.05, yaw +0.00 | pass: pitch -0.05, yaw +0.00 | 2 of 2 and 2 of 2 |
| 2011_09_30_0018 | not run | pass: roll +0.25, pitch +0.05, yaw -0.10 | 3 of 3 and 3 of 3 |

* **The three false alarms are gone.** On 0029 the candidate's own optimum stands (pitch and
  yaw within 0.15 deg of KITTI). On 0032 yaw's control is at `t` 2.30, so yaw is not judged.
  On 0056 the far optimum is no longer taken; at the candidate's own optimum no control is
  detected, and the check returns `inconclusive` (its unjudged estimate is 4.6 deg off in
  roll).
* **The cost is coverage:** 86 of 108 axes are judged, against 97. Every axis lost
  apart from 0056's had its control at `t` 2.02 to 2.49, the same in both versions.
* **Three 3 deg turns are missed:** 0029 yaw and 0070 and 0091 pitch, which their turned
  runs leave unjudged. None is judged right.
* **On 2011_09_30_0018** the check passes that day's calibration and flags all six turns.

The scores, with the driver's record, are in
[`docs/assets/kitti_camera_lidar_v2/development/`](../assets/kitti_camera_lidar_v2/development/scores.json);
the runs are reproducible from commit `3a05cf8`, whose `src/` and run tools the v2 audit
freezes.

The sections above this one (Hilti, AIST, the sensor_calib_tas bag, the thirteen KITTI
drives) were run with `t >= 2` and the earlier start rule, and are not re-run.

## Limitations

* Rotation only; translation is reported unchecked on every run.
* On a recording with little structure an axis is not judged: exp07 (a corridor) and exp02-exp04
  judge a single axis. The verdict says which.
* A start more than about 3 deg away could leave the search in a spurious optimum; the per-axis
  profile starts (above) now judge KITTI 0005 and 0009 pitch +5 deg and every 3 deg turn of the
  13 KITTI drives, and the short-recording selection (above) KITTI 0005 yaw +5 deg. An axis the
  recording does not resolve stays unchecked whatever the start, and a start beyond the +-8 deg
  scan is not covered.
* A rig that stands still for most of a recording keeps few frames (KITTI 0057: 9 of 37) and
  judges little.
* An axis can be judged at a wrong optimum: on 3 of 23 unseen KITTI drives the check flagged
  KITTI's calibration, judging axes 0.8 to 9.6 deg off with held-out controls just over their
  threshold (`t` 2.09 to 2.33), two of them from an alternative search start (audit above).
  The control bound and the start rule were changed after that audit (above); a pre-registered
  audit on KITTI's other four days tests the changed rules
  ([v2](kitti_camera_lidar_v2_check.md)).
* The jackknife std ignores systematic effects, so a std of 0.03 deg does not mean 0.03 deg
  accuracy (see exp07); the 0.5 deg floor is the resolution.
* Intrinsics are not handled. Rig motion is compensated only with a velocity topic connected to
  the LiDAR, the rolling sweep only with a per-point time field (above; evaluated on KITTI
  and one handheld Hilti recording), and an image in the camera body frame is only named, not
  corrected (above); a
  solid-state LiDAR is handled through angular-neighbour depth edges, evaluated only on the static
  AIST Livox recordings above; the thresholds (2 deg control, 0.5 deg std bound, t >= 2.5) come
  from these dev recordings only.
