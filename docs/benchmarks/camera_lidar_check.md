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

## Limitations

* Rotation only; translation is reported unchecked on every run.
* On a recording with little structure an axis is not judged: exp07 (a corridor) and exp02-exp04
  judge a single axis. The verdict says which.
* A start more than about 3 deg away could leave the search in a spurious optimum; the per-axis
  profile start (above) now judges KITTI 0005 and 0009 pitch +5 deg, and the short-recording
  selection (above) KITTI 0005 yaw +5 deg. An axis the recording does not resolve stays
  unchecked whatever the start, and a start beyond the +-8 deg scan is not covered.
* The jackknife std ignores systematic effects, so a std of 0.03 deg does not mean 0.03 deg
  accuracy (see exp07); the 0.5 deg floor is the resolution.
* Intrinsics are not handled. Rig motion is compensated only with a velocity topic connected to
  the LiDAR, the rolling sweep only with a per-point time field (above; neither was available
  together on a real recording here), and an image in the camera body frame is only named, not
  corrected (above); a
  solid-state LiDAR is handled through angular-neighbour depth edges, evaluated only on the static
  AIST Livox recordings above; the thresholds (2 deg control, 0.5 deg std bound, t >= 2) come from these dev
  recordings only.
