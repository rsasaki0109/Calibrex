# Hilti 2022 fisheye stereo falsification (development run)

The KITTI data-only decision
([KITTI Camera-LiDAR falsification](kitti_camera_lidar_falsification.md)) repeated on a
second dataset: two handheld Hilti 2022 recordings whose camera pair is an
equidistant fisheye (alphasense cam0 and cam1, 0.108 m apart) and whose LiDAR is a
PandarXT-32. Both recordings share one calibration (`calib_3`). These are development
runs, not a published artifact: the bags are not redistributed and no CI test pins the
numbers below.

| recording | decision | reference | known-bad |
| --- | --- | --- | --- |
| `exp21_outside_building` | `pass` | roll, pitch, yaw `pass`; z unjudged | pitch `fail`; z unjudged |
| `exp01_construction_ground_level` | `pass` | roll, pitch, yaw, z `pass` | pitch and z `fail` |

`exp01` was chosen as the second recording before it was scored: it is an indoor
construction site, where `exp21` is outdoors. Both runs use the tool unchanged and the
first 6 GB of each bag.

## Setup

* **Candidates.** The reference `T_cam0_lidar` is composed from the dataset's Kalibr
  camchain (cam0 to IMU) and its `lidar_calibration.yaml` (IMU to LiDAR). The known-bad
  is declared before scoring, the same perturbation in camera terms as KITTI's: the
  reference turned 5 deg about the camera's y axis (pitch) and moved 0.25 m along its
  optical axis. Its declared axes are pitch and z.
* **Frames.** What `calibrex check` reads for each candidate, as it would on this bag:
  * its selection of cam0 images and scans (the short-recording policy);
  * each image paired with the scan at the time the candidate's visible points were
    measured, read from the cloud's per-point `timestamp` field;
  * the points turned to the image time with the IMU's gyro. This corrects rotation
    only: the bag has no velocity topic, so the rig's translation during the sweep is
    not compensated.
  * Each image comes with the cam1 image of the same stamp.
* **Rotation** from edge alignment on the raw fisheye cam0 images (equidistant model).
* **Translation along the optical axis** from stereo depth on the rectified pair
  (`calibrex.evaluation.fisheye_stereo`): both cameras are re-projected onto a common
  pinhole whose x axis is the baseline. cam1 sits to the left of cam0, so the rectified
  frame is turned 180 deg about the optical axis; the transform to it is applied
  to every candidate the same way.
* **Rule and decision.** Unchanged from KITTI: `calibrex check`'s `judge_pair`, and the
  decision passes when the reference passes and the known-bad fails.

## Result: `exp21` (outside a building)

**`pass`, on rotation.** The reference passes on roll, pitch and yaw, and the known-bad
fails on pitch. z goes unjudged on both candidates. The run used 48 frame pairs from
the start of `exp21` (git commit `fb599b2`), with a vertical offset of
−0.83 px removed and the gyro turning each scan by 0.14 deg (median; at most 0.39 deg).

| candidate | roll (deg) | pitch (deg) | yaw (deg) | z (m) | verdict |
| --- | --- | --- | --- | --- | --- |
| dataset reference | −0.10 / 0.50 `pass` | −0.20 / 0.50 `pass` | +0.15 / 0.63 `pass` | −0.096, unjudged | `pass` |
| known-bad | unjudged | +4.95 / 0.50 `fail` | +0.20 / 0.50 `pass` | +0.154, unjudged | `fail` |

Each judged cell gives the candidate's error, then its tolerance.

* **The known-bad's pitch is judged.** The decision is therefore a pass, not
  `inconclusive`: pitch is one of its declared axes.
* **Stereo depth sees the 0.25 m move but is not trusted.** It puts the known-bad
  0.250 m beyond the reference, exactly the declared move. Its jackknife std, 0.053 m
  and 0.051 m, is just above the 0.05 m gate, so z is not judged.
* **Depth still reads the reference about 0.1 m short** (−0.096 m). The joint stereo
  fit (cross-check; it judges nothing) disagrees.
  * It reads the reference's z at +0.029 ± 0.036 m and the known-bad's at
    +0.276 ± 0.035 m, both trusted.
  * It fits a disparity scale of −2.6%; depth alone has no scale term.
  * Its yaw of the reference, −2.1 ± 0.4 deg, disagrees with the edges' +0.15 deg. Its
    x and y are untrusted.
* **The rectification is not the source of the scale.** A synthetic check projects
  points at 2, 5 and 10 m through the rectified pair. Disparity equals fx·B/Z to 1e-4 px,
  and both maps land on the fisheye projections to 1e-4 px.

## Result: `exp01` (construction site, ground level)

**`pass`, on rotation and depth.** All four axes are judged on both candidates. The
reference passes on every one, and the known-bad fails on its two declared axes. The
run used 48 frame pairs from the start of `exp01` (git commit `3ba887d`), with a
vertical offset of +0.33 px removed. The gyro turned each scan by 0.14 deg (median; at
most 1.48 deg).

| candidate | roll (deg) | pitch (deg) | yaw (deg) | z (m) | verdict |
| --- | --- | --- | --- | --- | --- |
| dataset reference | −0.05 / 0.50 `pass` | −0.15 / 0.50 `pass` | +0.10 / 0.50 `pass` | +0.005 / 0.042 `pass` | `pass` |
| known-bad | −0.05 / 0.50 `pass` | +4.85 / 0.50 `fail` | +0.05 / 0.50 `pass` | +0.245 / 0.027 `fail` | `fail` |

* **Stereo depth is trusted on both candidates.**
  * Reference: jackknife std 0.014 m on 20,588 points; the control response is 1.03.
  * Known-bad: std 0.009 m; the control response is 0.99.
  * Depth puts the known-bad 0.240 m beyond the reference, against a declared move of
    0.25 m.
* **`exp21`'s depth offset does not repeat.** On `exp01` depth reads the reference at
  +0.005 m, against −0.096 m on `exp21`. The offset belongs to that recording, not to
  the rig or the estimator.
* **The joint stereo fit diverges here.** It trusts no axis on either candidate (its
  jackknife std exceeds its gates) and fits a disparity scale of −7.1% and −4.3%. As on
  KITTI, it judges nothing.
* **The vertical offset changes sign between recordings**: +0.33 px here, −0.83 px on
  `exp21`. The next section traces it to the cameras' relative pitch.

## The vertical offset: the cameras' relative pitch, different per recording

The decision removes one row offset per recording, measured from its rectified images.
`tools/hilti_vertical_offset.py` samples every frame's matched patches and separates
the explanations. It fits the offset as a constant plus terms in disparity, column and
row, on the patches whose texture constrains rows, with a jackknife over frames.

| | `exp21` (2022-04-21) | `exp01` (2022-03-03) |
| --- | --- | --- |
| patches (constraining rows) | 12,367 (5,117) | 7,606 (4,301) |
| constant | −0.809 ± 0.078 px | +0.378 ± 0.020 px |
| per px of disparity | −0.008 ± 0.009 px | +0.004 ± 0.002 px |
| per 100 columns | +0.000 ± 0.021 px | +0.027 ± 0.013 px |
| per 100 rows | −0.040 ± 0.047 px | −0.050 ± 0.043 px |
| first half, second half (frame medians) | −0.86, −0.84 px | +0.32, +0.36 px |
| correlation with the gyro's rate about the baseline | −0.17 (implied −0.16 ms) | +0.11 (implied +0.19 ms) |

* **A constant: a relative pitch.** Each recording's offset is almost all constant and
  stable over the recording. At fx = 351 px it is a relative pitch of −0.13 deg on
  `exp21` and +0.06 deg on `exp01` against the camchain. That is 0.19 deg between two
  recordings seven weeks apart, both against the same calibration file.
* **Not the scene.** A vertical baseline component would grow with disparity. It is at
  most 0.004 px per px here, so a near indoor scene and a far outdoor one would differ
  by a fraction of the 1.19 px between them.
* **Not timing.** A time offset between the cameras would make the offset follow the
  rig's angular rate. With rates up to 26 deg/s, the correlation is weak, and the
  implied time offsets (under 0.2 ms) disagree in sign between the two recordings.
* **What this does not tell.** Whether the rig moved between sessions or the camchain
  was already off for both cannot be told from two recordings. Measuring the offset per
  recording from its own images, as the decision does, is right either way. A stereo
  extrinsic that drifts by 0.1 to 0.2 deg between sessions is itself a finding: it is
  the kind of drift `calibrex check` exists to catch.

The `camera-camera` pair of `calibrex check` now measures this rotation on any stereo
bag. On these recordings it reads −0.12 deg about cam0's x on `exp21` and +0.05 deg on
`exp01`, and fails a camchain turned 1 deg on `exp01`; see
[the tutorial](../tutorials/calibrex_check.md#camera-camera-a-stereo-pairs-rotation).

## The other camera pairs: within and between recordings (2026-10-13)

With all five cameras, `calibrex check` reads the side and down pairs on both
recordings. Where an axis is judged on both, cam1–cam3's roll reads −0.10 deg on
`exp01` and +0.21 deg on `exp21`, about six times its jackknife std (see
[the tutorial](../tutorials/calibrex_check.md#camera-camera-a-stereo-pairs-rotation)).
Two explanations were open: the rig changed between the sessions, or the estimate is
biased at the edges of the fisheye images, where the side pairs overlap.

`tools/hilti_camera_pair_reproducibility.py` reads the frames where all five cameras
fired together, 1 s apart: 78 on `exp01` and 80 on `exp21`, the bags' first 79 and 81 s.
It estimates every pair on five subsets of them:

* `all` frames;
* the `first` and `second` half in time, which see different parts of the scene;
* the `even` and `odd` frames, which see the same parts.

Each estimate is the check's own, with the same options.

### Within a recording

For each pair of subsets, *z* is the difference over the two subsets' combined std.
Judged axes are those observable in both subsets.

| subsets | axes compared (judged in both) | RMS *z* (judged) | \|*z*\| > 2 (judged) | largest judged difference |
| --- | --- | --- | --- | --- |
| first vs second half | 39 (19) | 1.69 (1.49) | 6 (2) | 0.19 deg |
| even vs odd frames | 42 (25) | 0.46 (0.37) | 0 (0) | 0.08 deg |

* **The std measures the spread within the frames analysed.** Interleaved subsets see
  the same scene and agree to better than their std.
* **Other parts of the scene move the reading by more.** Halves in time differ by 1.5
  to 1.7 times their combined std (RMS). The largest difference on an axis judged in
  both halves is 0.19 deg (cam0–cam4 yaw, `exp21`). cam0–cam1's pitch on `exp21`
  differs by 0.05 deg, 4.8 times its std.
* **Consequence.** The 0.5 deg tolerance of the verdicts is not at risk. A comparison of
  two readings "within their std" is: a side pair's std understates how its reading
  moves with the scene by about 1.5 times.

### Between recordings

On all 78 and 80 frames, cam0–cam1 changes by (−0.177, −0.059, −0.013) deg between
`exp01` and `exp21`, with stds of at most 0.04 deg. Its halves agree within each
recording. This is the 1.19 px row offset of the previous section.

Suppose only cam1 turned relative to the other cameras:

* each cam1–camN pair changes by cam0–cam1's change, turned into cam1's axes and with
  its sign reversed;
* the cam0–camN pairs do not change.

| pair, axis (judged on both) | change, exp21 − exp01 | only cam1 turned | *z* | no change, *z* |
| --- | --- | --- | --- | --- |
| cam1–cam3 roll | +0.241 ± 0.053 deg | +0.177 deg | +1.2 | +4.5 |
| cam1–cam3 yaw | −0.046 ± 0.057 deg | +0.014 deg | −1.1 | −0.8 |
| cam0–cam4 roll | −0.130 ± 0.075 deg | 0 | −1.7 | −1.7 |

* **The model removes the one large difference.** Over all 18 axes, judged or not,
  χ²/n is 2.15 with the model and 2.78 without. Scaling by the 1.5 times above gives
  an expected χ²/n of about 2.2.
* **One exception: cam1–cam4's pitch.** It changes by −0.28 deg (*z* −4.5). It is
  unjudged on both recordings, and pitch is the axis the side pairs constrain least.
* **cam1, not cam0.** If cam0 had turned instead, the cam0–camN pairs would change and
  the cam1–camN pairs would not. cam1–cam3's roll, the largest judged change, rules
  that out.

**Conclusion.** cam1–cam3's six-sigma difference between the recordings is cam1 itself:
the same 0.18 deg turn about the baseline that cam0–cam1 measures to 0.005 deg. It is
not a bias at the image edges. The rest is the scene dependence measured within each
recording.

Reproduce (about 50 minutes per recording):

```bash
python tools/hilti_camera_pair_reproducibility.py exp21_head.bag \
    --camchain calib_3_cam0-1-camchain-imucam.yaml \
    --camchain calib_3_cam2-camchain-imucam.yaml \
    --camchain calib_3_cam3-camchain-imucam.yaml \
    --camchain calib_3_cam4-camchain-imucam.yaml \
    --output exp21_reproducibility.json
```

## Re-centring the rectification (2026-10-13)

`camera-camera` measured row offsets in a rectification built from the candidate,
searching ±4 rows and only positive disparities. Two things went wrong with a
candidate a degree off:

* **The turn left the search.** A 1 deg turn moves rows by about 6 px at 350 px.
* **Far points went negative.** A turn about the rectified y axis moves far points to
  negative disparity.

The test turned cam3's camchain 1 deg about its own y axis on `exp01`. Its three camera
pairs became `estimator_failed`, and the turn went unseen.

**The fix.** The estimator now measures coarsely first (±12 rows, disparities from
−30 px, 100 patches per pair). If the coarse reading exceeds 0.2 deg, it re-centres the
rectification on the estimate and measures as before, up to three times. The error
composes the two. With the test's 1 deg turn, cam0–cam3 and cam1–cam3 `fail` with pitch
+1.37 and +1.40 deg, and the check's attribution names cam3.

**Does the reading depend on where it starts?**
`tools/hilti_camera_pair_start_dependence.py` starts each side pair from the camchain
turned −0.5, 0 and +0.5 deg in pitch. It removes the start's own turn from the reading,
so a start-independent reading comes out the same each time. It does this with the
re-centring and with the single measurement used before. `exp01`, 48 frames where all
five cameras fired together:

| pair | start (pitch) | re-centred: pitch, residual | single: pitch, residual |
| --- | --- | --- | --- |
| cam0–cam3 | −0.5 deg | +0.38, 0.17 px | +0.38, 0.17 px |
| cam0–cam3 | 0 | +0.36, 0.14 px | +0.31, 0.66 px |
| cam0–cam3 | +0.5 deg | +0.37, 0.32 px | −0.30 (unjudged), 1.60 px |
| cam1–cam3 | −0.5 deg | +0.38, 0.18 px | +0.38, 0.18 px |
| cam1–cam3 | 0 | +0.39, 0.13 px | +0.25 (unjudged), 0.43 px |
| cam1–cam3 | +0.5 deg | +0.44, 0.11 px | −0.78 (unjudged), 0.97 px |
| cam0–cam4 | −0.5 deg | −0.37 (unjudged), 0.12 px | `inconclusive` |
| cam0–cam4 | 0 | −0.48 (unjudged), 0.13 px | −0.32 (unjudged), 0.29 px |
| cam0–cam4 | +0.5 deg | −0.43, 0.20 px | −0.43, 0.20 px |
| cam1–cam4 | −0.5 deg | −0.04, 0.20 px | +0.15, 0.40 px |
| cam1–cam4 | 0 | −0.02, 0.19 px | −0.02, 0.19 px |
| cam1–cam4 | +0.5 deg | +0.00, 0.28 px | −0.03, 0.23 px |

The pitch is in degrees against the camchain. Where the start lands within 0.2 deg of
the reading, the re-centring does nothing and the two methods agree.

* **The re-centred readings agree across the starts within 0.11 deg.** Their residuals
  are 0.1 to 0.3 px.
* **The single measurement does not.** When the reading is 0.4 to 0.9 deg from the
  start, the residual grows to 0.29 to 1.60 px. The reading is then attenuated (+0.31
  and +0.25 instead of +0.37), lands on the wrong side (−0.78), or collapses
  (`inconclusive`). The check's earlier readings were single measurements from the
  candidate, the 0 rows.
* **cam3's candidate pitch is about 0.38 deg off.** cam0–cam3 and cam1–cam3 agree, and
  cam0–cam3 reads +0.40 deg on `exp21`.
* **cam0–cam4 and cam1–cam4 disagree on pitch by about 0.45 deg.** Each is stable
  across starts. cam0 and cam1 agree to 0.05 deg, so one of the two pairs carries a
  bias of that size, specific to its view. The check's closure loops go through the
  IMU, whose camera-IMU pitch is unjudged, so they cannot see it.

## Earlier run: stamp pairing, no motion

The first run paired images and scans by their stamps and left the points where they
were measured, which is not what the check does. The in-view points were then measured
61 ms after the image (59 to 62 ms per frame); with the check's pairing the gap is 11 ms
(−14 to +12 ms). That run read the reference's z at −0.137 m against a 0.111 m tolerance
(`warn`), so the decision was `inconclusive`. It is reproduced with `--stamp-pairing
--no-motion`.

## A disparity scale in the depth estimate: tried and rejected

The joint fit's scale suggested adding a scale term to the depth estimate. A pilot fitted
both variants on the same residuals at the declared reference, without edge alignment,
with 50 frames per KITTI run and 47 frames of the earlier Hilti run.

| run | current fit: reference / +0.25 m reads | free scale: reference (std) | joint fit's scale removed: reference (std) |
| --- | --- | --- | --- |
| KITTI 0005, spread | +0.048 / +0.258 | +0.023 (0.047) | +0.058 (0.016) |
| KITTI 0005, first 50 | +0.026 / +0.268 | +0.072 (0.049) | +0.093 (0.009) |
| KITTI 0009, spread | +0.002 / +0.233 | −0.114 (0.112) | +0.020 (0.038) |
| KITTI 0009, first 50 | −0.025 / +0.113 | −0.186 (0.047) | +0.044 (0.028) |
| Hilti `exp21`, stamp pairing | −0.124 / +0.103 | +0.398 (0.380) | −0.016 (0.044) |

* **A free scale is not separable from depth.** On camera-facing surfaces it trades off
  against the offset. The std grows 3 to 10 times, the fitted scales disagree in sign
  between runs, and on Hilti the control response falls to −0.15.
* **The joint fit's scale is not the rig's.** On one KITTI rig and calibration it is
  −0.2% to −0.3% on spread frames but −1.3% to −1.4% on the first 50. Removing it
  brings Hilti's reference to −0.016 m, but moves KITTI 0005's first-50 reference to
  +0.093 m with a 0.009 m std, beyond its tolerance.
* **The estimator is unchanged.**

## What was decided after seeing results

* **The pair's vertical offset is removed.** In a first run the rectified rows of cam0
  and cam1 disagreed by 0.79 px. Stereo depth then read the reference with a 0.11 m
  jackknife std, more than its 0.05 m gate, so z went unjudged on both candidates. The
  offset is now measured from the rectified images themselves (the median row shift of
  matched patches, `estimate_vertical_offset`) and removed from cam1's map before
  matching. The measurement uses no LiDAR and no candidate.
* **The frames follow the check.** The tool's description already said it used the
  check's frames. The visible-time pairing and the gyro were added after the
  `inconclusive` run, when the tool was found not to do what the check does. This
  turned the decision from `inconclusive` to `pass`.

## Limits

* **Translation during the sweep is not compensated.** The bag has no velocity topic.
  With the check's pairing, the in-view points are within about 14 ms of the image.
* **Two recordings, 48 frames each,** from the start of each bag.
* **z is judged on `exp01` only.** On `exp21` the depth std is just above its gate, and
  the reference's depth reading there, about −0.1 m, is unexplained. The joint fit reads
  +0.03 m there, the scale that would reconcile the two does not carry over to KITTI,
  and `exp01` does not show the offset.
* **The reference is not ground truth.** Its readings are findings about the data and
  the estimators, not evidence that Hilti's calibration is wrong.

## Reproduce

```bash
python tools/hilti_stereo_falsification.py exp21_head.bag \
    --camchain calib_3_cam0-1-camchain-imucam.yaml \
    --lidar-calibration lidar_calibration.yaml --output hilti_exp21.json
```

The vertical offset's analysis (about 6 minutes per recording):

```bash
python tools/hilti_vertical_offset.py exp21_head.bag \
    --camchain calib_3_cam0-1-camchain-imucam.yaml \
    --lidar-calibration lidar_calibration.yaml --output exp21_vertical.json
```

`exp01_head.bag` is made the same way as `exp21_head.bag`
([camera-LiDAR check](camera_lidar_check.md)):

```bash
curl -L -r 0-6442450943 -o exp01_head.bag \
  https://huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2022/resolve/main/rosbags/exp01_construction_ground_level.bag
python tools/trim_partial_rosbag1.py exp01_head.bag
```

About 6 minutes on 8 cores. The output records the git commit, the bag's name and
size, the SHA-256 of both calibration files, the per-point time field and the gyro's
correction per candidate. `--stamp-pairing --no-motion` reproduces the earlier run.

The scale-term pilot (about 4 minutes per run):

```bash
python tools/stereo_depth_scale_pilot.py kitti 2011_09_26_drive_0005_sync spread \
    --fixed-scale -0.0019
python tools/stereo_depth_scale_pilot.py hilti exp21_head.bag \
    --camchain calib_3_cam0-1-camchain-imucam.yaml \
    --lidar-calibration lidar_calibration.yaml --stamp-pairing --no-motion \
    --fixed-scale -0.0267
```

The fixed scales are the joint fit's for the reference in each published run (KITTI
0005: −0.0019 spread, −0.0129 first 50; 0009: −0.0029, −0.0131) and in the earlier
Hilti run.
