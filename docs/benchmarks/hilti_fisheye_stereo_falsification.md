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
  `exp21`. It is measured from each recording's own images and removed before matching.
  Both recordings use the same calibration file, and why the offset differs is not
  known.

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
