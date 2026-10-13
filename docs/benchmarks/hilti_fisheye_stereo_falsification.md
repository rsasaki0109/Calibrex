# Hilti 2022 fisheye stereo falsification (development run)

The KITTI data-only decision
([KITTI Camera-LiDAR falsification](kitti_camera_lidar_falsification.md)) repeated on a
second dataset: a handheld Hilti 2022 recording whose camera pair is an equidistant
fisheye (alphasense cam0 and cam1, 0.108 m apart) and whose LiDAR is a PandarXT-32.
This is a development run, not a published artifact: the bag is not redistributed and
no CI test pins the numbers below.

## Setup

* **Candidates.** The reference `T_cam0_lidar` is composed from the dataset's Kalibr
  camchain (cam0 to IMU) and its `lidar_calibration.yaml` (IMU to LiDAR). The known-bad
  is declared before scoring, the same perturbation in camera terms as KITTI's: the
  reference turned 5 deg about the camera's y axis (pitch) and moved 0.25 m along its
  optical axis. Its declared axes are pitch and z.
* **Frames.** The bag-level check's own selection of cam0 images and scans (the short
  recording policy), each with the cam1 image of the same stamp.
* **Rotation** from edge alignment on the raw fisheye cam0 images (equidistant model).
* **Translation along the optical axis** from stereo depth on the rectified pair
  (`calibrex.evaluation.fisheye_stereo`): both cameras are re-projected onto a common
  pinhole whose x axis is the baseline. cam1 sits to the left of cam0, so the rectified
  frame is turned 180 deg about the optical axis; the transform to it is applied
  to every candidate the same way.
* **Rule and decision.** Unchanged from KITTI: `calibrex check`'s `judge_pair`, and the
  decision passes when the reference passes and the known-bad fails.

## Result

**`inconclusive`.** The known-bad fails on pitch, but the reference reads `warn` on
`z`. The run used 47 frame pairs from the first two minutes of `exp21`, with a vertical
offset of −0.83 px removed (git commit `5abcde3`).

| candidate | pitch (deg) | z (m) | verdict |
| --- | --- | --- | --- |
| dataset reference | +0.00 / 0.63 `pass` | −0.137 / 0.111 `warn` | `warn` |
| known-bad | +5.00 / 0.66 `fail` | +0.091 / 0.133 `pass` | `fail` |

Each cell gives the candidate's error, then its tolerance. Edge alignment judged only pitch
on these frames; roll and yaw went unjudged on both candidates.

* **Depth is trusted on both candidates.**
  * Reference: jackknife std 0.037 m; its 0.1 m control moved the estimate by 0.092 m.
  * Known-bad: std 0.044 m; its control moved the estimate by 0.076 m.
  * The edges ran a second time from each candidate with its depth reading removed, as
    on KITTI.
* **The depth reading separates the two, but is offset.** Depth puts the known-bad
  0.228 m beyond the reference (the declared move is 0.25 m), but it reads the
  reference itself at −0.137 m. The reference therefore exceeds its tolerance, and the
  known-bad lands inside its own.
* **The joint stereo fit (cross-check) explains the offset.** The fit includes the
  rig's scale and roll.
  * It reads the reference's z at −0.007 ± 0.049 m and the known-bad's at
    +0.246 ± 0.048 m. Both readings are trusted, with control responses 1.08.
  * It also fits a disparity scale of −2.7% (KITTI's published runs: −0.2% to −1.4%).
  * Depth alone has no scale term. At range Z, a scale error s looks like a depth
    error s·Z, so −2.7% reads as
    −0.137 m at 5.1 m. The run did not record the matched points' ranges, so this
    is a consistency check, not a measurement.
  * Its other readings of the reference: x +0.036 ± 0.045 m (control 0.87); y
    untrusted (std 0.086 m); yaw −1.28 ± 0.47 deg.
  * As on KITTI, the cross-check judges nothing.
* **The scale is not the rectification's.** A synthetic check projects points at 2, 5
  and 10 m through the rectified pair. Disparity equals fx·B/Z to 1e-4 px, and both
  maps land on the fisheye projections to 1e-4 px. The −2.7% comes from the data or
  their calibration: the camchain's focal lengths and baseline, the equidistant model
  far from the centre, or the LiDAR's range. This run cannot tell which.

## What was decided after seeing results

* **The pair's vertical offset is removed.** In a first run the rectified rows of cam0
  and cam1 disagreed by 0.79 px. Stereo depth then read the reference with a 0.11 m
  jackknife std, more than its 0.05 m gate, so z went unjudged on both candidates. The
  offset is now measured from the rectified images themselves (the median row shift of
  matched patches, `estimate_vertical_offset`) and removed from cam1's map before
  matching. With it, the std fell to about 0.04 m and the depth reading is trusted.
  The measurement uses no LiDAR and no candidate.

## Limits

* **No motion compensation.** The recording is handheld; each scan is used as it was
  swept, not moved to the image time.
* **One recording, 47 frames,** the first two minutes of `exp21`.
* **The rule was not changed for this dataset.** Adding a scale term to the depth
  estimate would remove the offset above. It would also change the published KITTI
  decision, so it has to be tried on KITTI first and published there before it is
  used here.
* **The reference is not ground truth.** Its depth reading is a finding about the data
  and the estimator, not evidence that Hilti's calibration is wrong.

## Reproduce

```bash
python tools/hilti_stereo_falsification.py exp21_head.bag \
    --camchain calib_3_cam0-1-camchain-imucam.yaml \
    --lidar-calibration lidar_calibration.yaml --output hilti_exp21.json
```

About 6 minutes on 8 cores. The output records the git commit, the bag's name and
size, and the SHA-256 of both calibration files.
