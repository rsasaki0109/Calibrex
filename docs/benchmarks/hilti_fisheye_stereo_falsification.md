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

RESULT_PLACEHOLDER

## What was decided after seeing results

* **The pair's vertical offset is removed.** In a first run the rectified rows of cam0
  and cam1 disagreed by 0.79 px. Stereo depth then read the reference with a 0.11 m
  jackknife std, more than its 0.05 m gate, so z went unjudged on both candidates. The
  offset is now measured from the rectified images themselves (the median row shift of
  matched patches, `estimate_vertical_offset`) and removed from cam1's map before
  matching. With it, the std fell to about 0.045 m and the depth reading is trusted.
  The measurement uses no LiDAR and no candidate.

## Limits

* **No motion compensation.** The recording is handheld; each scan is used as it was
  swept, not moved to the image time.
* **One recording, 47 frames,** the first two minutes of `exp21`.
* **The reference is not ground truth.** Its depth reading is a finding about the data
  and the estimator, not evidence that Hilti's calibration is wrong.

## Reproduce

```bash
python tools/hilti_stereo_falsification.py exp21_head.bag \
    --camchain calib_3_cam0-1-camchain-imucam.yaml \
    --lidar-calibration lidar_calibration.yaml --output hilti_exp21.json
```

About 12 minutes on 8 cores. The output records the git commit, the bag's name and
size, and the SHA-256 of both calibration files.
