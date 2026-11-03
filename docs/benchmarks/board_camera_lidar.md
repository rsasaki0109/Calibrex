# Camera–LiDAR from a checkerboard (development)

`calibrex.evaluation.board_camera_lidar` measures `T_camera_lidar` from a checkerboard held
still at several poses in front of both sensors. It is the path for the camera-LiDAR
**translation**, which the targetless `camera-lidar` check does not judge (its edge objective
is flat within ±5 cm; see [camera-LiDAR check](camera_lidar_check.md)). This page records the
method and its first run on real data. It is development, not an audit: there is no
reference extrinsic for the data used here, and no claim is made.

## Method

Per board pose:

1. **Camera.** OpenCV finds the inner corners (`findChessboardCornersSB`, with
   `findChessboardCorners` and sub-pixel refinement as a fallback). The corners are
   undistorted with the camera model (`radtan`, `equidistant` or none) and the board pose
   comes from PnP (`IPPE`, refined by Levenberg-Marquardt). The board plane in the camera
   frame is its normal and offset.
2. **LiDAR.** The scan is mapped into the board frame the camera saw, through the current
   extrinsic. Points within a margin of the squares and of the board plane are candidates;
   RANSAC (2 cm) finds the plane, and least squares refines it to a fixed point, keeping
   the points within 3 robust sigmas of it (at least 2 cm): a Livox board's points spread
   over several centimetres at a distance. A plane turned more than 25 deg from the
   camera's board is rejected.

All poses together go to the Zhang–Pless plane solver of
`calibrex.solvers.planar_board_lidar_camera_solver` (rotation from the normals, then
translation from the offsets, with Huber reweighting). Each pose is weighted by
`(1 m / distance)^2`: a focal-length error scales the PnP distance, so the camera's
plane-offset error grows with the distance. The solve runs three times, the first
from a rough prior (only the axis permutation between the LiDAR and the optical frame is
needed) with a generous crop, then with the solution and a tighter crop.

**Observability.** A plane fixes the translation only along its normal. The information
`sum w n n^T` over the camera-frame normals has an eigen-direction with less than 1% of the
largest eigenvalue when the boards were only turned one way. The translation along such a
direction stays at the prior's value, so the next crop does not drift off the board. A
translation axis is reported as not observed when such a direction has 30% or more of it.
Every axis also gets a jackknife std: one pose left out at a time. An axis counts as
observed when its std is within 2 cm, or 0.3 deg for rotations, and its direction is
observed. Synthetic boards tilted both left/right and up/down recover all six axes to 1 cm
and 0.2 deg, and boards turned only left and right leave the vertical translation
unobserved (`tests/unit/test_board_camera_lidar.py`).

## First real data: the ACSC sample

The sample of ACSC (Cui et al., *ACSC: Automatic Calibration for Non-repetitive Scanning
Solid-State LiDAR and Camera Systems*; repository `HViktorTsoi/ACSC`, GPL-3.0; its data
are not redistributed here) holds 21 static board poses.

| Item | Value |
| --- | --- |
| LiDAR | Livox Horizon, one accumulated cloud per pose (1.2 M points) |
| Camera | 2048 x 1536, the sample's intrinsics and radial-tangential distortion (its 2 px skew dropped) |
| Board | 7 x 5 inner corners, 8 cm squares |
| Poses | 21, at 3.3–12.0 m, turned left and right only |
| Archive | `acsc_sample.zip`, sha256 `38459093f5cbac4b7acbd7331f42bf2aba76f2596af90ab1ae4f972104a58ca3` |
| Prior | the axis permutation only (identity rotation between the LiDAR and optical axes, zero translation) |

The camera found the board in all 21 images (reprojection 0.04–0.32 px), and the LiDAR plane
in all 21 (0.5–1.9 cm plane RMS).

Reproduce with the extracted sample (needs `opencv-python-headless`):

```bash
python tools/board_camera_lidar_acsc.py ACSC/sample --output acsc_board.json
```

| Axis (camera frame) | Value | Jackknife std | Observed |
| --- | --- | --- | --- |
| roll (about x) | 2.22 deg from the prior | 0.35 deg | no (std over 0.3 deg) |
| pitch (about y) | −0.25 deg from the prior | 0.299 deg | yes, just within the bound |
| yaw (about z) | −0.62 deg from the prior | 0.41 deg | no (std over 0.3 deg) |
| x | 13.3 cm | 1.6 cm | yes |
| y | −0.7 cm (the prior's 0, kept) | — | **no**: the board normals have almost no vertical component |
| z | 15.2 cm | 0.8 cm | yes |

What this shows:

- **Turned only left and right, the boards cannot fix the vertical translation.** The board
  normals' singular values are 4.20, 1.83 and 0.067; the weak direction is the camera's
  y axis. With the y translation free, it drifted by tens of centimetres between solves and
  pulled the next crop off the board; holding it at the prior and reporting it as not
  observed is what keeps the other axes stable.
- **The plane offsets disagree more for far boards.** Residuals are within 4 cm at
  3–4.4 m, up to 7 cm at 6–7 m and up to 9 cm at 9–12 m. A small focal-length or
  distortion error in the sample's intrinsics would do this; the distance weighting limits
  its effect, and boards within about 6 m are better.
- **Rotation is limited by the LiDAR plane normals.** A Livox plane fit on a 0.64 x 0.48 m
  board disagrees with the camera's by 0.3–3.1 deg per pose, so with 21 poses two rotation
  axes stay above the 0.3 deg bound and the third is just within it.
- **A fixed RANSAC band made the plane depend on the point order.** The board's Livox
  points spread over 4–6 cm at 6–7 m. With the first version's 2 cm band, a different
  slice was kept for each RANSAC sample: reordering a pose's points moved its plane by up
  to 3 deg and 10 cm. The fixed-point refinement removed that (the plane now moves by
  less than 0.001 deg and 0.1 mm under reordering), and the table is from it.

**For a capture:** tilt the board up and down as well as left and right, keep it within
about 6 m, hold it still, and use a dozen or more poses.

## The same poses through `calibrex estimate --board`

`calibrex.evaluation.board_camera_lidar_bag` finds the poses in a bag: images searched four
times a second, a pose wherever the board's centre stays within 1 cm and its normal within
0.5 deg for a second or more, and the LiDAR scans stamped inside that, less 0.2 s at either
end, accumulated. The 21 ACSC poses were packed into a rosbag2 for it: each pose as four
identical PNG images over 1.2 s (`CompressedImage`, with a `CameraInfo` holding the sample's
intrinsics) and its cloud split into ten `PointCloud2` scans inside the hold, then one second
without data. `calibrex estimate BAG --board 7x5x0.08 --pairs camera-lidar` (no `--tf`)
found 21 still poses and gave the result above to the last digit: the same axes observed,
x 13.27 cm and z 15.22 cm with the same stds. Without a prior that connects the two frames,
the pair is `partial` and no frame is exported; with a rough `--tf` prior, the axes the
boards do not observe are filled from it and marked NOT MEASURED, as for the other pairs.

## Next

A real capture tilted up and down as well, to observe the vertical translation and all
three rotations, and the same option in `calibrex check`, so that a deployed camera-LiDAR
translation can be checked and not only estimated.
