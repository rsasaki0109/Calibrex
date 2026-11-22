# Camera intrinsics without a target

`calibrex intrinsics rotate` estimates a camera's intrinsics from a recording of the camera
turned in place: no checkerboard. When a camera only rotates, where a scene point appears
in one frame fixes where it appears in every other frame through the intrinsics and the
rotation alone, so panning, tilting and rolling the camera over a textured, static scene
determines the focal lengths, the principal point and the lens distortion. The result is a
Kalibr camchain entry (`pinhole` / `radtan`: `k1 k2 p1 p2`), like
[`calibrex intrinsics guide`](intrinsics_guide.md) writes.

## Run it

From a ROS 2 bag, a video, or a live camera (recorded for `--seconds`, default 20):

```bash
calibrex intrinsics rotate --bag my_bag --topic /camera/image_raw
calibrex intrinsics rotate --video turning.mp4
calibrex intrinsics rotate --camera 0 --resolution 1280x720 --seconds 20
```

It needs OpenCV (`pip install 'calibrex[opencv]'`). `--output` (default
`intrinsics_camchain.yaml`) and `--camera-name` (default `cam0`) name the camchain; `--json`
prints the result as JSON.

## How to record

- **Turn the camera about its lens.** This is what matters most. A camera on a tripod or a
  pan-tilt head is ideal; by hand, hold it close to the lens and rotate it, not your arm.
- **Face a static, textured scene, as far away as you can.** Outdoors, a street or a
  building façade tens of metres away; indoors, a cluttered room. Nothing should move.
- **Pan, tilt and roll**, each by 20 deg or more, for 10-20 s. Turning about one axis only
  leaves the focal length undetermined; the command warns when fewer than two axes turned
  20 deg.
- Turn smoothly: features are tracked frame to frame and dropped when tracking them back
  misses by more than 0.3 px.

## What it reports

The camchain, the intrinsics with their stds, the transfer RMS (how far a tracked feature
lands from where the fitted rotation and intrinsics put it, in pixels), the number of
keyframes, how far the camera turned about its x (tilt), y (pan) and z (roll) axes, and
warnings.

The stds come from the fit's Jacobian, scaled for each tracked point entering several
keyframe pairs. They do not include the bias below.

## The limit: translation

A hand-held camera also translates as it turns, and the intrinsics take up the parallax of
that translation **without raising the residual**, so the data alone cannot show it. On a
simulated camera (480 x 360, barrel distortion k1 = -0.12, a textured room 8 m across):

| Lens offset from the rotation centre | Scene distance | Focal-length error |
| --- | --- | --- |
| 0 (turned about the lens) | 4 m | -0.08 % (principal point within 0.2 px, k1 -0.1202) |
| 5 cm | 4 m | +0.9 % |
| 5 cm | 2 m | +1.9 % |
| 15 cm | 4 m | +2.8 % |
| 15 cm | 10 m | +1.1 % |

The focal length comes out too long by about 0.8 times the offset over the distance. For
0.5 % (enough to start a camera-LiDAR or camera-IMU calibration), turn the camera about its
lens, or keep the offset under about 1/150 of the scene distance. When the focal length
must be known to 0.1 %, use a checkerboard ([`calibrex intrinsics guide`](intrinsics_guide.md)).

## How it works

1. Features are tracked with pyramidal Lucas-Kanade, forward-backward checked; a keyframe
   is taken whenever they moved 15 px (median).
2. The focal length and k1 are found together on a grid: with them, the undistorted
   correspondences between keyframes four apart must be related by a rotation. Searching
   the focal length alone is biased by tens of percent, because barrel distortion bends the
   image motion the way a shorter lens does (on the simulated camera, 490 px instead of
   420).
3. A least-squares fit refines the eight intrinsics and every keyframe's rotation from the
   transfer error: a feature seen in one keyframe is undistorted, rotated into another up
   to four keyframes later and projected with the distortion. The scene points are not
   unknowns, which keeps the fit small (eight intrinsics and three angles per keyframe)
   and quick to converge.

A model of the translation (a fixed lens offset from the rotation centre plus each point's
depth) was tried on the simulated camera and did not converge to the true offset, so the
translation remains a documented limit rather than an estimated quantity.
