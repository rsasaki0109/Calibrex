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

With a live camera a window guides the turning (below) and the calibration starts as soon
as the turns are enough; `--no-window` records for `--seconds` without it.

It needs OpenCV (`pip install 'calibrex[opencv]'`; the live window needs `opencv-python`,
not the headless build). `--output` (default
`intrinsics_camchain.yaml`) and `--camera-name` (default `cam0`) name the camchain; `--json`
prints the result as JSON.

### Stereo

A stereo rig turned in place gives both cameras' intrinsics and their relative rotation.
Pass the right camera as a second topic, a second video, or a side-by-side stream:

```bash
calibrex intrinsics rotate --bag my_bag --topic /left/image_raw --right-topic /right/image_raw
calibrex intrinsics rotate --video left.mp4 --right-video right.mp4
calibrex intrinsics rotate --video stereo.mp4 --side-by-side --baseline-m 0.12
calibrex intrinsics rotate --camera 0 --side-by-side --seconds 20
```

The camchain gets `cam0` (left) and `cam1` (right). Both cameras turn together, so between
any two moments the right camera's rotation is the left camera's seen in the right
camera's axes; aligning the two rotation sequences (keyframes matched by time, left
orientations interpolated) gives the relative rotation. Bag topics are paired by their
header stamps, so the cameras must share a clock. Two separate live cameras cannot be
recorded in step here: record them to a bag or to videos.

The baseline is harder. Its **length** is not observable from rotation, so `T_cn_cnm1` is
written only with `--baseline-m`. Its **direction** comes from features matched between
simultaneous left and right images, and a rotation error tilts it by about the rotation
error over the parallax angle (in a narrow field a rotation about the camera's y axis and a
translation along x look alike). It is reported only when the matches' median parallax is
at least 2 px and ten times the rotation's error in pixels; otherwise the command says so
and leaves `T_cn_cnm1` out. On a simulated 10 cm rig turned in a room 16 m across, the
intrinsics came within 0.4 % (focal) and 4 px (principal point), the relative rotation
within 0.7 deg (0.2 deg in a room 40 m across), and the parallax (under 1 px) was too small
for a direction. With the true rotation and objects 2-3.5 m away the direction came within
3 deg, but a rotation 0.5 deg off is already too coarse for it. Use a checkerboard when the
full stereo extrinsics matter.

Turning a rig about its middle also offsets each lens by half the baseline from the rotation
centre, which lengthens the focal lengths by about 0.8 (b/2)/D: face a distant scene.

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

## The guide window

`--camera` shows the live image with what to do next, and `--video --guide` shows the same
over a recording (`--no-window --record annotated.avi` writes it to a video instead):

- **Point the camera at a static scene with more texture**: fewer than 60 features are
  tracked.
- **Slower**: the features moved more than 6 % of the image width in one frame, or half
  of them were lost.
- **Pan**, then **Tilt**, then **Roll**, with how far the camera has turned about that axis
  and the 25 deg it should reach. A bar per axis fills as you turn and turns green.
- **Done**: all three axes turned 25 deg over at least 20 keyframes. The live camera then
  stops and calibrates; `s` calibrates earlier, `q` quits.

The guide turns the image motion between keyframes into a rotation with a guessed focal
length (0.8 times the image width, about 64 deg across), so its angles are approximate: a
guess 20 % off reads pan and tilt 20 % off, while roll is exact. The 25 deg target leaves
room above the 20 deg the calibration needs. On a simulated camera (280 px focal length,
k1 = -0.12) a person sweeping each axis as told reached done in about 80 frames (8 s at 10 fps),
pan and tilt read within 10 % of the truth, roll within 1.5 deg, and the calibration of
those frames gave the focal length within 0.3 %.

The guide cannot tell whether you turn about the lens: the lens distortion bends the image
motion as much as a few centimetres of translation, so it only reminds you.

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
