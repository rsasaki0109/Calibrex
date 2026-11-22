# Calibrate camera intrinsics with live guidance

`calibrex intrinsics guide` calibrates a camera's intrinsics with a checkerboard and tells
you, frame by frame, how to move the board next. It keeps only the frames that add
something, re-estimates the intrinsics after each one, and says when the focal lengths and
the principal point are known well enough. The result is a Kalibr camchain entry, which
Calibrex's camera pairs (`camera-imu`, `camera-lidar`, `camera-focal`) read.

![Guidance at four stages: move to the highlighted cell, tilt, vary, done](../assets/intrinsics_guide.png)

## Run it

You need OpenCV *with* window support: the `opencv` extra installs the headless build, so
install `opencv-python` instead for the live window.

```bash
python -m pip install "calibrex @ git+https://github.com/rsasaki0109/Calibrex.git" opencv-python
calibrex intrinsics guide --board 9x6x0.025 --camera 0 --resolution 1280x720
```

`--board` gives the **inner** corners and the square size in metres (a board of 10 x 7
squares has 9 x 6 inner corners). Prefer a board with one even and one odd number of
squares: a board that looks the same turned half a turn is still usable for intrinsics,
but its corners may come in either order.

Keys: `s` saves the camchain (default `intrinsics_camchain.yaml`, entry `cam0`;
`--output`, `--camera-name`), `q` quits.

A recorded video works too, without a window:

```bash
calibrex intrinsics guide --board 9x6x0.025 --video board.mp4 --no-window \
    --record annotated.avi --output camchain.yaml
```

`--no-window` runs to the end of the video, or until the calibration is done, and saves
what it has. `--record` writes the annotated frames, which is also a way to see the
guidance on a machine whose OpenCV has no window support.

## What the screen shows

- **The advice**, at the top, in this order:
  1. *Show the whole board to the camera*: no board found; every inner corner must be
     in view.
  2. *Hold the board still*: a frame is kept only when the corners moved at most 1 px
     since the previous frame, so motion blur and rolling-shutter skew stay out.
  3. *Move the board to the highlighted area*: the image is cut into 5 x 4 cells
     (`--grid`); a yellow rectangle and an arrow point to the uncovered cell nearest to
     the board. Covered cells are tinted green. The corners and the image edges matter
     most for the distortion and the principal point.
  4. *Tilt ~30 deg: turn the board's left (right, top, bottom) edge away*: the board
     must be seen tilted all four ways (more than 25 deg). A board facing the camera
     leaves the focal length undetermined, so the first calibration waits for six views
     and two tilts.
  5. *Vary distance and tilt*: the cells and tilts are complete but the estimate is not
     yet precise enough; frames whose pose differs from every kept one are added.
  6. *Done*: at least 15 views, the focal lengths' std below 0.3 % of their value, the
     principal point's std below 2 px and a reprojection RMS below 1 px.
- **The board**: its corners in green when still and red when moving; a white outline
  flashes when the frame is kept.
- **The status**, at the bottom: views kept, cells covered, tilts seen (`left+` seen,
  `left-` missing), and the current estimate of `fx fy cx cy k1 k2` with the stds.

## The model and its limits

The model is OpenCV's pinhole with radial-tangential distortion `k1 k2 p1 p2` (`k3` fixed
at 0), which Kalibr calls `pinhole` / `radtan`. Fisheye lenses need another model and are
not supported here.

The stds are OpenCV's, from the reprojection residuals of the kept views. They assume
independent corner errors, so they understate the error when the corners carry an error
shared within a view (a board that is not flat, a print scaled differently from its
nominal square size, lens effects the model lacks). On a synthetic camera rendered with
~0.1 px of aliasing on the corners, following the advice gave the focal lengths within
0.5 % and the principal point within 1.5 px, a few stds from the truth. Check the square
size of the printed board with a ruler: it scales the poses, not the intrinsics, but a
board that is not flat biases them.
