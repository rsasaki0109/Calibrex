# From camera intrinsics to the robot's extrinsics

The camera pairs of `calibrex check` and `calibrex estimate` (camera-imu, camera-focal,
camera-lidar and camera-camera) need each camera's intrinsics. A bag often has none: no
`CameraInfo` topic, or one with an uncalibrated (all-zero) matrix. Then calibrate them
first and hand the camchain over with `--intrinsics`.

## 1. Calibrate the intrinsics

With a checkerboard ([guide](intrinsics_guide.md)) or without one ([rotate](intrinsics_rotate.md)),
from a live camera, a video or the robot's own bag:

```bash
calibrex intrinsics guide --board 9x6x0.025 --camera 0 --output front_camchain.yaml
calibrex intrinsics rotate --bag turning_bag --topic /front/image_raw --output front_camchain.yaml
```

Add `--distortion equidistant` for a fisheye lens. From a bag, each camera's entry also
records its topic (`rostopic`), and the command prints the next step.

## 2. Check or estimate the extrinsics with them

```bash
calibrex check robot_bag --intrinsics front_camchain.yaml
calibrex estimate robot_bag --intrinsics front_camchain.yaml --output estimate/
```

`--intrinsics [TOPIC=]CAMCHAIN[:camN]` is repeatable, and is matched to the bag's image
topics in this order:

1. `TOPIC=` names the topic: `--intrinsics /front/image_raw=front_camchain.yaml`. The
   file's first camera is used, or `:camN`, for example the right camera of a stereo
   camchain: `--intrinsics /right/image_raw=stereo_camchain.yaml:cam1`.
2. An entry whose `rostopic` is the topic (what `intrinsics guide` and `rotate` write from
   a bag), so one stereo camchain serves both cameras.
3. With one camera in the bag, the only `--intrinsics` file applies to it (if it holds
   one camera, or `:camN` picks one).

The intrinsics come before every other source (a `--tf` camchain, a FAST-LIVO camera file,
`CameraInfo`); the progress line of each pair names the one used, for example
`intrinsics: front_camchain.yaml cam0 (--intrinsics)`. Only the intrinsics are taken: the
extrinsics still come from the bag's `/tf_static` or `--tf`. A `--tf` camchain, by
contrast, is matched by frame name, which on most bags is not the camera's header frame.

`calibrex estimate` then writes `camchain-imucam.yaml` next to the frames YAML: the
intrinsics from the file, `T_cam_imu` from the bag, the format Kalibr and VIO systems read.

## 3. Or run one camera pair directly

`calibrex camera-imu rotation` and `calibrex camera-imu focal` read the camchain with
`--camchain`; a camchain from a bag already names its image topic:

```bash
calibrex camera-imu rotation robot_bag --camchain front_camchain.yaml --imu-topic /imu ...
```
