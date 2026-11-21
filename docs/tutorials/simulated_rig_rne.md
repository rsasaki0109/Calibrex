# Check a simulated rig (Robot Native Engine)

[Robot Native Engine](https://github.com/rsasaki0109/RobotNativeEngine) (RNE) is a
Rust robot simulator. Its
[example 149](https://github.com/rsasaki0109/RobotNativeEngine/tree/main/examples/149_calibrex_rig_bag)
moves a hand-held rig through a room and records what its sensors see as a ROS 2 bag:
two spinning LiDARs, an IMU and a camera, each mounted with a known rotation and lever
arm. The bag comes with the true mountings and with candidates in which one sensor is
turned 3°, so `calibrex check` can be run on a recording whose truth is known.

It is simulated data. A pass shows that the estimators and the verdict work end to end
on a recording that looks like a real one (motion-distorted scans with per-point time,
IMU noise and bias, a noisy camera); it is not evidence of accuracy on hardware.

## Run it

You need a Rust toolchain (RNE's `rust-toolchain.toml` selects it) and Calibrex
installed from this repository, with OpenCV for the camera pairs:

```bash
git clone https://github.com/rsasaki0109/RobotNativeEngine.git
cd RobotNativeEngine
cargo run --release -p calibrex_rig_bag --example 149_calibrex_rig_bag -- \
    --out /tmp/rne_rig --duration-s 40

python -m pip install "calibrex[opencv] @ git+https://github.com/rsasaki0109/Calibrex.git"
calibrex check /tmp/rne_rig/rig.mcap --tf /tmp/rne_rig/good.yaml
calibrex check /tmp/rne_rig/rig.mcap --tf /tmp/rne_rig/turned_lidar2.yaml
```

The bag is MCAP with CDR payloads, written by RNE without a ROS installation, and
Calibrex reads it with its own MCAP reader. It holds:

| Topic | Type | Rate |
|---|---|---|
| `/imu/data` | `sensor_msgs/Imu` | 200 Hz |
| `/lidar/points`, `/lidar2/points` | `sensor_msgs/PointCloud2` (`x y z intensity ring time`) | 10 Hz |
| `/camera/image_raw` | `sensor_msgs/Image` (`mono8`, 480 × 360) | 10 Hz |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | 10 Hz |
| `/tf_static` | `tf2_msgs/TFMessage` (the true mountings in `lidar`) | once |

`--sensors imu` records only the LiDAR and the IMU (a 20 s bag is 29 MB and checks in
about 30 s). Each `turned_*.yaml` turns one sensor 3°: `turned_imu.yaml` (also
`turned.yaml`), `turned_lidar2.yaml` and `turned_camera.yaml`.

## What to expect

On a 40 s LiDAR + IMU bag, `good.yaml` passes imu-lidar (roll 0.002°, pitch 0.006°,
yaw 0.009° against 0.5°) and `turned.yaml` fails on yaw (2.99°).

On a 90 s bag of all four sensors (411 MB, 7 min 41 s to check on four cores):

| Pair | `good.yaml` | Largest judged error |
|---|---|---|
| imu-lidar (`lidar`) | pass (roll, pitch, yaw, y) | 0.008°, 0.2 mm |
| imu-lidar (`lidar2`) | pass (roll, pitch, yaw) | 0.024° |
| lidar-lidar | pass (all six axes) | 0.019°, 1.0 mm |
| camera-imu | warn (pitch, yaw) | yaw 1.12° against 0.84° |
| camera-focal | inconclusive | — |
| camera-lidar (`lidar`) | fail (roll, pitch) | roll 2.65° |
| camera-lidar (`lidar2`) | inconclusive | — |

`turned_imu.yaml` fails both imu-lidar pairs and passes lidar-lidar;
`turned_lidar2.yaml` fails imu-lidar (`lidar2`) and lidar-lidar and passes imu-lidar
(`lidar`). At 40 s camera-imu is inconclusive; its next steps ask for about 60 s.

## What the simulated rig shows about Calibrex

Two results are on Calibrex's side, not the simulator's:

- **The imu-lidar time offset is biased by about −1.5 ms.** Both sensors are stamped
  from one simulation clock, yet the check reports −1.87 ms. Bags with an ideal IMU, an
  exact accelerometer, a noiseless LiDAR or a 1–2 kHz IMU report −1.7 to −1.8 ms.
  Calibrex's own `calibrex.data.synthetic_lidar_imu_bag` (no RNE code) reports a true
  offset of 0 as −1.46 ms and +7 ms as +5.52 ms; at 5, 10 and 20 Hz LiDAR the
  zero-offset bag reports −1.13, −1.46 and −1.73 ms. The offset is reported, not judged;
  read it as accurate to about 2 ms.
- **camera-lidar misjudges roll under fast hand-held motion.** With the rig still, the
  bag's LiDAR points projected through the published `T_lidar_camera` and `CameraInfo`
  land on the rendered depth (median difference under 5 mm), yet camera-lidar reports
  roll errors of 2.4–3.0° on 40–90 s bags, at full and at half speed. Its evidence
  shows `median_rings: 0`: the edge objective recovers LiDAR beams from point elevation
  after moving the points to the image time, which recovers no beams on this motion
  (about 1.9° per frame), and it falls back to unstructured edges. The bag's `ring`
  field gives the beams directly. Treat camera-lidar on fast hand-held recordings with
  care.

The [RNE simulated rig workflow](https://github.com/rsasaki0109/Calibrex/blob/main/.github/workflows/rne-simulated-rig.yml)
runs the LiDAR + IMU case on changes to the check and imu-lidar code, with
[`tools/check_rne_simulated_rig.py`](https://github.com/rsasaki0109/Calibrex/blob/main/tools/check_rne_simulated_rig.py):
the true mounting must pass with every rotation axis judged, and the turned one must
fail on yaw by 2–4°.
