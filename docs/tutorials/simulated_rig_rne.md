# Check a simulated rig (Robot Native Engine)

[Robot Native Engine](https://github.com/rsasaki0109/RobotNativeEngine) (RNE) is a
Rust robot simulator. Its
[example 149](https://github.com/rsasaki0109/RobotNativeEngine/tree/main/examples/149_calibrex_rig_bag)
moves a hand-held rig through a room and records what its sensors see as a ROS 2 bag:
two spinning LiDARs, an IMU and a camera, each mounted with a known rotation and lever
arm. The bag comes with the true mountings and with candidates in which one sensor is
turned 3° or the IMU is moved 5 cm, so `calibrex check` can be run on a recording whose
truth is known.

It is simulated data. A pass shows that the estimators and the verdict work end to end
on a recording that looks like a real one (motion-distorted scans with per-point time,
IMU noise and bias, a noisy camera); it is not evidence of accuracy on hardware.

## Run it

You need a Rust toolchain (RNE's `rust-toolchain.toml` selects it) and Calibrex
installed from this repository, with OpenCV for camera-imu:

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

`--sensors imu` records only the LiDAR and the IMU. Each `turned_*.yaml` turns one
sensor 3°: `turned_imu.yaml` (also `turned.yaml`), `turned_lidar2.yaml` and
`turned_camera.yaml`; `shifted_imu.yaml` moves the IMU 5 cm along the LiDAR x axis.
`--rotation-speed 2` turns the rig twice as fast, which makes the IMU's lever arm
observable (below).

The room has a ring of thin floor-to-ceiling posts, two in three leaning 35°, so that
wherever the camera looks the LiDAR beams cross depth discontinuities in the image,
some of them slanted.

## What to expect

On a 90 s bag of all four sensors (393 MB, 7 min 7 s to check on four cores):

| Pair | `good.yaml` | Largest judged error |
|---|---|---|
| imu-lidar (`lidar`) | pass (roll, pitch, yaw) | 0.020° |
| imu-lidar (`lidar2`) | pass (roll, pitch, yaw) | 0.045° |
| lidar-lidar | pass (all six axes) | 0.037°, 1.0 mm |
| camera-lidar (`lidar`) | pass (roll) | 0.10° |
| camera-lidar (`lidar2`) | pass (roll, pitch, yaw) | 0.25° |
| camera-imu | inconclusive (no axis judged; see below) | — |
| camera-focal | inconclusive | — |

Each turned candidate fails exactly its sensor's pairs: `turned_imu.yaml` both imu-lidar
pairs (yaw 3.01° and 2.85°); `turned_lidar2.yaml` imu-lidar (`lidar2`), lidar-lidar and
camera-lidar (`lidar2`); `turned_camera.yaml` both camera-lidar pairs (roll 3.0° and
3.4°). Every other pair passes as with `good.yaml`.

On a 90 s `--sensors imu --rotation-speed 2` bag imu-lidar judges all six axes:
`good.yaml` passes (lever arm within 5 mm), `turned.yaml` fails on yaw (3.01°) and
`shifted_imu.yaml` fails on x (4.7 cm) while its rotation passes.

## What the simulated rig shows about Calibrex

- **camera-lidar needs depth discontinuities, slanted ones included.** It aligns LiDAR
  depth edges with image edges and drops frames with fewer than 150 edge points in the
  image. In the room without posts a 16-beam LiDAR gave 35–105 per frame
  (inconclusive). With upright posts only, pitch (a sideways image shift) was judged and
  roll and yaw were not: an upright edge does not move when the image shifts along it.
  Leaning posts make slanted edges, and all three rotation axes are judged.
  - Before rsasaki0109/Calibrex#394 camera-lidar recovered LiDAR beams from point
    elevation after moving the points to the image time; a fast sweep (1.9° of motion
    against a 2° beam spacing) left no beams (`median_rings: 0`), and the unstructured
    fallback reported a false 2.65° roll error. Beams, azimuth order and range jumps now
    come from the points as measured.
- **camera-imu's rotation is biased by the rig's translation.** Its yaw estimate is
  about 1° off (1.03–1.12° on several bags), and the check marks the pair inconclusive
  (a 1° roll control is not detected, pitch and yaw are not constrained to the bound).
  The camera is not the cause: the same check passes with errors under 0.02° when the
  rig only rotates, and neither an ideal IMU nor a camera at the rig's rotation centre
  changes the error. Comparing the tracked frame-to-frame rotations with the simulated
  ones shows a cross-coupling of about 0.9 % between axes, which turns into the
  extrinsic error.
  - Most frame pairs (243 of 300) are fitted as pure rotations: with about half a pixel
    of parallax, `recoverPose` puts few points in front of both cameras and the
    essential matrix is rejected. A pure rotation takes up the parallax of the rig's
    translation.
  - Keeping the essential-matrix decomposition nearest the pure rotation for every pair
    did not help (0.95° instead of 0.52° of coupling): the parallax is too small for it.
  - On a hand-held rig the translation follows the rotation (here their frequencies beat
    over about a minute), so the error does not average out over a recording.
- **The imu-lidar time offset is biased by a few milliseconds.** Both sensors are stamped
  from one simulation clock, yet the check reports −3.1 to −3.6 ms in the room with posts
  (−1.9 to −2.2 ms without them, −1.0 ms when the rig turns twice as fast).
  - The rotation solver is exact on true poses: given them and the gyro (with its bias),
    it returns 0 and +7 ms to the microsecond, at any IMU or scan rate.
  - Calibrex's own noise-free `calibrex.data.synthetic_lidar_imu_bag` reports a true 0 as
    −1.1 to −1.7 ms at 5–20 Hz LiDAR (−1.3 ms at 10 Hz, −1.1 ms with the translation
    switched off), and +7 ms as +5.5 ms.
  - The gyro-deskewed odometry poses are not shifted in time (+0.2 ms), and fitting
    increments over two scans instead of one gives the same offset, so it is not
    per-scan noise: the scan-to-scan odometry's rotation errors follow the motion and
    the scene. Fitted jointly with the rotation and the gyro bias, the weakly constrained
    offset takes them up the same way in every window (with the rotation and bias held at
    the truth the cost is lowest near +0.3 ms; without the robust reweighting the offset
    is −0.6 ms instead of −1.1 ms).
  - Neither the analytic nor the jackknife std (0.2–0.6 ms) sees an error shared by every
    window. `calibrex check` therefore prints the offset with its data-only std and a note
    that one recording's offset is accurate to a few ms; recordings of one rig and scene
    share the error, so `calibrex drift` comparisons are unchanged. The offset
    stays reported, not judged.
- **The imu-lidar lever arm needs brisk rotation.** At the default speed its data-only
  std is 1.1–1.8 cm against the 1 cm bound, and a 4 min recording does not shrink it;
  the segment-duration sensitivity (1.0–3.3 cm) is over the bound too. The lever arm
  reaches the accelerometer through the angular velocity and acceleration, and turning
  twice as fast makes all three axes observable in 90 s.

The [RNE simulated rig workflow](https://github.com/rsasaki0109/Calibrex/blob/main/.github/workflows/rne-simulated-rig.yml)
runs both cases on changes to the check, imu-lidar, odometry and camera-lidar code, with
[`tools/check_rne_simulated_rig.py`](https://github.com/rsasaki0109/Calibrex/blob/main/tools/check_rne_simulated_rig.py):

- a 20 s bag of all sensors but camera-imu's (`--sensors imu,lidar2,camera`): with the
  true mountings imu-lidar (both LiDARs), lidar-lidar and camera-lidar (both LiDARs) must
  pass, and each turned sensor must fail exactly its own pairs with a rotation error
  within 2–4° (1 min 51 s locally);
- a 90 s bag with `--sensors imu --rotation-speed 2` (`--profile lever-arm`): the true
  mounting must pass with x, y and z judged, the turned IMU must fail, and the shifted IMU
  must fail with a translation error within 3–7 cm while its rotation passes (3 min 53 s
  locally).
