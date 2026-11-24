# LiDAR-LiDAR on a Unitree Go2

This page calibrates `T_reference_target` between an added LiDAR on a Go2 (a Livox MID360,
a Hesai XT16, ...) and the Go2's built-in Unitree L1 under the chin. The method is the
map registration of [NTU VIRAL LiDAR-LiDAR](../benchmarks/ntu_viral_lidar_lidar.md). On
top of it, each L1 point is placed along the reference trajectory at the time it was
captured.

!!! warning "No real-data result yet"
    This mode is tested only on a synthetic Go2 scene
    (`tests/unit/test_lidar_lidar_trajectory.py`). No Go2 recording has been scored, and
    lidar-lidar has no SOTA claim (the NTU VIRAL audit was refuted).
    On real data, the mode has run on a legged robot with an upside-down MID360
    ([GrandTour development](../benchmarks/grand_tour_lidar_lidar_dev.md)). The MID360 is
    ten times denser than an L1, so that run says nothing about accumulation.

## Why the L1 needs more than the default

| L1 property | Problem with one rigid scan per sample | What `--preset unitree-go2-l1` does |
| --- | --- | --- |
| About 15 Hz, roughly 1,000–3,000 points per message, non-repetitive pattern | One scan is too sparse to register reliably | Accumulates 10 consecutive scans per sample, with 5 cm voxels |
| On the head of a walking robot: turning, gait roll, pitch and bounce | One pose per scan smears the points | Places every point by the reference trajectory at its own `time` offset |
| Stamped by the robot's own clock (reported about 609 s off the PC) | Points are placed at the wrong poses | `--target-time-offset receive-time` gives a coarse offset; `--time-offset-search` refines it |
| Sees the robot's own legs and body | Self-points become false correspondences | Drops points closer than 0.5 m |

Use the denser added LiDAR as the **reference**: its odometry supplies the map and the
trajectory. The L1 is the **target**.

## Record

- Record both clouds raw, with per-point times: `/utlidar/cloud` (fields `x y z intensity
  ring time`) and the reference driver's topic (for example `/livox/lidar` as a
  `CustomMsg`).
- Do not use `/utlidar/cloud_deskewed`. It is in `odom` and has no per-point time.
- Do not use go2_ros2_sdk's WebRTC `point_cloud2`. It is a voxel map stamped with the
  host clock.
- Walk 60–120 s in a room or corridor with walls, floor, ceiling and some furniture. The
  scene must hold planes facing three directions.
- **Vary the turn rate.** Turn left and right, and stop and go. At a constant body twist
  (a steady circle, or a straight line at constant speed), a clock offset cannot be told
  apart from an extrinsic rotation and translation. The synthetic test shows the
  estimate drifting under such motion.

## Initial value

The L1's design mount in Unitree's `go2_description.urdf` is joint `radar_joint`, parent
`base`, `xyz="0.28945 0 -0.046825" rpy="0 2.8782 0"`: about 165 deg of pitch, mounted
inverted. As `T_base_l1`:

```yaml
T: [[-0.965512, 0.0, 0.260358, 0.28945],
    [0.0, 1.0, 0.0, 0.0],
    [-0.260358, 0.0, -0.965512, -0.046825],
    [0.0, 0.0, 0.0, 1.0]]
```

Write `T_base_reference` for your added LiDAR from your mount's CAD. Pass both files with
`--body-transforms`. Their composition `inv(T_base_reference) T_base_l1` is then both the
initial value and the design reference the result is compared against. The design
reference is not ground truth.

## Run

```bash
calibrex lidar-lidar ros2 go2_bag \
  --reference-topic /livox/lidar --point-time-field offset_time \
  --target-topic /utlidar/cloud --preset unitree-go2-l1 \
  --body-transforms T_base_mid360.yaml T_base_l1.yaml \
  --target-time-offset receive-time --time-offset-search 0.1 0.005 \
  --dataset-family go2 --dataset-license private --output go2_l1.yaml
```

- `--target-time-offset` defines `t_reference = t_target + offset`. Give it in seconds, or
  `receive-time` for the median header-minus-receive difference between the two topics.
- `--time-offset-search HALF STEP` refits the extrinsic for every offset on that grid,
  around the given offset. It scores only the **train** blocks, so held-out blocks never
  choose the offset. The artifact's `time_offset` field records:
    - the value;
    - a block-jackknife std;
    - the profile;
    - whether the minimum sat on the search edge (if it did, the verdict is at best `warn`).
- Explicit flags override the preset: `--target-accumulate-scans`, `--target-voxel-size`,
  `--target-min-range`, `--target-max-range`, `--target-point-time-field`.
- The per-point trajectory mode runs on the CPU registration backend only.
- The preset also sets `--window-sensitivity 0`. The extrinsic is refit with each L1
  sample mapped onto only the reference scans it spans, and each DoF's shift is added to
  its reported std (`std_systematic`). On GrandTour, a ±5-scan map biased the translation
  by up to 2 cm through odometry drift
  ([post hoc analysis](../benchmarks/grand_tour_lidar_lidar.md#post-hoc-where-the-cross-mission-scatter-comes-from)).

## Reading the result

The verdict and evidence are the same as for every lidar-lidar run:

- held-out blocks;
- the block jackknife;
- 1 deg and 5 cm known-bad controls.

On the synthetic Go2 scene, an inverted L1 at 300 points per scan came back within 0.1 deg
and 1 cm. Each comparison below was made on that scene:

| Variant | Result |
| --- | --- |
| Ignore the 37 ms clock lag | The extrinsic is off by more than 0.5 deg |
| Single rigid L1 scans instead of accumulated, per-point-timed ones | At least twice the rotation error |
| Clock offset profile | Lag recovered within 6 ms |

## Limitations

- The reference trajectory is interpolated between reference scans. Gait vibration
  faster than half the reference scan rate is not resolved.
- The reference odometry's drift over a sample's map window is noise for every sample.
