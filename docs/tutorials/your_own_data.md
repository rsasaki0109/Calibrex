# Calibrating Your Own Sensors

This tutorial walks you through calibrating your own LiDAR pair or
LiDAR–camera combination using a rosbag — no ROS installation required.

---

## Prerequisites

```bash
pip install calibrex
calibrex doctor          # checks environment and optional dependencies
```

Save a schema-valid readiness report for CI or support tickets:

```bash
calibrex doctor --output readiness.yaml
calibrex validate readiness.yaml --kind environment-readiness
```

The artifact (`slac.environment_readiness/v0.1`) records Calibrex and Python
versions, optional dependency availability, and—when you pass a dataset
path—inspected streams plus `workflow_suggestions` that point at starter
templates under `examples/sensor_templates/`.

Inspect a rosbag before calibrating:

```bash
calibrex doctor /path/to/my_bag --output readiness.yaml
```

Text output lists each suggestion's `template_path` and a `next_command` you
can copy (for example `calibrex init --template velodyne_vlp16_pair_rosbag2 ...`).

---

## Scenario A — Two spinning LiDARs (rosbag2 / ROS 2)

This is the most common starting point for multi-LiDAR robots.

### 1. Record a bag

Move the robot (or rotate it) for 30–120 s in a scene with clear planar
surfaces (walls, floors, ceilings).  Both sensors must see overlapping
geometry from at least 6 distinct viewpoints.

```bash
# ROS 2 example
ros2 bag record /velodyne_left/velodyne_points /velodyne_right/velodyne_points \
    -o my_lidar_calib
```

Tip: record indoors — more planar surfaces per unit distance than outdoors.

### 2. Pick a template

```bash
calibrex init --template velodyne_vlp16_pair_rosbag2 --output my_calib/config.yaml
calibrex init --list-templates
```

Or copy manually from `examples/sensor_templates/`.

### 3. Edit the config (three fields)

Open `my_calib/config.yaml` and change:

```yaml
dataset:
  path: /path/to/my_lidar_calib          # ← your bag folder or .mcap path

sensors:
  lidar_left:
    topic: /velodyne_left/velodyne_points  # ← your actual topic
  lidar_right:
    topic: /velodyne_right/velodyne_points # ← your actual topic
```

Check your topics with:

```bash
ros2 bag info /path/to/my_lidar_calib
# or for rosbag1:
rosbag info recording.bag
```

### 4. Run calibration

```bash
calibrex calibrate my_calib/config.yaml
```

### 5. Read the result

```bash
cat outputs/velodyne_vlp16_pair_rosbag2/result.yaml
```

The result contains:

- `transforms` — calibrated extrinsics (`T_parent_child`, meters, quaternion xyzw)
- `metrics` — holdout scores and grades
- `assessment` — PASS / WARN / FAIL against the evidence contract

Commit the result as your baseline once you trust it. On later changes, gate
pull requests with [Calibration CI](calibration_ci.md):

```bash
calibrex ci outputs/my_calib/result.yaml \
  --baseline calibration/baseline.yaml \
  --output-dir outputs/calibration-ci \
  --enforce
```

Copy the PR workflow from [`examples/ci/`](../../examples/ci/) into
`.github/workflows/` for GitHub Actions.

Example excerpt:

```yaml
transforms:
  T_lidar_left_lidar_right:
    translation_m: [0.503, 0.012, -0.004]
    rotation_quat_xyzw: [0.0, 0.0, 0.001, 1.0]
    provenance:
      evidence_level: algorithmically_refined
```

A `grade: pass` in `metrics.lidar_rig_point_to_plane_rmse_m` means the
holdout error is within spec.  If you see `grade: warn` or `grade: fail`,
see [Troubleshooting](#troubleshooting) below.

---

## Scenario B — Two spinning LiDARs (rosbag1 / ROS 1)

Same as Scenario A, but use:

```bash
cp -r examples/sensor_templates/velodyne_vlp16_pair_rosbag1 my_calib
```

And set `dataset.type: rosbag1` and `dataset.path: /path/to/recording.bag`.

---

## Scenario C — Ouster OS1 pair (rosbag2)

Ouster publishes standard `sensor_msgs/msg/PointCloud2`.  Use:

```bash
cp -r examples/sensor_templates/ouster_os1_pair_rosbag2 my_calib
```

If your Ouster driver publishes per-point timestamps in a field called `t`,
uncomment `point_time_field: t` in both sensor blocks.  This enables
point-level time alignment across scans.

---

## Scenario D — Spinning LiDAR + Camera (checkerboard)

This workflow requires pre-extracted target poses (plane normal + camera
detection per frame).

### 1. Extract poses

Use a tool such as
[lidar_camera_calib](https://github.com/ankitdhall/lidar_camera_calib) or
[Kalibr](https://github.com/ethz-asl/kalibr) to produce a CSV with columns:

```
frame_id, target_normal_x, target_normal_y, target_normal_z,
target_offset_m, image_u_px, image_v_px
```

### 2. Copy and edit the template

```bash
cp -r examples/sensor_templates/spinning_lidar_camera_planar_board my_calib
```

Edit `my_calib/config.yaml`:

- `dataset.path` → directory containing `poses.csv`
- `sensors.camera0.intrinsics` → your camera's fx, fy, cx, cy, distortion
- `sensors.lidar0.model` → your lidar model
- `frames.lidar0.transform.estimate.initial.translation` → rough guess (metres)

### 3. Run and verify

```bash
calibrex calibrate my_calib/config.yaml
```

Check `metrics.planar_board_normal_rmse_deg` and
`metrics.point_plane_center_rmse_m` for pass/fail grades.

---

## Troubleshooting

### `grade: fail` on `lidar_rig_point_to_plane_rmse_m`

Likely cause | Fix
-------------|----
Too few planar surfaces | Record in a more structured environment (corridor, warehouse)
Insufficient overlap | Increase `max_source_messages` / `max_target_messages`
Initial translation very wrong | Update `frames.<sensor>.transform.estimate.initial`
`voxel_size_m` too large | Reduce from 0.5 to 0.3 for high-density sensors

### `solver_adapter_status: unavailable`

The solver reported that data could not be loaded.  Check:

1. `dataset.path` points to the right file or folder
2. `sensors.<name>.topic` matches a topic in the bag
   (`ros2 bag info` or `rosbag info`)
3. The bag contains `sensor_msgs/[msg/]PointCloud2` messages

### Very high translation error but low rotation error

Often means the initial translation guess (in `frames`) is off by more than
the prior sigma.  Increase `prior_sigma.translation_m` or tighten the guess.

---

## Understanding result grades

Grade | Meaning
------|--------
`pass` | Holdout metric within the spec threshold
`warn` | Holdout metric outside spec — result may be usable but should be verified
`fail` | Holdout metric far outside spec — result should not be trusted

---

## Next steps

- **Validate with empirical uncertainty:**
  ```bash
  calibrex camera-lidar empirical-uncertainty \
    correspondence.yaml problem.yaml --stability-only \
    --output uncertainty.yaml
  ```

- **Import results from external tools:**
  See `docs/tutorials/lidar_camera_adapter.md` for iKalibr / Kalibr integration.

- **Public dataset benchmarks:**
  See `examples/public_datasets/` and `docs/tutorials/public_datasets.md`.
