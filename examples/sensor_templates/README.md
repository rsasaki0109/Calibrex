# Sensor Templates

Ready-to-edit configuration templates for common sensor calibration scenarios.
Each template is a working starting point — edit the fields marked `EDIT` and
run `calibrex calibrate <template>/config.yaml`.

## Available templates

| Template | Sensors | Data format | Backend |
|----------|---------|-------------|---------|
| `velodyne_vlp16_pair_rosbag2/` | Velodyne VLP-16 × 2 | ROS 2 bag (`.db3` / `.mcap`) | `native_lidar_point_to_plane` |
| `velodyne_vlp16_pair_rosbag1/` | Velodyne VLP-16 × 2 | ROS 1 bag (`.bag`) | `native_lidar_point_to_plane` |
| `ouster_os1_pair_rosbag2/` | Ouster OS1 × 2 | ROS 2 bag | `native_lidar_point_to_plane` |
| `spinning_lidar_camera_planar_board/` | Spinning LiDAR + camera | Filesystem (poses CSV) | `native_planar_board` |

## Quick decision guide

```
I have my own rosbag…
  ├── ROS 1 (.bag) ──→ velodyne_vlp16_pair_rosbag1  or  ouster_os1_pair_rosbag* (change dataset.type: rosbag1)
  └── ROS 2 (.db3 / .mcap) ──→ velodyne_vlp16_pair_rosbag2  or  ouster_os1_pair_rosbag2

I have a checkerboard and pre-extracted target poses (CSV)…
  └──→ spinning_lidar_camera_planar_board
```

## Minimum data requirements

| Scenario | Minimum recording | Notes |
|----------|------------------|-------|
| LiDAR–LiDAR | ~30 s driving / rotation | Needs overlapping planar surfaces from ≥6 viewpoints |
| LiDAR–Camera (board) | 20+ checkerboard poses | At least 5 poses per quadrant of the image |

## Adapting a template to other sensors

The `velodyne_vlp16_pair_rosbag2` template works for any spinning LiDAR that
publishes `sensor_msgs/msg/PointCloud2` — change `model:` to `spinning` and
update `topic:` to match your driver.

For Ouster with per-point timestamps, uncomment `point_time_field: t` in the
sensor block.

## Public dataset examples

See `examples/public_datasets/` for end-to-end examples with downloadable data:

```bash
python3 tools/download_public_dataset.py tiers_livox_lidars_cali --output-dir data/public
calibrex calibrate examples/public_datasets/tiers_livox_lidars_cali/config.yaml
```

For a full walkthrough, see `docs/tutorials/your_own_data.md`.
