# AIST LiDAR-camera calibration dataset (Livox Avia)

Public dataset: [Zenodo 7779880](https://zenodo.org/records/7779880), CC BY 4.0. The
`livox.tar.gz` archive holds five static rosbag2 recordings of a Livox Avia
(`/livox/points`, `sensor_msgs/msg/PointCloud2`) and an STC-MCS500POE camera
(`/image`, `/camera_info`). The data stay outside this repository.

`koide_reference_frames.yaml` is a `slac.check_frames` candidate: the
`T_lidar_camera` that direct_visual_lidar_calibration estimated on this dataset
(its documented example `calib.json`). It is a reference from another tool, not
ground truth.

The camera and the LiDAR stamp their headers on separate clocks, so rewrite the
stamps with the bag receive time first (valid only because the rig is static):

```bash
tar xzf livox.tar.gz
python tools/restamp_rosbag2_headers.py livox/rosbag2_2023_03_09-13_46_54 restamped_13_46_54 \
  --topic /image --topic /camera_info --topic /livox/points
calibrex check restamped_13_46_54 --pairs camera-lidar \
  --tf examples/public_datasets/aist_lidar_camera_livox/koide_reference_frames.yaml
```

Results: [camera-lidar check benchmark](../../../docs/benchmarks/camera_lidar_check.md).
