# Public Datasets

Calibrex examples should prefer public datasets over synthetic-only workflows.
Large datasets are not committed to the repository. Instead, Calibrex keeps
small manifests, configs, and source metadata.

List known public datasets:

```bash
calibrex public-datasets list
calibrex public-datasets show tum_rgbd_freiburg1_xyz
```

RGB-D / Open3D SLAC:

```bash
python tools/download_public_dataset.py tum_rgbd_freiburg1_xyz --output-dir data/public
calibrex calibrate examples/public_datasets/tum_rgbd_freiburg1_xyz/config.yaml
```

Solid-state LiDAR-to-LiDAR evidence demo:

```bash
python tools/download_public_dataset.py livox_horizon_horizon_pcd_sample --output-dir data/public
python tools/generate_calibration_evidence_gif.py
calibrex public-datasets show livox_horizon_horizon_pcd_sample --json
calibrex inspect data/public/livox_horizon_horizon_pair --type livox-pcd --json
calibrex calibrate examples/public_datasets/livox_horizon_horizon_pcd_sample/config.yaml
calibrex evaluate examples/public_datasets/livox_horizon_horizon_pcd_sample/precomputed_result.yaml --export-html
```

Autonomous driving:

```bash
calibrex inspect examples/public_datasets/kitti_raw_2011_09_26_drive_0005 --type kitti-raw
calibrex compile examples/public_datasets/kitti_raw_2011_09_26_drive_0005/config.yaml
calibrex inspect data/public/nuscenes --type nuscenes
```

KITTI raw data and nuScenes require their official download flows and terms.
After downloading, point the dataset manifest paths at the local extracted
dataset.

nuScenes inspection reads the JSON metadata tables directly, without requiring
the nuScenes SDK. It discovers sensor streams from `sensor.json`,
`calibrated_sensor.json`, and `sample_data.json`, then reports channel counts,
modality counts, keyframe counts, missing local sample files, ego pose table
coverage, and calibrated sensor transforms normalized to Calibrex `xyzw`
quaternions. This is the first nuScenes step for validating that Camera, LiDAR,
Radar, and ego-pose metadata fit the same Calibrex dataset inspection model
used by KITTI.
When `calibrex calibrate` runs on `dataset.type: nuscenes`, Calibrex imports
those `calibrated_sensor` entries into result `reference_extrinsics` with
`T_parent_child` convention, `ego` as parent, and channel-derived child frame
names such as `lidar_top`, `cam_front`, and `radar_front`. These are dataset
reference values, not optimized Calibrex estimates, so they stay separate from
`transforms` and future `optimized_extrinsics` outputs.
Calibrex also stores config-derived initial transforms in `candidate_extrinsics`.
When a candidate and reference share the same parent/child edge, the report
adds `extrinsic_reference_*` metrics for pair count, maximum translation delta,
and maximum rotation delta. This is the first comparison path for evaluating
dataset calibration, manual candidate files, and future external baseline
outputs without mixing their meanings in one field.
External candidates can be supplied without editing the dataset config:

```bash
calibrex calibrate config.yaml --candidate-extrinsics candidates/manual.yaml
```

The candidate file may contain a top-level `candidate_extrinsics` mapping, a
`transforms` mapping from an existing Calibrex-style result, or a direct mapping
from transform name to `T_parent_child` transform fields. Imported candidates
override config-derived candidates with the same transform name and are recorded
under `run.provenance.external_candidate_extrinsics`.

For downloaded KITTI sequences, `calibrex inspect --type kitti-raw --json`
includes sampled Velodyne diagnostics: frame count, sampled point count, XYZ
bounds, intensity range, local planarity, roughness, map sharpness proxy, and
malformed file warnings.
Without `--json`, the same command prints a human-readable summary with streams,
warnings, LiDAR quality, point-to-plane metrics, and collection recommendations.
When KITTI OXTS files are present, `inspect` also reports motion excitation
diagnostics such as duration, mean speed, speed range, and yaw/pitch/roll
excitation. KITTI timestamp files are also compared to report Camera-LiDAR and
LiDAR-OXTS nearest timestamp deltas.
When image and Velodyne files are present, `inspect` also records concrete
camera-LiDAR frame pairs selected by nearest timestamp. These pairs include the
image path, `.bin` path, signed `lidar_time - camera_time` delta, and LiDAR point
count so overlay artifacts can reference real public-dataset frames.
When KITTI `calib_velo_to_cam.txt` and `calib_cam_to_cam.txt` are available,
`calibrex calibrate` and `calibrex visualize --export-html` project a sampled
Velodyne frame into the selected camera image inside
`artifacts/camera_lidar_overlay.html`.
The same projection feeds report metrics for projected point count, projection
ratio, median projected depth, depth span, and normalized horizontal/vertical
image coverage. A lightweight edge-alignment proxy also checks how many
projected LiDAR points land near image intensity edges.
For depth-aware checking, Calibrex also marks projected LiDAR points with
nearby depth jumps and scores whether those depth discontinuities land near
image edges. When more than one inspected camera-LiDAR frame pair is available,
these projection metrics are reported with train/holdout values instead of
judging the fixed-LiDAR calibration from a single frame.
Use `evaluation.kitti.max_projection_pairs` to increase the number of public
dataset frames used for those metrics. For quick checks, 3 frames is enough; for
fixed vehicle LiDAR validation, use 10 to 50 frames when the sequence is
available locally.
KITTI evaluation can also run known perturbation sweeps around the dataset
reference calibration:

```yaml
evaluation:
  kitti:
    perturbation_rotation_deg: [0.5, 1.0]
    perturbation_translation_m: [0.05, 0.10]
```

The resulting `lidar_camera_perturbation_*` metrics show whether roll, pitch,
yaw, x, y, and z offsets make projection and edge diagnostics worse. Treat this
as sensitivity and ranking evidence, not as absolute ground truth.

Those diagnostics feed the first LiDAR quality metrics in result files:
`lidar_frame_coverage`, `lidar_point_coverage`, and
`lidar_spatial_coverage_m`, plus geometry proxies such as
`lidar_local_planarity`, `lidar_map_roughness_m`, and `lidar_map_sharpness`.
The same local voxel planes also provide the alpha
`lidar_point_to_plane_rmse_m` metric with train/holdout values.
When KITTI OXTS packets are available, Calibrex additionally builds an
OXTS-projected LiDAR train map and scores holdout frames against that map. The
resulting `lidar_world_map_point_to_plane_rmse_m`,
`lidar_world_map_point_to_plane_median_holdout_m`, and
`lidar_world_map_point_to_plane_p95_holdout_m` metrics are the first
fixed-rig LiDAR map-consistency signals for vehicle extrinsic validation.
The matching `lidar_world_map_perturbation_*` metrics rerun that OXTS-projected
map consistency check after known extrinsic perturbations, so the report can
say whether a sequence has enough signal to rank bad roll, pitch, yaw, x, y, or
z candidates worse than the reference.
Calibrex also reports `lidar_world_map_sensitivity_*` and
`lidar_world_map_weak_dof_count`. A weak DoF warning means the evaluated public
dataset segment did not move the world-map holdout metric enough for that
direction, so the result should be treated as under-observed rather than
trusted.
The HTML report includes a dedicated LiDAR World-Map Diagnostics section with
the summary metrics and a DoF sensitivity table, so weak directions are visible
without digging through the full metric list.
For fixed vehicle LiDAR ranking, Calibrex also reports `lidar_perturbation_*`
metrics. These rerun the LiDAR voxel-plane proxy after known roll, pitch, yaw,
x, y, and z perturbations, then report train/holdout RMSE deltas. Positive
deltas mean the dataset reference calibration ranked better than the perturbed
candidate under the current LiDAR-only proxy.
Calibrex also turns these diagnostics into provisional degeneracy warnings, for
example when KITTI Velodyne samples contain too few planar neighborhoods, weak
vertical structure, or holdout point-to-plane degradation.
The report recommendations then say what to recollect, such as a longer
fixed-LiDAR sequence, more vertical structure, static planar surfaces, or
cleaner holdout passes with less dynamic traffic. Weak OXTS motion excitation
adds recommendations for longer driving logs, acceleration/braking, and turns.
Large timestamp deltas add a recommendation to verify synchronization or enable
time-offset estimation.

The KITTI raw example models the Velodyne LiDAR as a fixed-mounted sensor on
the vehicle and compiles a `fixed_lidar_mount_prior`
factor together with camera reprojection, LiDAR surfel, LiDAR-camera alignment,
and IMU preintegration descriptors.

KITTI is also the first public dataset target for SceneCalib/SST-Calib-inspired
development: targetless camera-LiDAR calibration, time-offset estimation, and
fixed-rig LiDAR quality gates should be exercised here before adding private or
synthetic-only workflows.

Import KITTI fixed-LiDAR calibration initial values:

```bash
calibrex kitti import-calib /path/to/2011_09_26 --output /tmp/kitti_transforms.yaml
```

This reads `calib_velo_to_cam.txt` and exports Calibrex `T_parent_child`
transforms such as `T_camera0_lidar0`.

Official sources:

- TUM RGB-D dataset: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- KITTI raw data: https://www.cvlibs.net/datasets/kitti/raw_data.php
- nuScenes: https://www.nuscenes.org/nuscenes

Research references:

- SceneCalib: https://arxiv.org/abs/2304.05530
- SST-Calib: https://arxiv.org/abs/2207.03704
- Decentralized multi-LiDAR SLAC: https://arxiv.org/abs/2007.01483
- M-LOAM: https://arxiv.org/abs/2010.14294
- Koide et al. LiDAR-camera calibration: https://arxiv.org/abs/2302.05094
