# Targetless LiDAR-Camera Adapter

Calibrex treats Koide-style targetless LiDAR-camera calibration as an external
baseline first. The core does not vendor external calibration code. It records
adapter readiness, input streams, optional command availability, loaded
transforms, and license boundary in `result.yaml`.

Enable the adapter from a pipeline factor:

```yaml
pipeline:
  type: multi_sensor_slac
  factors:
    koide_lidar_camera:
      enabled: true
      options:
        command: direct_visual_lidar_calibration
        result_path: outputs/external/koide_result.yaml
```

By default the command is not executed. Calibrex only records whether it is
available. To run a local external command, opt in explicitly:

```yaml
pipeline:
  factors:
    koide_lidar_camera:
      enabled: true
      options:
        command:
          - direct_visual_lidar_calibration
          - --dataset
          - "{dataset_path}"
          - --output
          - "{result_path}"
        execute: true
        timeout_sec: 600
        result_path: outputs/external/koide_result.yaml
```

Supported placeholders are `{dataset_path}`, `{result_path}`,
`{camera_streams}`, and `{lidar_streams}`. Commands are executed without a shell,
and stdout, stderr, timeout state, and return code are stored in provenance.

`result_path` may point to a Calibrex-compatible transform file:

```yaml
transforms:
  T_camera0_lidar0:
    convention: T_parent_child
    parent: camera0
    child: lidar0
    translation_m: [0.0, 0.0, 0.0]
    rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
```

When the transform is present, Calibrex converts it into the configured frame
graph, for example updating `T_base_link_lidar0` through `T_base_link_camera0 *
T_camera0_lidar0`.

The adapter reports these metrics:

- `koide_lidar_camera_adapter_available`
- `koide_lidar_camera_input_ready`
- `koide_lidar_camera_result_available`
- `koide_lidar_camera_execution_success`

Calibrex also reports cross-modal evaluation scaffolding in the same result:

- `lidar_camera_transform_pairs`
- `lidar_camera_overlay_readiness`
- `lidar_camera_overlay_score`
- `lidar_camera_projection_frame_count`
- `lidar_camera_projected_points`
- `lidar_camera_projection_ratio`
- `lidar_camera_projection_depth_median_m`
- `lidar_camera_projection_depth_span_m`
- `lidar_camera_projection_horizontal_coverage`
- `lidar_camera_projection_vertical_coverage`
- `lidar_camera_edge_alignment_score`
- `lidar_camera_edge_gradient_mean`
- `lidar_camera_depth_discontinuity_points`
- `lidar_camera_depth_edge_alignment_score`
- `lidar_camera_depth_edge_gradient_mean`
- `lidar_camera_mutual_information_score`

When these metrics exist, `calibrex calibrate` and
`calibrex visualize --export-html` write
`artifacts/camera_lidar_overlay.html`. In the alpha this is a diagnostic
scaffold with readiness, score, transform, selected KITTI camera-LiDAR frame
pairs, and recommendation context. For KITTI raw sequences with camera and
Velodyne calibration files, the artifact also projects sampled LiDAR points into
the selected camera image. Projection metrics are aggregated over the inspected
KITTI camera-LiDAR frame pairs, with train/holdout fields when at least two
pairs are available. The score is still alpha-level, but the visualization now
uses concrete public-dataset frames.

For KITTI raw, configure the number of frame pairs used by these metrics:

```yaml
evaluation:
  kitti:
    max_projection_pairs: 20
    projection_sample_points: 800
    perturbation_rotation_deg: [0.5, 1.0]
    perturbation_translation_m: [0.05, 0.10]
```

The perturbation settings run known roll/pitch/yaw/x/y/z offsets around the
dataset reference calibration and report whether projection, edge, and
depth-edge metrics worsen. This is a ranking and sensitivity diagnostic, not an
absolute ground-truth accuracy claim.

This is the first step toward comparing Koide-style single-shot targetless
calibration against native SLAC factors under the same evaluation report.

## Try it: KITTI raw camera-LiDAR evidence demo

`calibrex demo kitti-lidar-camera-evidence` runs this whole flow end to end on
a KITTI raw sequence:

```bash
calibrex demo kitti-lidar-camera-evidence --output-dir outputs/kitti_lidar_camera_evidence
```

By default it uses a small synthetic fixture bundled at
`examples/public_datasets/kitti_lidar_camera_evidence/` (KITTI raw data itself
requires the official login-gated download flow, so Calibrex cannot fetch it
automatically the way it does the Livox pair demo). Pass
`--dataset-path /path/to/2011_09_26/2011_09_26_drive_0005_sync` to point at a
real, locally downloaded KITTI raw sequence instead.

The demo:

- uses the dataset's `calib_velo_to_cam.txt` extrinsic as the reference/output
  `T_base_link_lidar0` transform (`run.provenance.dataset_initialization`
  records `source: kitti_raw`, `status: loaded`, and the applied transform
  name; the transform's own provenance records
  `producer: dataset_provider`, `evidence_level: dataset_provided`);
- enables the `koide_lidar_camera` adapter factor without `execute: true` or a
  `result_path`, so `koide_lidar_camera_*` metrics report readiness only —
  the external tool is not invoked and no transform is imported from it;
- reports the projection, edge-alignment, and `lidar_camera_perturbation_*`
  known-bad roll/pitch/yaw/x/y/z probes described above as diagnostic overlay
  evidence on the LiDAR candidate extrinsic, not a standalone camera
  calibration result;
- writes `result.yaml`, `evidence.json`, `assessment.json`, `protocol.json`,
  `policy.json`, `transforms.json`, and `bundle.json`, and verifies the
  bundle (KITTI raw does not yet populate per-frame raw-input-file hashes the
  way the Livox pair demo does, so verification does not gate on raw
  recomputation here).

A cached example result for report rendering without rerunning the pipeline is
available at
`examples/public_datasets/kitti_lidar_camera_evidence/cached_evidence_result.yaml`.
