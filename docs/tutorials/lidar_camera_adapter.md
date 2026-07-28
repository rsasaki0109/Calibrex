# External Calibration Adapters

Calibrex records imported and externally executed calibration outputs in a
shared `slac.external_calibration_run/v0.1` artifact. The contract captures
tool/version/commit/license identity, the process or import boundary, bounded
stdout/stderr digests, input/output digests, frame and time conventions,
training-data isolation, typed outputs, warnings, and provenance. External
fitness values are always marked non-comparable; Calibrex recomputes its own
holdout and known-bad evidence after applying a candidate.

Validate any such artifact with:

```bash
calibrex validate external-run.json --kind external-run
```

## Import a Kalibr camchain

The importer reads a completed Kalibr camchain without importing Kalibr, ROS,
or any Kalibr runtime dependency:

```bash
calibrex external-run import-kalibr camchain-imucam.yaml \
  --output external-run.json \
  --input-artifact recording.bag \
  --tool-version 0.3.2 \
  --source-commit <kalibr-commit> \
  --training-isolation-declared
```

Repeat `--input-artifact` for every fitting input. Add
`--expected-source-sha256` when an upstream pipeline supplies an expected
camchain digest; a mismatch is materialized as `digest_mismatch`, not silently
accepted.

The importer preserves Kalibr's documented conventions:

- `T_cam_imu` maps IMU coordinates into the camera;
- `T_cn_cnm1` maps the previous camera into the current camera;
- `timeshift_cam_imu` uses `t_imu = t_cam + timeshift_cam_imu`.

See Kalibr's official
[YAML format documentation](https://github.com/ethz-asl/kalibr/wiki/Yaml-formats)
and [top-level license](https://github.com/ethz-asl/kalibr/blob/master/LICENSE).
The default SPDX declaration is `BSD-4-Clause`; override it explicitly if the
exact external distribution being recorded has a different audited license.

## Koide-style targetless LiDAR-camera execution

Calibrex treats Koide-style targetless LiDAR-camera calibration as an external
baseline first. The core does not vendor external calibration code. It records
adapter readiness, input streams, optional command availability, loaded
transforms, and license boundary in `result.yaml`, and writes the typed
`external-run.json` beside it.

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
  selected camera, Velodyne, and calibration input SHA-256 digests with the
  raw-recomputation gate enabled.

A cached example result for report rendering without rerunning the pipeline is
available at
`examples/public_datasets/kitti_lidar_camera_evidence/cached_evidence_result.yaml`.

## Full-scale KITTI falsification benchmark

After downloading the official KITTI raw
`2011_09_26_drive_0005_sync` sequence through KITTI's login-gated flow, run:

```bash
calibrex demo kitti-falsification-benchmark \
  /data/2011_09_26/2011_09_26_drive_0005_sync \
  --output-dir outputs/kitti_falsification \
  --enforce
```

The default frozen protocol selects the first 50 time-matched frame pairs,
samples 4,000 LiDAR points per frame, uses seed `20260729`, and evaluates two
predeclared candidates: the dataset calibration and that calibration
right-composed with +5° camera-frame yaw and +0.25 m camera-frame x. Both trials
use the same config, holdout split, gates, and perturbation controls.

Each trial materializes `result.yaml`, `evidence.json`, `policy.json`,
`assessment.json`, `observability.json`, `protocol.json`, `transforms.json`,
`bundle.json`, and `verification.json`. The top-level `benchmark.json` pins the
selected frame IDs, source metadata, calibration/image/scan digests, candidates,
trial artifact digests, and generation provenance.

`--enforce` succeeds only if the dataset reference assessment is PASS, the
declared known-bad assessment is FAIL, and both raw-recomputed bundles verify.
Otherwise the frozen run remains a useful FAIL or INCONCLUSIVE result; do not
retune gates after observing it.
