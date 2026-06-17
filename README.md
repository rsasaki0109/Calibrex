# Calibrex

**Universal Sensor Calibration Framework for Robotics**

Calibrex turns multi-sensor calibration logs into comparable evidence:
schema-valid results, candidate/reference extrinsic comparisons, train/holdout
metrics, PASS/WARN/FAIL quality gates, portable HTML reports with a
first-screen Calibration Scoreboard, and schema-valid machine-readable report
sidecars.

<p align="center">
  <img src="docs/assets/readme-calibration-report.svg" alt="Calibrex public dataset calibration report overview" width="100%">
</p>

<p align="center">
  <a href="https://github.com/rsasaki0109/Calibrex/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/rsasaki0109/Calibrex/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776ab">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-2f855a">
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-f59e0b">
</p>

```bash
calibrex calibrate config.yaml --candidate-extrinsics candidates/manual.yaml
```

## What Calibrex Shows

| Public dataset workflow | What Calibrex reads | What appears in `result.yaml`, `report.html`, and sidecars |
|---|---|---|
| KITTI raw | Velodyne frames, camera frames, OXTS motion, timestamps, calibration files | LiDAR map consistency, perturbation sensitivity, timestamp diagnostics, camera-LiDAR overlay checks |
| nuScenes | `sample_data`, `ego_pose`, `sensor`, `calibrated_sensor` metadata | `reference_extrinsics`, `candidate_extrinsics`, candidate/reference translation and rotation deltas |
| TUM RGB-D | RGB/depth associations and trajectory-style metadata | Open3D SLAC adapter boundary, schema-normalized result and quality report |

| Output | Why it matters |
|---|---|
| `candidate_extrinsics` vs `reference_extrinsics` | Keeps manual candidates, dataset references, and solver outputs separate. |
| Train/holdout metrics | Makes calibration quality harder to overfit to one frame or one sequence segment. |
| Observability and degeneracy warnings | Reports when a dataset cannot support a trusted estimate for some DoF. |
| `summary.json`, `metrics.json`, `observability.json`, `degeneracy.json` | Gives CI, notebooks, and benchmark scripts stable machine-readable report inputs, including metric-family rollups. |
| `calibrex validate` | Verifies configs, results, manifests, and report sidecars from their `schema_version`. |
| HTML report, Calibration Scoreboard, and overlays | Gives reviewers a portable artifact instead of a one-off notebook screenshot. |

Examples are public-dataset workflows. Large raw logs are never committed to the
repository; Calibrex keeps configs, manifests, schemas, and reproducible
evaluation code.

## Vision

Robotic calibration is fragmented across many sensor-specific tools. Calibrex
aims to make calibration reproducible, inspectable, extensible, and
quality-driven.

Calibrex is built around SLAC: Simultaneous Localization and Calibration.
Instead of estimating extrinsics as a standalone afterthought, Calibrex treats
calibration parameters, trajectory, timing, and map variables as parts of a
unified estimation problem.

## Current Alpha Scope

- Typed config and result schema
- ROS-independent core abstractions
- Camera, LiDAR, IMU, and Radar sensor declarations
- Frame graph validation with explicit transform convention
- Dataset inspection for filesystem inputs and MCAP stubs
- Dataset manifest schema for portable stream declarations
- Public dataset catalog for TUM RGB-D, KITTI raw, and nuScenes references
- KITTI raw loader for fixed-mounted vehicle LiDAR calibration
- Velodyne `.bin` reader with sampled point cloud diagnostics in `inspect`
- LiDAR coverage metrics for frame, point, and spatial extent checks
- LiDAR local planarity, roughness, and map sharpness proxy metrics
- Alpha `lidar_point_to_plane_rmse_m` metric from voxel-plane train/holdout splits
- OXTS-projected LiDAR world-map train/holdout point-to-plane consistency metrics
- OXTS-projected LiDAR world-map perturbation ranking metrics
- Weak LiDAR DoF warnings from world-map perturbation sensitivity
- KITTI LiDAR point-to-plane perturbation sensitivity metrics for roll/pitch/yaw/x/y/z
- LiDAR degeneracy warnings from point cloud coverage, vertical structure, and holdout gaps
- Actionable LiDAR recollection recommendations in result quality reports
- Human-readable KITTI `inspect` output with LiDAR diagnostics and recommendations
- SDK-free nuScenes metadata inspection for LiDAR, camera, radar, ego pose, and calibrated sensors
- nuScenes `calibrated_sensor` import into result `reference_extrinsics`
- Candidate extrinsics stored separately from solver output and compared against references
- External candidate extrinsic YAML import with `--candidate-extrinsics`
- HTML Calibration Scoreboard summarizing grade counts, weak DoF, candidate/reference matches, and artifacts
- Schema-valid machine-readable report sidecars for summary, metrics, observability, and degeneracy
- KITTI OXTS motion excitation diagnostics for speed, duration, and yaw checks
- KITTI Camera-LiDAR and LiDAR-OXTS timestamp alignment metrics
- KITTI camera image and Velodyne frame-pair extraction for overlay artifacts
- KITTI Velodyne-to-camera projection overlay using public dataset calibration files
- Camera-LiDAR projection metrics for point count, ratio, depth, image coverage,
  and multi-frame train/holdout aggregation
- Configurable KITTI projection frame count via `evaluation.kitti.max_projection_pairs`
- KITTI camera-LiDAR known perturbation sensitivity metrics for roll/pitch/yaw/x/y/z
- Camera-LiDAR edge-alignment proxy from projected LiDAR points and PNG gradients
- Camera-LiDAR depth-discontinuity edge alignment proxy
- Camera-LiDAR overlay readiness, overlay score proxy, and MI score placeholders
- Camera-LiDAR overlay HTML artifact scaffold linked from the report
- LiDAR SLAC design notes based on SceneCalib, SST-Calib, multi-LiDAR SLAC,
  M-LOAM, and Koide et al. ICRA 2023
- Koide-style targetless LiDAR-camera adapter boundary with external result import
  and opt-in subprocess execution
- Unified `calibrate`, `evaluate`, `visualize`, `schema`, `inspect`, and `export` commands
- PASS/WARN/FAIL quality gates and HTML report generation
- Metric registry and threshold profiles, including an autonomous-driving profile
- RGB-D Open3D SLAC adapter boundary
- Autonomous-driving profile hooks, including Radar and Autoware export stubs

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
calibrex doctor
calibrex schema dataset-manifest --output schemas/dataset_manifest.schema.json
calibrex schema report-summary --output schemas/report_summary.schema.json
calibrex init camera-lidar-imu --output config.yaml
calibrex compile config.yaml
calibrex calibrate config.yaml --output-dir outputs/example
calibrex validate outputs/example/result.yaml --json
calibrex validate outputs/example/summary.json --json
calibrex evaluate outputs/example/result.yaml --export-html
calibrex visualize outputs/example/result.yaml --export-html
calibrex export outputs/example/result.yaml --format ros-tf --output outputs/example/tf.yaml
```

Public dataset examples:

```bash
calibrex public-datasets list
calibrex inspect examples/public_datasets/kitti_raw_2011_09_26_drive_0005 --type kitti-raw
calibrex compile examples/public_datasets/kitti_raw_2011_09_26_drive_0005/config.yaml
calibrex kitti import-calib /path/to/2011_09_26 --output /tmp/kitti_transforms.yaml
calibrex calibrate config.yaml --candidate-extrinsics candidates/manual.yaml
calibrex calibrate examples/public_datasets/tum_rgbd_freiburg1_xyz/config.yaml
```

Open3D SLAC adapter example:

```bash
calibrex calibrate examples/rgbd_open3d_slac/config.yaml
```

Install `calibrex[open3d]` to execute Open3D-backed processing. Without the
optional dependency, Calibrex still emits a schema-valid result with WARN
diagnostics and adapter provenance.

Targetless LiDAR-camera adapter results can be imported, or executed through an
explicit `execute: true` subprocess boundary, with
`koide_lidar_camera` pipeline options. See
`docs/tutorials/lidar_camera_adapter.md`.

## Design Principles

- Core is ROS-independent.
- MCAP and filesystem datasets are first-class inputs.
- Configs compile to backend-neutral variables, factors, priors, and gauges.
- Every result includes provenance and quality metrics.
- Evaluation is mandatory.
- Sensor-specific algorithms are plugins.
- Existing tools can be used as adapters or baselines.
- GPL-based tools must stay behind optional adapter or subprocess boundaries.

## License

Apache-2.0 is recommended for the Calibrex core.
