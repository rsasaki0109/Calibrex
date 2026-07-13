# Calibrex — simultaneous localization and calibration

Universal sensor calibration evidence framework for robotics.

v0.3 adds trajectory evidence (`slac.trajectory/v0.1`) with ground-truth-free
gates, native-deskew KISS-ICP odometry at bag conversion, anchored temporal
time-offset estimation with joint separability reporting, and LiDAR-IMU rotation
evidence. On TIERS Indoor02 real data, native deskew cut identity selftest
translation error from ~9.5 cm to ~4.3 cm (~55%) at the cost of modest rotation
regression (~1.2° → ~2.0°); trajectory cross-segment drift proxy PASS at
0.143 m over the full 42 s track. Gates refuse to certify what they cannot
falsify — two-pass odometry FAILs its own trajectory gate (0.366 m); adapted
temporal mode absorbs injected offsets while anchored mode tracks them exactly;
yaw IMU probes stay uninformative under z-dominated excitation.

<p align="center">
  <a href="https://github.com/rsasaki0109/Calibrex/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/rsasaki0109/Calibrex/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-2f855a">
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-f59e0b">
</p>

Calibrex compares calibration estimates as evidence, not just matrices.

It lets you evaluate dataset references, manual candidates, native estimates,
online estimates, and external tool outputs through one schema, one CLI, one
metric suite, and one report.

<p align="center">
  <img src="docs/assets/calibrex-motion-calibration-loop.gif" alt="Calibrex simultaneous localization and calibration on TIERS Indoor02 real moving-platform data" width="100%">
</p>

<p align="center">
  <sub>Real TIERS Indoor02 run (<code>online_kissicp_config.yaml</code>): Velodyne VLP-16 builds a motion-compensated world map from KISS-ICP <code>/odom</code>; Ouster OS1 target batches converge online through holdout gates on a moving platform. Generated from a real <code>calibrex calibrate --online</code> replay (106/108 batches adopted).</sub>
</p>

```bash
calibrex calibrate config.yaml
calibrex evaluate outputs/result.yaml
calibrex render outputs/result.yaml --format html
```

<table>
  <tr>
    <td width="33%">
      <img src="docs/assets/calibration-evidence-demo.gif" alt="Livox public solid-state LiDAR evidence animation" width="100%">
    </td>
    <td width="33%">
      <img src="docs/assets/a2d2-multilidar-evidence-demo.gif" alt="A2D2 public fixed front LiDAR evidence animation" width="100%">
    </td>
    <td width="33%">
      <img src="docs/assets/a2d2-front-rear-evidence-demo.gif" alt="A2D2 public fixed front-rear LiDAR evidence animation" width="100%">
    </td>
  </tr>
  <tr>
    <td>
      <sub><b>Livox Horizon ↔ Horizon</b>: public solid-state 3D LiDAR PCD sample with known-bad controls and holdout point-to-plane evidence.</sub>
    </td>
    <td>
      <sub><b>A2D2 front pair</b>: public NPZ point cloud, <code>lidar_id 0 → 1</code>, using public fixed-rig sensor metadata.</sub>
    </td>
    <td>
      <sub><b>A2D2 front-rear pair</b>: public NPZ point cloud, <code>lidar_id 1 → 3</code>, showing a different fixed LiDAR baseline. No toy geometry.</sub>
    </td>
  </tr>
</table>

<p align="center">
  <sub>README visuals are generated from public raw samples. Calibrex records protocol metadata, known-bad probes, and PASS / FAIL / INCONCLUSIVE policy gates in <code>evidence.json</code> and <code>assessment.json</code>. The GIF source manifest is <code>docs/assets/readme-gif-gallery.json</code>.</sub>
</p>

| View | Public input | Calibration view | What to inspect |
|---|---|---|---|
| Online calibration loop | A2D2 NPZ range sample | Streaming fixed-rig estimate replay | Rolling residual, holdout gate, provenance lock |
| Livox Horizon ↔ Horizon | Official Livox PCD sample | Solid-state LiDAR pair | Known-bad rejection and holdout residual evidence |
| A2D2 front pair | A2D2 NPZ range sample | Fixed front LiDAR baseline | Multi-LiDAR metadata and support accounting |
| A2D2 front-rear pair | A2D2 NPZ range sample | Longer fixed-rig LiDAR baseline | Different overlap and support behavior |

<p align="center">
  <img src="docs/assets/readme-calibration-report.svg" alt="Calibrex calibration report overview" width="100%">
</p>

## Why Calibrex

Most calibration tools output a transform. Calibrex tries to answer the next
question: **should this transform be trusted?**

It records:

- candidate, reference, and optimized extrinsics separately
- producer, role, execution mode, and evidence level for each estimate
- train/holdout metrics, perturbation sensitivity, and temporal stability
- observability and degeneracy warnings
- map/correspondence lineage and leakage checks
- portable HTML and machine-readable report sidecars

## Current Alpha (v0.3)

| Area | Status |
|---|---|
| Typed config/result schemas | Stable alpha |
| `calibrate`, `evaluate`, `render`, `visualize`, `compare`, `inspect`, `export` CLI | Stable alpha |
| Online/streaming LiDAR calibration (`calibrate --online`) | Stable alpha (rosbag1 + rosbag2) |
| rosbag2/MCAP reader + odometry motion compensation + per-point deskew | Stable alpha; validated on TIERS Indoor02 |
| Trajectory evidence (`trajectory.json`, ground-truth-free gates) | Stable alpha; Indoor02 cross-segment PASS at 0.143 m over 42 s |
| Native-deskew KISS-ICP odometry at bag conversion | Stable alpha; identity selftest ~4.3 cm / ~2.0° (vs ~9.5 cm / ~1.2° baseline) |
| KITTI raw fixed-vehicle LiDAR evaluation | Experimental end-to-end |
| nuScenes metadata and reference extrinsic import | Experimental |
| TUM RGB-D / Open3D SLAC adapter boundary | Experimental |
| Fixed-trajectory and motion-compensated SE(3) LiDAR extrinsic solver | Native prototype |
| Koide-style LiDAR-camera result import / subprocess boundary | Adapter-only |
| Camera-LiDAR projection / edge-alignment evidence | Evaluated ADR-0004 protocol rows (holdout, known-bad probes, policy gates); not standalone camera calibration |
| Camera-LiDAR per-point capture-time solver | Native fixed-twist scalar offset solver; 17 ms synthetic truth and authenticated TIERS public application (honest weak-motion INCONCLUSIVE) |
| Temporal time-offset probes + 1D holdout-RMSE estimator | Anchored (`time_offset_anchor: initial`) and adapted dual-mode evaluation; selftest injection tracked exactly in anchored mode; cross-sensor separability degenerate on Indoor02 |
| LiDAR-IMU rotation evidence | Evaluated `lidar_imu` family (holdout rotation-rate RMSE 3.56 deg/s, gravity support 4.42°); yaw probes weak under z-dominated excitation |
| Solid-state LiDAR-to-LiDAR evidence visualization | Public Livox Horizon-Horizon PCD demo |
| ICP/GICP/NDT registration comparison | Native rematching curvature/Jaccard/multi-start plus MIT Open3D GICP and license-declared NDT boundaries; Livox GICP honestly FAILs holdout |
| Multi-LiDAR fixed-rig and online evidence | Public A2D2 VLP-16 demo; TIERS LidarsCali (static + mixed spinning×solid-state) |
| Hand-eye and robot-world/hand-eye calibration | Seven native `AX=XB` baselines plus Zhuang-Roth-Sudhakar, Shah, Li-Wang-Wu, and Dornaika-Horaud closed-form/nonlinear `AX=YB`; ETHZ common holdout and 24 controls per method |
| Backend-neutral joint SLAC optimizer | Grouped holdout, robust LM, typed Schur pose elimination, TUM RGB-D 48→8 shared-calibration solve |
| Radar extrinsic velocity-consistency check (nuScenes) | Experimental; `radar_lidar_velocity_consistency` scores static-target radial Doppler residuals against ego motion, not full radar calibration |

<p align="center">
  <img src="docs/assets/lidar-calibration-coverage.svg" alt="Calibrex LiDAR calibration coverage map" width="100%">
</p>

## Outputs

Calibrex writes schema-valid artifacts instead of one-off notebook state:

```text
result.yaml
comparison.json
report.html
summary.json
metrics.json
evidence.json
assessment.json
protocol.json
policy.json
observability.json
degeneracy.json
bundle.json
artifacts/rig_3d.html
```

The 3D rig viewer can show reference, candidate, and online/estimated
extrinsics together.

Report sidecars record evidence materialization:

- `metrics_origin`: `recomputed`, `cached`, or `unknown`
- `data_verified`: whether raw input file digests were verified for this artifact
- `evidence.json.input_files`: raw file paths, SHA-256 digests, sizes, and known public source URLs when available
- `protocol.json`: declared evidence protocols, transform conventions, support parameters, and known-bad controls
- `policy.json`: falsification gates and thresholds applied to the evidence

`calibrex render` renders an existing result and does not recompute metrics.
`calibrex report` remains a deprecated alias for HTML rendering. `Calibrex
evaluate` reapplies quality gates to a result. Cached inputs emit CLI and HTML
warnings, and `calibrex assess evidence.json` applies the falsification policy
without treating a scientific `FAIL` or `INCONCLUSIVE` verdict as a software
execution error. Use `calibrex assess --enforce evidence.json` when a non-pass
assessment should return a non-zero exit code.
`calibrex evidence result.yaml --output evidence.json` materializes the
evidence sidecar from an existing result without recomputing metrics.
`calibrex compare` shows materialization for both sides. `bundle.json` records
artifact SHA-256 digests and can be checked with `calibrex verify bundle.json`;
report outputs also include `verification.json` with that check materialized.
`calibrex verify verification.json` recomputes the source bundle check and
detects stale or edited verification records. Saved verification artifacts use
relative paths to colocated bundles where possible. When
`evidence.json.input_files` is present, `verify` also checks those raw input
file sizes and digests.

Committed JSON schemas are regenerated with:

```bash
calibrex schema all --output-dir schemas
```

## Install

```bash
pip install -e ".[dev]"
```

Optional backends stay optional:

```bash
pip install -e ".[dev,open3d]"
```

## Quickstart

```bash
calibrex doctor
calibrex init camera-lidar-imu --output config.yaml
calibrex calibrate config.yaml --output-dir outputs/example
calibrex validate outputs/example/result.yaml --json
calibrex evidence outputs/example/result.yaml --output outputs/example/evidence.json
calibrex evaluate outputs/example/result.yaml --export-html
calibrex render outputs/example/result.yaml --output-dir outputs/example/rendered
calibrex visualize outputs/example/result.yaml --export-html
```

Public dataset examples:

```bash
calibrex public-datasets list
calibrex demo livox-evidence --output-dir outputs/livox_horizon_horizon_pcd_sample
python3 tools/download_public_dataset.py livox_horizon_horizon_pcd_sample --output-dir data/public
python3 tools/generate_calibration_evidence_gif.py --readme-gallery
calibrex public-datasets show livox_horizon_horizon_pcd_sample --json
calibrex inspect data/public/livox_horizon_horizon_pair --type livox-pcd --json
calibrex calibrate examples/public_datasets/livox_horizon_horizon_pcd_sample/config.yaml
calibrex evidence outputs/livox_horizon_horizon_pcd_sample/result.yaml \
  --output outputs/livox_horizon_horizon_pcd_sample/evidence.json
calibrex render examples/public_datasets/livox_horizon_horizon_pcd_sample/cached_evidence_result.yaml
calibrex public-datasets show tiers_livox_lidars_cali --json
calibrex inspect examples/public_datasets/kitti_raw_2011_09_26_drive_0005 --type kitti-raw
calibrex demo kitti-lidar-camera-evidence --output-dir outputs/kitti_lidar_camera_evidence
calibrex calibrate examples/public_datasets/tum_rgbd_freiburg1_xyz/config.yaml
```

`--readme-gallery` regenerates every README GIF from public raw samples and
fetches the small A2D2 range sample when it is not already present locally.

Compare two calibration runs:

```bash
calibrex compare outputs/reference/result.yaml outputs/candidate/result.yaml \
  --output outputs/comparison.json
calibrex compare outputs/reference/result.yaml outputs/candidate/result.yaml \
  --enforce-compatible
```

`comparison.json` includes `protocol_compatibility` so cached, recomputed, and
mismatched evidence protocols are not silently ranked together. Use
`--enforce-compatible` when a protocol warning or non-comparable comparison
should fail a shell pipeline.

Compare more than two labeled runs in one report artifact:

```bash
calibrex report-compare \
  reference=outputs/dataset/result.yaml \
  perturbed=outputs/perturbed/result.yaml \
  koide=outputs/koide_adapter/result.yaml \
  native=outputs/Calibrex/result.yaml \
  --reference reference \
  --output outputs/report_comparison.json
```

`report_comparison.json` reuses the pairwise comparison machinery for every
compared pair, keeps per-pair `protocol_compatibility` gating, records each
entry's transform provenance (producer, role, evidence level), and adds a
cross-result ranking table per metric family. `--reference LABEL` compares one
baseline against each other entry instead of all pairs, and
`--enforce-compatible` fails the pipeline unless every compared pair is
protocol-compatible.

## Design Principles

- Sensor-agnostic core; sensor-specific factors live at the edges.
- Core stays ROS-independent.
- Public dataset calibration is treated as a reference, not absolute truth.
- Evaluation is mandatory.
- Every result must carry provenance.
- GPL tools stay behind optional adapter or subprocess boundaries.

## Docs

- [SLAC concept](docs/concepts/slac.md)
- [Frame conventions](docs/concepts/frame_conventions.md)
- [License boundaries](docs/concepts/license_boundaries.md)
- [Public datasets](docs/tutorials/public_datasets.md)
- [LiDAR-camera adapter](docs/tutorials/lidar_camera_adapter.md)
- [Support](SUPPORT.md)

## License

Apache-2.0.
