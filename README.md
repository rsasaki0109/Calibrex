# Calibrex

**Universal Sensor Calibration Evidence Framework for Robotics**

Calibrex compares calibration estimates as evidence, not just matrices.

It lets you evaluate dataset references, manual candidates, native estimates,
online estimates, and external tool outputs through one schema, one CLI, one
metric suite, and one report.

```bash
calibrex calibrate config.yaml
calibrex evaluate outputs/result.yaml
calibrex render outputs/result.yaml --format html
```

<p align="center">
  <img src="docs/assets/calibration-evidence-demo.gif" alt="Calibrex calibration evidence animation" width="100%">
  <br>
  <sub>Livox public Horizon-Horizon PCD demo: fixed solid-state 3D LiDAR-to-LiDAR extrinsic comparison on real point-cloud returns.</sub>
  <br>
  <sub>Calibrex evidence metrics on the sample: 47,587 points, 167 shared 1 m voxels, 16,884 / 23,666 fixed-denominator holdout plane matches, P90 |point-to-plane| 0.806 m, 100% known-bad controls detected.</sub>
  <br>
  <sub>Evidence protocol metadata, known-bad probes, and PASS / FAIL / INCONCLUSIVE policy gates are exported in <code>evidence.json</code> and <code>assessment.json</code>.</sub>
</p>

<p align="center">
  <img src="docs/assets/readme-calibration-report.svg" alt="Calibrex calibration report overview" width="100%">
</p>

<p align="center">
  <a href="https://github.com/rsasaki0109/Calibrex/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/rsasaki0109/Calibrex/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-2f855a">
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-f59e0b">
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

## Current Alpha

| Area | Status |
|---|---|
| Typed config/result schemas | Stable alpha |
| `calibrate`, `evaluate`, `render`, `visualize`, `compare`, `inspect`, `export` CLI | Stable alpha |
| KITTI raw fixed-vehicle LiDAR evaluation | Experimental end-to-end |
| nuScenes metadata and reference extrinsic import | Experimental |
| TUM RGB-D / Open3D SLAC adapter boundary | Experimental |
| Fixed-trajectory SE(3) LiDAR extrinsic correction solver | Native prototype |
| Koide-style LiDAR-camera result import / subprocess boundary | Adapter-only |
| Solid-state LiDAR-to-LiDAR evidence visualization | Public Livox Horizon-Horizon PCD demo |
| Multi-LiDAR fixed-rig evidence visualization | Public A2D2 VLP-16 demo / TIERS LidarsCali planned |
| Radar native calibration | Planned |

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
`calibrex report` remains a deprecated alias for HTML rendering. `calibrex
evaluate` reapplies quality gates to a result. Cached inputs emit CLI and HTML
warnings, and `calibrex assess evidence.json` applies the falsification policy
without treating a scientific `FAIL` or `INCONCLUSIVE` verdict as a software
execution error. Use `calibrex assess --enforce evidence.json` when a non-pass
assessment should return a non-zero exit code.
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
calibrex evaluate outputs/example/result.yaml --export-html
calibrex render outputs/example/result.yaml --output-dir outputs/example/rendered
calibrex visualize outputs/example/result.yaml --export-html
```

Public dataset examples:

```bash
calibrex public-datasets list
calibrex demo livox-evidence --output-dir outputs/livox_horizon_horizon_pcd_sample
python3 tools/download_public_dataset.py livox_horizon_horizon_pcd_sample --output-dir data/public
python3 tools/generate_calibration_evidence_gif.py
calibrex public-datasets show livox_horizon_horizon_pcd_sample --json
calibrex inspect data/public/livox_horizon_horizon_pair --type livox-pcd --json
calibrex calibrate examples/public_datasets/livox_horizon_horizon_pcd_sample/config.yaml
calibrex render examples/public_datasets/livox_horizon_horizon_pcd_sample/cached_evidence_result.yaml
calibrex public-datasets show tiers_livox_lidars_cali --json
calibrex inspect examples/public_datasets/kitti_raw_2011_09_26_drive_0005 --type kitti-raw
calibrex calibrate examples/public_datasets/tum_rgbd_freiburg1_xyz/config.yaml
```

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
