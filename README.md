# slac — simultaneous localization and calibration

Formerly Calibrex. Universal sensor calibration evidence framework for robotics.

v0.2 adds moving-platform online calibration (pure-Python rosbag2/MCAP reader,
odometry motion compensation, optional per-point deskew), camera-LiDAR evidence
rows under the ADR-0004 protocol, and temporal time-offset probes with a 1D
holdout-RMSE estimator. On TIERS Indoor02 real data, motion compensation cut
identity self-consistency extrinsic error roughly 10× (~9.6 cm / ~1.2° vs ~98 cm
/ ~37° for the static control). Gates refuse to certify what they cannot
falsify — inconclusive or failed probe power is recorded honestly, not smoothed
over.

<p align="center">
  <a href="https://github.com/rsasaki0109/slac/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/rsasaki0109/slac/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-2f855a">
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-f59e0b">
</p>

slac compares calibration estimates as evidence, not just matrices.

It lets you evaluate dataset references, manual candidates, native estimates,
online estimates, and external tool outputs through one schema, one CLI, one
metric suite, and one report.

<p align="center">
  <img src="docs/assets/online-calibration-loop.gif" alt="Online solid-state Livox LiDAR calibration evidence loop" width="100%">
</p>

<p align="center">
  <sub>Online replay on the TIERS LidarsCali static rig: Livox Horizon (non-repetitive scan) streams build a fixed source map while Livox Avia target batches converge through rolling residuals, holdout gates, and provenance-backed assessment. Generated from a real <code>slac calibrate --online</code> run on the public rosbag.</sub>
</p>

```bash
slac calibrate config.yaml
slac evaluate outputs/result.yaml
slac render outputs/result.yaml --format html
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
  <sub>README visuals are generated from public raw samples. slac records protocol metadata, known-bad probes, and PASS / FAIL / INCONCLUSIVE policy gates in <code>evidence.json</code> and <code>assessment.json</code>. The GIF source manifest is <code>docs/assets/readme-gif-gallery.json</code>.</sub>
</p>

| View | Public input | Calibration view | What to inspect |
|---|---|---|---|
| Online calibration loop | A2D2 NPZ range sample | Streaming fixed-rig estimate replay | Rolling residual, holdout gate, provenance lock |
| Livox Horizon ↔ Horizon | Official Livox PCD sample | Solid-state LiDAR pair | Known-bad rejection and holdout residual evidence |
| A2D2 front pair | A2D2 NPZ range sample | Fixed front LiDAR baseline | Multi-LiDAR metadata and support accounting |
| A2D2 front-rear pair | A2D2 NPZ range sample | Longer fixed-rig LiDAR baseline | Different overlap and support behavior |

<p align="center">
  <img src="docs/assets/readme-calibration-report.svg" alt="slac calibration report overview" width="100%">
</p>

## Why slac

Most calibration tools output a transform. slac tries to answer the next
question: **should this transform be trusted?**

It records:

- candidate, reference, and optimized extrinsics separately
- producer, role, execution mode, and evidence level for each estimate
- train/holdout metrics, perturbation sensitivity, and temporal stability
- observability and degeneracy warnings
- map/correspondence lineage and leakage checks
- portable HTML and machine-readable report sidecars

## Current Alpha (v0.2)

| Area | Status |
|---|---|
| Typed config/result schemas | Stable alpha |
| `calibrate`, `evaluate`, `render`, `visualize`, `compare`, `inspect`, `export` CLI | Stable alpha |
| Online/streaming LiDAR calibration (`calibrate --online`) | Stable alpha (rosbag1 + rosbag2) |
| rosbag2/MCAP reader + odometry motion compensation + per-point deskew | Stable alpha; validated on TIERS Indoor02 |
| KITTI raw fixed-vehicle LiDAR evaluation | Experimental end-to-end |
| nuScenes metadata and reference extrinsic import | Experimental |
| TUM RGB-D / Open3D SLAC adapter boundary | Experimental |
| Fixed-trajectory and motion-compensated SE(3) LiDAR extrinsic solver | Native prototype |
| Koide-style LiDAR-camera result import / subprocess boundary | Adapter-only |
| Camera-LiDAR projection / edge-alignment evidence | Evaluated ADR-0004 protocol rows (holdout, known-bad probes, policy gates); not standalone camera calibration |
| Temporal time-offset probes + 1D holdout-RMSE estimator | Evaluated on motion-compensated online runs; synthetic ±5 ms recovery; real-data probe power limited after solver adaptation |
| Solid-state LiDAR-to-LiDAR evidence visualization | Public Livox Horizon-Horizon PCD demo |
| Multi-LiDAR fixed-rig and online evidence | Public A2D2 VLP-16 demo; TIERS LidarsCali (static + mixed spinning×solid-state) |
| IMU extrinsic candidate evaluation | Planned; OXTS motion excitation checks exist today but do not evaluate IMU extrinsics |
| Radar extrinsic velocity-consistency check (nuScenes) | Experimental; `radar_lidar_velocity_consistency` scores static-target radial Doppler residuals against ego motion, not full radar calibration |

<p align="center">
  <img src="docs/assets/lidar-calibration-coverage.svg" alt="slac LiDAR calibration coverage map" width="100%">
</p>

## Outputs

slac writes schema-valid artifacts instead of one-off notebook state:

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

`slac render` renders an existing result and does not recompute metrics.
`slac report` remains a deprecated alias for HTML rendering. `slac
evaluate` reapplies quality gates to a result. Cached inputs emit CLI and HTML
warnings, and `slac assess evidence.json` applies the falsification policy
without treating a scientific `FAIL` or `INCONCLUSIVE` verdict as a software
execution error. Use `slac assess --enforce evidence.json` when a non-pass
assessment should return a non-zero exit code.
`slac evidence result.yaml --output evidence.json` materializes the
evidence sidecar from an existing result without recomputing metrics.
`slac compare` shows materialization for both sides. `bundle.json` records
artifact SHA-256 digests and can be checked with `slac verify bundle.json`;
report outputs also include `verification.json` with that check materialized.
`slac verify verification.json` recomputes the source bundle check and
detects stale or edited verification records. Saved verification artifacts use
relative paths to colocated bundles where possible. When
`evidence.json.input_files` is present, `verify` also checks those raw input
file sizes and digests.

Committed JSON schemas are regenerated with:

```bash
slac schema all --output-dir schemas
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
slac doctor
slac init camera-lidar-imu --output config.yaml
slac calibrate config.yaml --output-dir outputs/example
slac validate outputs/example/result.yaml --json
slac evidence outputs/example/result.yaml --output outputs/example/evidence.json
slac evaluate outputs/example/result.yaml --export-html
slac render outputs/example/result.yaml --output-dir outputs/example/rendered
slac visualize outputs/example/result.yaml --export-html
```

Public dataset examples:

```bash
slac public-datasets list
slac demo livox-evidence --output-dir outputs/livox_horizon_horizon_pcd_sample
python3 tools/download_public_dataset.py livox_horizon_horizon_pcd_sample --output-dir data/public
python3 tools/generate_calibration_evidence_gif.py --readme-gallery
slac public-datasets show livox_horizon_horizon_pcd_sample --json
slac inspect data/public/livox_horizon_horizon_pair --type livox-pcd --json
slac calibrate examples/public_datasets/livox_horizon_horizon_pcd_sample/config.yaml
slac evidence outputs/livox_horizon_horizon_pcd_sample/result.yaml \
  --output outputs/livox_horizon_horizon_pcd_sample/evidence.json
slac render examples/public_datasets/livox_horizon_horizon_pcd_sample/cached_evidence_result.yaml
slac public-datasets show tiers_livox_lidars_cali --json
slac inspect examples/public_datasets/kitti_raw_2011_09_26_drive_0005 --type kitti-raw
slac demo kitti-lidar-camera-evidence --output-dir outputs/kitti_lidar_camera_evidence
slac calibrate examples/public_datasets/tum_rgbd_freiburg1_xyz/config.yaml
```

`--readme-gallery` regenerates every README GIF from public raw samples and
fetches the small A2D2 range sample when it is not already present locally.

Compare two calibration runs:

```bash
slac compare outputs/reference/result.yaml outputs/candidate/result.yaml \
  --output outputs/comparison.json
slac compare outputs/reference/result.yaml outputs/candidate/result.yaml \
  --enforce-compatible
```

`comparison.json` includes `protocol_compatibility` so cached, recomputed, and
mismatched evidence protocols are not silently ranked together. Use
`--enforce-compatible` when a protocol warning or non-comparable comparison
should fail a shell pipeline.

Compare more than two labeled runs in one report artifact:

```bash
slac report-compare \
  reference=outputs/dataset/result.yaml \
  perturbed=outputs/perturbed/result.yaml \
  koide=outputs/koide_adapter/result.yaml \
  native=outputs/slac/result.yaml \
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
