# Solid-state LiDAR calibration

Calibrex treats a solid-state LiDAR capture as more than an unordered point
cloud.  A result records the acquisition semantics that affect calibration:

- `architecture` and `scan_pattern` (for example, `non_repetitive`);
- the source field, reference, and unit for per-point capture time;
- the integration window;
- the source and representation of internal calibration data;
- the temperature condition and whether temperature compensation was applied.

The contract is ROS-independent.  ROS 1/ROS 2, MCAP, Livox PCD, and external
tool adapters may populate it, but `src/calibrex` does not import ROS or GPL
code.  A configured profile is copied into `result.yaml` under `solid_state`
with config/dataset SHA-256 provenance.  The standalone artifact contract is
available as `schemas/solid_state_context.schema.json` and through:

```bash
calibrex schema solid-state-context --output schemas/solid_state_context.schema.json
calibrex validate outputs/run/result.yaml
```

## Configuration

The existing `point_time_field` remains the adapter mapping from a
`PointCloud2` field.  `solid_state.point_time_*` describes its semantics; the
loader checks that both declarations agree and mirrors the field into the
profile.

```yaml
sensors:
  livox_avia:
    type: lidar
    model: livox_avia
    topic: /livox/lidar
    point_time_field: offset_time
    solid_state:
      architecture: solid_state
      scan_pattern: non_repetitive
      point_time_field: offset_time
      point_time_reference: message_stamp
      point_time_unit: seconds
      point_time_available: true
      integration_window_s: 0.005
      intrinsic_calibration:
        source: manufacturer
        model: manufacturer_native
        artifact_path: calibration/livox_avia_intrinsics.yaml
      temperature:
        sensor_c: 34.2
        source: sensor_telemetry

evaluation:
  solid_state:
    enabled: true
    require_point_time: true
    require_temperature: false
    require_independent_holdout: true
    min_temperature_span_c: 5.0
    min_distance_span_m: 5.0
    min_fov_azimuth_deg: 20.0
    min_fov_elevation_deg: 5.0
    distance_bins_m: [0.0, 10.0, 20.0, 40.0, 80.0]
    min_points_per_distance_bin: 100
    min_known_bad_detectable_fraction: 0.5
```

Missing time or temperature is a `WARN` by default.  Set the corresponding
`require_*` option when a motion-compensated or temperature-transfer claim is
not acceptable without that metadata.

## Evidence protocol

The solid-state metrics are deliberately separated from accuracy claims:

1. Existing point-to-plane metrics score a train/source map against a holdout
   query and record whether the holdout is independent.
2. Existing Livox controls perturb roll, pitch, yaw, and translation by known
   amounts.  Their detectable fraction is surfaced as
   `solid_state_known_bad_detectable_fraction` in addition to the detailed
   `lidar_pair_known_bad_*` metrics.
3. Livox PCD diagnostics report sampled range, azimuth/elevation FOV, and
   distance-bin support.  These are coverage/stratification metrics, not a
   replacement for independent spatial accuracy.
4. ROS 1/ROS 2 online replay records capture-window timestamps and, when deskew
   is actually applied from decoded per-point offsets, marks the capture
   window's point-time observation as available.  A PCD file index is never
   promoted to a physical sensor timestamp.
5. ROS 1 inspection samples report range, azimuth/elevation FOV, distance-bin
   support, and decoded point-time bounds for each supported LiDAR topic. The
   TIERS native adapter uses a temporal source-map boundary and later target
   windows for its offline holdout; the online adapter records the same
   boundary decision in provenance.

The public Horizon-Horizon PCD sample is intentionally limited: it is a
single source/target pair, so its point-to-plane result is useful geometry
evidence but not independent temporal validation.  For a publishable claim,
use multiple temporally separated windows or a held-out recording and set
`require_independent_holdout: true`.

For Livox ROS 1 data, `timebase` is a sensor clock whose relationship to
ROS/VRPN time must be checked for the recording. Calibrex does not silently
pretend that this raw value is a ROS epoch. During ROS 1 replay the bag-record
timestamp is the replay-clock authority and the decoded `offset_time` is added
to it:
`capture_ns = bag_record_timestamp_ns + offset_time_ns`.  The raw Livox
timebase is retained for diagnostics, and provenance records the fitted
timebase-to-bag mapping residual and a stable/unstable status.  If no stable
mapping can be established, pose interpolation is rejected or the result
remains non-independent; this is the correct evidence outcome until a
documented clock mapping is supplied.

## Ground-truth solver gate

Public sensor-pair recordings generally do not include independently surveyed
extrinsics or an absolute clock reference. Keep that limitation separate from
the solver-mechanics check by running the deterministic synthetic benchmark:

```bash
python tools/run_solid_state_synthetic_benchmark.py \
  --output outputs/solid_state_synthetic_benchmark_v01.yaml \
  --markdown-output outputs/solid_state_synthetic_benchmark_v01.md \
  --enforce
calibrex validate outputs/solid_state_synthetic_benchmark_v01.yaml \
  --kind solid-state-synthetic-benchmark
```

The reference case uses a known three-plane motion scene, a perturbed initial
extrinsic, and a **+30 ms** clock offset. The checked gate recovers the
reference to below 1e-12 numerical error in this deterministic fixture and rejects a
fixed-clock known-bad control with a 30 ms time error. Its output records the
truth, estimate, parameter errors, thresholds, and generator SHA-256 in
`provenance`. This is a correctness gate for the implementation, not evidence
of accuracy on a physical sensor pair; real-data claims still require
independent extrinsic/clock ground truth.

The checked artifact is [YAML](../assets/solid-state-synthetic-benchmark-v01.yaml)
with a compact [Markdown report](../assets/solid-state-synthetic-benchmark-v01.md).

## Livox workflow

The checked-in workflow is:

```bash
calibrex demo livox-evidence \
  --output-dir outputs/livox_horizon_horizon_pcd_sample
calibrex validate \
  outputs/livox_horizon_horizon_pcd_sample/result.yaml
```

The configuration is
[`examples/public_datasets/livox_horizon_horizon_pcd_sample/config.yaml`](https://github.com/rsasaki0109/Calibrex/blob/main/examples/public_datasets/livox_horizon_horizon_pcd_sample/config.yaml).
It declares the Livox non-repetitive pattern and explicitly leaves unavailable
point-time and temperature observations unknown rather than inferring them
from PCD file order.

## Adapter and license boundary

The core implementation is limited to typed metadata, native evidence, and
adapters.  GPL implementations must remain subprocess/reference adapters;
they are not imported into `src/calibrex`.  The first-party and permissive
references audited for integration are:

- [Livox camera-LiDAR calibration](https://github.com/Livox-SDK/livox_camera_lidar_calibration), MIT;
- [Koide direct visual-LiDAR calibration](https://github.com/koide3/direct_visual_lidar_calibration), MIT;
- [Livox automatic calibration](https://github.com/Livox-SDK/Livox_automatic_calibration),
  whose repository license and transitive dependencies must be re-audited at
  release time;
- [Multi_LiCa](https://github.com/TUMFTM/Multi_LiCa), LGPL-3.0, suitable only
  for an isolated adapter boundary;
- ACSC, FAST-Calib, LiDAR_IMU_Init, and `hku-mars/livox_camera_calib`, which
  are useful research references but GPL-licensed and therefore excluded from
  the core dependency graph.

## Research references

The implementation choices are informed by, but do not reproduce, these
methods:

- [García-Gómez et al., *Geometric Model and Calibration Method for a Solid-State LiDAR*](https://www.mdpi.com/1424-8220/20/10/2898)
  for variable angular resolution and device-specific angular distortion;
- [Huang et al., *Global Unifying Intrinsic Calibration for Spinning and Solid-State LiDARs*](https://arxiv.org/abs/2012.03321)
  for a unified geometric calibration model;
- [Kettelgerdes et al., *Precise intrinsic calibration and geometrical stability analysis of automotive solid-state LiDAR sensors over temperature and lifetime*](https://doi.org/10.1016/j.optlaseng.2026.109761)
  for treating temperature as a calibration condition rather than an
  incidental note;
- [PBACalib](https://fusionportable.github.io/files/PBACalib.pdf),
  [Koide](https://arxiv.org/abs/2302.05094), and
  [FAST-Calib](https://arxiv.org/abs/2507.17210) for targetless/target-based
  extrinsic evidence on non-repetitive sensors;
- [Liu et al., *Targetless Extrinsic Calibration of Multiple Small FoV
  LiDARs and Cameras using Adaptive Voxelization*](https://arxiv.org/abs/2109.06550)
  for the small-overlap problem and multi-LiDAR bundle adjustment;
- [OA-LICalib](https://arxiv.org/abs/2205.03276) for information-based segment
  selection and singular-value observability checks. Its reference repository
  is GPL-3, so it is an external comparison/adapter candidate rather than a
  core dependency;
- [Mints et al., *Online Calibration of Extrinsic Parameters for Solid-State LIDAR Systems*](https://doi.org/10.3390/s24072155)
  for the practical distinction between manufacturer-supplied intrinsic
  calibration and estimated extrinsics.
