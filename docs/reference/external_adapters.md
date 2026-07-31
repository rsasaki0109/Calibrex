# External Calibration Adapters

Calibrex keeps external calibration implementations behind a ROS-independent,
subprocess/import boundary. It does not import UniCalib, Koide's direct
visual-LiDAR calibration implementation, model weights, ROS, CUDA, or GPL code
into `src/calibrex`.

Every Koide or UniCalib adapter result includes an
`external_calibration_run` mapping. Set `external_run_output` to materialize
the same mapping as a standalone `slac.external_calibration_run/v0.1`
YAML/JSON artifact:

```yaml
pipeline:
  factors:
    unicalib_lidar_camera:
      enabled: true
      options:
        result_path: outputs/unicalib/transform.yaml
        input_paths:
          - inputs/fixed_split_manifest.yaml
        expected_result_sha256: "<sha256>"
        external_run_output: outputs/unicalib/external_run.yaml
        tool_version: "wacv-2026"
        source_repository: https://github.com/han-15/UniCalib
        source_commit: "<full commit>"
        license_spdx: MIT
        training_isolation_declared: true
        training_isolation_evidence: "checkpoint training split excludes evaluation sequence"
```

Commands may be a string or argv list. Placeholders are substituted per argv
element and no shell is enabled:

```yaml
command:
  - /opt/unicalib/run
  - --dataset
  - "{dataset_path}"
  - --output
  - "{result_path}"
execute: true
timeout_sec: 600
```

Supported placeholders are `{dataset_path}`, `{result_path}`,
`{camera_streams}`, and `{lidar_streams}`. Container execution should declare
`execution_mode: container` and a content-addressed `container_digest`.

The imported transform format is deliberately adapter-neutral:

```yaml
transforms:
  T_camera0_lidar0:
    convention: T_parent_child
    parent: camera0
    child: lidar0
    translation_m: [0.0, 0.0, 0.0]
    rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
```

External objective values can be recorded under `external_metrics`, but the
schema labels them non-comparable with Calibrex holdout evidence by default.
The normal calibration pipeline recomputes its camera-LiDAR evidence after the
adapter transform is applied; an external training loss never substitutes for
those gates.

## License and provenance gate

Release-ready runs pin tool version, source commit, SPDX identifier, adapter
version, input/output SHA-256 digests, frame/time conventions, execution mode,
and training-data isolation. Missing fields remain schema-valid but produce a
WARN provenance metric. Digest mismatch makes the external-run artifact
`failed`.

All allowed execution modes are out-of-process execution or result import.
There is no in-process external-code mode. GPL tools therefore remain separate
executables or containers and are never copied into the Calibrex core.
