# Koide input readiness

`calibrex doctor --workflow koide` is a preflight for the optional
`direct_visual_lidar_calibration` provider. It does not fit an extrinsic and it
does not replace the independent holdout/known-bad evaluation. The artifact is
the auditable answer to “is this capture safe to attempt?”

```bash
calibrex doctor --workflow koide \
  --config examples/public_datasets/kitti_lidar_camera_evidence/config.yaml \
  --output artifacts/koide-readiness.yaml
calibrex validate artifacts/koide-readiness.yaml --kind koide-readiness
```

The artifact records each check's status, observed value, evidence, next
action, and source digests. Missing evidence is `unknown` on the check and
produces an overall `warn`; missing camera/LiDAR fitting inputs, finite
intrinsics/CameraInfo, or explicit frame binding produces `blocked`. A warning
is therefore never silently promoted to `ready`.

Readiness thresholds are typed and belong to the preflight protocol, not the
evaluation holdout protocol:

```yaml
pipeline:
  factors:
    koide_lidar_camera:
      enabled: true
      options:
        profile: commercial
        hardware_profile: ouster  # livox, ouster, velodyne, or generic
        readiness:
          thresholds:
            max_sync_delta_ms: 50.0
            min_sampled_point_count: 500
            min_image_texture_score: 0.03
```

Hardware profiles are never inferred from a model name. If omitted, the
artifact records `generic` guidance and a warning. The Livox guidance asks for
point-time preservation, Ouster guidance asks for broad scan overlap, and
Velodyne guidance asks for multiple timestamped scans and vertical structure.

For a production run, pass the artifact to the typed runner and enable the
strict gate:

```yaml
options:
  execution_mode: container
  readiness_artifact_path: artifacts/koide-readiness.yaml
  readiness_artifact_sha256: <sha256-of-the-artifact-file>
  strict_readiness: true
```

Strict execution is fail-closed: the artifact status must be exactly `ready`,
every readiness check must be `pass` (so no `warn` or `unknown` evidence can
enter execution), and all declared input digests must still verify before an
external stage starts. The Koide implementation, ROS dependencies, model
weights, and GPL code remain outside the Calibrex core.

For the official commercial handoff, validate
`examples/official/koide_execution_lock.yaml` and run
`calibrex camera-lidar koide-handoff`; that lock pins the upstream source/image,
selects manual initialization, and excludes SuperGlue. The A2D2 readiness
artifact under `outputs/koide_pilot_a2d2_blocked` is intentionally diagnostic
only and is not an official Koide execution or score.
