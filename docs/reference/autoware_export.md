# Autoware export

`calibrex export --format autoware` is a ROS-independent adapter for an
Autoware sensor kit. It consumes the selected `transforms` from a validated
Calibrex result and writes three artifacts:

* `sensor_kit_calibration.yaml`, with the Autoware mapping
  `<parent_frame>: <child_frame>: {x, y, z, roll, pitch, yaw}` in metres and
  radians;
* a deterministic `*.static_tf.launch.py` using
  `tf2_ros/static_transform_publisher`'s explicit quaternion options; and
* a schema-valid `*.manifest.yaml` containing source/result digests, export
  configuration, tool/version/git/protocol provenance, byte-level output
  digests, and a canonical manifest digest.

For example, when the result frame graph root is `sensor_kit_base_link`:

```bash
calibrex export results/result.yaml \
  --format autoware \
  --output individual_params/config/vehicle/sensor_kit_calibration.yaml \
  --base-frame sensor_kit_base_link \
  --sensor-frame lidar_top \
  --sensor-frame CAM_FRONT/camera_link
```

Copy the generated calibration YAML into the vehicle's
`individual_params/config/<vehicle>/<sensor-kit>/` package and include the
launch sidecar while bringing up the sensor kit. Validate the provenance
sidecar before installing it:

```bash
calibrex validate individual_params/config/vehicle/sensor_kit_calibration.manifest.yaml \
  --kind autoware-export
calibrex schema autoware-export --output autoware_export.schema.json
```

The export uses Calibrex's `T_parent_child` convention: a transform maps a
point from the child frame into the parent frame. Thus `T_sensor_kit_base_link`
is not silently treated as `T_base_sensor`; pass `--invert T_sensor_kit_base_link`
when that is the measured direction. Inversion is the exact SE(3) inverse,
including the rotated negative translation. The static-TF command uses
`--qx --qy --qz --qw` and the same `xyzw` order recorded in the manifest.

The adapter rejects missing or ambiguous frame selections, empty/non-finite
values, zero or materially non-unit quaternions, unsupported conventions, and
duplicate output child frames. It does not infer a vehicle `base_link` from a
sensor name: select the exact parent frame used by the sensor-kit package.
Autoware's calibration loader consumes the parent-frame mapping at the top of
the YAML; Calibrex metadata and the legacy transform list are retained for
audit and are not a substitute for updating the vehicle's URDF/Xacro and
sensor topics. Re-run the downstream TF and perception checks after changing
an installed calibration.

Existing Python callers can continue to use `export_autoware_yaml(result,
path)` or `export_autoware_transforms(mapping, path)`. Production callers
should pass `AutowareExportConfig`, `source_path`, and sidecar output paths to
`build_autoware_export`/`write_autoware_export`, and should use `overwrite=True`
only for an intentional replacement.
