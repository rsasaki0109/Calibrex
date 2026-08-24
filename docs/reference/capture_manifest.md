# Capture manifest

`calibrex capture inspect` creates a schema-versioned, ROS-independent
inventory for a recording or a directory of files:

```bash
calibrex capture inspect recording --type rosbag2 \
  --capture-id drive-001 --session-id session-001 \
  --vehicle-id car-07 --sensor-kit-id kit-2026 \
  --sensor lidar-front:lidar:SN123:VLP-16:5.0:mount-front:lidar_front \
  --output capture-manifest.yaml
```

The artifact records source SHA-256 and size, storage/index/CRC evidence,
message/topic or file streams, timestamps and rate, duplicate/out-of-order
counts, frame and clock bindings, physical sensor identity, and the command,
tool, configuration, source digests, and git revision used to produce it.
`artifact_sha256` is a canonical self-digest; `calibrex validate` verifies it.
Provenance `source_sha256` keys are stable `source_id` values and every digest
is a full lowercase SHA-256 hex string.

Self-digest validation and input validation are intentionally separate:

```bash
# Portable check: validates schema and artifact_sha256 only.
calibrex validate capture-manifest.yaml --kind capture-manifest

# Production check: also re-hashes source/config paths relative to the
# manifest's directory and fails on missing, unknown, or changed inputs.
calibrex capture verify capture-manifest.yaml --json
calibrex validate capture-manifest.yaml --kind capture-manifest --verify-inputs
```

`capture verify` emits a typed `slac.capture_manifest.verification/v0.1`
report. A copied manifest can therefore still be loaded for portability, but
it must not be adopted until the production verification report is valid.

Unknown is represented explicitly.  For example, an MCAP fast-write file may
omit the optional Summary/Index section or encode a zero (unavailable) CRC.
`capture inspect --type mcap --json` now records bounded framing, Header,
DataEnd, Footer, schema/channel/message-link, Summary/Index, and directly
verifiable CRC evidence in `sources[].mcap_integrity`. Missing optional
Summary/Index or unavailable CRC fields remain `unknown` and produce a
warning; malformed framing, dangling links, and CRC mismatches block intake.
Compressed chunks whose codec is recognised but not available for this scan
remain unknown, while an unrecognised codec is explicitly unsupported and
blocking. The wire-level rules are defined by the
[official MCAP format specification](https://mcap.dev/spec), including the
optional Summary/Index sections and DataEnd/Footer CRC fields. For rosbag2,
metadata alone does not prove storage index or container CRC verification, so
those fields remain `unknown`; an adapter only marks QoS fields known when the
storage metadata exposes named values. Vendor-specific or malformed message
types without a decoder are retained as `unsupported` or `unknown` streams;
required streams cannot make a capture `ready` until an adapter is available.
The built-in ROS 1 (`sensor_msgs/Image`,
`sensor_msgs/CameraInfo`) and ROS 2 CDR (`sensor_msgs/msg/Image`,
`sensor_msgs/msg/CameraInfo`) adapters validate dimensions, encoding/stride/data
lengths, CDR alignment, matrix finiteness, and ROI bounds. Inventory runs use
`include_data=false`, so image pixels are validated without retaining a second
full image buffer. A ROS-valid uncalibrated CameraInfo (for example `K[0] == 0`)
is retained as decoded evidence but receives `calibration_status: uncalibrated`
and blocks a required-camera readiness decision until usable intrinsics are
supplied. The standard `radar_msgs` RadarScan adapter is documented in the
[RadarScan adapter reference](radar_scan_adapter.md); it records bounded
return-count, range, angular, Doppler, duplicate, and frame evidence and
blocks weak required radar diversity.

The default `strict` readiness profile treats every discovered stream as
required and fails closed. For recordings containing debug topics, use the
`declared` profile and record the policy explicitly:

```bash
calibrex capture inspect recording --type rosbag2 \
  --readiness-profile declared \
  --required-stream /points --optional-stream /debug/diagnostics \
  --required-kind camera_info
```

Non-required, non-critical topics are non-gating under this profile. Any
discovered Image, CameraInfo, or Radar topic must nevertheless be declared
required or explicitly optional; it cannot be silently ignored. Duplicate
IDs, dangling sensor/stream references, inconsistent timestamps/rates, and
ambiguous alias spellings are rejected before an artifact is emitted.

The identity fields are intentionally required for a ready intake decision.
Use `--json` for machine-readable output or `calibrex schema capture-manifest`
for the public JSON Schema. Use `calibrex schema capture-manifest-verification`
for the verification report schema. Existing `calibrex inspect` commands and
the legacy `slac.dataset_manifest/v0.1` remain unchanged.
