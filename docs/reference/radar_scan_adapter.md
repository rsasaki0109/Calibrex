# RadarScan intake adapter

Calibrex accepts the standard `radar_msgs` point-cloud message through the
ROS-independent bag adapters. The exact supported types are:

| Bag | Type | Pinned definition | License |
|---|---|---|---|
| ROS 2 / CDR | `radar_msgs/msg/RadarScan` (0.2.2) | [`radar_msgs` ROS 2 commit `47d2f26906ef38fa15ada352aea6b5aad547781d`](https://github.com/ros-perception/radar_msgs/tree/47d2f26906ef38fa15ada352aea6b5aad547781d/msg) | Apache-2.0 |
| ROS 1 / rosmsg | `radar_msgs/RadarScan` (0.2.2 Noetic) | [`radar_msgs` ROS 1 commit `bfd6d5487405500957dfa1b39a7e513ec172536a`](https://github.com/ros-perception/radar_msgs/tree/bfd6d5487405500957dfa1b39a7e513ec172536a/msg) | Apache-2.0 |

The message is a stamped sequence of `RadarReturn` values. Each return is
exactly five `float32` fields, in this order: `range` (m), `azimuth` (rad),
`elevation` (rad), `doppler_velocity` (m/s), and `amplitude` (dB). Calibrex
normalizes these into typed `RadarReturn` and `RadarScanMessage` records and
retains the source type/commit/license in the message provenance.

Both decoders are bounded and fail closed. They reject truncated or oversized
payloads, invalid sequence counts, non-finite values, negative range,
out-of-domain spherical angles, and trailing bytes. Exact duplicate returns
are reported as a diagnostic because the upstream message has no return ID.
`RadarScanMessage.to_xyz()` provides a pure-Python spherical-to-sensor-frame
conversion; it does not imply a vehicle-frame transform.

## Capture readiness

`calibrex capture inspect` marks an exact official type `supported` only after
at least one sampled payload decodes successfully. A malformed sample becomes
`unknown`; a vendor-specific radar type remains `unsupported` and recommends
an adapter. The manifest records sampled return count, range bounds, angular
spans, Doppler span, duplicate count, frame, and a bounded diversity status.
Empty or weak required radar evidence blocks intake. The same evidence on an
optional radar stream produces a warning, so declared profiles remain explicit
about what is and is not gating.

Frame IDs must remain stable across sampled messages. A conflict is retained
as evidence and blocks a required stream until the recording or frame binding
is corrected. A missing frame remains a warning under the existing readiness
policy.

Autoware consumes `radar_msgs/msg/RadarScan` as radar pointcloud data. See the
[Autoware radar data-message reference](https://autowarefoundation.github.io/autoware-documentation/1.8.0/design/autoware-architecture-v1/components/sensing/data-types/radar-data/reference-implementations/data-message/),
the [radar pointcloud preprocessing design](https://autowarefoundation.github.io/autoware-documentation/1.8.0/design/autoware-architecture-v1/components/sensing/data-types/radar-data/radar-pointcloud-data/),
and the [`radar_scan_to_pointcloud2` converter](https://autowarefoundation.github.io/autoware_universe/main/sensing/autoware_radar_scan_to_pointcloud2/).
Those documents describe the sensor-frame message and conversion to
`sensor_msgs/msg/PointCloud2`; they do not replace Calibrex's frame-binding or
calibration-adoption gates.
