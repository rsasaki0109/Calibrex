# Camera-LiDAR capture-time semantics research note

## Motivation and primary references

A spinning LiDAR scan is a sequence of measurements, not an instantaneous
point cloud. Projecting every point into an image at the message timestamp can
turn platform motion into an apparent extrinsic error. Calibrex therefore
separates spatial calibration from the timestamp and deskew policy used to
construct each cross-modal observation.

The contract is informed by:

- Taylor and Nieto, *Motion-Based Calibration of Multimodal Sensor Extrinsics
  and Timing Offset Estimation*, IEEE Transactions on Robotics 32(5), 2016,
  DOI [`10.1109/TRO.2016.2596771`](https://doi.org/10.1109/TRO.2016.2596771),
  which makes temporal offset part of a motion-based multimodal calibration
  state and emphasizes excitation.
- Park et al., *Spatiotemporal Camera-LiDAR Calibration: A Targetless and
  Structureless Approach*, RA-L 5(2), 2020, DOI
  [`10.1109/LRA.2020.2969164`](https://doi.org/10.1109/LRA.2020.2969164),
  arXiv `2001.06175`, which estimates spatial and temporal registration jointly
  from continuous sensor motion.
- Lv et al., *Targetless Intrinsics and Extrinsic Calibration of Multiple
  LiDARs and Cameras with IMU using Continuous-Time Estimation*, arXiv
  `2501.02821`, which evaluates asynchronous LiDAR and camera observations on a
  continuous-time trajectory.
- OA-LICalib, *Observability-Aware Intrinsic and Extrinsic Calibration of
  LiDAR-IMU Systems*, arXiv `2205.03276`, which explicitly models individual
  LiDAR measurement timestamps and motion distortion in continuous time.
- Koide et al., *General, Single-shot, Target-less, and Automatic
  LiDAR-Camera Extrinsic Calibration Toolbox*, ICRA 2023, arXiv `2302.05094`,
  which provides the independent direct-registration baseline kept behind the
  Calibrex subprocess/precomputed-result adapter.

No implementation code from these projects is copied into Calibrex.

## Clock and point-time contract

Calibrex retains its global convention:

```text
sensor_time + dt_sensor = reference_time
```

Each LiDAR point additionally carries a signed capture offset relative to its
message stamp. The frontend must declare whether the offset is seconds or
integer nanoseconds and whether the stamp denotes scan start, midpoint, or
end. The label is provenance; it never silently changes the supplied numeric
offset. Camera exposure time is the deskew reference instant.

The native primitive transforms a point through `T_body_lidar`, applies the
inverse constant-body-twist motion from point capture to camera exposure, and
returns it to the LiDAR frame at the exposure instant. It records minimum and
maximum capture times, maximum deskew duration, point IDs, policy, body twist,
`T_body_lidar`, and method. Policy mappings reject unknown units, unknown stamp
references, and non-finite clock offsets before evidence is produced.

## Implemented fixed-twist specialization

The typed `CameraLidarTimedCapture` input contains one LiDAR message stamp
`t_L`, one camera exposure `t_C`, measured signed point offsets `delta_k`, a
local body twist `xi`, `T_body_lidar`, and optional cross-modal reference point
positions. With Calibrex's clock convention, point `k` is captured at

```text
t_k(dt_L) = t_L + delta_k + dt_L
Delta_k   = t_C - t_k(dt_L).
```

The fixed-twist specialization moves the return into the body frame, applies
the implemented first-order translational and exact constant-angular-velocity
motion over `Delta_k`, and moves it back into the LiDAR frame at camera time:

```text
p_B,k(t_C) = R(-omega Delta_k) p_B,k(t_k) - v Delta_k
p_L,k(t_C) = T_body_lidar^-1 p_B,k(t_C).
```

This is an independent, ROS-free specialization of the timing state used by
Taylor--Nieto and the continuous-time projection used by Park et al.; it is not
a reproduction of either complete joint spatial-temporal optimizer. When every
capture supplies reference positions, the native solver minimizes train point
RMSE over a bounded scalar `dt_L` grid followed by golden-section refinement.
The unchanged holdout is evaluated once. When references are absent, it applies
measured firing times and reports `evidence_only`; it cannot claim an estimated
clock offset.

Observability is not inferred from sample count. A centered finite difference
in `dt_L` produces the RMS point sensitivity in m/s. The scalar rank is one only
when that sensitivity meets the predeclared threshold. Six held-out controls
inject `-20`, `-10`, `-5`, `+5`, `+10`, and `+20` ms and must move held-out
points by the declared margin. Static or weak-motion captures return
`degenerate_motion`, rank zero, a `time_offset` weak direction, and materialized
failed controls rather than an empty probe list.

Synthetic truth uses independently generated reference positions and recovers
a `+17 ms` LiDAR offset to numerical tolerance with disjoint capture IDs. Tests
also cover evidence-only operation, static-motion degeneracy, input ordering,
mixed reference availability, invalid options, and measured firing-time endpoint
retention in the ROS 1 adapter.

## Public TIERS LidarsCali application

`camera_lidar_capture_time_config.yaml` reads the 7,178,470,523-byte TIERS
`LidarsCali.bag` directly. It pairs `/cam_1/color/image_raw` header timestamps
with `/velodyne_points`, decodes the measured PointCloud2 `time` field, and
derives local constant twists from `/vrpn_client_node/UWBTest/pose`. The raw bag
is authenticated by expected size and first-MiB digest before `data_verified`
is set; the evidence bundle additionally records and verifies the full SHA-256.

The public run retains 23 pairs: 18 train captures (2,304 points) and five
holdout captures (640 points). Mean firing-time span is 100.82 ms on train and
100.81 ms on holdout. Mean camera-to-LiDAR stamp delta is 11.52/8.48 ms and mean
point-to-exposure delta is 46.91/47.20 ms. The resulting deskew displacement is
0.91 mm train RMS and 1.90 mm holdout RMS; it is a motion-correction magnitude,
not an alignment error.

The predeclared minimum time sensitivity is `0.05 m/s`. Measured sensitivity is
only `0.01562 m/s`, giving rank zero, and none of the six signed controls exceeds
the 1 mm detection margin (the largest, at 20 ms, moves held-out points by about
0.557 mm). The public verdict is therefore **INCONCLUSIVE**, without weakening
the gate. TIERS does not publish cross-modal point identities for this sequence,
so `camera_lidar_time_offset_estimated=0`. VRPN `UWBTest` is also only a motion
proxy because its constant alignment to the `velodyne` message frame is
unpublished. The run
does not claim clock accuracy, spatial extrinsic accuracy, or metrology truth.
The config's identity `T_camera0_lidar0` only connects the required frame graph;
provenance marks it as an unused, unestimated placeholder and the backend removes
it from both output and candidate transform collections.

## Evidence boundary

Constant twist is a transparent baseline, not a claim that arbitrary vehicle
motion is constant during a scan. A production adapter may replace the motion
source with IMU integration, odometry interpolation, or a continuous-time
spline while preserving the same capture-time contract. If a dataset lacks
per-point times, Calibrex must report that limitation rather than inventing
timestamps from point order unless an explicit sensor firing model is
configured.

Primitive tests cover seconds/nanoseconds, sensor clock offset, translation,
rotation, non-identity LiDAR mounting, independent per-point capture times,
invalid policy values, and empty scans. The comparison layer evaluates external
and dataset-reference Camera-LiDAR transforms under the exact same frame split
and structured capture-time declaration.

## Common external-baseline comparison

KITTI dataset-reference, Calibrex-applied, and optional Koide-style external
transforms are projected over one shared seeded frame split. Each candidate
gets fresh projection ratio, image-edge alignment, and depth-edge alignment on
the same frame IDs. With only two public fixture frames, the splitter still
reserves one complete frame for holdout instead of silently producing an empty
holdout.

An external transform is not labelled independently validated unless its
adapter declares training isolation. Otherwise the same numbers remain useful
post-hoc diagnostics but receive WARN with an explicit leakage limitation.
Backend-native scores are not compared against Calibrex projection metrics.
For every non-reference candidate, Calibrex records the SE(3) translation and
rotation delta from the KITTI dataset reference together with candidate-minus-
reference holdout projection, edge, and depth-edge deltas. This exposes cases
where meaningfully different transforms are indistinguishable under weak image
evidence. These deltas remain WARN diagnostics because no metrology acceptance
threshold is declared. The provenance records each transform, split IDs,
pairwise deltas, structured capture-time evidence, and training-isolation
declaration.

The Koide-style adapter additionally materializes a typed external-tool
identity containing the tool name/version, source repository and commit,
SPDX license identifier, adapter version, command, result path, result size,
and result SHA-256. `koide_lidar_camera_provenance_complete` passes only when
the version, repository, commit, license, and output digest are all present.
An incomplete identity does not erase a readable transform, but it remains a
WARN provenance boundary. When an external transform is applied, the same
tool identity is copied into the schema-valid per-transform provenance instead
of being reduced to a generic adapter label.

KITTI Velodyne `.bin` payloads contain `(x, y, z, intensity)` but no per-point
capture offset. The configured policy is therefore validated and serialized,
while `per_point_times_available=false` and `deskew_applied=false` are explicit.
Calibrex does not infer firing times from array order. Projection remains a
rigid scan at the LiDAR message timestamp, and the limitation is a WARN metric.

The checked two-frame synthetic KITTI-shaped fixture materializes one train
and one holdout frame. Both the dataset reference and currently applied candidate
produce holdout projection ratio 1.0, edge alignment 0.5, and depth-edge
alignment 0.5. These equal values are not evidence of method equivalence: the
fixture is deliberately tiny and its calibration transform is identity. The
dataset reference is marked training-isolated; the applied candidate remains
WARN because its training isolation is undeclared. Bundle verification
recomputes SHA-256 for both images, both Velodyne scans, and two calibration
files (six raw inputs) with zero issues.
