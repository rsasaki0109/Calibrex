# Camera-LiDAR capture-time semantics research note

## Motivation and primary references

A spinning LiDAR scan is a sequence of measurements, not an instantaneous
point cloud. Projecting every point into an image at the message timestamp can
turn platform motion into an apparent extrinsic error. Calibrex therefore
separates spatial calibration from the timestamp and deskew policy used to
construct each cross-modal observation.

The contract is informed by:

- Park et al., *Spatiotemporal Camera-LiDAR Calibration: A Targetless and
  Structureless Approach*, arXiv `2001.06175`, which estimates spatial and
  temporal registration jointly from continuous sensor motion.
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

## Evidence boundary

Constant twist is a transparent baseline, not a claim that arbitrary vehicle
motion is constant during a scan. A production adapter may replace the motion
source with IMU integration, odometry interpolation, or a continuous-time
spline while preserving the same capture-time contract. If a dataset lacks
per-point times, Calibrex must report that limitation rather than inventing
timestamps from point order unless an explicit sensor firing model is
configured.

Synthetic tests cover seconds/nanoseconds, sensor clock offset, translation,
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
