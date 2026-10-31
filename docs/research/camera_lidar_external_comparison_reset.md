# Camera–LiDAR: freeze local tuning and establish an external comparison

The October 28 local geometry/restart result is a development checkpoint, not
SOTA evidence. Freeze it at `308885cd6d85eb9432586996eb421571ccc7b56e`
(results: `6b6379a9`). Stop changing the objective, restart rules and thresholds
on the two repeatedly inspected drives. The next milestone is an executed,
comparable external baseline and then a separately registered fresh-data test.

## Task equivalence comes first

The recent experiment counts nuisance-profiled rotation controls with translation
fixed. Koide and UniCalib estimate six-DoF extrinsics. Comparing those control
counts with their published pose errors would be invalid. A dataset reference
translation must not be silently supplied to just one six-DoF competitor.

The primary comparison will be **six-DoF calibration recovery**. Before execution,
lock one existing native Calibrex six-DoF estimator entry point and its complete
configuration, not the research profile checker. The recent checker remains a
separate calibration-verification experiment. Do not modify official competitors
to hold translation fixed and then label the result as their published method.

All methods must receive the same selected raw image/cloud/time intervals,
intrinsics, starting `T_camera_lidar`, perturbations and resource budget. Their
method-specific preprocessing is allowed but must be recorded and included in
end-to-end runtime. No manually clicked correspondences or reference correction
is allowed after seeing an evaluation case. When supplying an initial transform
to Koide's native initialization field, record that it is externally supplied,
not a run of its manual or learned initialization procedure.

## Baseline selection checked against primary sources

| Method | Role | Pinned source / current limitation |
| --- | --- | --- |
| [Koide direct visual LiDAR calibration](https://github.com/koide3/direct_visual_lidar_calibration) | Required training-free external baseline | Existing MIT execution lock: `02a0dc039f5509708f384be4ff3228e0ae09352d`; NID BFGS, 16 bins, default culling. Spent-data runtime smoke completed below; not a comparative score. |
| [UniCalib](https://github.com/han-15/UniCalib) ([WACV 2026 paper](https://openaccess.thecvf.com/content/WACV2026/html/Han_UniCalib_Targetless_LiDAR-camera_Calibration_via_Probabilistic_Flow_on_Unified_Depth_WACV_2026_paper.html)) | Required learned baseline, reported separately | Source reviewed at `0336aa6c4aef4a25db8a51daef8ef86f813dfcef`. Official instructions require a CUDA visibility extension and monocular-depth preprocessing. Checkpoints, depth-provider identity and training-data overlap are not locked; this workspace exposes no NVIDIA device. Not executed. |
| [TLC-Calib](https://github.com/SNU-VGILab/TLC-Calib) | Additional 2026 comparison candidate | Source reviewed at `ceb723923a40ed5a031b67cf8c6d71ccb0ca21b6`. Neural Gaussian approach with CUDA dependencies. Confirm license, dependencies and comparable input assumptions before inclusion; not executed. |

Koide alone is a useful baseline, not proof of leadership over current methods.
Do not substitute paper-reported numbers for results from the same inputs.
Missing required baselines block the comparative claim, not merely its footnote.

## Fresh-data reservation — not an approved evaluation protocol

No fresh sensor payload was downloaded or read in this reset. Existing exposure
records mark all 55 eligible KITTI raw drives and KITTI-360 0000, 0002 and 0003
as spent. Raw/odometry aliases and learned-model training exposure must be audited
before declaring anything independent.

Reserve KITTI-360 0004–0007, 0009 and 0010 as **candidates only**, subject to the
existing exposure audit. Select intervals from metadata by a committed rule,
not by looking at their images or motion quality. KITTI-360 provides one family
and one rig; multiple sequences do not satisfy the broader cross-family claim.
A second non-KITTI family/rig must be identified, licensed, accessible and cleared
for exposure before any broad claim. The older research plan's requirements
(including the spent-drive development coverage) remain unmet and are not waived.

Before any fresh read, commit a separate executable preregistration containing:

- Exact sequence/interval inventory, aliases, reference hashes and failure policy.
- Source, configuration, checkpoint and container identities for every method.
- A common six-DoF initialization/perturbation schedule and random seeds.
- Full rotation geodesic error and translation norm; report median, p95 and every
  failure. The existing strict `<0.5 deg / <0.20 m` recovery event can be reused
  only as an explicitly scoped benchmark definition, not a relaxed earlier gate.
- Failure-aware recovery rate over **all** scheduled cases; crash, timeout and
  invalid output count as failures, never disappear from accuracy summaries.
- Recording-level paired inference (not frame-level pseudo-replication), with a
  fixed resampling seed and 95% interval for recovery-rate improvement against
  each required comparator. A superiority claim requires its lower bound above
  zero and the existing absolute quality/coverage gates to pass; numerical gates
  must be derived and frozen using the required development evidence.
- End-to-end time, peak memory, hardware and preprocessing cost. Different CPU/GPU
  profiles are separate tracks, not a speedup claim from unlike hardware.

**Status: blocked draft.** The six-DoF native entry point, shared exporter,
point-time convention, learned checkpoint isolation, second family/rig and
complete development-derived gates are not yet locked. Do not consume the fresh
candidates to fill those gaps. This document is not permission to run an audit.

## What was actually executed

A three-frame smoke input was copied byte-for-byte from the already-spent KITTI
0005 ROS 2 capture: three Image, CameraInfo and PointCloud2 messages each. The
official Koide preprocessor produced a cloud/image bundle. This uses only the
first three messages, not a representative evaluation interval.

The official image digest is
`sha256:f7363cc3deadc2419f3527a7b9ec7bdabaa3031fd8e1b9c6cc46666ae1301bbb`.
Two normal pulls failed while the managed Docker `vfs` driver expanded layers.
After retaining logs, the same pinned image was exported with `crane v0.22.1`
and imported as a flattened local runtime. The source checkout inside the runtime
matches the pinned commit. The derived runtime is not claimed to have the original
image digest. Reproducible old pytest scratch fixtures and clean detached source
worktrees were removed for space; committed code, experiment inputs and reports
were retained.

The preprocessor warns about negative per-point timestamps from the existing
azimuth-derived time convention. This must be resolved on spent data before a
fair benchmark; successful parsing is not validation of temporal correctness.
The first calibration attempt failed because even `--background` initializes
GLFW and needs a display. Xvfb/software Mesa was added without modifying Koide
source. Ubuntu package indexes were available; the unneeded ROS apt repository
returned 403 during update, so display packages were installed using the already
retrieved Ubuntu indexes. TLS checks and the network proxy were retained.

A fixed diagnostic seed applies Euler xyz `[1,-1,0.5]` degrees and translation
`[0.1,-0.05,0.05]` metres to the dataset reference. It is written as the inverse
`T_lidar_camera` quaternion/translation required by the native JSON format.
This reference-derived runtime smoke is **not** a fair head-to-head result,
a fresh-data audit, a convergence guarantee, or evidence of production readiness.


The Xvfb run exited **0** and produced a finite rigid transform. Diagnostic
reference disagreement changed from **1.503 to 2.227 degrees**, and from
**0.12247 to 0.12230 metres**. Rotation got worse: successful execution is not
successful recovery. Wolfe line-search warnings are retained. No representative
accuracy or end-to-end timing claim follows from this smoke.

[Machine-readable report](../assets/external_comparison_reset_20261028/smoke-report.json)
and its [schema](../assets/external_comparison_reset_20261028/smoke-report.schema.json)
include source identity, input/message hashes, seed/output transforms, runtime
logs and artifact hashes. The runtime container was observed exited with code 0,
not OOM-killed, starting at `2026-10-28T13:32:51.241309434Z` and finishing at
`2026-10-28T13:33:02.78282884Z`; that interval includes display-package installation
and excludes preprocessing, so it is not calibration benchmark time.

The executed preprocessing command, inside the flattened official runtime with
ROS Humble and the workspace setup sourced, was:

```sh
ros2 run direct_visual_lidar_calibration preprocess /input /output -a -d \
  --camera_model plumb_bob \
  --camera_intrinsics 721.5377,721.5377,609.5593,172.854 \
  --camera_distortion_coeffs 0,0,0,0,0
```

After copying the fixed seed into `calib.json`, the successful command was:

```sh
LIBGL_ALWAYS_SOFTWARE=1 timeout 180s xvfb-run -a /ros_entrypoint.sh \
  ros2 run direct_visual_lidar_calibration calibrate /data --background --auto_quit
```

Both stages used a 4-CPU / 4-GB container limit. Preprocessing mounted input
read-only and disabled networking; calibration used the prepared bundle as a
writable mount and retained networking from the display-package installation.

## Next deliverable

Build one shared spent-data six-DoF comparison using the existing native estimator
and this official Koide execution path. First reconcile point-time semantics and
lock matching inputs, initialization and failure accounting. Then expand to the
required development coverage and run the learned baseline on suitable hardware.
Do not resume two-drive objective tuning or read fresh candidates in place of
this deliverable. If Calibrex fails that comparison, use the measured failure to
choose a substantive estimator change before preregistering fresh evaluation.


### Execution follow-up

The [shared-bundle six-DoF comparison](camera_lidar_shared_koide_bundle.md) now
preserves point acquisition times when adapting KITTI to Koide and executes both
Koide and the existing native Pandey MI solver on identical preprocessed inputs
and seed. Both miss the scoped recovery threshold. This is optimizer-only
execution evidence; the independent D2D/end-to-end comparison remains pending.
