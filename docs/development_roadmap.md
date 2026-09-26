# Development Roadmap

This document is the current development decision for Calibrex as of
2026-09-27. Earlier revisions were dated 2026-08-19 and 2026-08-20. It
reconciles the implemented code, the older ADR direction, public-data evidence,
and relevant research/OSS. It is a planning inventory, not a claim that every
listed method is production-ready.
[ADR 0010](adr/0010-v0-5-reconcile-v0-4-and-operational-trust.md) records why
the v0.4 pillars close and what v0.5 carries forward.

**Strategic direction (2026-08-19):** Calibrex is positioned as a practical
calibration tool first, and a research evidence framework second.  New work is
evaluated against two questions: "Does this let a user with a real rosbag
produce a trusted result more easily?" and "Does this let a developer trust the
result more deeply?"  Both matter, but user-facing friction comes first.

Calibrex remains an evidence framework rather than a solver collection. A new
method is complete only when its inputs and results are typed and
schema-validatable, its provenance is recorded, its observability limits are
reported, and unchanged holdout data and known-bad controls can falsify it.

## Maturity vocabulary

| Label | Meaning |
| --- | --- |
| Alpha | Native path is CLI-wired, tested, and has at least one meaningful public-data workflow. |
| Experimental | Native path exists, but public evidence is limited, FAIL, or INCONCLUSIVE. |
| Evidence only | Calibrex evaluates a supplied candidate but does not natively solve the complete calibration. |
| Adapter | An optional external or precomputed tool boundary exists; it is not part of the default core. |
| Research | Typed solver/factor work exists, but it is not yet a generally supported end-user path. |

`PASS`, `WARN`, `FAIL`, and `INCONCLUSIVE` describe one evidence run. They do
not by themselves change method maturity. In particular, an honest public-data
`FAIL` is stronger research evidence than a synthetic-only green example.

## Reconciliation with existing direction

[ADR 0008](adr/0008-v0-4-scale-and-absolute-time.md) is still marked
**Proposed** and has been partially overtaken by implementation:

- Its Radar pillar said Radar lacked a protocol, holdout, probes, and gates.
  The repository now contains candidate consistency, robust ego velocity,
  trajectory rotation/yaw, lever-arm/time, joint spatiotemporal solvers, and a
  nuScenes adapter. The remaining gap is strong public excitation and a
  conclusive full-system Radar result, not absence of native algorithms.
- Its full-scale KITTI Camera-LiDAR pillar remains open. The committed runnable
  demo is deliberately a small synthetic KITTI-shaped fixture, so it cannot
  establish real full-scale perturbation power.
- Its absolute capture-time pillar remains directionally valid. Per-point
  capture-time evidence exists, but TIERS real-data evidence remains
  INCONCLUSIVE and message-time versus capture-time semantics still require an
  end-to-end acceptance demonstration.
- Later work substantially expanded planar-board Camera-LiDAR, hand-eye,
  robot-world/hand-eye, registration comparison, targetless Camera-LiDAR, and
  backend-neutral joint optimization beyond the ADR 0008 snapshot.

The portfolio below, rather than the older ADR context section, is the current
source for deciding what to build next. ADRs remain useful records of why work
was started and are not silently rewritten as if they had predicted later
implementation.

## Current calibration portfolio

All native rows use independent Apache-2.0 Calibrex code unless the boundary
column says otherwise. “Synthetic” means deterministic recovery and
falsification tests exist, but no independent public sensor capture establishes
accuracy.

| Calibration method | Delivery | Evidence implemented | Public-data evidence | Maturity | License boundary |
| --- | --- | --- | --- | --- | --- |
| Fixed-trajectory LiDAR point-to-plane | Native CLI backend | Spatial-block holdout, 6-DoF probes, spectrum | A2D2 and Livox workflows; no metrology truth | Alpha | Core Apache-2.0; A2D2 CC BY-ND 4.0; Livox upstream terms |
| Solid-state LiDAR acquisition contract and Livox evidence | Native schema/evaluation path | Scan-pattern, point-time, integration, intrinsic, temperature metadata; distance/FOV bins; holdout and known-bad controls | Livox Horizon PCD workflow; PCD pair is limited non-independent holdout evidence | Alpha | Core Apache-2.0; sensor/dataset terms remain external |
| Online motion-compensated LiDAR point-to-plane | Native streaming pipeline | Rolling holdout, adoption gates, trajectory and deskew evidence | A2D2, Livox, and TIERS workflows | Alpha | Core Apache-2.0; dataset terms remain external |
| Robust point-to-point ICP | Native solver/comparison backend | Voxel holdout, fresh rematching, curvature, Jaccard, multi-start | Livox comparison is an honest FAIL under low overlap | Experimental | Core Apache-2.0 |
| Chen-Medioni point-to-plane ICP | Native solver | Shared registration holdout and observability diagnostics | Synthetic/unit evidence; no dedicated public accuracy claim | Research | Core Apache-2.0 |
| Open3D Generalized ICP | Optional adapter | Common Calibrex split, rematching and curvature metrics | Pinned Open3D Livox integration test when data/dependency are installed | Adapter | Open3D MIT; optional dependency |
| PCL/Autoware NDT | Subprocess or precomputed adapter | Common split plus explicit train-isolation declaration | No committed conclusive public comparison | Adapter | PCL permissive; TIER IV CalibrationTools is GPL-3.0 and subprocess-only |
| Open3D RGB-D SLAC | Optional adapter | Calibrex result/report evidence | Runnable example; not a native accuracy claim | Adapter | Open3D MIT; optional dependency |
| Backend-neutral joint SLAC | Native graph and typed Schur LM | Group holdout, robust solve, per-block probes, joint spectrum | A2D2 frontend and TUM multi-window work | Experimental | Core Apache-2.0 |
| TUM RGB-D joint trajectory/extrinsic/depth calibration | Native CLI backend | Frame holdout, shared-variable probes, cross-window transfer, reassociation stability | Real TUM `fr1/xyz`; primary public result honestly FAILs transfer/reference gates | Experimental | Core Apache-2.0; TUM dataset terms |
| Planar-board Camera-LiDAR plane alignment | Native CLI backend | Capture holdout, normal/offset closure, normal-span rank, 12 probes | ACFR 40-pose real data: method gates PASS | Alpha | Core Apache-2.0; ACFR source Apache-2.0 |
| Planar-board Camera-LiDAR point+plane | Native CLI backend | Shared capture split, joint spectrum, 12 probes | ACFR real data: method gates PASS | Alpha | Core Apache-2.0; external feature extraction |
| Horn point-only Camera-LiDAR | Native CLI backend | Shared split, eigengap, 6D spectrum, 12 probes | ACFR real data: method gates PASS | Alpha | Core Apache-2.0; external feature extraction |
| Planar-board line+plane | Native solver | Rotation/translation spectra and edge-angle gate | Synthetic/unit evidence; no CLI-wired public extractor | Research | Core Apache-2.0; extractor must remain an adapter |
| Pandey mutual-information Camera-LiDAR | Native CLI backend | Frame holdout, 12 probes, objective curvature | Real A2D2 image/reflectivity run FAILs; inputs are already camera-registered and are not independent accuracy evidence | Experimental | Core Apache-2.0; A2D2 CC BY-ND 4.0 |
| Levinson-Thrun online edge Camera-LiDAR | Native CLI backend | Temporal holdout, online window diagnostics, curvature | Two-pair A2D2 run is WARN/INCONCLUSIVE | Experimental | Core Apache-2.0; A2D2 CC BY-ND 4.0 |
| Koide-style direct visual-LiDAR | Executable/precomputed adapter | Generic external-run artifact, input readiness, tool identity, result digest, common Camera-LiDAR evidence | Boundary is implemented; no maintained full-scale comparison result | Adapter | Upstream toolbox MIT; ROS/PCL/GTSAM/Ceres remain external |
| Kalibr Camera-IMU/camera chain | YAML importer plus native evaluator | Generic external-run artifact with transforms, intrinsics, time shifts, digests, conventions, and isolation declaration. Independent evidence (`external-run evaluate-camera-imu`) checks rotation and time shift with an angular-rate holdout, a train-only gyro bias, excitation gates, and eight signed controls, and blocks leakage | Synthetic Kalibr-producer proof only: truth PASS, 3°/20 ms errors FAIL, yaw-only INCONCLUSIVE. No public Kalibr run has been evaluated, and lever arm is not evaluated | Adapter | Top-level BSD-4-Clause; ROS/Kalibr runtime stays external |
| Per-point Camera-LiDAR capture time | Native solver plus TIERS adapter | Disjoint capture holdout, six time controls, timing observability | Synthetic 17 ms recovery; TIERS real-data result INCONCLUSIVE | Experimental | Core Apache-2.0; TIERS dataset terms |
| Anchored moving-platform time offset | Native online evidence | Injection controls, adapted/anchored comparison, separability row | TIERS shared-clock/injection evidence; absolute convention is not yet closed | Evidence only | Core Apache-2.0 |
| LiDAR-IMU rotation consistency | Native evaluator | Angular-rate holdout, gravity support, per-axis excitation, bad-rotation probes | TIERS Indoor02 evidence; no translation/time native solve | Evidence only | Core Apache-2.0 |
| Radar Doppler candidate consistency | Native evaluator | Frame holdout, yaw controls, static/excitation gates | nuScenes mini path; often INCONCLUSIVE | Evidence only | Core Apache-2.0; nuScenes non-commercial terms |
| Robust Radar ego velocity | Native solver | Huber IRLS, LOS rank/condition, outlier tests | Used by Radar pipeline; automotive geometry may be planar-only | Experimental | Core Apache-2.0 |
| Radar-to-trajectory yaw/rotation | Native solvers | Deterministic holdout, direction diversity, signed rotation controls | nuScenes adapter; limited excitation prevents a general PASS claim | Experimental | Core Apache-2.0; nuScenes terms |
| Radar lever arm and clock offset | Native solver | Profiled time grid, lever-arm spectrum, holdout, eight controls | nuScenes path is intentionally INCONCLUSIVE when excitation is weak | Experimental | Core Apache-2.0; nuScenes terms |
| Radar joint spatiotemporal calibration | Native solver | Joint observability, holdout and control composition | Synthetic and adapter-level evidence; no conclusive public full-system result | Research | Core Apache-2.0 |
| Motion hand-eye `AX=XB` | Eight native baselines: Park-Martin, Tsai-Lenz, Shiu-Ahmad, Daniilidis, Horaud-Dornaika closed/nonlinear, Chou-Kamel, Andreff | Common motion split, closure, spectra/nullspaces, 12 probes | ETHZ real robot-arm comparison remains INCONCLUSIVE because several relative-motion methods lack probe power | Alpha | Independent core implementations; ETHZ data non-commercial and pinned upstream code BSD-3-Clause |
| Robot-world/hand-eye `AX=YB` or `AX=ZB` | Five native baselines: Shah, Li-Wang-Wu, Dornaika-Horaud closed/nonlinear, Zhuang-Roth-Sudhakar | Absolute-pose holdout, joint rank/gaps, 24 controls | ETHZ real-data method-specific gates include passing cases, while combined run stays INCONCLUSIVE | Alpha | Independent core implementations; ETHZ dataset boundary |

### Portfolio gaps that matter

1. **Evidence scale:** Camera-LiDAR has an opt-in full-scale KITTI falsification
   pair (`@pytest.mark.kitti`), with a vendor reference PASS and a declared
   known-bad FAIL. The remaining gap is to publish that result as a maintained
   benchmark artifact, not only as an opt-in test.
2. **Common external execution adoption:** the external-run artifact is proven
   by Koide (fail-closed handoff, readiness, and real-pilot finalization),
   Kalibr (independent Camera--IMU holdout evidence), and iKalibr (import).
   RIs-Calib, Open3D, and NDT should migrate to it incrementally. Kalibr
   lever-arm evidence and an evaluated public Kalibr run are still missing.
3. **Empirical uncertainty:** `slac.empirical_se3_uncertainty/v0.1` is
   CLI-wired, with synthetic and opt-in KITTI ground-truth integration tests
   and FeatDepth-gated correspondences. Maintained provider artifacts and
   portfolio evidence are still unpublished.
4. **LiDAR-IMU user path:** every continuous-time IMU factor exists. These are
   rotation with gyro bias, lever arm, clock offset, accelerometer
   bias/gravity, diagonal intrinsics, and sliding-window marginalization. Each
   is only a separate synthetic `recover-*` command, though. No single user
   command runs them jointly on a real capture.
5. **Absolute capture time:** `capture_time_reference` exists, but ADR 0008
   Pillar 3 has no end-to-end acceptance run, and TIERS real-data evidence
   stays INCONCLUSIVE.
6. **Lifecycle in CI:** the append-only lifecycle registry and the Calibration
   CI gate exist separately. Candidate/baseline comparisons from pull requests
   are not yet recorded as registry events with rollback records.
7. **Default suite health:** fixed on 2026-09-27. Three legacy result
   fixtures lacked `run.provenance` and now use `build_result_provenance`. On
   Windows `core.autocrlf` checkouts, digest-bound JSON fixtures were rewritten
   to CRLF, which broke their declared SHA-256. `.gitattributes` now pins
   `*.json` to `eol=lf`.

## Research and OSS map

These projects are references or adapter targets, not sources to copy into the
core.

| Work | Relevant capability | Roadmap use | Boundary decision |
| --- | --- | --- | --- |
| [iKalibr, T-RO 2025](https://arxiv.org/abs/2407.11420) / [OSS](https://github.com/Unsigned-Long/iKalibr) | Unified targetless Camera/LiDAR/Radar/RGB-D/IMU spatial and temporal calibration; continuous-time refinement and rolling-shutter readout | Primary external comparison and continuous-time design reference | Adapter/container first; audit the top-level and bundled third-party licenses before redistribution |
| [Targetless multi-LiDAR/multi-camera/IMU continuous-time calibration, 2025](https://arxiv.org/abs/2501.02821) | IMU spline, LiDAR voxel-map BA, visual BA, intrinsics, extrinsics, and time offsets | Factor decomposition and multi-sensor benchmark design | Paper reference; do not reproduce learned feature dependencies in the core |
| [RIs-Calib](https://arxiv.org/abs/2408.02444) / [OSS](https://github.com/Unsigned-Long/RIs-Calib) | Continuous-time multi-Radar/multi-IMU spatiotemporal calibration | Radar external baseline and message adapter target | MIT upstream; optional executable/container |
| [OA-LICalib](https://github.com/APRIL-ZJU/OA-LICalib) | Observability-aware continuous-time LiDAR-IMU intrinsic/extrinsic calibration | Excitation and observability reference for a future native solver | GPL-3.0; paper/reference or subprocess only |
| [Koide direct visual-LiDAR](https://github.com/koide3/direct_visual_lidar_calibration) | Automatic targetless single-shot Camera-LiDAR calibration for varied sensor models | First real generic external-run adapter | MIT upstream; ROS and native dependencies stay optional |
| [García-Gómez et al., solid-state geometric calibration](https://www.mdpi.com/1424-8220/20/10/2898) | Device-specific angular distortion and variable angular resolution | Motivation for explicit scan architecture/FOV metadata and intrinsic model provenance | Paper reference; no source copied into core |
| [Huang et al., unified spinning/solid-state intrinsic calibration](https://arxiv.org/abs/2012.03321) | Unified geometric model and tetrahedral target arrangement | Intrinsic-calibration adapter/protocol reference | Paper reference; native implementation remains independent |
| [PBACalib](https://fusionportable.github.io/files/PBACalib.pdf) | Targetless plane-constrained LiDAR-camera bundle adjustment on Livox data | External camera-LiDAR comparison target for non-repetitive scans | Paper/reference; no source copied into core |
| [FAST-Calib](https://arxiv.org/abs/2507.17210) / [OSS](https://github.com/hku-mars/FAST-Calib) | Target-based solid-state/mechanical LiDAR-camera calibration | Known-bad and runtime comparison reference | GPL-2.0; subprocess/reference only |
| [Mints et al., online solid-state extrinsic calibration](https://doi.org/10.3390/s24072155) | FPFH/FGR extrinsic calibration across Blickfeld, Cepton, and Livox | Practical external baseline; explicitly separates manufacturer intrinsics from estimated extrinsics | Paper/reference; no GPL code copied |
| [Livox camera-LiDAR calibration](https://github.com/Livox-SDK/livox_camera_lidar_calibration) | Board-corner camera-LiDAR calibration for Livox sensors | Permissive adapter target and public workflow cross-check | MIT upstream; runtime stays optional |
| [Kalibr](https://github.com/ethz-asl/kalibr) | Camera chain and Camera-IMU spatial/temporal calibration | External-run contract, YAML import/export, reference comparison | Permissive BSD-style top-level license; ROS/toolchain stays external |
| [OpenCalib](https://github.com/PJLab-ADG/SensorsCalibration) | Multi-sensor autonomous-driving toolbox | Result-format and benchmark comparison target | Apache-2.0 upstream; still prefer an adapter over copied ecosystem code |
| [TIER IV CalibrationTools](https://github.com/tier4/CalibrationTools) | ROS 2/Autoware Camera/LiDAR/Radar/ground calibration workflows | Autoware integration and export validation | GPL-3.0; subprocess/container boundary only |
| [Uncertainty-aware online extrinsic calibration, 2025](https://arxiv.org/abs/2501.06878) | Conformal prediction intervals evaluated by coverage and width on KITTI and DSEC | Motivation for solver-neutral empirical coverage evidence | Implement statistical evidence independently; neural estimator is optional and external |

## Priority order

### Adoption track — Calibration CI

This track packages existing evidence capabilities into a low-friction public
workflow without weakening the research priorities below:

1. extend `calibrex doctor` into a schema-versioned environment and dataset
   readiness artifact with path-based type inference, provisional quality
   evidence, workflow suggestions, and provenance; **done**
   (`slac.environment_readiness/v0.1`);
2. publish a reusable GitHub Action that validates, compares, and assesses
   calibration artifacts in pull requests; **done** (`examples/ci/`,
   `action.yml`, `docs/tutorials/calibration_ci.md`);
3. complete the generic external-run contract and prove it with Koide and
   Kalibr producers; **done** (Koide fail-closed real-pilot handoff;
   `calibrex external-run evaluate-camera-imu` for Kalibr);
4. finish the full-scale KITTI falsification benchmark; **done** (opt-in
   `@pytest.mark.kitti` integration test); and
5. cut an installable release only after clean-wheel, schema-drift, and
   quickstart checks pass; **deferred** (maintainer decision 2026-09-27: no
   release for now; readiness is tracked in Next issue 1).

The adoption track reuses the core models and adapters. It must not add ROS,
GPL, visualization-server, or external-solver dependencies to `src/calibrex`.

### P0 — Evidence and integration hardening

The three committed P0 issues below are **implemented** (2026-08-20). The next
tranche focuses on continuous-time factor integration and lifecycle monitoring.

### P1 — Continuous-time foundation — implemented

`ContinuousTimeTrajectoryContract` (`slac.continuous_time_trajectory/v0.1`),
SE(3) manifold Jacobians, a sparse GN/LM fitter, native LiDAR point-to-plane
factors, IMU gyro pre-integration (rotation plus shared gyro bias), IMU
lever-arm factors, a shared IMU clock offset, accelerometer bias/gravity, and
sliding-window Schur marginalization against the knot path are implemented and
CLI-wired. Diagonal IMU intrinsics completed the factor set on 2026-08-20.

### P2 — Native LiDAR-IMU and sliding-window evidence — factors implemented; joint user path open

Build in stages: rotation plus gyro bias, lever arm, clock offset,
accelerometer bias/gravity, then optional intrinsics. Every stage requires
disjoint temporal holdout, per-axis excitation, separability diagnostics, and
injected signed controls. Add marginalization only with explicit gauge and
consistency evidence.

Every stage now exists as a synthetic recovery with holdout and a signed
control. What remains is a single joint command on real data (Next issue 2).

### P3 — Calibration lifecycle

Represent drift detection and adoption as schema-valid events with incumbent,
candidate, evidence window, decision, and rollback provenance. A weakly
observable window must not update the installed transform.

**Delivered:** `decide_calibration_lifecycle_window()` adoption gates,
synthetic replay artifact `slac.calibration_lifecycle/v0.1`, CLI
`calibrex lifecycle simulate`. On 2026-08-24 the append-only lifecycle
registry v0.2 added physical sensor identity, input digests, operator
provenance, and a hash-chained event log.

## Completed since the 2026-08-20 revision

- **Operational trust platform** (2026-08-24) — capture manifest intake
  adapters (`calibrex capture inspect`); digest-bound raw calibration replay
  (`calibrex replay`); the append-only lifecycle registry
  (`calibrex lifecycle`); multi-LiDAR, camera--IMU, and radar
  service-replacement gates with READY/HOLD decisions; `radar_msgs/RadarScan`
  intake; and hardened Autoware promotion and smoke gates. See
  [operational KPIs](reference/operational_kpis.md).
- **Fail-closed Koide real-pilot handoff** — `koide-real-plan`,
  `koide-real-verify`, and `koide-real-finalize` never start Docker, ROS, or
  Koide, and cannot make an official claim from synthetic evidence. The
  checked-in A2D2 pilot is intentionally `BLOCKED`.
- **Required result provenance** (2026-08-25) — `slac.result.provenance/v0.1`
  on every pipeline result, enforced by `CalibrationResult.save()` and
  `calibrex validate --kind result`
  ([result provenance](reference/result_provenance.md)).
- **Independent Camera--IMU evidence for external runs** (2026-09-27) —
  `calibrex external-run evaluate-camera-imu` evaluates an imported
  `T_cam_imu` and `timeshift_cam_imu` on a separate
  `slac.camera_imu_motion_recording/v0.1`, producing
  `slac.external_camera_imu_evidence/v0.1`. It uses a middle-block holdout
  with guard gaps, a train-only gyro bias, excitation gates, and eight signed
  rotation/time controls. It blocks a recording that matches a fitting input.
  `calibrex external-run synthesize-camera-imu-fixture` provides the
  nonphysical Kalibr-producer proof
  ([external adapters](reference/external_adapters.md)).

## Next implementation issues

Revised 2026-09-27 and ordered by the practical-tool criterion. Each issue
names the portfolio gap it closes. Large public datasets stay outside the
repository and are read from environment-variable paths by opt-in tests.

### 1. Release readiness (the release itself is deferred)

**Gap:** adoption track item 5.

**Outcome:** the tree stays release-ready so that a GitHub Release wheel can
be cut later without catch-up work. As of 2026-09-27, the maintainer has
decided not to cut a release for now.

**Acceptance conditions:**

- The default test suite (`-m "not kitti"`) is green on Linux and on a Windows
  `core.autocrlf` checkout. **Done** on 2026-09-27 (portfolio gap 7).
- The clean-wheel install smoke, the schema-drift test
  (`test_static_schema_files_match_generated_schemas`), and the quickstart in
  `docs/tutorials/your_own_data.md` pass from the built wheel.
- When a release is decided, the CHANGELOG `Unreleased` section is
  consolidated under the new version and ADR 0010 is accepted.

### 2. Joint LiDAR-IMU calibration command

**Gap:** portfolio gap 4.

**Outcome:** `calibrex lidar-imu calibrate` runs rotation, gyro bias, lever
arm, clock offset, and accelerometer bias/gravity jointly on the knot path from
a real capture, and reports per-parameter holdout and signed controls in one
schema-valid artifact.

**Acceptance conditions:**

- Synthetic joint recovery meets the budgets of the individual `recover-*`
  commands, each parameter keeps its signed control, and parameter
  separability is reported.
- An opt-in real-data run on the existing TIERS Indoor02 capture reports
  honestly (PASS, WARN, FAIL, or INCONCLUSIVE) without retuned gates. It needs
  no new download.

### 3. Lifecycle registry events from Calibration CI

**Gap:** portfolio gap 6.

**Outcome:** a Calibration CI comparison can append a candidate/adoption or
rollback event to a lifecycle registry, bound to the CI artifact digests.

**Acceptance conditions:**

- An example workflow and an integration test cover adopt, hold, and rollback
  paths, and a weakly observable candidate never mutates the head.
- There is no in-place overwrite, and registry verification stays green after
  every path.

### 4. Absolute capture-time acceptance (ADR 0008 Pillar 3)

**Gap:** portfolio gap 5.

**Outcome:** an end-to-end demonstration that message time and capture time
converge under the declared `capture_time_reference`.

**Acceptance conditions:**

- A synthetic shared-clock selftest puts the anchored baseline within ±10 ms,
  and the injection differential stays exact.
- The TIERS real-data result is re-reported under the declared convention,
  whatever its status.

### 5. Kalibr lever-arm evidence and a public Kalibr run

**Gap:** portfolio gap 2.

**Outcome:** extend the external Camera--IMU evidence with
accelerometer-based `T_cam_imu` translation checks and signed lever-arm
controls, then evaluate one public Kalibr camchain on a separately captured
recording.

**Acceptance conditions:**

- The synthetic truth passes and a ±5 cm lever-arm error fails.
- The public run is opt-in, reports honestly, and records the dataset license
  boundary. Its data live outside the repository.

### 6. Planar-board line+plane CLI wiring

**Gap:** the Research row "Planar-board line+plane".

**Outcome:** a CLI backend that uses an adapter-side edge extractor, with the
shared capture split and 12 probes, evaluated on the ACFR planar-board data
already used by the plane and point+plane rows.

## Completed in the 2026-08-20 revision (2026-08-19 baseline)

- **`slac.environment_readiness/v0.1` for `calibrex doctor`** — schema-valid
  artifact with Python/Calibrex/optional dependency versions, dataset path
  checks, `workflow_suggestions` pointing at `examples/sensor_templates/`, and
  provenance; validated in unit tests and release smoke.
- **Full-scale KITTI Camera-LiDAR falsification benchmark** — frame-graph
  candidate fix, dataset reference consistency gate, `@pytest.mark.kitti` opt-in
  integration test (vendor reference PASS, known-bad FAIL), and explicit
  `falsification_passed` CLI flag.
- **Empirical SE(3) uncertainty — ground-truth workflow** — correspondence
  bootstrap, pixel mean jitter, degenerate-spread policy; synthetic CLI
  integration test and opt-in official KITTI test (`CALIBREX_KITTI_RAW_0005` +
  `CALIBREX_KITTI_DEPTH_PROVIDER`).
- **Independent FeatDepth correspondence for empirical uncertainty** —
  `build_featdepth_correspondence_from_problem()` with depth-provider identity
  and FeatDepth depth gating; opt-in KITTI ground-truth test updated.
- **Empirical SE(3) uncertainty — public stability-only workflow** (`--stability-only`
  flag, integration test on a KITTI-shaped synthetic fixture, `INCONCLUSIVE` policy
  with honest reporting). Schema `slac.empirical_se3_uncertainty/v0.1`.
- **iKalibr external-run adapter** — pure-Python importer for
  `ikalibr-prog-result.yaml`/`.json`; covers cereal SO3d/Vector3d formats, time
  offsets, digest binding, and `BSD-3-Clause` boundary.
- **rosbag2 support in `native_lidar_point_to_plane`** — `.db3` and `.mcap`
  containers are now accepted without ROS installation.
- **Sensor templates** — four ready-to-edit configs in `examples/sensor_templates/`
  covering Velodyne VLP-16 × 2 (rosbag1/2), Ouster OS1 × 2, and
  spinning LiDAR + camera (planar-board).
- **Quickstart tutorial** — `docs/tutorials/your_own_data.md` covering the
  end-to-end flow from bag recording to result interpretation.
- **Continuous-time sliding-window marginalization** — overlapping knot-window
  fits with Schur-complement priors, batch consistency checks, holdout, and a
  skip-marginalization control (`calibrex trajectory recover-sliding-window`,
  `slac.continuous_time_sliding_window/v0.1`).
- **Continuous-time IMU accelerometer bias and gravity** — specific-force
  residuals ``a_kin + b - R^T g`` on the screw-linear knot path, Jacobian
  tests, synthetic recovery with disjoint temporal holdout and a +0.3 m/s^2 z
  known-bad accel-bias control (`calibrex trajectory recover-imu-accel-bias`,
  `slac.continuous_time_imu_accel_bias/v0.1`).
- **Continuous-time IMU diagonal intrinsics** — per-axis gyro and
  accelerometer scale factors on the screw-linear knot path, Jacobian tests,
  synthetic recovery with disjoint temporal holdout and a +0.05 z gyro-scale
  known-bad control (`calibrex trajectory recover-imu-intrinsics`,
  `slac.continuous_time_imu_intrinsics/v0.1`).
- **Calibration lifecycle adoption and rollback** — schema-valid incumbent,
  candidate, evidence-window, and rollback events with holdout/observability
  gates; weakly observable windows leave the installed transform unchanged
  (`calibrex lifecycle simulate`, `slac.calibration_lifecycle/v0.1`).
- **Continuous-time IMU clock offset** — a shared scalar delay mapping IMU
  timestamps onto the screw-linear knot path, Jacobian tests, synthetic
  recovery with disjoint temporal holdout and a +0.02 s known-bad control
  (`calibrex trajectory recover-imu-clock-offset`,
  `slac.continuous_time_imu_clock_offset/v0.1`).
- **Continuous-time IMU lever arm** — gravity-compensated specific-force
  residuals of a displaced IMU origin on the screw-linear knot path, Jacobian
  tests, synthetic recovery with disjoint temporal holdout and a +0.15 m z
  known-bad control (`calibrex trajectory recover-imu-lever-arm`,
  `slac.continuous_time_imu_lever_arm/v0.1`).
- **Continuous-time IMU gyro pre-integration** — rotation residuals against
  the screw-linear knot path with a shared gyro bias, Jacobian tests, synthetic
  recovery with disjoint temporal holdout and a +0.08 rad/s z known-bad control
  (`calibrex trajectory recover-imu-preintegration`,
  `slac.continuous_time_imu_preintegration/v0.1`).
- **Continuous-time LiDAR point-to-plane factors** — signed-distance residuals
  on the screw-linear knot path, Jacobian tests, synthetic recovery with
  disjoint temporal holdout and a +0.15 m z known-bad control
  (`calibrex trajectory recover-point-to-plane`,
  `slac.continuous_time_lidar_point_to_plane/v0.1`).
- **Calibration CI PR workflow template** — `examples/ci/calibration-ci-pr.yml`
  with README, action output contract test, artifact upload in CI smoke, and
  links from `calibration_ci.md` / `your_own_data.md`.

## Implementation issues from the 2026-08-19 revision (all implemented)

These were ordered by the practical-tool criterion: user-facing friction first,
then evidence depth.

### 1. Reusable Calibration CI GitHub Action for pull requests — implemented

**Why now:** `calibrex ci` and the local action wrapper exist, but users still
need a copy-pasteable workflow that validates, compares, and assesses calibration
artifacts on every PR without reading the monorepo action source.

**Outcome:** publish a documented, version-pinned GitHub Action (or workflow
composite) that runs `calibrex validate`, `compare`, and `assess` on candidate vs
baseline artifacts and uploads schema-valid outputs.

**Acceptance conditions:**

- Workflow example under `.github/workflows/` or `examples/ci/` that runs on
  pull requests with configurable candidate/baseline paths.
- Action outputs `status`, artifact path, and summary path (matching the
  existing local action contract).
- Document usage in `docs/tutorials/your_own_data.md` or a dedicated CI tutorial.
- No new dependencies in `src/calibrex`.

### 2. Independent public correspondence for empirical uncertainty — implemented

**Why now:** the opt-in KITTI ground-truth test builds correspondences by
projecting LiDAR at the vendor initial pose, which yields honest but often
degenerate uncertainty intervals. Independent depth/feature correspondence is
the next step toward trustworthy PASS/WARN/FAIL coverage on real data.

**Outcome:** wire a maintained FeatDepth (or equivalent) correspondence provider
into `calibrex camera-lidar empirical-uncertainty` without self-consistency at
the initial transform.

**Acceptance conditions:**

- Document pinned provider commit, checkpoint path, and env vars in test skip
  messages and provenance.
- Opt-in integration test reports `policy_status in {"pass", "warn"}` when data
  is present, without retuning coverage thresholds after seeing results.
- Provider stays outside `src/calibrex` (adapter/artifact boundary only).

**Delivered:** `build_featdepth_correspondence_from_problem()` gates LiDAR
projections with a frozen FeatDepth depth-provider artifact, propagates provider
lineage into correspondence provenance, and relies on pixel jitter plus resample
bootstrap (Issue 3) instead of projection at the vendor initial pose alone;
opt-in KITTI test uses `CALIBREX_KITTI_DEPTH_PROVIDER` (pinned commit in skip
message).

### 3. Continuous-time LiDAR point-to-plane factors — implemented

**Why now:** trajectory contract, SE(3) Jacobians, and sparse GN/LM fitter
exist; native IMU and LiDAR factors against the knot path are the blocker for
multi-sensor continuous-time evidence.

**Outcome:** add native LiDAR point-to-plane residuals tied to
`ContinuousTimeTrajectoryContract` with holdout and injected signed controls.

**Acceptance conditions:**

- Factor evaluates on synthetic trajectory recovery with frozen seed and digest
  provenance.
- Disjoint temporal holdout and at least one known-bad control refute over-tight
  reports.
- Unit tests cover Jacobians and schema-valid fit artifacts; no GPL in core.

**Delivered:** `TrajectoryPointToPlaneMeasurement` in the sparse fitter,
schema-valid measurements/fit fields, synthetic recovery artifact
`slac.continuous_time_lidar_point_to_plane/v0.1` with mid-span temporal
holdout and a signed +0.15 m z control, CLI
`calibrex trajectory recover-point-to-plane`.

### 4. Continuous-time IMU gyro pre-integration — implemented

**Why now:** LiDAR plane residuals are on the knot path; LiDAR-IMU evidence
still needs native inertial factors before lever arm and clock offset.

**Outcome:** add gyro pre-integration residuals with a shared bias, holdout,
and a signed known-bad control.

**Delivered:** `TrajectoryImuPreintegrationMeasurement` in the sparse fitter,
optional `estimate_gyro_bias`, synthetic recovery artifact
`slac.continuous_time_imu_preintegration/v0.1`, CLI
`calibrex trajectory recover-imu-preintegration`.

### 5. Continuous-time IMU lever arm — implemented

**Why now:** gyro pre-integration recovers rotation and bias; LiDAR-IMU
translation still needs a native lever-arm factor before clock offset.

**Outcome:** add gravity-compensated specific-force residuals of a displaced
IMU origin, holdout, and a signed known-bad control.

**Delivered:** `TrajectoryImuLeverArmMeasurement` in the sparse fitter,
optional `estimate_lever_arm`, synthetic recovery artifact
`slac.continuous_time_imu_lever_arm/v0.1`, CLI
`calibrex trajectory recover-imu-lever-arm`.

### 6. Continuous-time IMU clock offset — implemented

**Why now:** rotation, bias, and lever arm are on the knot path; LiDAR-IMU
still needs a native time delay before accelerometer bias.

**Outcome:** add a shared IMU clock offset, holdout, and a signed known-bad
control.

**Delivered:** `estimate_imu_clock_offset` in the sparse fitter, synthetic
recovery artifact `slac.continuous_time_imu_clock_offset/v0.1`, CLI
`calibrex trajectory recover-imu-clock-offset`.

### 7. Continuous-time IMU accelerometer bias and gravity — implemented

**Why now:** rotation, bias, lever arm, and clock are on the knot path;
accelerometer specific force still needs a native bias and gravity split
before optional IMU intrinsics.

**Outcome:** add accelerometer bias and world-frame gravity in the
specific-force residual, holdout, and a signed known-bad control.

**Delivered:** `estimate_accel_bias` and `estimate_gravity` in the sparse
fitter, synthetic recovery artifact `slac.continuous_time_imu_accel_bias/v0.1`,
CLI `calibrex trajectory recover-imu-accel-bias`.

### 8. Continuous-time sliding-window marginalization — implemented

**Why now:** all single-window IMU/LiDAR factors are on the knot path; streaming
and long-horizon use needs explicit gauge and cross-window consistency evidence.

**Outcome:** add overlapping window fits with Schur knot priors, batch overlap
checks, holdout, and a skip-marginalization control.

**Delivered:** `knot_marginalization_priors` in the sparse fitter, synthetic
recovery artifact `slac.continuous_time_sliding_window/v0.1`, CLI
`calibrex trajectory recover-sliding-window`.

### 9. Continuous-time IMU diagonal intrinsics — implemented

**Why now:** bias, gravity, lever arm, and clock are on the knot path; per-axis
scale mismatch is the next optional IMU correction before lifecycle artifacts.

**Outcome:** add diagonal gyro and accelerometer scale factors with holdout and a
signed known-bad gyro-scale control.

**Delivered:** `estimate_gyro_scale` and `estimate_accel_scale` in the sparse
fitter, synthetic recovery artifact `slac.continuous_time_imu_intrinsics/v0.1`,
CLI `calibrex trajectory recover-imu-intrinsics`.

### 10. Calibration lifecycle adoption and rollback — implemented

**Why now:** continuous-time IMU factors are on the knot path; streaming adoption
needs explicit drift, rejection, and rollback provenance separate from batch fit
results.

**Outcome:** represent incumbent/candidate decisions as schema-valid lifecycle
events; weakly observable windows must not mutate the installed transform.

**Delivered:** `decide_calibration_lifecycle_window()` gates, synthetic replay
artifact `slac.calibration_lifecycle/v0.1`, CLI `calibrex lifecycle simulate`.

## Previous P0 issues (implemented 2026-08-20)

### Reusable Calibration CI GitHub Action for pull requests

- `examples/ci/calibration-ci-pr.yml` PR template, README, integration test for
  `run_calibration_ci_action.py`, docs in `calibration_ci.md` and
  `your_own_data.md`; existing composite action exposes `status`, `artifact`,
  `summary`.

### `calibrex doctor` environment readiness artifact

- Schema `slac.environment_readiness/v0.1`, CLI YAML artifact, workflow
  suggestions to `examples/sensor_templates/`, unit tests and round-trip.

### Full-scale KITTI Camera-LiDAR falsification benchmark

- Vendor-reference PASS and known-bad FAIL via `@pytest.mark.kitti` opt-in test;
  `falsification_passed` CLI flag; dataset reference consistency gate.

### Empirical SE(3) uncertainty — ground-truth Camera-LiDAR workflow

- Synthetic CLI integration test and opt-in KITTI test with correspondence
  bootstrap and pixel jitter; `build_probabilistic_correspondence_from_problem()`
  for digest-verified projection when official data env vars are set.

## Completion gate for this roadmap

The roadmap is considered superseded only by a newer dated roadmap or an
accepted ADR that explicitly reconciles this inventory. New method proposals
must identify the portfolio gap they close, their independent evaluation, and
their license boundary before implementation begins.
