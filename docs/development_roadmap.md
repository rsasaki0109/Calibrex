# Development Roadmap

This document is the current development decision for Calibrex as of
2026-08-19 (updated 2026-08-19). It reconciles the implemented code, the older ADR direction,
public-data evidence, and relevant research/OSS. It is a planning inventory,
not a claim that every listed method is production-ready.

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
| Kalibr Camera-IMU/camera chain | YAML importer | Generic external-run artifact with transforms, intrinsics, time shifts, digests, conventions, and isolation declaration | Import boundary is implemented; no maintained Calibrex comparison result | Adapter | Top-level BSD-4-Clause; ROS/Kalibr runtime stays external |
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

1. **Evidence scale:** Camera-LiDAR has several native methods but lacks a
   reproducible, raw, unregistered, full-scale public PASS/known-bad FAIL pair.
   The full-scale KITTI benchmark is implemented but no PASS+FAIL public pair
   result has been recorded yet.
2. **Common external execution adoption:** the schema-versioned external-run
   artifact is implemented and proven by Koide, Kalibr, and iKalibr. Remaining
   adapters such as RIs-Calib, Open3D, and NDT should migrate incrementally.
3. **Empirical uncertainty:** `slac.empirical_se3_uncertainty/v0.1` is now
   implemented, CLI-wired, and exercised in a stability-only integration test.
   The remaining gap is a public workflow with independent ground-truth coverage
   assessment (PASS/WARN/FAIL rather than INCONCLUSIVE).
4. **Continuous time:** `ContinuousTimeTrajectoryContract`, SE(3) manifold
   Jacobians, and sparse GN/LM fitter are implemented. The remaining gap is
   native IMU and LiDAR factor integration and sliding-window marginalization.
5. **Complete LiDAR-IMU solve:** rotation evidence exists; native translation,
   clock offset, bias, and intrinsic estimation do not.
6. **Lifecycle monitoring:** online adoption gates exist, but there is no
   schema-defined drift event, incumbent/candidate history, or rollback
   decision artifact.

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
   evidence, workflow suggestions, and provenance;
2. publish a reusable GitHub Action that validates, compares, and assesses
   calibration artifacts in pull requests;
3. complete the generic external-run contract and prove it with Koide and
   Kalibr producers;
4. finish the full-scale KITTI falsification benchmark; and
5. cut an installable release only after clean-wheel, schema-drift, and
   quickstart checks pass.

The adoption track reuses the core models and adapters. It must not add ROS,
GPL, visualization-server, or external-solver dependencies to `src/calibrex`.

### P0 — Evidence and integration hardening

Finish the three implementation issues below. They turn existing breadth into
repeatable comparative evidence and should land before another paper baseline.

### P1 — Continuous-time foundation — partially implemented

`ContinuousTimeTrajectoryContract` (`slac.continuous_time_trajectory/v0.1`),
SE(3) manifold Jacobians, and a sparse GN/LM fitter are implemented and
CLI-wired. Remaining work: native IMU pre-integration factors, LiDAR
point-to-plane factors against the continuous knot path, and sliding-window
marginalization with gauge and consistency evidence.

### P2 — Native LiDAR-IMU and sliding-window evidence

Build in stages: rotation plus gyro bias, lever arm, clock offset, then
accelerometer bias/gravity and optional intrinsics. Every stage requires
disjoint temporal holdout, per-axis excitation, separability diagnostics, and
injected signed controls. Add marginalization only with explicit gauge and
consistency evidence.

### P3 — Calibration lifecycle

Represent drift detection and adoption as schema-valid events with incumbent,
candidate, evidence window, decision, and rollback provenance. A weakly
observable window must not update the installed transform.

## Completed since last roadmap revision (2026-08-19)

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
- **Empirical SE(3) uncertainty — ground-truth workflow (synthetic + opt-in KITTI)**
  — integration tests for coverage assessment without `--stability-only`;
  `build_probabilistic_correspondence_from_problem()` projects digest-verified
  LiDAR into the camera for official KITTI runs when
  `CALIBREX_KITTI_RAW_0005` and `CALIBREX_KITTI_DEPTH_PROVIDER` are set.

## Next implementation issues

Ordered by the practical-tool criterion: user-facing friction first, then
evidence depth.

### 1. `calibrex doctor` environment readiness artifact — implemented

**Why now:** this is the first command a new user runs.  The current output is
ad-hoc text.  Making it a schema-valid artifact enables automated CI checks,
clearer first-run guidance, and diagnostic provenance — all high-value for
practical users.

**Outcome:** promote `calibrex doctor` from an ad-hoc check to a
schema-versioned `slac.environment_readiness/v0.1` artifact with path-based
dataset-type inference, workflow suggestions, and provenance.

**Acceptance conditions:**

- Define `slac.environment_readiness/v0.1` JSON Schema and Pydantic model.
- CLI produces a schema-valid YAML artifact covering Python version, installed
  Calibrex version, optional dependency versions, and dataset path checks.
- For each detected dataset path, emit at least one `workflow_suggestion` row
  pointing to the appropriate template in `examples/sensor_templates/`.
- Unit tests validate schema and round-trip; no new external dependencies
  enter `src/calibrex`.

### 2. Full-scale KITTI Camera-LiDAR falsification benchmark

**Why now:** the only existing Camera-LiDAR evidence run (`kitti_raw_2011_09_26_drive_0005`)
uses the full KITTI processing stack (KDE correspondence + multi-sensor SLAC),
which is not directly comparable to the planar-board baseline.  A dedicated
falsification run with vendor reference transform closes the evidence gap and
demonstrates that Calibrex can detect a known-bad calibration on real data.

**Outcome:** run the existing KITTI falsification pipeline on the full
`2011_09_26_drive_0005_sync` sequence and record an honest PASS/FAIL pair.

**Acceptance conditions:**

- Build a `CameraLidarCalibrationProblem` with the KITTI vendor reference
  transform.
- Run `calibrex camera-lidar benchmark-rotation` (or `benchmark-six-dof`) and
  record the `falsification_passed` flag without retuning thresholds.
- Add a `@pytest.mark.kitti` opt-in integration test asserting
  `falsification_passed=True` for the positive case and `False` for at least
  one known-bad rotation perturbation (e.g. 10 deg about Z).
- Provenance must record sequence ID, frame selection policy, problem SHA-256,
  and vendor calibration source.

### 3. Empirical SE(3) uncertainty — ground-truth Camera-LiDAR workflow

**Status:** synthetic public-workflow integration test and opt-in official KITTI
test are implemented; remaining gap is a maintained public correspondence source
that does not rely on depth-lidar projection from the initial pose.

**Why now:** the stability-only (`--stability-only`) path is implemented and
tested.  The full coverage assessment path (with a vendor reference transform)
is the natural next step and closes the gap between "the tool ran" and "the
tool told me whether my calibration is trustworthy".

**Outcome:** run `calibrex camera-lidar empirical-uncertainty` without
`--stability-only` on a public dataset with a vendor or independent reference
transform, and report PASS/WARN/FAIL coverage.

**Acceptance conditions:**

- Use the ACFR planar-board result or the KITTI falsification result as the
  correspondence and problem source.
- Freeze seed, correspondence input, and problem digest in provenance.
- Report whether the Sidak-corrected joint family coverage gate is PASS, WARN,
  or FAIL; do not retune thresholds after seeing the result.
- Add an opt-in `@pytest.mark.integration` test asserting
  `policy_status in {"pass", "warn"}` when the data is present.

## Completion gate for this roadmap

The roadmap is considered superseded only by a newer dated roadmap or an
accepted ADR that explicitly reconciles this inventory. New method proposals
must identify the portfolio gap they close, their independent evaluation, and
their license boundary before implementation begins.
