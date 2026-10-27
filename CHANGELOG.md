# Changelog

## Unreleased

- Validate sliding-window fault-injection translations as exactly three finite
  real components before fitting. Prevent scalar/single-component broadcasting
  from silently changing controls; preserve intended anchor and overlap shifts
  in both normal and residual-space marginalization modes.

- Validate sliding-window layout integer types before the batch reference fit.
  Reject boolean/fractional sizes and overlaps, and convert NumPy integer counts
  to Python integers before coverage arithmetic to avoid fixed-width overflow.

- Require integer elimination dimensions and knot counts across marginalization
  APIs, rejecting booleans and oversized requests before arithmetic. Validate
  Schur input symmetry even for zero elimination; preserve valid no-op copies.

- Validate native and B-spline solver settings before optimization: finite
  damping/tolerances, integer iteration budgets and non-reversing native damping
  scales. Preserve zero-iteration B-spline evaluation, zero B-spline stall
  tolerances and unit native damping multipliers.

- Reject nonfinite interior knot timestamps, mismatched knot states in dense
  factor assembly, and out-of-range single-knot priors before fit evaluation.
  Audit mixed weighted observations and all supported knot prior representations
  against the native fitter's objective, normal gradient and manifold derivatives.

- Exclude rows unrelated to eliminated variables and exactly absent columns from
  residual-space rank decisions. Prevent zero-row padding from manufacturing
  retained information while preserving unrelated observations and constant
  costs. Generator 0.11 records total and involved residual row counts.

- Replace the full residual-space SVD basis with thin SVD and implicit
  Householder projection during square-root marginalization. Avoid quadratic
  storage in residual rows at fixed column counts, preserving the rank rule
  and complete profiled cost within floating-point accuracy (generator 0.10).

- Compress outgoing residual-space window priors with QR of the augmented
  Jacobian and residual, preserving constant cost and bounding stored rows by
  retained variables plus one. Record before/after row counts in generator 0.9
  evidence. Elimination itself still uses the dense residual-space SVD.

- Add experimental `--square-root-marginalization` sliding-window mode. Transfer
  rectangular residual factors directly between anchored knot windows without
  converting them back to information priors. Record separate SVD elimination
  evidence and the selected mode in generator 0.8 artifacts. Keep normal Schur
  elimination as the default and the existing failed control gate unchanged.

- Add a dense residual-space marginalization reference API that retains weighted
  Jacobian and residual factors, avoiding cancellation from Schur subtraction.
  Record normalized-column SVD rank and threshold; test weak retained information,
  coupled nullspaces and complete profiled cost. Sliding-window fitting continues
  to use its existing normal-equation path.

- Compute Schur corrections in diagonally scaled coordinates, avoiding an
  overflowing eliminated solution when the retained information and gradient
  remain representable. Cover full-rank and coupled-nullspace systems with
  subnormal weights; reject nonfinite retained results explicitly.

- Add opt-in `project_observable_subspace` updates to the native sparse fitter.
  Solve the damped model only within the diagonally scaled observed eigenspace;
  reject a material gradient in numerical null directions. Record each attempted
  projected step's rank, threshold and range check in window evidence. The
  default solver is unchanged; synthetic generator 0.7 records the selected mode.

- Support coupled numerical nullspaces in Schur elimination with diagonally
  scaled rank and range checks. Reject retained coupling or gradients outside
  the eliminated information range instead of silently projecting them away.
  Keep full-rank solves and exact-zero handling, and record elimination rank,
  thresholds and range errors in optional per-window evidence (generator 0.6).

- Allow Schur elimination of exactly unobserved coordinates when their complete
  rows, columns, and gradient entries are zero. Eliminate only the informative
  subblock, preserving weak positive information without a pseudoinverse cutoff
  or artificial prior. Reject inconsistent missing-information normal equations;
  coupled singular active blocks remain unsupported.

- Solve sparse trajectory updates only on columns with nonzero information,
  keeping exactly unobserved coordinates unchanged without adding priors or
  diagonal floors. Reassess the active columns every iteration and record the
  frozen coordinates in per-window evidence. Generator 0.5 solves the previously
  singular development window but retains the failed no-marginalization control.
  Make scalar and batched screw-interpolation endpoint Jacobians exactly zero
  or identity to prevent roundoff from manufacturing information on absent knots.

- Add per-window local information diagnostics, separately for observations and
  observations plus inherited marginal priors. Record scaled spectra, numerical
  rank/nullity, exact zero-information coordinates, and coupled-nullspace
  participation with global knot indices. Generator 0.4 exposes the synthetic
  first window's unobserved terminal knot without adding artificial constraints.

- Recompute sliding-trajectory costs, plane/pose errors, gradients, and final
  stationarity from its returned knots instead of copying batch diagnostics.
  Retain every window result, aggregate step maxima, and preserve earlier
  failures in the overall status. Add optional per-window statuses and stitched
  diagnostics to recovery artifacts. Generator 0.3 rejects nonconverged sliding
  fits; the retained development seed exposes an earlier singular window.

- Remove implicit initial-pose penalties from sliding-window fits against
  world-frame planes. Preserve caller-supplied pose observations, admitting
  each once when an incoming marginal prior is used. Retain the gauge audit's
  paired seed: the unchanged no-marginalization control gate now fails, so the
  synthetic recovery CLI reports failure rather than claiming demonstrated
  marginalization benefit. Bump its generator version to 0.2.

- Prevent repeated point-to-plane observations in sliding-window marginalization:
  an incoming prior replaces all rows through the preceding window endpoint,
  including that endpoint. Admit only later rows when the prior is used, while
  preserving raw overlap observations in no-prior controls. Validate advancing
  windows before fitting and allow prior-only trajectory problems.

- Preserve cross-knot curvature and retained gradients with joint anchored
  marginalization priors. Use the complete Schur prior in sliding-window fits
  and include its coupled columns in both normal-equation builders and candidate
  scoring. Keep the single-knot API and explicit block-diagonal helper available.

- Preserve the retained normal-equation gradient in single-knot marginalization
  priors. Evaluate the same shifted residual in linearization and candidate
  scoring; reject gradients outside the information range. Use a linear solve
  for Schur elimination. Multi-knot splitting remains a block-diagonal
  approximation that drops cross-knot curvature.

- Audit B-spline final stationarity with both feasible one-sided clock gradients
  at exact internal IMU knot ties. Central-difference cancellation no longer
  certifies convergence; report `stationarity_not_confirmed` when necessary.
  Retain the existing optimization steps and covariance calculation.

- Guard B-spline clock estimation with the common gyro/accelerometer time
  domain. Reject infeasible coupled updates before constructing candidates,
  retry with higher damping, and use inward finite differences at boundaries.
  Record clock-domain diagnostics and reject zero-width estimation domains;
  fixed-clock fitting remains supported.

- Guard screw-linear clock updates with a shared feasible time domain, retry
  rejected coupled candidates with more damping, and record domain diagnostics.
  Unify derivative sides across IMU factors and check both feasible sides before
  certifying stationarity at exact gyro knot ties. Retain three paired synthetic
  development runs and boundary/knot regression tests.

- Add frozen empirical per-axis error intervals using a finite-sample order
  statistic on independent complete-pipeline recordings. Reject calibration/test
  overlap and incompatible scopes; abstain with insufficient data and retain
  failed attempts. Register model/evaluation schemas and retain the 20+10 bag
  cohort, including three undercovered axes. Correct the P0 gravity comparison:
  unaligned synthetic and odometry world frames cannot be compared component-wise.

- Distinguish stalled continuous-time fits from stationary solutions in both
  screw-linear and B-spline solvers. Use unit-invariant gradient and step
  diagnostics, retain explicit termination reasons, extend result schemas with
  `stalled`, and warn for a nonstationary final bag fit. Historical results remain
  readable. Add independent uncertainty-coverage evaluation with every failed
  trial retained; 20 known-plane and five full-pipeline synthetic bag draws
  expose fixed-map covariance undercoverage without retuning intervals.

- Add a method demo gallery for LiDAR–LiDAR, IMU deskew, Camera–IMU,
  Camera–LiDAR and fisheye stereo, with provenance and real/synthetic labels.
- Support unified/MEI and Double Sphere cameras in native stereo rectification
  and adjacent Kalibr pair loading alongside KB4 and pinhole/radtan. Add model
  domain checks, wide-ray inverse tests and synthetic rectification/depth demos;
  generic bag-check adapters retain their existing camera-model support.

- Replace the misleading planning-only sample entry with an immediate real-data
  point-cloud comparison: NTU VIRAL before/after reveal, orbit, zoom and saved
  residual metrics. Keep the tiny synthetic bag as a clearly labelled planning
  test. The display payload is reproducible and bound to source digests;
  comparison playback is explicitly distinguished from a live solve.

- Add an interactive, dependency-free 3D rig viewer to browser planning and
  results. Compose frame transforms in their parent frames, inspect position
  and quaternion, and rotate or zoom with pointer/keyboard controls. Keep
  disconnected roots separate and label exported prior-filled axes and their
  descendants as NOT MEASURED; omitted estimate frames are not drawn.

- Show real IMU rotation training residuals in the browser's live dashboard,
  separated by worker and independent fit. Report initialization and each robust
  IRLS round as opt-in progress diagnostics; calibration results, evidence schemas
  and held-out verdicts remain unchanged.

- Return an inconclusive camera focal check when held-out rotations cannot
  support every rate ratio. Missing held-out ratios previously bypassed the
  disagreement gate through NaN comparisons. Reject nonfinite evidence before
  constructing numeric artifacts; keep schemas and thresholds unchanged.

- Record the first pre-registered INS-LiDAR rotation audit across KITTI and
  Boreas: **refuted, 6/8 gates**. All four fits completed and all eight estimated
  axes agree within 0.286 deg, with eight detected 1 deg controls. One KITTI
  recording estimates no rotation axis, failing completeness and empty-control
  integrity. Preserve all cases, source/input pins, receipts and audit evidence;
  add the independent Boreas adapter and failure-aware frozen tooling. Seven
  pairs now have an audit; supported pairs remain three. No gate was relaxed.

- Differentiate native screw gyro clock offsets with exact interval angular
  velocity and explicit one-sided knot limits, replacing the 1 ms sampled
  rate. Preserve residuals, all nonclock blocks, stopping and schemas. Retain
  all three spent development cases: policies and all 32 controls pass, while
  iterations change from 21/23/29 to 20/32/19 and the middle case stops early
  with `max_iterations` before its configured budget of 60. A status erratum
  corrects the initial limit description without changing stored outcomes. No speed or general convergence improvement is claimed.

- Batch fixed-state screw interpolation and point-to-plane evaluation; share
  accelerometer stencils and avoid unused translation logarithms. Preserve
  complete results, covariance, stopping and schemas. Against the previous
  maintained implementation, twelve fresh CPU workers measure paired median
  complete-API ratios of 2.023x for spent synthetic LiDAR–IMU recovery and
  4.709x for public GLIM geometry, with exact complete pair equality.
  Fresh-worker ratios are 1.462x and 2.189x; one joint API pair is 1.641x.
  Retain every observation and CPU-only evidence replay.

- Share one gyro integral across residual and mounting-rotation derivatives;
  avoid discarded IMU derivatives and perturbed clock samples when scoring.
  Preserve complete inputs, fitted results, covariance, stopping and schemas.
  Retain all six synthetic joint-recovery workers: 1.452x paired median for
  the complete CPU API, 1.292x including worker startup/output. The declared
  2x whole-API target was missed; this is an incremental execution cleanup.

- Reuse fixed-state screw interval preparation, poses, Jacobians and IMU body
  samples; share exponentials and avoid discarded geometric RMSE derivatives.
  Preserve exact complete results, covariance, stopping and schemas. Paired
  complete APIs measured 2.395x for synthetic LiDAR–IMU recovery and
  2.765x for spent public GLIM geometry. Retain both complete cohorts,
  including the first geometry attempt that missed the unchanged 2x gate,
  all 32 detected controls per joint case, and CPU-only archive replay.

- Report the native sparse trajectory gradient at the returned state. Reuse the
  last linearization while its state is current, refreshing it after an accepted
  update or state restoration and sharing it with covariance. Keep stopping
  criteria, accepted states, covariance and schemas unchanged. Retain all three
  paired spent LiDAR-IMU development cases: every other fit field and native
  metric matches exactly, with all 32 controls detected in each case. Native
  assembly counts are unchanged; no timing or independent accuracy claim.

- Correct native trajectory information-prior whitening so the weighted
  residuals and normal equations match the declared information quadratic.
  Preserve zero and weak information without adding a positive floor. Validate
  finite real 6x6 information, symmetry and positive semidefiniteness after
  diagonal scaling. Keep prior centering, Log derivatives, solver settings and
  serialized schemas unchanged.
  Retain all historical/current development results: the correlated-prior
  window changes from 26 to 7 iterations with a smaller overlap error; the
  overall maximum translation error is unchanged and the native holdout/control
  fixture still passes. No timing or independent accuracy claim is made.

- Correct native `screw_linear` gyro factor Jacobians for left/right SO(3)
  logarithm perturbations and the exponential differential of integrated
  increments. Include diagonal gyro scale in bias derivatives. Check nonzero
  residuals, mounting rotation, nonuniform sample times, zero and small motion,
  actual sparse assembly, weighted gradients and known constant-bias recovery.
  Keep residual values, sampling, convergence settings and schemas unchanged.
  Retain all three spent development outcomes and exact residual/rotation
  parity on ten fixtures; all quality gates pass, with mixed iteration counts.

- Include analytic trajectory-knot derivatives in native `screw_linear`
  accelerometer factors, including all supports of their discrete kinematic
  stencil across segment boundaries. Apply accelerometer scale to lever-arm
  derivatives after mounting rotation. Check all-knot finite differences,
  nonunit scales, endpoint clamping, dense nonuniform knots, actual sparse
  assembly and weighted objective gradients. Preserve residual values,
  convergence settings and serialized schemas. Retain every outcome on the
  three spent development seeds: policy changes from two passes and one
  failure to three passes, while some rotation maxima increase slightly.

- Make the native `screw_linear` trajectory acceptance objective use squared
  measurement weights, matching its existing inverse-sigma normal equations
  and covariance. Cover all five measurement families, diagonal information
  priors, a finite-difference gradient, and closed-form heterogeneous-weight
  fits. Keep convergence thresholds and artifact schemas unchanged. Record the
  research direction for reliability under a fixed compute budget.

- Avoid discarded geometric Jacobians in sparse screw trajectory objective
  scoring and use exact fixed-size FP64 cross products in the SE(3) Jacobian.
  Preserve full derivatives in linearization and covariance. Three paired
  synthetic LiDAR–IMU recovery calls measured a 2.367x complete-API speedup;
  three paired fits with spent public GLIM geometry measured 3.143x.
  Fresh-worker ratios are 2.204x and 2.879x. Retain all complete inputs,
  trajectories, calibration covariance, controls and unchanged policy failures,
  with exact pair equality and a CPU-only archive auditor.

- Prepare offline camera–LiDAR edge images and LiDAR discontinuities in bounded,
  ordered CPU workers with `EdgeAlignmentOptions(preparation_workers=8)`.
  Preserve default single-worker behavior, frame selection, all fits and controls.
  Over 48 spent public KITTI pairs, three fresh workers per variant measured
  preparation 5.203 s to 1.341 s (3.881x), complete estimation API 6.543 s to
  2.522 s (2.595x), and the larger fresh-worker scope 13.822 s to 9.156 s (1.510x).
  Retain both complete six-worker cohorts, the initial Git line-ending audit
  failure and its correction, and CPU-only record replay.

- Share native point rotations within the optional Numba online camera–LiDAR
  grid and evaluate bounded chunks in up to four CPU workers. Preserve all
  candidate scores, tracking and diagnostics. Against the previous maintained
  CPU backend, 48 public KITTI pairs measured complete solve time 22.019 s to
  5.965 s (3.692x), median updates 0.682 s to 0.127 s (5.387x), and fresh-worker
  time 32.604 s to 16.220 s (2.010x). Retain all six exact-result observations,
  both source snapshots, and CPU-only archive auditing.

- Add a typed PNG luminance array reader that avoids Python pixel tuples for
  array consumers, preserving the existing rows API and validation. Connect it
  to online A2D2 preparation and KITTI workers. On 48 public KITTI images, the
  complete PNG-to-array read stage measured 2.855 s to 0.769 s (3.713x), with
  all 21,773,760 float64 samples unchanged. Retain separate pilot/formal
  evidence, a CPU-only auditor and full public-calibration parity tests.

- Add an opt-in Numba CPU path for online camera–LiDAR projection and image
  spilloff. Preserve ordered transforms, reductions, tracking, holdout and
  curvature without global patches. Use separable image neighborhoods and
  support ordered preparation workers. Add a cold-worker comparison that
  reuses existing KITTI inputs and bounds evidence storage. The maintained CPU
  path measured 51.174 s to 23.724 s (2.157x) against production NumPy, with
  complete parity, matching the existing compiled benchmark control's level.

- Add opt-in CUDA image spilloff and guarded rolling-grid evaluation for the
  Levinson–Thrun camera–LiDAR tracker. Retain native ambiguous comparisons,
  finalists, holdout controls and curvature, and preserve default CPU result
  serialization. Replace dense beam assignment with exact sorted-neighbor
  lookup. Overlap CUDA initialization with ordered CPU preparation, with
  optional parallel loading. Add complete-worker CPU/CUDA comparison and
  CPU-only audit tools.

- Redact Windows drive and UNC paths in check and estimate HTML reports on
  either platform, and write rosbag2 metadata with LF line endings on every
  platform. Keep cache-home and SQLite filename tests portable, and compare
  only the synthetic sample's three trigonometric float64 fields within two
  ULPs across platforms; other sample bytes and metadata remain exact.

- Load CLI commands, solver exports and artifact models on demand. An 18-worker
  comparison against the batched refiner measured complete refinement CLI time
  of 7.758 s to 3.003 s (2.584x), with identical full results and 130 unchanged
  schema/ledger exports. Artifact validation measured 3.830x in fresh processes.

- Batch probabilistic Camera–LiDAR pose evaluation on CPU and retain invariant
  distributions per solve. Preserve complete candidate traces, acceptance and
  holdout exactly; a four-worker projected KITTI geometry control measured
  29.920 s to 10.352 s for complete workers. Tiny inputs retain scalar evaluation.
  Add source-bound measurements, reproduction tools and a CPU-only archive audit.

- Add `calibrex convert kitti-360` for rectified perspective images, timed
  Velodyne scans, sparse published poses and calibration in the official split
  layout. Frame windows retain their original IDs, source files are digest-pinned,
  and failed conversions preserve existing output bags. Synthetic round trips
  are tested; real KITTI-360 conversion remains unverified while official-host
  access is blocked. This adds no new SOTA claim.

- Add an opt-in `start_scan_min_control_t` margin for adopting camera-LiDAR
  profile-start fits. A rejected fit retains the current transform and evidence;
  defaults preserve v2 behavior pending spent-data replays. Publish a post hoc
  inventory of the 55 spent KITTI raw drives with new-rule metrics explicitly
  unknown. Add a spent-data-only seven-check input capture and paired replay
  tool with lossless snapshots, digest verification, real-check parity and
  retained failures. Expose the margin as `--camera-lidar-far-control-t` in
  `check`, `estimate` and `drift`, recording it in their schema-valid results
  and separating estimator cache entries. This work introduces no new SOTA
  claim or audit thresholds.

- **The KITTI camera-LiDAR check audit v2 is refuted: 8 of 9 gates.**
  - **The runs.** All 18 pre-registered drives of 2011_09_28, 2011_09_29, 2011_09_30 and
    2011_10_03 ran once from the frozen code.
    - Every archive matched its pinned size and ETag, and every calibration its SHA-256.
    - One fetch (2011_10_03_0027) dropped its connection and was retried under the
      rerun policy; both attempts are recorded.
    - All 18 conversions and 126 checks exited 0. Three drives were not usable: the rig
      stood too long, leaving too few distinct views.
  - **What held.**
    - On all 15 scored drives the check passed that day's calibration (v1 flagged
      KITTI's on 3 of 23).
    - Every judged axis was within 0.45 deg of it, and 36 of 45 axes were judged.
    - All 36 3 deg and 36 1 deg turns about judged axes were flagged, and none was
      judged right.
  - **What failed.** On 2011_09_28_0002, four turned runs took a far optimum that judged
    roll 1.45 to 1.50 deg off the day's calibration. Its held-out control was at `t` 2.58
    to 2.65 (post hoc). The 3 deg yaw run's 1.503 deg contradicts the 1.5 deg
    turned-accuracy gate.
  - **Leaderboard.** `camera-lidar` stays `refuted` (2 refuted audits). The page is
    `docs/benchmarks/kitti_camera_lidar_v2_check.md`; the runs, scores and audit are in
    `docs/assets/kitti_camera_lidar_v2/evaluation/`.

- **camera-lidar v2 is pre-registered: KITTI's other four days, each with its own
  calibration.**
  - **What changed in the check, after the refuted v1 audit.**
    - A held-out control now needs `t` 2.5 (was 2). In v1's runs, every axis judged more
      than 0.75 deg off KITTI's calibration had its control at `t` 2.09 to 2.33.
    - An optimum found from the profile maxima replaces the candidate's only when it
      observes more rotation axes, at least two, and none at the search bound. In v1, a tie
      went to the higher objective (0029: yaw 9.6 deg off), and a yaw at the search bound
      was accepted (0056).
  - **Development.** All 36 2011_09_26 drives and 2011_09_30_0018 ran through the v2
    harness.
    - KITTI's calibration passes on all 37; v1's rules flagged 3 of the 36.
    - Every judged axis is within 0.45 deg of it (0.89 deg in turned runs).
    - 86 of 89 3 deg turns and 89 of 89 1 deg turns are flagged; none is judged right.
    - The cost is coverage: 86 of 108 axes are judged on the 2011_09_26 drives, against 97.
    - 2011_09_30_0018 comes from another day, with its own calibration. There the check
      passes that day's calibration and flags all six turns.
  - **The claim.** It covers 18 drives of 2011_09_28, 2011_09_29, 2011_09_30 and 2011_10_03
    that no camera-LiDAR method has read, each checked against its own day's calibration.
    - The gates and thresholds are v1's: no flags on KITTI's calibration; judged axes
      within 0.75 deg (1.5 deg in turned runs); at least half of the axes judged.
    - At least 90 percent of 3 deg turns and 75 percent of 1 deg turns must be flagged, and
      no 3 deg turn may be judged right.
    - At least 12 drives must be scored. The code is frozen at `f0afa94`.
  - **New tools.**
    - `tools/kitti_drive_fetch.py` fetches only a drive's first 600 frames (all the check
      reads) by HTTP range. Every range carries `If-Match` on the archive's pinned ETag,
      and every member's CRC-32 is checked.
    - `tools/kitti_camera_lidar_v2_evaluate.py` is the driver.
      `tools/score_kitti_camera_lidar_v2.py` and `tools/build_kitti_camera_lidar_v2_audit.py`
      score the runs and build the audit.
  - **Docs.** The page is `docs/benchmarks/kitti_camera_lidar_v2_check.md`. The check's
    earlier tables (Hilti, AIST, KITTI's thirteen drives) were run with `t >= 2` and are
    not re-run.

- **The KITTI camera-LiDAR check audit is refuted: 5 of 9 gates.**
  - **The runs.** All 23 pre-registered drives ran once from the frozen code. Every
    archive matched its pinned size and ETag, and all 23 conversions and 161 checks
    exited 0.
  - **What held.**
    - On 20 drives, KITTI's calibration passes, every judged axis within 0.45 deg of it.
    - 61 of 69 axes are judged.
    - 57 of 61 3 deg and 60 of 61 1 deg turns about judged axes are flagged (gates 90
      and 75 percent).
  - **What failed.** On three drives the check flagged KITTI's calibration (0029 and
    0056 `fail`, 0032 `warn`). It judged axes 0.8 to 9.6 deg off, and on 0056 judged a
    3 deg-turned roll right. Four gates are contradicted:
    - KITTI's calibration not flagged;
    - judged-axis accuracy (9.60 deg against 0.75);
    - turned-run accuracy (9.66 against 1.5);
    - no 3 deg turn judged right.
  - **Post-hoc, not part of the verdict.**
    - Every judged axis more than 0.75 deg off had a held-out control just over its
      threshold (`t` 2.09 to 2.33).
    - On 0029 and 0056 the estimate came from an alternative search start.
  - **Leaderboard.** `camera-lidar` stands `refuted` (0 supported, 1 refuted). The page
    is `docs/benchmarks/kitti_camera_lidar_check.md`; the runs, scores and audit are in
    `docs/assets/kitti_camera_lidar/evaluation/`. README and the docs home now list every
    audited pair's standing.

- **The KITTI camera-LiDAR check audit is pre-registered: honest per-axis verdicts.**
  - **The claim.** On 23 KITTI 2011_09_26 drives no camera-LiDAR method has read,
    `calibrex check --pairs camera-lidar`:
    - never flags KITTI's calibration, and every axis it judges there is within
      0.75 deg of it; it judges at least half of the axes;
    - started 1 or 3 deg off, about one camera axis, judges every axis within 1.5 deg
      of KITTI's calibration;
    - flags at least 90 percent of the 3 deg turns about axes it judged and never
      judges such a turned axis right, and flags at least 75 percent of the 1 deg turns.
  - **It is a check claim**, not a calibration-accuracy claim: the check starts at the
    candidate, and no external camera-LiDAR baseline is integrated.
  - **Drives.** Every unread 2011_09_26 drive with at least 125 frames, chosen from the
    archives' file lists; their sizes, ETags and the calibration archive's SHA-256 are
    pinned.
  - **Thresholds.** v3's rules on the 13 development drives (`5d72a6c`), which pass every
    requirement except the drive count, set for the evaluation.
  - **Tools**, frozen at `71a115b`: `tools/score_kitti_camera_lidar.py`,
    `tools/build_kitti_camera_lidar_audit.py`, and the KITTI driver
    `tools/kitti_camera_lidar_sota_evaluate.py`. The page is
    `docs/benchmarks/kitti_camera_lidar_check.md`.

- **camera-lidar no longer fails KITTI's own calibration.**
  - **The problem.** On 13 KITTI 2011_09_26 drives, the check failed KITTI's
    calibration on 5. The estimate settled 2 to 11 deg away and passed its own
    controls there.
  - **Four changes**, each tested on those drives:
    - The edge response has its local mean removed (`edge_background_sigma_px`, 20 px).
      The objective no longer pulls towards textured regions; on 0028 it had climbed
      11 deg in yaw.
    - A spinning LiDAR's frame that repeats the last kept view is dropped
      (`repeated_view_change`, 0.05). 0057 stands 29 s at a crossing.
    - `calibrex convert kitti-raw --velodyne-time azimuth` adds a per-point time,
      modelled from the azimuth and the sweep's stamps, so the rolling sweep is moved to
      the image time.
    - The search also starts from each axis's profile maximum on its own. From 3 deg
      off, the search had run along a ridge on 4 drives; on 0015 a 3 deg roll turn was
      judged right.
  - **Now.** KITTI's calibration passes on all 13 drives, 36 of 39 axes judged, every
    judged axis within 0.35 deg. Every 1 and 3 deg turn about a judged axis is flagged,
    on that axis only. Details in `docs/benchmarks/camera_lidar_check.md`.
  - **Test fix.** The drift test compared a rotation-vector component with the 2.5 deg
    turn; the component moves 1.84 deg. It now expects that.

- **camera-imu has a supported SOTA claim: the Hilti camera-IMU v3 audit, 7 of 7 gates.**
  - **The runs.**
    - Five recordings on which no camera-IMU estimate had been made: exp05, exp06,
      exp11, exp14 and exp18.
    - Every bag's SHA-256 matched the pinned one. All ten runs exited 0 on their only
      attempt, from the frozen code.
  - **The results.**
    - Every judged axis was within 0.528 deg of Kalibr (gate 1.0), and every clock
      offset within 0.470 ms (gate 0.75).
    - 26 of 30 axes were judged.
    - Across recordings, a judged axis spanned at most 0.408 deg (gate 1.0).
    - The cam0-cam1 rotation was within 0.444 deg per jointly judged axis (gate 0.75).
  - **What the method declined to judge.** The four axes it left unjudged include the
    two largest errors, exp18's pitch (-0.87 and -0.68 deg).
  - **It claims less than v1 (post-hoc, v1's unchanged scorer).** Under v1's rules the
    same runs are refuted: `constrained` fails on three units, and the whole relative
    rotation on exp14 is 0.57 deg.
  - **Leaderboard.** `camera-imu` stands `supported` (1 supported, 2 refuted). The
    results are in `docs/benchmarks/hilti_camera_imu.md` and
    `docs/assets/hilti2022_camera_imu_v3/`.

- **The Hilti camera-IMU v3 audit is pre-registered: honest per-axis verdicts.**
  - **Why.** v1 and v2 held their accuracy gates. Both failed on gates about axes the
    method itself reports it cannot pin down.
  - **The claim.**
    - Every axis the method judges is within 1.0 deg of Kalibr. An axis is judged when
      its std is at most 0.3 deg and its 1 deg control is detected.
    - Judged clock offsets are within 0.75 ms.
    - At least half of the axes are judged.
    - A judged axis spans at most 1.0 deg across recordings.
    - The cam0-cam1 rotation is within 0.75 deg per axis judged for both.
  - **Recordings.** exp05, exp06, exp11, exp14 and exp18, on which no camera-IMU
    estimate has been made.
  - **Thresholds.** 1.5 times the worst of the 22 development units, rounded up.
    Development scores `supported` (`docs/assets/hilti2022_camera_imu_v3/`).
  - **Tools**, frozen at `ea5ff40`:
    - `tools/score_hilti_camera_imu_v3.py`;
    - `tools/build_hilti_camera_imu_v3_audit.py`;
    - a `camera-imu-v3` mode in `tools/hilti_camera_sota_evaluate.py`.

- **Two pre-registered Hilti 2022 camera audits, both refuted.**
  - **How they ran.** `docs/benchmarks/hilti_camera_sota_preregistration.yaml` fixed
    both claims, their recordings and the recordings' SHA-256 before any estimator read
    them. The recordings were chosen from bag headers and indexes only. The code was
    frozen at `0482db6`. `tools/hilti_camera_sota_evaluate.py` downloaded, ran and
    deleted one recording at a time.
  - **`camera-camera`, a new target pair: refuted, 4 of 6 gates.** Twelve recordings,
    the first 5 GB of each. Every scored unit is within 0.33 deg of Kalibr, and 47 of
    60 units are scored. A 1 deg turn of cam3 or cam1 is detected in 16 of 23 trials
    (gate 90 %) and named in 11 (gate 50 %).
  - **Why it failed (post-hoc).**
    - When the turned axis (pitch) was judged, it read `warn` or `fail`: 24 of 24.
    - The misses are `unchecked` axes (28). In 27 of them the estimate still moved with
      the turn.
    - In 6, the turn itself raised the axis's jackknife std above the 0.1 deg bound.
  - **`camera-imu` v2: refuted, 5 of 7 gates.** v1's claim, scorer and thresholds, on
    five recordings of at least 200 s.
    - Accuracy holds: 0.94 deg and 0.48 ms at worst.
    - Yaw is not constrained on two units, of 260 and 321 s, so recording length does
      not decide observability.
    - cam0's estimates spread 0.81 deg across recordings (gate 0.75).
  - **Leaderboard.** `camera-camera` is a new target pair, standing `refuted`.
    `camera-imu` now has two refuted audits.
  - **Tools.**
    - Selection and download: `tools/hilti_bag_metadata.py`, `tools/hilti_download.py`.
    - Runs: `tools/hilti_camera_sota_run.py`.
    - Scoring and audits: `tools/score_hilti_camera_camera.py`,
      `tools/build_hilti_camera_camera_audit.py`, `tools/build_hilti_camera_imu_v2_audit.py`.
    - Post-hoc analysis: `tools/hilti_camera_camera_posthoc.py`.
  - **Where.** Every run, manifest, score and audit is under
    `docs/assets/hilti2022_camera_sota/`. The write-up is
    `docs/benchmarks/hilti_camera_sota.md`.

- **Hilti's common camera-IMU offset, split into a rig part and a motion part.**
  - **The question.** On `exp21`, attribution names the IMU: one 0.5 deg turn of its
    candidate explains every `camera-imu` error. Is that the rig, or the motion?
  - **New tool.** `tools/hilti_camera_imu_offset.py` fits every camera on all windows,
    on the two halves in time, and on interleaved windows, and pools the cameras into
    one rig offset.
  - **The rig part.** About IMU z (vertical in use), all five cameras read the same
    turn on both recordings: −0.39 ± 0.07 deg on `exp01` and −0.42 ± 0.04 deg on
    `exp21`. A full linear gyro fit keeps it, so it is not the gyro's scale or
    non-orthogonality. It is a rotation between the gyro frame and Kalibr's IMU frame;
    the data cannot tell which of the two is off.
  - **The motion part.** The horizontal components differ between the recordings and
    between `exp01`'s halves, by up to 0.3 deg, while interleaved windows agree.
  - **Docs.** `docs/benchmarks/hilti_camera_imu.md` has the numbers. No verdict rule
    changes.

- **Attribution reads translation too.**
  - **The change.** Pairs that estimate translation (`imu-lidar`, `lidar-lidar`,
    `ins-lidar`, `gnss-lidar`) now add their `t_candidate - t_estimate` to the
    attribution. Each frame's hypothesis is a turn and, where translation is observed, a
    shift along its own axes.
  - **The lever arm.** A parent's turn swings the child's origin: a pair reads
    `t_parent_child × w` from its parent's turn `w`. A 1 deg turn of an IMU with a LiDAR
    1.2 m away is named as the IMU turned, not as the LiDAR moved 2 cm.
  - **Floors.** Translation axes count at least at the translation floor over `sigma_k`
    (6.7 mm by default). Directions pinned down worse than the floors are not fitted.
  - **Unchanged without translation.** Groups with rotation only give the same
    hypotheses, statuses and summaries as before (59 random rigs: largest difference
    6e-16).
  - **On Hilti `exp21`** (first 40 s, `imu-lidar` with the lever arm, two
    `camera-imu`). The baseline is `consistent`. With the LiDAR's candidate moved 5 cm,
    `imu-lidar` reads `warn` and attribution is `ambiguous` between the IMU and the
    LiDAR, each moved 3.4 cm. The cameras are ruled out. One translation pair cannot
    tell its two frames apart.
  - **Schema.** New optional fields: `translation_m` and `translation_std_m` per
    hypothesis, `translation_observations` per group, `sigma_floor_m` on the report.

- **The all-camera check runs about twice as fast, with the same results.**
  - **The stereo matcher.** `camera-camera` searched each patch's ZNCC match with a
    Python loop. It now matches blocks of patches as array operations: box sums give
    each window's mean and energy, and an einsum over sliding windows gives the
    correlations. One pair's estimate on Hilti `exp01` went from 89 s to 27 s.
  - **The camera's pure-rotation RANSAC.** It now fits and scores all 200 hypotheses
    at once, from the same random draws. One `camera-imu` pair went from 76 s to 45 s
    (both under the profiler).
  - **The IMU read.** `load_ros2_imu` keeps its last two reads, keyed by the bag files'
    names, sizes and modification times. A check that pairs one IMU with five cameras
    reads it once instead of five times (14 s each on `exp01`). The returned arrays are
    read-only.
  - **Same results.** On Hilti `exp01` and `exp21` every pair's errors and stds match
    the earlier runs to 1e-11 deg (camera-imu exactly), with the same verdicts, closures
    and attribution.
  - **Run time.** The pairs' run times sum to 530 and 481 s, against 1132 and 1026 s
    before (568 and 543 s wall). Both were measured on a shared 4-core machine: the
    earlier runs beside two other checks, these beside a test run.
  - **Tests.** Both the matcher and the RANSAC are checked against the loops they
    replaced.

- **`camera-camera` fits a vertical scale mismatch instead of reading it as a turn.**
  - **The problem.** A focal length or distortion slightly off in either camera makes
    one rectified image a little taller than the other. The rotation model read that
    as a turn: a 0.5% fy error gives roll −0.17 / pitch +0.34 deg on a synthetic
    forward pair and pitch −0.66 deg on one 60 deg apart.
  - **On Hilti.** This was the cam0–cam4 / cam1–cam4 pitch disagreement (`warn`, and
    a closure `warn`). Refitted with the term, cam1–cam4's pitch is degenerate with it
    and the triangle closes (−0.49 → +0.02 deg).
  - **The fix.** The term is fitted with a 0.3% prior. The rotation is fitted with the
    result removed, and the remaining uncertainty goes into each axis's std
    (`scale_std_deg`) but not into the observability gate.
  - **The result.** The false `warn` is gone, a 1 deg turn of cam3 still fails, and
    attribution still names cam3. 29 camera-pair axes are judged, against 31.
  - **New evidence fields.** `row_scale_ratio` and its std are recorded (optional).
  - **New tool.** `tools/hilti_camera_edge_bias.py` refits a pair by image region and
    with the left-out terms.

- **Closure loops are built from the best-observed estimates first.**
  - **The problem.** The spanning forest took estimates in pair-type order. On a
    camera rig, every camera pair therefore closed through the IMU. Where
    `camera-imu` misses an axis (Hilti pitch), no loop could check that axis.
  - **The change.** The forest now takes the estimates that observe the most axes,
    with the smallest std, first.
  - **On Hilti `exp01`.** Judged loop axes rise from 12 to 14, and pitch checks from 1
    to 3. A camera-only loop flags the cam0–cam4 / cam1–cam4 pitch disagreement:
    −0.51 deg, `warn`.
  - **On `exp21`.** 9 axes as before; one pitch check moves out.

- **`camera-camera` re-centres its rectification and finds errors of several degrees.**
  - **The problem.** The matcher searched ±4 rows and only positive disparities. A
    candidate more than about 0.7 deg off (at 350 px) left the pair
    `estimator_failed`. A 1 deg turn of Hilti cam3 went unseen.
  - **The fix.** A coarse measurement comes first (±12 rows, disparities from −30 px).
    When it reads more than 0.2 deg, the rectification is re-centred on the estimate,
    and the final measurement composes with that turn.
  - **New evidence fields.** The artifact records the turn as `recentred_by_deg`
    (optional), and `vertical_offset_samples` takes `min_disparity_px`.
  - **Synthetic.** 3 deg about each axis, and 5 deg, are recovered to 0.06 deg.
  - **Hilti.**
    - A 1 deg turn of cam3 now fails its camera pairs, and attribution names cam3.
    - 31 camera-pair axes are judged on the two recordings, against 27 before.
    - Linearisation about a wrong candidate left residuals of 0.3 to 1.6 px; they now
      drop to 0.1 to 0.3 px.
    - The side pairs' pitch is now measured. It shows cam3's candidate about 0.38 deg
      off on both recordings, and a 0.5 deg disagreement between cam0–cam4 and
      cam1–cam4 (`exp01` cam0–cam4 is now `warn`).

- **`calibrex check` says which frame's rotation the pair errors point to.**
  - **The fit.** For each connected group of run pairs, every frame is assumed to be
    the only one off. Its turn is fitted to the pairs' rotation errors (a child turned
    by `w` reads `R_parent_child w`, a parent reads `-w`), and each fit's χ² is compared
    with nothing off.
  - **The status.** `consistent`, `attributed` (one frame), `ambiguous` or
    `unexplained`.
  - **Weights.** Weights use the verdicts' resolution: std at least rotation floor /
    `sigma_k`. A direction constrained worse than the rotation floor is not fitted.
  - **Where it appears.** The optional `rotation_attribution` field, the text report
    and the HTML report. Informational: no verdict changes.
  - **On Hilti.** `exp01` is consistent. On `exp21`, a common turn of the IMU of
    about 0.5 deg explains all the camera-IMU errors.
  - **Limit (fixed by the re-centring above).** A 1 deg turn of cam3 went unseen: its
    camera pairs failed to estimate, because the turn moved rows beyond the ±4 px
    search.

- **Hilti camera pairs: the difference between recordings is the rig, not a bias at the
  image edges.**
  - **The tool.** `tools/hilti_camera_pair_reproducibility.py` estimates each camera
    pair on halves of a recording:
    - halves in time, which see different parts of the scene;
    - interleaved halves, which see the same parts.
  - **Within a recording.** Interleaved halves agree within their std (RMS *z* 0.46).
    Halves in time differ by 1.5 to 1.7 times it, by up to 0.19 deg on a judged axis.
    The std is the spread within the frames analysed, not across scenes.
  - **Between recordings.** cam1–cam3's six-sigma change in roll is cam1 turning
    0.18 deg relative to the other cameras, as cam0–cam1 measures. The check's
    behaviour is unchanged.

- Fuse native LiDAR residuals and ordered squared-error accumulation on the
  optional CPU adapter for factors with at least 8,192 rows. Match CPython
  3.10/3.11 and 3.12 summation policies and retain public hooks and diagnostics.
  Avoid discarded optimizer residual lists. Publish full paired controls,
  actual work and the observed small-workload timing regression.

- Fuse optional CPU LiDAR plane scoring and observation gathers for batches of
  at least 8,192 rows. Preserve ordered FP64 scores, native distance ties,
  signed zeros, independent read-only output columns and small-batch fallback.
  Publish all fresh whole-process controls and actual GPU/CPU work; the
  incremental gain over the preceding GPU implementation is modest.

- Reuse the last exact CUDA LiDAR finite differences with bitwise transform/step
  keys, owned writable return arrays and a 64 MiB host-cache limit. Compute only
  initial solver rank for native compiled factors with at least 8,192 residuals;
  retain full final diagnostics and public evaluation hooks. Publish complete
  public dense GPU/CPU controls, actual launch/cache counts, small-workload timing
  regressions and stopped comparisons with source/input lineage.

- Query captured continuous-time LiDAR uniform/adaptive plane maps together on
  an owned CUDA stream, preserving neighborhood order and native boundary/tie
  repair. Batch large-frame deskew on the optional CPU adapter with exact
  timestamps, poses, world points and failure counters; keep small frames on
  the original replay path. Stream diagnostic XYZ without materializing unused
  observations. Record actual GPU/CPU work and publish complete paired controls,
  cold/slower runs and the preceding small-frame regression.

- Add optional CUDA numerical finite differences for dense
  continuous-time LiDAR refinement. Retain all thirteen evaluations, exact
  arithmetic and native ordered CPU normal equations. Share compiled kernels
  across owned streams, retain cheap standalone residuals on CPU, and keep
  small capture budgets on CPU without importing CUDA. Reuse native selected
  timestamp arrays and count backing allocations in the bounded pose cache.
  Record actual execution and publish full public-data comparisons,
  including slower preceding attempts and initially empty compiler caches.

- Keep optional compiled continuous-time LiDAR poses and observations in read-only
  numeric batches through association and MAD filtering. Fuse finite differences,
  avoid unused Jacobian conversions, and evaluate large independent clock
  candidates with three owned threads without nested worker pools. Preserve
  numerical values, diagnostic counters, custom-method fallbacks and schemas;
  publish complete public-data comparisons and all preceding attempts.

- Add optional fused CUDA point-to-plane LiDAR odometry, including on-device
  normal covariance preparation and captured spatial queries. Retain native
  time-gap scaling and deskew policies, record actual device work and native
  repairs, and expose explicit coordinate canonicalization for reproducible
  dense geometry. Preserve CPU defaults and publish full public temporal replays.

- Add an optional CUDA backend for LiDAR-LiDAR map registration and blocked
  evidence evaluation. Keep point queries, normal preparation, residuals,
  analytic Jacobians and Huber systems on device, with native uncertainty
  repairs and unchanged calibration/evidence policies. Record execution
  provenance and publish complete fresh-process public-data comparisons.

- Retain continuous-time numerical finite differences and robust normal equations
  in numeric arrays with the optional CPU backend. Reuse captured observation
  fields, batched ordered hybrid selection and prepared MAD arrays/medians,
  plus bounded pose batches and unambiguous adaptive gates,
  preserving exact output values, support-floor ties, custom-method fallbacks
  and schema stability. Publish independent initially empty compiler-cache
  comparisons against the preceding compiled-query version.

- Add optional compiled CPU kernels for continuous-time interpolation, world
  coordinates and repeated point-to-plane residuals, preserving arithmetic
  order and numerical finite differences. Batch exact adaptive centroid queries
  with scalar gate/tie handling and unsupported-map fallbacks. Record actual
  optional kernel execution in provenance; retain the Python default and result
  schemas. Publish complete public-data comparisons with empty initial compiler
  caches and all saved numerical results checked.

- Reuse captured uniform plane geometry, bounded adaptive neighborhoods and
  odometry poses/slerp constants during continuous-time refinement. Batch moving
  world coordinates in the scalar Hamilton order and overlap complete input
  hashing with computation, rejecting changed inputs. Publish separate complete
  public-data comparisons at the existing and default sampling budgets, with
  exact saved-artifact parity and unchanged temporal holdout.

- Search only occupied adaptive voxel cells in the original level/x/y/z order,
  preserving nearest ties, scalar distances and inclusive gates. Publish fresh
  complete continuous-time comparisons on the public GLIM Livox identity control,
  with exact saved-artifact parity, actual sampled train/holdout identities and
  verified raw/converted input lineage.

- Reuse captured Livox records for solid-state quality bins and native input
  loading; decode checked binary PCD views and preserve ordered statistics.
  Evaluate independent controls with bounded CPU workers and isolated query caches,
  defer unused spatial-tree imports, batch static plane association and accumulate
  native normal equations in the original product/addition order. Preserve solver
  history, diagnostics and quality; publish separate full-point and existing-budget
  complete-process comparisons against the preceding accelerated version.

- Batch native fixed-rig point-to-plane residuals in bounded observation chunks,
  preserving Hamilton products, normalization counts, finite differences and
  complete solver history. Retain scalar handling for small/nonfinite inputs and
  refresh observation arrays on replacement. Publish incremental full-process
  and optimizer timings on the existing public Livox configuration, with exact
  result parity and unchanged observability/quality judgments.

- Pack exact Livox voxel neighborhoods within the existing 8 MiB lookup budget
  and share read-only transformed coordinates between voxel and plane evidence.
  Preserve ordered voxel sums, first-seen keys, nearest ties and all control cases.
  Measure candidate/control evaluation within the same complete calibration run
  and publish phase and whole-process gains separately, with exact result parity.

- Share Livox captured inputs and statistics across inspection, candidate/control
  evaluation and registration within one calibration run. Reuse source voxel sums
  and immutable plane arrays, cap compact direct-index tables at 8 MiB each and
  avoid distances for empty neighbors. Preserve all numerical outputs and fresh
  inputs on the next run. Publish separate cumulative, incremental, intermediate
  and interrupted-worker timing evidence without mixing their measurement scopes.

- Accelerate Livox candidate and perturbation evaluation with invocation-scoped
  decoded records, invariant statistics and source planes. Batch exact 27-voxel
  associations using checked sparse integer keys, and preserve the scalar
  Hamilton-product ordering in batched coordinate transforms. Keep all input
  points, controls, support denominators and quality decisions; retain complete
  public calibration subprocess timings and numerical parity evidence.

- Vectorize native registration voxel centroids, point ICP information matrices
  and batched target-normal PCA, retaining input-order sums and existing numeric
  behavior. Reuse training coordinates and run independent multi-start trials
  with bounded CPU threads on larger clouds. `multi_start_workers=1` selects
  sequential execution; the automatic default preserves ordered diagnostics.
  Public raw-scan evidence includes fresh verification/preprocessing, both full
  ICP calls, sequential controls and all repeated numerical payloads.

- Add caller-owned immutable, verified Camera–LiDAR input snapshots for repeated
  six-DoF estimates. Public demos and sweeps opt in with
  `--reuse-verified-inputs`; pipeline summaries include initial capture once.
  Check the problem SHA-256 on every estimate, bind loaded arrays to verified
  bytes, and record snapshot scope/cost. Default runs still verify fresh files.

- Batch ordered CUDA D2D histograms in groups of at most 32 frames. Share pose
  and histogram-edge uploads, reuse invocation-local projection scratch, and
  download each group's counts together. Preserve native frame reduction,
  exact visibility/tie rules, and record histogram download counters.

- Fuse CPU D2D projection, visibility, and histogram accumulation in the
  optional `numba_cpu` backend. Freeze feature bins, reuse thread-local scratch,
  visit only touched pixels, and release the GIL for frame parallelism. Keep
  float64 projection, histogram edge rules, visibility counts, and tie order.

- Parallelize independent D2D frame evaluations inside one six-DoF estimate
  with persistent CPU threads. `estimate-six-dof --frame-workers` and public
  demos' `--candidate-frame-workers` preserve frame reduction order, full search
  traces, and holdout evaluation; results record requested/effective workers.
  Defaults remain serial and CUDA requires one frame worker.

- Add `calibrex demo kitti-six-dof-sweep` for six prespecified signed-axis
  initializations on the same public KITTI fit/holdout frames. Save a plan before
  optimization, retain all case evidence, and compare numerical traces within
  each case. CPU-only sweeps support isolated worker processes via `--workers`;
  CUDA comparisons stay serial. The overview links every case report.

- Add `--repetitions` to the public KITTI six-DoF demo. Reuse one frozen
  problem and initial pose, alternate CPU/candidate order, retain every attempt,
  and report successful-run timing medians/ranges. Exact parity is checked across
  repetitions as well as backends; image overlays keep the first pair.

- Add `calibrex demo kitti-six-dof` to compare NumPy and an explicitly selected
  CUDA/CPU backend on public KITTI raw data with a frozen depth provider. The
  command fixes a 20-fit/5-holdout split and controlled initial perturbation,
  saves standard results and a benchmark, and produces offline image-overlay
  HTML. Partial runs and backend/parity failures remain in the evidence. No
  rosbag or schema change is required.

- **`calibrex external-run import-sensor-calib-tas` imports a sensor_calib_tas result
  as a `--tf` file.**
  - **What sensor_calib_tas outputs.** Each sensor's pose in its reference sensor's
    frame, as `<frame>_calibrated` static transforms or the latched
    `~/sensor_poses` message. It writes no file.
  - **The import.** It reads either one (from a bag, or from `ros2 topic echo` YAML)
    and writes a `slac.check_frames` file under the sensors' own frame names. Given
    with `--tf`, that file overrides the deployed calibration on a driving bag.
  - **The test.** On a synthetic driving bag whose deployed LiDAR rotation is 3 deg
    off, the check fails; with the imported result it passes.
  - **Schema.** `slac.check_frames` gains an optional `provenance`: generator, tool and
    license, and the source's SHA-256.

- **A rectified image stream's intrinsics come from CameraInfo's `P`, without
  distortion.**
  - **The bug.** An `image_rect*` topic (ROS image_pipeline) carries undistorted images,
    but the check applied the raw image's `K` and `D` to it.
  - **Where it showed.** sensor_calib_tas's demo bag. Its front camera publishes
    `image_rect_color` next to a plumb_bob CameraInfo (k1 = −0.27).
  - **On that bag.** The demo bag itself is a parked, 4.9 s target session: the check
    plans its camera-LiDAR and stereo pairs but cannot judge them (see
    `docs/benchmarks/camera_lidar_check.md`).

- **`camera-camera` skips cameras that share no view, and the check covers all five of
  Hilti's alphasense cameras.**
  - **The skip.** The pair measures the share of the first camera's view the second
    sees under the candidate. Below 5% it is skipped with the new reason code
    `no_shared_view`, and rectification refuses cameras that face opposite ways.
  - **All five cameras.** With Hilti's per-camera camchains, `calibrex check` plans 10
    camera pairs and 5 `camera-imu` pairs.
  - **Camera pairs.** cam0–cam1 passes on every axis on both recordings. The side and
    down pairs share 12 to 34% of their view, so they are judged only in part, and
    their readings differ between recordings by up to 0.3 deg. The opposite side
    cameras are skipped.
  - **`camera-imu`.** It passes on 9 of 10 runs; `exp21` cam1 is `warn` at 0.51 deg.
    Its clock offsets (1.5 to 2.0 ms) agree with Kalibr's `timeshift_cam_imu`
    (1.7 to 1.9 ms).

- **`camera-camera` checks unsynchronised stereo pairs with a gyro, and reports the
  cameras' clock offset.**
  - **When it applies.** An IMU must be connected to the first camera in the candidate
    tree. Images are then paired with the other camera's nearest image within 30 ms.
  - **The method.** Each pair is rectified for the rig's turn over its stamped gap. The
    rotation error and the clock offset are fitted together, and the offset is reported
    (not judged) as the pair's `time_offset`.
  - **Evidence.** The artifact gains optional `time_offset_s`, its std and status,
    `timing_source` and `max_stamp_gap_s`.
  - **Pairing.** It now takes the nearest image within the tolerance.
  - **Tests.** Rendered pairs taken 25 ms apart (at 25 and 80 deg/s) read their
    rotation within 0.05 deg and a zero offset. A 5 ms stamp offset reads −5 ms.
  - **Hilti (`tools/stereo_timing_injection.py`).**
    * The synchronised bags read 0 ± 0.15 ms.
    * A 5 ms stamp offset reads −4.9 ms.
    * Pairing each image with the next frame (25 ms) still passes. The rotation std
      grows to 0.08 deg, and without the gyro one axis goes unobserved on `exp01`.

- **`calibrex check` judges a stereo pair's rotation: the `camera-camera` pair.**
  - **Planning.** It is planned for every two cameras the candidate tree connects
    (`T_first_second`). Intrinsics come from a Kalibr camchain or `CameraInfo`, and
    the images must be taken within 2 ms of each other.
  - **The method.** It rectifies the pair with the candidate and fits the right image's
    row offsets (`estimate_stereo_rotation`). It judges roll, pitch and yaw about the
    first camera, each with a block jackknife and a 0.2 deg control. The translation
    is not judged, and `calibrex estimate` does not export the pair.
  - **Evidence.** A `slac.stereo_rotation/v0.1` artifact (new schema). The pair-name
    enums of the check, estimate and drift schemas widen.
  - **Supporting changes.** `--camera` keeps a pair when either of its cameras is
    selected. Rectification also takes radial-tangential pinhole cameras
    (`RadtanCamera`, checked against OpenCV). Patch matching is vectorised: the same
    samples, ten times faster.
  - **Hilti 2022 (`exp01`, `exp21`).**
    * The dataset camchain passes, with every axis observed (std 0.002 to 0.06 deg).
    * A camchain with cam1 turned 1 deg fails on `exp01` (1.01 deg). On `exp21` it
      reads 0.895 deg, so `warn`: the turn about the vertical axis is the weakest.

- **The Hilti pair's vertical offset is a relative pitch of the cameras that differs
  between recordings (development run).**
  - **The new function.** `vertical_offset_samples` returns every matched patch's
    vertical offset with its pair, place, disparity and whether it constrains rows.
    `estimate_vertical_offset` is their median, unchanged.
  - **The tool.** `tools/hilti_vertical_offset.py` fits the offset's constant,
    disparity, column and row terms, and correlates each frame's offset with the
    gyro's rate about the baseline.
  - **Result.** The offset is a constant, stable within each recording: −0.81 px on
    `exp21` (2022-04-21) and +0.38 px on `exp01` (2022-03-03). That is a relative pitch
    0.19 deg apart, against one calibration file.
  - **Ruled out.**
    * The scene: the disparity term is at most 0.004 px per px.
    * Timing: the correlation with an angular rate of up to 26 deg/s is weak, and the
      implied time offsets disagree in sign.

- **The Hilti data-only decision passes on a second recording, `exp01`, with depth
  judged (development run).**
  - **The choice.** The indoor construction site `exp01_construction_ground_level` was
    chosen before it was scored, because `exp21` is outdoors. The tool and the
    calibration are unchanged.
  - **Result.** The reference passes on roll, pitch, yaw and z (+0.005 ± 0.014 m). The
    known-bad fails on pitch (4.85 deg) and z (+0.245 m).
  - **What it says about `exp21`.** Its −0.1 m depth offset does not repeat here, so it
    belongs to that recording.
  - **Caveats.** The joint stereo fit diverges on `exp01`. The pair's vertical offset
    is +0.33 px here against −0.83 px on `exp21`.

- **The Hilti data-only decision reads its frames as `calibrex check` does, and
  passes on rotation (development run).**
  - **The fix.** `tools/hilti_stereo_falsification.py` said it used the check's frames
    but paired images and scans by their stamps and left the points where they were
    measured. It now does what the check does for each candidate: it pairs each image
    with the scan at the time the candidate's visible points were measured (per-point
    `timestamp`), and turns the points to the image time with the IMU's gyro.
    `--stamp-pairing --no-motion` reproduces the earlier run.
  - **Result on `exp21` (48 frames): `pass`.**
    * The reference passes on roll, pitch and yaw; the known-bad fails on pitch
      (4.95 deg).
    * Stereo depth puts the known-bad 0.250 m beyond the reference. Its std, 0.053 m,
      is just above the 0.05 m gate, so z is not judged.
    * Depth still reads the reference about 0.1 m short, which is unexplained.
  - **Tried and rejected: a disparity scale in the depth estimate**
    (`tools/stereo_depth_scale_pilot.py`).
    * A free scale is not separable from depth on camera-facing surfaces: the std
      grows 3 to 10 times.
    * Removing the joint fit's scale fixes Hilti's reference, but moves KITTI 0005's
      first-50 reference beyond its tolerance. That scale also differs between frame
      selections of one KITTI rig (−0.2% against −1.3%).
    * The estimator is unchanged; see
      [the page](docs/benchmarks/hilti_fisheye_stereo_falsification.md).

- **Fisheye stereo pairs feed the stereo depth checks, and the data-only decision runs
  on Hilti 2022 (development run).**
  - **Rectification.** `calibrex.evaluation.fisheye_stereo` rectifies an equidistant
    (Kannala-Brandt) pair read from a Kalibr camchain onto one pinhole camera.
    `estimate_vertical_offset` measures the rectified pair's leftover row offset from
    the images alone, and `with_right_row_shift` removes it.
  - **Reuse.** `judge_candidate` runs the KITTI decision's per-candidate steps on any
    frames. KITTI's results are unchanged. `tools/hilti_stereo_falsification.py` runs
    the decision on a Hilti bag.
  - **First result on `exp21` (47 frames): `inconclusive`,** superseded by the entry
    above: the reference's z read −0.137 m against a 0.111 m tolerance.

- **The joint stereo fit models the stereo rig's scale and roll.**
  - **The change.** The rig's disparity scale and the roll between its cameras are
    fitted beside its offset (`rig_terms`, recorded as `disparity_scale` and
    `rig_roll_px`). Each iteration's correction is bounded (0.2 m, 2 deg per axis).
  - **Effect on KITTI.** The reference's x readings across drives and frame selections
    now span 8 cm instead of 11.4 cm, three of four within 1.3 cm. x and y stay unjudged.
  - **Ruled out.** Compensating the LiDAR sweep with OXTS motion did not reduce the
    spread (development run).

- **A known-bad that passes with its error unseen makes the KITTI data-only decision
  `inconclusive`, not `fail`.**
  - **The declared axes.** They come from the protocol's declaration of the known-bad (in
    the camera's frame, beyond 0.5 deg or 0.02 m from the declared reference): pitch and
    z. Each decision records them as `known_bad_axes`.
  - **The rule.** A known-bad that passes with none of them judged is `inconclusive`. One
    that passes with any of them judged still `fail`s. The trial's own verdict (what
    `calibrex check` reports) is unchanged.
  - **Effect.** The only row that changes is 0009's first 50 frames: `fail` becomes
    `inconclusive`.

- **KITTI data-only rotation is judged after removing the depth error stereo measures.**
  - **The change.** When stereo depth trusts its reading of the candidate's error along
    the optical axis, the edges run again from the candidate with it removed, and the
    depth is measured again at their new rotation. Rotation is still judged as the
    candidate's error, and the shift is recorded (`edge_start_depth_shift_m`).
  - **Result on 0009.** On the spread frames the known-bad's pitch is now judged and fails
    (control t 1.99 to 2.2).
  - **Result on 0005.** The reference passes on both frame selections; its readings
    barely move (z 0.050 m, yaw 0.50 to 0.40 deg) but fall inside the tolerance, so
    the data-only status is `pass` (was `inconclusive`). The page says the order was
    added for 0009's pitch.
  - **Still failing.** On 0009's first 50 frames the known-bad still passes.

- **Joint six-DoF stereo fit, recorded as a cross-check.**
  - **Method.** `estimate_stereo_alignment` fits translation and a small rotation together
    from stereo depth on planar LiDAR surfaces of any facing (normals from neighbourhood
    PCA). It re-linearises at the corrected transform, with a jackknife per axis and a
    moved-candidate control per translation axis. It is tested on a rendered street with
    house fronts. The facing-surface depth estimate shares its matcher, unchanged.
  - **KITTI.** The data-only trials record it as `stereo_alignment`, and the rendered
    summary lists its readings. `x` and `y` stay unjudged: the reference's x reads from
    -4.6 to +7.3 cm across drives and frame selections, against per-run jackknife stds of
    1 to 3 cm, and y is untrusted on three of four runs.

- **Fixed: KITTI camera-LiDAR pairs after a missing scan loaded the wrong scan.**
  - **The bug.** KITTI raw streams can skip frames while `timestamps.txt` keeps a blank
    line for each; drive 0009 has no scans 177-180. `find_camera_lidar_pairs` took a scan's
    timestamp from its line but its file from its position in the directory, so every pair
    after the gap loaded the scan four frames (0.4 s) later. The LiDAR-OXTS sampling had
    the same lookup.
  - **The fix.** Files are now taken by the frame number in their name, as the drive
    loader and the rosbag converter already did. Drives without gaps (0005) are
    unchanged.
  - **The data-only decision.** It also drops pairs more than 60 ms apart, as the
    bag-level check does.
  - **Published results.** The 0009 replication was republished. Every rotation axis of
    the reference is now judged, and the spread decision passes. The known-bad's pitch is
    not judged there (control t = 2.0), and on 0009's first 50 frames it still passes.

- **The KITTI data-only decisions are repeated on drive 0009.**
  - **Why 0009.** It was recorded with the same calibration files (checked by SHA-256),
    so its reference and known-bad candidates are the same transforms.
  - **Artifact and tool.** The artifact gains `data_only_replications` (optional,
    schema-compatible); `tools/publish_kitti_falsification_benchmark.py` gains
    `--replicate`.
  - **Result: the 5 cm belongs to drive 0005.** On 0009 the reference reads under 1 cm
    along the optical axis, against 5 cm on 0005, and the known-bad's 0.25 m fails again.
  - **Result: rotation on 0009.** The edges judge no rotation on its spread frames. On
    its first 50 frames the known-bad passes with its pitch unjudged, a `fail` of the
    data-only decision.
  - The benchmark page and the published test record all of it.

- **The KITTI data-only decision also judges translation along the optical axis.**
  - **Why edges cannot.** Moving the candidate along the optical axis keeps a road point on
    the road and a wall point on its wall, so the shifted cloud still lines up. In a
    development run the edge estimate ran to its search bound, with its 5 cm controls
    undetected.
  - **Stereo depth.** `calibrex.evaluation.stereo_lidar_depth` block-matches LiDAR points
    on camera-facing surfaces between the rectified stereo images. It fits the disparity
    residual as a rig offset plus the candidate's depth error (which fades as `1 / Z^2`),
    with a block jackknife, a moved-candidate control and an occlusion filter. It uses
    only the stereo pair's own calibration.
  - **Artifact.** Each data-only trial records the stereo estimate (`depth`), its `z`
    judgement (`depth_axis`) with the `calibrex check` rule, and
    `unchecked_translation_axes`. All fields are optional and schema-compatible.
  - **Result.** The known-bad fails on its 0.25 m (0.26 to 0.28 m against 0.035 to
    0.082 m) as well as its 5 deg, on both frame selections. The reference is not
    passed: on the spread frames stereo depth puts it 0.050 m along the optical axis
    against a 0.050 m tolerance (`warn`). The data-only status is therefore
    `inconclusive` (it was `pass` with rotation alone). The benchmark page says which
    choices were made after seeing results.

- **The KITTI falsification is also decided from the images and scans alone.**
  - **Method.** `calibrex.evaluation.kitti_edge_falsification` starts the camera-LiDAR
    edge alignment (block jackknife, held-out known-bad controls) at each candidate and
    judges it with the per-axis rule of `calibrex check`. It passes when the reference
    passes and the known-bad fails.
  - **Artifact.** `data_only_status` and `data_only` (optional, schema-compatible) carry
    one decision on 50 frames spread over the drive and one on the protocol's first 50.
    The protocol's own status is unchanged.
  - **Result.** Spread frames: the reference passes (within 0.1 deg) and the known-bad
    fails on its 5 deg turn (pitch 5.0 against 0.5 deg). The protocol's first 50
    frames: the reference sits at its yaw tolerance (`warn`), so that row is
    inconclusive. Translation is not judged. The page says the selection was chosen
    after both were run.

- **KITTI camera-LiDAR falsification published as a maintained artifact.**
  - **Published files.** `docs/assets/kitti-camera-lidar-falsification.json` (schema-valid) and
    its rendered summary, generated by `tools/publish_kitti_falsification_benchmark.py` on
    the official drive 0005 (50 frame pairs).
  - **CI test.** `tests/integration/test_kitti_falsification_published.py` checks the
    committed files on every CI run.
  - **New per-trial fields.** Each trial now records its policy gates, its headline metrics
    and its LiDAR-camera evidence checks; trial file paths are relative to the run.
  - **Provenance.** The git commit is read at the start of the run.
  - **Result.** The reference passes and the declared known-bad (+5 deg yaw, +0.25 m)
    fails. The known-bad is rejected only by its comparison with KITTI's own calibration;
    every check of the images and scans passes it. The benchmark page says so.

- **VLS-128 packets are decoded.** `velodyne_msgs/VelodyneScan` from a VLS-128 / Alpha Prime
  (product id 0xA1) is decoded in single-return mode (strongest or last). Like an HDL-64E it
  needs the unit's calibration (`db.xml` or ROS YAML).
  - **Decoding.** The four laser banks of a firing sequence are read, along with the firing
    timing (2.89 us cycle, firmware 5.2.3.0 and later). The azimuth advances inside a sequence
    with LidarView's 0.01 deg rounding.
  - **Checked against LidarView's 2019 `HDL_128` reference.** The channel order is read back
    from frame 0, and frames 1 and 2 match exactly (median 0.000 mm). At the earlier
    firmware's 2.666 us cycle, every point time matches up to one constant.
  - **What is not tested.** That reference used an unpublished calibration, so the laser table
    is not tested by it. `Alpha_Prime.xml` as published gives this recording flatter
    surfaces than the 2019 order does.
  - **Calibration files.** `load_velodyne_calibration` reads 128-laser files, including
    LidarView's `Alpha_Prime.xml`, which repeats its XML declaration and spells
    `horizOffsetCorection`.
  - **Data.** `lidarview_velodyne` downloads the `HDL_128` recording and `Alpha_Prime.xml`.
    VLS-128 dual return stays refused by name.

- **Documented: the camera-LiDAR known-bad control near its threshold.** Doubling the frames,
  8 or 12 blocks, and a rank test were each tried on the KITTI pairings where an axis went
  unjudged. In each case a held-out block prefers the turned fit, so the data, not the rule,
  are the limit. Twelve blocks of 96 frames would judge one more axis at about four times the
  jackknife cost. The rule and the defaults are unchanged.

- **Camera-LiDAR rolling sweep evaluated on a handheld recording.** Hilti 2022 exp21 (first
  6 GiB, the PandarXT-32 cloud with per-point time, the Alphasense gyro picked automatically):
  - **Reference.** Moving each point to the image time raises the edge-alignment objective at the
    reference from 0.360 to 0.387. About 6 % comes from the motion itself and 2 % from per-point
    timing.
  - **Perturbations.** All six rotation errors are flagged (five as recorded: roll +3 deg was
    inconclusive). Yaw +1 deg moved from fail to warn because the yaw std grew.
  - **New tool.** `tools/trim_partial_rosbag1.py` cuts a range-downloaded ROS 1 bag after its
    last complete record.

- **Camera-LiDAR check moves LiDAR points to the image time.**

  On a moving rig the LiDAR has turned and moved between a point's measurement and the image.
  `calibrex check --pairs camera-lidar` now moves every point to the image time with the rig's
  velocity.
  - **Source.** An odometry or twist topic, else an IMU (rotation only), whose frame is
    connected to the LiDAR in the frame tree. Each point is moved by its own time when the
    cloud has a per-point time field.
  - **Off switch and record.** `--camera-lidar-motion none` keeps scans as recorded (recorded in
    the run options). The edge evidence gains an optional `motion_compensation` block.
  - **New tool.** `tools/camera_lidar_motion_stress.py` pairs every scan with the image one
    frame away to emulate an unsynchronized camera.

  **Results.**
  - Synchronized KITTI: 0005 now flags all nine rotation errors (yaw +1 deg moved from pass to
    warn, yaw tolerance 1.35 to 0.5 deg) and 0009 pitch +1 deg moved from warn to fail. The
    references still pass; AIST Livox is unchanged (no velocity topic).
  - Emulated 0.1 s pairings: 13 of 16 rotation errors flagged (12 as recorded, which never
    judged pitch). On 0005 with the earlier image, yaw went unjudged and yaw +1 and +3 deg
    passed; the benchmark doc gives every case.

- **Camera-LiDAR check names an image in the camera body frame.**

  When most frames of `calibrex check --pairs camera-lidar` put no LiDAR edge in the image, the
  frames are read once more as if the image frame were the camera's REP 103 body frame
  (x forward) instead of its optical frame. If that makes them usable, the `inconclusive`
  reason says so and names the fix: an optical child frame in a `slac.check_frames` file and
  `--frame-map TOPIC=FRAME`. On KITTI 0005 with the camera frame replaced by its body frame the
  reason names it (16 of 16 frames), and the named fix brings the check back to `pass`. Nothing
  is corrected automatically.

- **Camera-LiDAR check: short recordings use their frame budget.** A recording shorter than 48
  scans 1 s apart (2.5 s with `--max-duration-s 120`) now spreads up to 48 scans, at least 0.25 s
  apart, over its length.
  - **When it applies.** Only when the view changes: the median mean absolute difference of
    consecutive images must reach 0.05 of full scale. Static recordings repeat one view and keep
    the regular selection.
  - **What stays the same.** Long recordings get exactly the frames they got before. The evidence
    notes say which selection was used.
  - **Options.** `EdgeAlignmentOptions.short_recording_spacing_s` (0.25) and
    `short_recording_min_view_change` (0.05). `load_camera_lidar_frames` gains the matching
    arguments and a `report`.

  **Results.**
  - KITTI 0005 goes from 16 to 48 frames. Yaw +5 deg is now flagged (`fail`, 4.9 / 1.14 deg),
    and the reference passes with roll, pitch and yaw judged.
  - All three +5 deg cases the first benchmark left unchecked are now flagged. The verdict rule
    (2 deg control, 0.5 deg std bound, t >= 2) is unchanged.
  - The AIST Livox verdicts are unchanged. The Hilti bags were not re-run.

- **Kalibr lever-arm evidence and a public Kalibr run.**

  `calibrex external-run evaluate-camera-imu` now checks the imported `T_cam_imu` translation
  against the accelerometer when the recording carries camera positions and accelerometer
  samples. These are the new optional `position_m` and `imu_accel` fields of
  `slac.camera_imu_motion_recording/v0.1`.
  - **Comparison.** The camera trajectory's second divided difference over +-0.25 s is
    compared with the accelerometer integrated against the same kernel. A bias and the
    gravity vector are fitted on training windows only.
  - **Controls.** Six signed +-5 cm controls must raise the holdout residual. One that lowers
    it fails the candidate; insensitive controls leave the translation inconclusive.
  - **Output.** The result is the new optional `lever_arm` block of
    `slac.external_camera_imu_evidence/v0.1`. Its gates are new optional threshold fields.
  - **Synthetic fixture.** It gains camera positions, a biased accelerometer in a tilted
    world frame, `--translation-error-m` and `--no-accelerometer`. The truth passes and
    +-5 cm errors on every axis fail.

  New `tools/aprilgrid_camera_imu_recording.py` (OpenCV, outside the core) turns a Kalibr
  AprilGrid recording into such a recording.

  **Public run.** UZH-FPV's Kalibr camchain for the Snapdragon forward camera, fitted on the
  indoor calibration recording, is evaluated on the separately captured outdoor one
  (`tools/download_public_dataset.py uzh_fpv_kalibr_camera_imu`; dataset license not stated,
  data outside the repository).
  - The published translation passes: holdout specific-force RMSE 0.102 m/s^2, 6/6
    controls. Every +-5 cm perturbation fails.
  - Rotation and time are not admissible there, because frame-to-frame AprilGrid pose noise
    gives 0.171 rad/s against the 0.05 budget.
  - The opt-in test is `tests/integration/test_uzh_fpv_kalibr_camera_imu.py`.

  Schemas regenerated; the additions are optional and compatible.
- **Velodyne packets: dual return, HDL-64E, and checks against reference clouds.**
  `calibrex.data.velodyne_packets` now decodes:
  - VLP-16 dual return, dropping a second return identical to the first;
  - the HDL-64E, single and dual return. It needs the sensor's own calibration
    (`load_velodyne_calibration`: Velodyne `db.xml` or ROS `velodyne_pointcloud` YAML), passed
    with `--calibration` to `tools/velodyne_scan_to_pointcloud2.py`, and the conversion summary
    records its digest.

  The VLP-32C decoder applies the head's turn between laser pairs within a block. New public helpers
  are `velodyne_packet_format` and `decode_velodyne_packet` (per-laser ring, return and time).

  `tools/validate_velodyne_decode.py --pcap X.pcap --reference X_n.vtp` compares the decoding with
  Kitware's Apache-2.0 LidarView test recordings, fetched by
  `tools/download_public_dataset.py lidarview_velodyne` (pinned commit, SHA-256 checked). Median
  point distances are 0.17 mm (VLP-16 dual), 0.05 mm (VLP-16 and VLP-32C single; 1.2 mm before
  the intra-block azimuth step) and 0.000 mm (HDL-64E single and dual). Per-point times agree to
  0.2 to 0.35 us. The opt-in test is `tests/integration/test_velodyne_lidarview_reference.py`.

  **Behaviour change:** product ids 0x21 (HDL-32E) and 0x24 (VLP-16 Hi-Res) were decoded as a
  VLP-16 and are now refused, as are VLP-32A/B (0x23; an 11 mm median miss with VLP-32C
  constants), VLS-128, Velarray and VLP-32C dual return. `calibrex check` names the packet format
  of a skipped `VelodyneScan` topic. No schema change.
- **Planar-board line+plane in the CLI.** The `native_planar_board` backend runs a line+plane row
  (Zhou et al.) next to the plane, point+plane and Horn rows, on the same seeded capture split. A
  new adapter, `calibrex.solvers.acfr_board_edges`, turns ACFR's camera and LiDAR board corners
  (rows 2-5, 8-11) into four boundary lines per capture. It orders each corner loop about its
  oriented board normal, picks the LiDAR labeling (the board is symmetric under a 180° turn) by
  edge-direction agreement under the train-split plane-only rotation, and rejects pairings whose
  edge lengths differ by over 15 %. Every step is recorded per capture. The line+plane solver
  gained twelve held-out ±5 cm / ±5° controls (`LinePlaneProbe`). On ACFR's 40 VLP-16 captures, all
  match (6 symmetric) and the row passes its default gates: held-out boundary distance 1.39 cm,
  direction 1.51°, 12/12 controls. Keeping the source row order instead gives 37 cm. There are
  nine new `line_plane_*` metrics and the opt-in test `tests/integration/test_acfr_line_plane.py`.
  No schema change.
- **Camera-LiDAR check: wider start and solid-state LiDAR.** The edge-alignment rotation search
  also starts from the per-axis maxima of the objective profile (+-8 deg) and keeps that start only
  when more rotation DoFs pass their held-out control and std bound there (on the Hilti exp07
  corridor a plain "higher objective" rule picked a ridge and lost the judged roll). KITTI 0005
  pitch +5 deg (was `pass`, pitch unchecked) and KITTI 0009 pitch +5 deg (was `inconclusive`) now
  `fail` at 4.95 / 0.5 deg; the other KITTI perturbations and the unmodified references on KITTI
  0005, 0009 and Hilti exp21, exp07 read as before, and KITTI 0005 yaw +5 deg stays unchecked
  (jackknife std 0.52 deg over the 0.5 deg bound). Clouds without beam structure (Livox-style
  non-repetitive scans) now get near-side depth edges among angular neighbours, with a jump that
  must reach 10 % of the range. On the public AIST Livox Avia dataset (Zenodo 7779880, static) the
  check judges all three rotation axes on two of five recordings where it previously dropped every
  frame. New `tools/restamp_rosbag2_headers.py` writes the bag receive time into header stamps for
  static recordings whose sensors stamp on separate clocks; `examples/public_datasets/
  aist_lidar_camera_livox/` holds the reference candidate. The verdict rule (2 deg control,
  0.5 deg std bound, t >= 2) is unchanged. No schema change.
- **Calibration CI decisions recorded in the lifecycle registry.** New `calibrex lifecycle
  record-ci <calibration-ci.json>` (`calibrex.calibration_ci_lifecycle`, artifact
  `slac.calibration_ci_lifecycle/v0.1`) copies the CI decision, candidate result, CI assessment
  and CI policy into a content-addressed `ci/<sha256>/` snapshot in the registry (written once;
  differing bytes are refused) and evaluates the candidate with the decision as evidence: PASS with
  PASS promotion and smoke evidence is promoted with `--promote`, INCONCLUSIVE/WARN is held and
  FAIL is rejected without touching the incumbent, and `--rollback-on-fail` rolls back a promoted
  incumbent that a later decision fails, with the decision bound to the rollback event
  (`rollback_lifecycle` gained an optional `evidence` reference). A candidate changed after the
  decision is refused, a decision cannot be recorded twice, and the registry is verified after
  every path. Integration test on real candidates (KITTI evidence demo PASS, precomputed example
  INCONCLUSIVE, the KITTI candidate under a 90 % known-bad policy FAIL); example workflow
  `examples/ci/calibration-ci-lifecycle.yml`.
- **Declared LiDAR capture-time reference in online replay (ADR 0008 Pillar 3).** New
  `sensors.<lidar>.capture_time_reference` (`scan_start` | `scan_midpoint` | `scan_end`) with
  `scan_period_s` declares where a message stamp sits in its sweep. Motion-compensated
  rosbag1/rosbag2 replay places a sensor that is not deskewed per point at its mid-sweep capture
  time (stamp + period / 2 for `scan_start`, - period / 2 for `scan_end`); per-point offsets, when
  used, already carry the convention and no shift applies. Provenance records each declared
  reference with `applied_shift_s` and `basis`; other dataset types refuse the field. Synthetic
  shared-clock acceptance: a rig whose source stamps at the sweep start and whose target stamps at
  the sweep end (60 ms gap) reads an anchored offset of -59 ms undeclared and +0.1 ms declared, and
  an injected +50 ms moves the declared estimate by -50.0 ms. TIERS Indoor02 re-report (new example
  configs `online_selftest_motion_temporal_anchored_declared*` and
  `online_kissicp_temporal_anchored_declared`): the anchored selftest baseline reads -19.2 ms
  undeclared and +18.9 ms declared, injection differentials -46.4 / -35.6 ms, so the real-data
  condition is not met. A self-pair cannot observe a stamp convention, so the earlier reading of
  its -25 ms baseline as VLP-16 sweep-end semantics is withdrawn; the Velodyne-to-Ouster pair stays
  degenerate. Additive config schema change (`schemas/config.schema.json`).
- **`calibrex imu-lidar joint` on TIERS Indoor02: INCONCLUSIVE; gravity-magnitude prior and
  opt-in IMU low-pass.** The real-data run (Ouster OS1 points and its 100 Hz IMU, 30 s odometry
  segment) reads INCONCLUSIVE without retuned gates. An 18 Hz roll/pitch chassis vibration
  dominates the IMU (gyro 0.19 rad/s RMS on x/y) while the motion below 2 Hz is almost pure yaw, so
  no rotation, gyro-bias or lever-arm control is detected, and without tilt gravity and the vertical
  accelerometer bias collapsed (gravity -2.0, bias 7.8 m/s^2, std 15.9). The B-spline fit now takes
  an optional prior on the gravity magnitude (`BSplineLidarImuProblem.gravity_magnitude_m_s2`),
  which the bag front end sets to 9.80665 m/s^2 with sigma 0.03; on Indoor02 gravity reads
  -9.805 m/s^2. New opt-in `--imu-lowpass-hz` (`LidarImuBagOptions.imu_lowpass_hz`) low-passes the
  IMU with a zero-phase Butterworth filter and resamples it at twice the cutoff before the fit,
  widening the knots to at least 3 / cutoff (closer knots fit the filtered noise: held-out RMS
  near 3). `SyntheticLidarImuBagSpec` can inject a roll/pitch vibration of the rig: a 12 s bag
  with 18 Hz, 0.25 rad/s vibration reads INCONCLUSIVE unfiltered (lever-arm error 15 cm) and, at
  10 Hz, detects all 32 controls with rotation error 0.055 deg (WARN, rounds not converged). On
  Indoor02 the filter cuts the IMU noise from 0.154 to 0.038 rad/s, but the roll/pitch of the
  mounting stays at 0.52 deg formal std, the yaw is unobservable (5.5 deg) and the run stays
  INCONCLUSIVE. Artifacts: `docs/assets/tiers_lidar_imu_joint/`. Options are recorded in the
  artifact's `options`; no schema change.
- **Fix: per-point time offsets under a since-boot header clock.** `classify_point_time` called
  point times "absolute" when their median lay within 1000 s (or 1e12 ns) of the header stamp,
  so an Ouster OS1 whose headers run on its since-boot clock (about 750 s after boot in TIERS
  Indoor02) had its 0-0.1 s offsets read as times 750 s before the header, breaking deskew and
  per-point timing. Absolute times must now lie within 5 s of the header. `calibrex imu-lidar
  joint` also decodes the bag once (one `ScanStore` for the angular-rate passes, the odometry
  segment and the map rounds). No schema change.
- **`calibrex imu-lidar joint`: joint LiDAR-IMU calibration of a recorded bag.** Runs the cubic
  B-spline fit of `calibrex.core.continuous_time_bspline_fit` on a bag
  (`calibrex.evaluation.lidar_imu_bag_calibration`): the angular-rate estimate seeds the rotation,
  clock offset and gyro bias; gyro-deskewed odometry over the longest reliable segment seeds the
  spline; map rounds build a plane map from the placed scans (voxel centroids, planar kNN
  normals), associate per-point-timed returns and refit the IMU rotation, gyro bias, lever arm,
  clock offset, accelerometer bias and gravity, re-estimating the gyro, accelerometer and plane
  noise from the residuals; the first round holds the rotation and clock offset at the angular-rate
  estimate. Every third IMU sample is held out and scores 32 signed controls (Delta chi2 >= 9).
  New artifact `slac.lidar_imu_bag_calibration/v0.1`: `T_lidar_imu`, clock offset, per-block
  estimate, formal std, last round change and verdict, the angular-rate estimate and the
  difference from it, round trace, noise, holdout, controls, separability, PASS / WARN / FAIL /
  INCONCLUSIVE and provenance with the bag digest. New `calibrex.data.synthetic_lidar_imu_bag`
  writes a rosbag2 of a 16-beam spinning LiDAR (per-point `offset_time`) and a 200 Hz IMU in a
  room with a known calibration. Synthetic results: a 12 s bag passes with rotation error
  0.025 deg, lever arm within about 3 mm, clock offset within 0.05 ms and the injected noise recovered;
  an 8 s bag stops after four rounds at 0.13 deg and reads WARN (not converged). The formal std
  holds the plane map fixed and understates the error (about 3 sigma on the 12 s bag). The B-spline
  fitter now takes the position-control-point Jacobian analytically (positions enter linearly)
  and skips calibration blocks a residual family does not depend on, about 2x faster with the
  same results. The TIERS Indoor02 run is recorded in the entry above.
- **Fix: `calibrex init --template` works from the installed wheel.** The sensor templates lived
  only in `examples/sensor_templates`, so outside a repository checkout the quickstart's second
  step failed with "sensor templates are unavailable". The wheel now ships a copy under
  `calibrex/resources/sensor_templates` (a unit test keeps it byte-identical to the examples), and
  `tools/local_release_smoke.py` runs `init --template` from the clean wheel. The quickstart in
  `docs/tutorials/your_own_data.md` and the Hilti camera-IMU page installed `calibrex` from PyPI,
  where it is not published; they now use the GitHub Release wheel. No schema change.
- **Optional CUDA backend for the Borer depth-to-depth six-DoF search.**
  `calibrex camera-lidar benchmark-six-dof --projection-backend cuda` (and
  `tools/run_borer_six_dof_chunk.py`) moves LiDAR projection, nearest-range visibility and
  histogram counting to the GPU through CuPy (new extra `calibrex[cuda]`, `cupy-cuda12x`); entropy,
  MI and the optimizer stay on the original NumPy definitions. Pinhole, double-sphere and MEI
  cameras; FP64 kernels with fused multiply-add disabled; `--resume` checks the full backend
  identity; one worker per GPU. `calibrex camera-lidar estimate-six-dof` (#130) runs one
  estimation from the problem's declared prior on any backend, with optional `--holdout-frame-id`
  frames, and saves a standard `slac.result/v0.1` result. The default stays `numpy`, and `check` /
  `estimate` / `drift` do not expose the backend. Real data: on KITTI raw 0018 the converged frozen-protocol trial is
  exactly equal between NumPy and CUDA (325 evaluations, every candidate record), and the six-DoF
  search is 15.2x faster on KITTI raw 0018 and 11.3x on KITTI-360 0002 (CUDA; schema-valid
  timing reports under `docs/assets/gpu_acceleration/`,
  [tutorial](docs/tutorials/gpu_acceleration.md)). No schema change.
- **`calibrex check` / `estimate` / `drift` read ROS 1 `.bag` files and every MCAP chunk
  compression directly.** New `calibrex.data.bag_formats` (format detection, ROS 1 index reader:
  topics/types/counts without a scan, chunks without the requested topics skipped, unindexed bags
  scanned) and `calibrex.data.ros1_cdr` (a generic ROS 1 wire format to CDR transcoder driven by
  the connection's `message_definition`; numpy-vectorised for record arrays such as Livox points).
  The existing reader functions in `calibrex.data.rosbag2` dispatch on the detected format, so no
  estimator changed. Real data: byte-identical payloads against rosbag2/MCAP conversions of the
  same bags (NTU VIRAL, an Autoware-style recording in ROS 1 none/lz4/bz2 versus db3, MCAP,
  message-zstd and file-zstd, Aqua beach pond bag versus MCAP). Fixes: MCAP `lz4` chunks are LZ4
  *frames* in real files (the reader only accepted raw blocks), `zstd` chunks without a content
  size, and a rosbag2 file-compressed `.mcap.zstd` is now streamed (`.db3.zstd` gets a clear
  error instead of "file is not a database"). No schema change.
- **`calibrex check` on the Autoware sample bag: raw Velodyne packets, VelocityReport, motion
  explanation.** New ROS-independent decoder `calibrex.data.velodyne_packets` for
  `velodyne_msgs/msg/VelodyneScan` (VLP-16 and VLP-32C single return, per-point firing
  time) and `tools/velodyne_scan_to_pointcloud2.py` (`calibrex.data.rosbag2_velodyne`), which
  writes a derived bag with per-sensor clouds in the sensor frames; `VelodyneScan` topics are
  planned as an ignored `lidar` with that hint. Autoware `VelocityReport`
  (`autoware_auto_vehicle_msgs`, `autoware_vehicle_msgs`) is decoded and classified as a wheel
  twist topic. A `no_judgeable_axes` `imu-lidar`/`camera-imu` reason now ends with "recording too
  short or too static: N s with integrated IMU rotation ...; need ..." computed from the
  recording's gyro. Real data: all-sensors-bag1 (36.5 s, straight 97 m drive, 1.5 deg total yaw)
  gives real `lidar-lidar` verdicts on the converted bag (front/left and front/right `pass`;
  +1/+3 deg and +5 cm perturbations `fail`), while the IMU and GNSS pairs stay `inconclusive`
  with the exact reason. The decoder is verified against the bag's concatenated cloud
  (`tools/validate_velodyne_decode.py`: VLP-16 median 0.03 cm, VLP-32C 1.8 to 3.1 cm, Autoware
  x-forward axes by default; dual return and other models unverified); `VelocityReport` is
  verified on the 873 messages of the Autoware `sample-rosbag`. No schema change (reason text
  only).
- **Browser check page verified on real bags; `--topic-kind` and `--camera` in the page.** The
  Pyodide run was compared with `calibrex check` on real data with the same cap: Koide
  IMU-LiDAR (4.4 GB bag), NTU VIRAL `tnp_01` LiDAR-LiDAR (22 GB bag, identical), RTK-SLAM
  IMU-LiDAR / GNSS-LiDAR / GNSS-IMU, Hilti exp21 camera-IMU / camera-focal and KITTI
  LiDAR-wheel odometry: same verdicts, numbers equal to 1e-6 or better except the camera pairs
  (OpenCV 4.10 against 4.14, estimates within one to two sigma). New Topic kinds box and Camera
  field on the page (`topic_kind`, `camera` in the request). Fix: non-finite floats (an
  unobservable `rate_ratio_std`) made the camera-focal reply invalid for `JSON.parse`; they are
  `null` now. `tools/check_browser_check_page.mjs` gains `--topic-kind`, `--camera` and prints
  the run progress. The feasibility table in the tutorial has the numbers. No schema change.
- **Faster first (uncached) runs, identical results.** LiDAR odometry computes target normals
  only for the points a registration hits, the gyro tables are built once per IMU series, the
  IMU CDR decode is one unpack per message, and several small numpy reductions in the voxel
  filter and the registration loop are replaced by exactly equivalent forms. CPU time on real
  bags drops by about 20 % (KITTI lidar-vehicle, RTK-SLAM gnss-lidar), 25 % (RTK-SLAM imu-lidar)
  and 50 % (Koide rigid-scan imu-lidar); every artifact is bit-identical
  (`tools/compare_check_artifacts.py`). No schema change.
- **`camera-lidar` in `calibrex check`, `estimate` and `drift`.** The pair that was missing from
  the check: targetless edge alignment (the Levinson-Thrun objective, image edges against LiDAR
  near-side depth discontinuities, re-implemented natively) pooled over up to 48 frames and
  maximised from the candidate, with a leave-one-block-out jackknife std and a held-out known-bad
  control per axis (2 deg, both directions, t >= 2); the usual verdict rule applies. **Only the
  rotation is judged**: translation is always `unchecked` (parallax resolves it too weakly; a 5 cm
  error is not detected). Intrinsics come from a `CameraInfo` or a Kalibr camchain (like
  `camera-imu`), no OpenCV; `--camera` selects the camera; the estimate is cached under a key that
  includes the candidate; `calibrex estimate` needs a rough `--tf` prior (like `lidar-lidar`);
  `calibrex drift` includes it, and now re-expresses near-half-turn rotation vectors so an
  optical-to-LiDAR rotation does not read as a 254 deg change. New evidence artifact
  `slac.camera_lidar_edge/v0.1` (`edge_alignment`), registered as `camera-lidar-edge`;
  `slac.calibration_check`, `slac.bag_estimate` and `slac.calibration_drift` gain the
  `camera-lidar` pair name (additive). `calibrex convert kitti-raw --camera image_02` also writes
  the rectified images, `CameraInfo` and the `imu_link` to camera tf, so the vendor calibration can
  be checked. Real data (Hilti 2022 `cam0` + PandarXT-32, KITTI development drives 0005 and 0009;
  [benchmark](docs/benchmarks/camera_lidar_check.md)): the reference calibration passes on all 8
  recordings (judged axes only), 23 of 27 rotation errors of 1, 3 and 5 deg are flagged, the rest
  `inconclusive` or `pass` with the axis unchecked; first-run time 20-100 s. Development evidence,
  no leaderboard claim.
- **C2 cubic B-spline trajectory and a joint LiDAR-IMU fit on it.** New
  `calibrex.core.continuous_time_bspline`: a uniform cumulative cubic B-spline on SO(3) x R^3 with
  vectorized evaluation of pose, body angular velocity and acceleration (Sommer et al. recursion)
  and world linear velocity and acceleration; derivatives are checked against finite differences
  and continuity across knots. New `calibrex.core.continuous_time_bspline_fit` fits the control
  points together with the IMU rotation, gyro bias, lever arm, clock offset, accelerometer bias and
  gravity from per-return LiDAR point-to-plane residuals and direct gyro and accelerometer factors
  (whitened, Levenberg--Marquardt, control-point Jacobians by vectorized central differences,
  marginal covariance by Schur complement). `calibrex trajectory recover-lidar-imu-joint
  --trajectory cubic_bspline` runs the joint recovery against a smooth continuous truth that the
  spline can only approximate, with 1000 LiDAR returns/s and 100 Hz IMU, every third IMU sample held
  out: on seeds 20261010, 1 and 2 it passes with trajectory error under 1 cm and 0.3 deg, rotation
  error 0.01-0.03 deg, clock offset within 0.25 ms, lever arm within 6 mm, holdout RMSE at the noise
  level, every block within 2.6 sigma and every control detected (Delta chi2 >= 448). The
  screw-linear runner is unchanged. No real-data front end yet.
- **Joint continuous-time LiDAR-IMU recovery (synthetic), with the IMU mounting rotation.** The
  sparse knot-path solver gains `R_body_imu` (`estimate_imu_rotation`, right-perturbation analytic
  Jacobians for gyro pre-integration and specific force, checked against finite differences) and
  reports the marginal covariance of the calibration blocks with the knots marginalized
  (`calibration_covariance`, `calibration_correlation`). New
  `calibrex trajectory recover-lidar-imu-joint` estimates the knot path from LiDAR point-to-plane
  returns (no truth pose gauges) together with the IMU rotation, gyro bias, lever arm, clock offset,
  accelerometer bias and gravity in one fit, holds out every third gyro window and accelerometer
  sample, runs 32 signed per-axis known-bad controls (whitened holdout chi-square, Delta chi2 >= 9)
  and reports the correlation of every block pair (`slac.continuous_time_lidar_imu_joint/v0.1`).
  On seeds 20261010, 1 and 2 the fit starts 4.05 deg from the true mounting rotation and recovers
  it to 0.05-0.10 deg; every block is within 4 sigma of truth and within the budget of its
  single-parameter `recover-imu-*` runner, and every control is detected. The
  strongest couplings are accelerometer bias with gravity (correlation 0.89, just under the 0.9
  entanglement flag) and lever arm with accelerometer bias (0.77). Synthetic only:
  the screw-linear knot path has piecewise-constant velocity, so its acceleration sits at the knots
  and the clock offset is observed only by gyro windows that straddle a knot; a real capture needs
  a smoother trajectory model first. The existing `recover-*` artifacts are numerically unchanged.
- **Docs: one workflow, estimate -> check -> drift.** New tutorial
  `docs/tutorials/workflow.md` (first in Get started) with real, abridged outputs from
  `calibrex estimate`, `calibrex check --tf frames.yaml` and `calibrex drift` on Koide
  `indoor_easy_01`/`02` (plus a KITTI development-drive set for a ground vehicle), the browser
  page, the translation-observability hints and the `calibrex --help` grouping. The README and
  the landing page now open with the workflow. New `docs/assets/calibrex-workflow.gif`
  (`tools/workflow_gif.py`, from the digest-bound transcript
  `docs/assets/calibrex_workflow/transcript.json`; registered in `readme-data-gifs.json`) and a
  screenshot of the estimate HTML report. Stale statements removed (check "no HTML report",
  browser page "only plans"). Docs and tools only; no schema change.
- **`calibrex drift` compares clock offsets.** Each pair's estimated time offset (row `dt`, ms) is
  tested across bags like an axis (pairwise z with a floor + chi-square) and enters the verdict,
  the attribution, the text and the HTML report; floors 1 ms for `camera-imu`, 2 ms otherwise
  (`--time-offset-floor-ms`), only offsets marked `estimated`, `lidar-wheel_odometry` never.
  New `tools/shift_stamps_in_bag.py` shifts the header stamps of a topic by a known offset as a
  known-bad control. Real data: no false alarm on Hilti camera-IMU (4 bags) or Koide IMU-LiDAR
  (2 bags); a +1 ms IMU shift is detected and attributed on camera-IMU, +5 ms on IMU-LiDAR.
  `slac.calibration_drift/v0.1` (unreleased) gains optional fields only.
- **`calibrex estimate --html FILE`** (and `calibrex render DIR/bag_estimate.json --format html`):
  a self-contained HTML report of a `slac.bag_estimate/v0.1` artifact in the style of the check
  report (light/dark, phone width): per-pair axes with value, 1 sigma on a log-scale bar and
  observability status (non-observed axes greyed, "NOT MEASURED"), exported/omitted frame tree,
  export digests, copyable `calibrex check --tf` and `static_transform_publisher` commands. No
  schema change.
- **Run `calibrex check` and `calibrex estimate` in the browser.** The bag check page
  (`app/check.html`) now runs the pair estimators in the Pyodide worker after the plan: pick
  the pairs, a duration cap (default 60 s) and a mode, watch per-pair and per-scan progress
  (`calibrex.check.browser.EventProgress`, a `CheckProgress` that posts JSON events), and get
  the verdict table, the `--html` report, the `slac.calibration_check` JSON (or the
  `slac.bag_estimate` JSON and `frames.yaml`) to download. New Python entry point
  `calibrex.check.browser.run_request_json`. The page reads files through WORKERFS with a
  1 MiB read-ahead (3x faster than slice-per-page). Verified on KITTI `k0015` in headless
  Chrome against the CLI: same verdicts, numbers within 1e-5. Feasibility per pair is in the
  tutorial. No schema change.
- **Translation observability: why a lever-arm axis is unobservable, and what would fix it.**
  The accelerometer (imu-lidar) and GNSS antenna (gnss-lidar) lever arms are seen only through
  the rotation between paired poses, `(R_j - R_i) l`, so an axis is observable only when the
  platform rotates about the other two. Each estimator now derives, per translation axis, the
  data-only std from the information matrix, the rotation the recording supplies about each
  sensor axis, the share of that information that survives the nuisance parameters, the
  recording needed to reach the bound (`T (std / bound)^2`, same motion) and, when no amount of
  this motion would, which rotation is missing and how long a 30 deg/s reference motion would
  take. `calibrex check` and `calibrex estimate` print it in `next steps`, for example
  `gnss-lidar lever arm z: needs rotation about the sensor x or y axis ... Model estimate: about
  4 min of it at >= 30 deg/s would reach the 1 cm bound`. Additive schema change only: an
  optional `excitation` object on `slac.imu_lidar_translation` axes, `slac.gnss_lidar_lever_arm`
  DoFs, `slac.calibration_check` `unchecked_axes` and `slac.bag_estimate` axes. Predictions are
  validated against empirical stds on RTK-SLAM and Koide recordings: the analytic std is
  optimistic by 2 to 6.5 times, the extrapolated duration is good to about 3 times with listed
  failures (`docs/concepts/translation_observability.md`,
  `tools/validate_translation_observability.py`).
- **`calibrex drift`: did the rig's calibration change between recordings?** `calibrex drift
  <bag1> <bag2> [<bag3> ...] --output DIR [--html FILE]` estimates every bag with the
  `calibrex estimate` machinery (same flags, estimator cache; per-bag `bag_estimate.json` under
  `DIR/<bag-name>/`) and tests every axis that the data observed in at least two bags: pairwise
  differences against `max(sigma_k * combined std, floor)` (rotation 0.5 deg, rigid-scan
  rotation 1.5 deg, translation 2 cm, as in `check`), a chi-square homogeneity test, and, for
  three or more bags, the bag(s) whose removal restores consistency. Verdict per pair `stable` /
  `drift` / `inconclusive`, overall the worst; `--fail-on drift|inconclusive|never`. New schema
  `slac.calibration_drift/v0.1` (`calibration-drift`, `schemas/calibration_drift.schema.json`).
  Also `tools/rotate_imu_in_bag.py` / `calibrex.data.rosbag2_imu_rotate`, which writes a copy of
  a bag with its IMU "remounted" by a known rotation (the known-bad control of the drift
  evaluation). Results: `docs/benchmarks/drift_real_data.md`. Existing schemas unchanged.
- **`calibrex estimate` real-data validation** (KITTI vehicle pairs, RTK-SLAM GNSS pairs; see
  `docs/benchmarks/estimate_real_data.md`). Two fixes: the exported `topic_frames` now include
  the export root's topics (an IMU-rooted export lost `/imu -> imu`, so `check --tf frames.yaml`
  skipped `imu-lidar` and `gnss-imu` as `frame_not_in_tree`), and the first progress line of
  `estimate` no longer shows the machine's uptime as elapsed time. No schema change.
- **`calibrex estimate`: a calibration for a bag that has none.** `calibrex estimate <bag>
  --output DIR` runs the check machinery (topic roles, planner, native estimators; same
  `--frame-map`, `--topic-kind`, `--vehicle-frame`, `--pairs`, `--max-duration-s`, cache flags)
  without a candidate and keeps the estimates. New schema `slac.bag_estimate/v0.1`
  (`bag-estimate`, `schemas/bag_estimate.schema.json`): per pair the transform, each axis with
  its std and observability (`observed`, `unobservable`, `control_not_detected`, `no_estimate`,
  `not_estimated`), the time offset, estimator settings, evidence, bag digest and provenance,
  plus the exported and omitted frames. Exports in DIR: `frames.yaml` (`slac.check_frames`,
  loads with `calibrex check --tf`), `static_transforms.sh` and `static_transforms.launch.yaml`
  (ROS 2 `static_transform_publisher`), `joints.urdf.xml`, and `camchain-imucam.yaml` when `--tf`
  gave a Kalibr camchain. A frame whose transform has an axis the data did not observe is
  written only when a `--tf` prior supplies it (the YAML says `NOT MEASURED` per frame),
  otherwise it is omitted and listed in `next steps`. lidar-lidar needs a `--tf` prior to start
  from and is skipped with a hint without one; camera-focal (an intrinsics check) is not
  estimated. `slac.calibration_check/v0.1` and `slac.check_frames/v0.1` are unchanged;
  `build_calibration_check` now shares `classify_bag_topics` with the new command.
- **`calibrex check` says what to do next.** The text output ends with a deduplicated
  `next steps:` block (a hint per skip reason code and unmapped topic: `--tf`, `--frame-map`,
  `--vehicle-frame`, `--topic-kind`, expected message types), and
  `--write-frames-template FILE` writes a starter `slac.check_frames` YAML pre-filled with the
  bag's sensor frames. The template carries `schema_version: slac.check_frames_template/v0.1`
  and TODO placeholders; `--tf` refuses it until it is filled in and the version is changed to
  `slac.check_frames/v0.1`. Hints are text-only; the artifact schema is unchanged.
- **`calibrex check` progress**: a live report on stderr with the pair (`pair i/N`), the stage
  (reading, odometry + deskew pass k of up to n, estimation), scans processed / total, elapsed
  time and an ETA when the total is known, and `[cache hit]` for reused estimates. One updating
  line on a terminal; one line per event when stderr is redirected or with `--json`. `--quiet`
  silences it. Results are bit-identical with progress on or off (hooks in
  `calibrex.core.progress` only observe). The verdict column of the final table is coloured
  (pass green, warn yellow, fail red, inconclusive/skipped dim) only on a TTY with `NO_COLOR`
  unset.
- Docs site revamp: custom theme and logo, a landing page that leads with `calibrex check`,
  tabbed navigation (Get started, Benchmarks & SOTA, Concepts, Reference, Internals),
  and a restyled browser calibration page.
- **Bag check in the browser** (`docs/app/check.html`): plans `calibrex check --plan` on your
  own rosbag2 under Pyodide in a Web Worker. The bag (`.db3` or `.mcap`, multi-GB is fine) is
  mounted with Emscripten WORKERFS and read lazily, never copied into memory. Takes optional
  `--tf` calibration files, `--vehicle-frame` and `--frame-map`; shows the frame tree, topic
  roles and pair table, a plan-JSON download and the equivalent local command. A committed
  synthetic sample bag (`docs/app/samples/check_sample`, `tools/build_check_sample_bag.py`)
  loads with one click. Python side: `calibrex.check.browser.plan_bag_for_browser` and
  `plan_request_json`; `tools/check_browser_check_page.mjs` drives the page in headless Chrome.
  - `calibrex.data.rosbag2.open_sqlite_readonly`: percent-encodes the path in the sqlite URI
    and falls back to `immutable=1` only when a plain read-only open fails (a WAL-mode bag on a
    read-only file system); a bag that opens today is opened exactly as before.
  - The bag input digest is hashed in 8 MiB chunks (same digest, bounded memory).
  - `encode_image` and `encode_camera_info` in `calibrex.data.ros_cdr_writer`.
- **Docs refresh.** README, landing page, workflow page and Install page now cover ROS 1 `.bag` and
  MCAP input ("Your bag format"), camera-LiDAR (rotation only), the Autoware Velodyne note, the
  browser verification status and the faster first runs, and say that these are on `main` and not
  in the v0.5.1 wheel. New README GIF `docs/assets/calibrex-check-camera-lidar.gif` (two real
  `calibrex check --pairs camera-lidar` runs on KITTI development drive 0005: the vendor
  calibration passes, the same calibration turned +3 deg about the camera y axis fails), rendered
  by `tools/camera_lidar_check_gif.py` and registered in `docs/assets/readme-data-gifs.json`.

- Use analytic SO(3) spline-control and IMU calibration Jacobians in the
  joint B-spline LiDAR-IMU fitter, sharing base states and computing only the
  derivatives needed by each residual family. On a public GLIM 12-second clip,
  three alternating CPU pairs measured 2.55x complete-worker acceleration
  (325 to 127 s medians) and 6.32x across the four fits including covariance
  (252 to 40 s), with matching calibration/evidence decisions. Public APIs
  and all 130 exported schemas are unchanged. Complete source-bound trials,
  per-round covariance/trajectory arrays and a CPU-only audit are archived;
  the real-data calibration verdict remains INCONCLUSIVE. See
  `docs/benchmarks/bspline_lidar_imu_acceleration.md`.

- Add a CPU shared-rotation candidate for native gyro deskew and joint
  B-spline LiDAR-IMU fitting. The public GLIM 12-second comparison measures
  a modest incremental 1.42x complete-run and 1.52x fit speedup over the
  analytic-Jacobian baseline, with all three quality comparisons passing.
  The complete source-bound evidence and limits are recorded in
  `docs/benchmarks/lidar_imu_shared_rotations.md`.

- Connect optional CUDA odometry to joint LiDAR-IMU initialization, gyro
  feedback passes and final trajectory construction. Record actual per-pass
  execution and repairs, and expose explicit coordinate and returned-motion
  grids with disabled defaults. On an identical dense public GLIM six-second
  workload, three complete CPU/CUDA pairs measure 211.124 -> 108.970 s
  (1.94x), with identical estimates and evidence. Preserve the earlier rejected
  uncanonicalized comparison and all source-bound records in
  `docs/benchmarks/cuda_joint_lidar_imu.md`.

- Prune shared CUDA spatial queries with tight subtree boxes and capture
  validated SciPy tree metadata in bulk. Unknown layouts retain public-node
  traversal; exact candidate arithmetic, ties and native repairs are preserved.
  The same query implementation serves LiDAR maps, point-to-point ICP and
  temporal scan odometry. Complete public dense joint-calibration comparisons,
  source-bound results and a CPU-only audit are documented in
  `docs/benchmarks/cuda_spatial_pruning.md`.

- Fuse optional CUDA scan-odometry covariance, symmetric eigenspace and
  planarity preparation; reuse completed targets across deskew rounds.
  Preserve lazy sparse preparation and native repairs, with more conservative
  guards near repeated roots. Complete previous/current public CUDA trials
  record preparation time and actual work in
  `docs/benchmarks/cuda_normal_preparation.md`.

## 0.5.1 - 2026-10-02

Highlights:

- **`calibrex check` covers all ten sensor pairs**, now including `camera-focal`.
- Clouds without per-point time are handled as rigid scans for `imu-lidar`,
  `lidar-lidar`, and `gnss-lidar`, with floors set from measured bias.
- First runs on real third-party ROS 2 bags that ship their own `/tf_static`,
  with three integration fixes: CameraInfo padding, Draco-compressed clouds,
  and `degenerate_frames`.
- README hero for the zero-config check.
- A pre-registered Hilti camera-IMU audit is refuted (6/7 gates) and added to
  the SOTA leaderboard.

- **A pre-registered `camera-imu` audit is refuted** on four unseen Hilti 2022
  recordings (`hilti-camera-imu-v1`, 6/7 gates).
  - Against Kalibr, the forward cameras (cam0, cam1) are within 0.70 deg and
    0.33 ms on every unit, reproduce across recordings to 0.43 deg, and
    reproduce the Kalibr cam0-to-cam1 rotation to 0.41 deg (all gates pass).
  - The `constrained` gate fails: on the shortest recording (exp04, 126 s)
    one or two axes of both cameras are unobservable at the 0.3 deg bound.
  - Development was exp21 and exp07 only; exp01-exp04 are now spent.
  - Tools: `tools/score_hilti_camera_imu.py`,
    `tools/build_hilti_camera_imu_audit.py`.
- **`calibrex check` runs `lidar-lidar` and `gnss-lidar` on clouds without
  per-point time** as rigid scans, like `imu-lidar` (#99). Both pairs used the
  field only to deskew LiDAR odometry (`lidar-lidar`: the reference LiDAR's map;
  the target scans were never deskewed, so the target no longer needs the field;
  `gnss-lidar`: the odometry compared with GNSS), so rigid scans are defensible.
  `--lidar-lidar-deskew` and `--gnss-lidar-deskew` `auto|constant_velocity|none`
  (default `auto`: the field when present, bit-identical default path; rigid
  otherwise or when forced). `LidarLidarMapOptions.deskew` and
  `GnssLidarRunOptions.deskew`; artifacts record `options.deskew: none` and a
  limitation only in rigid mode. Measured rigid minus deskewed: lidar-lidar on
  NTU VIRAL tnp_01 at most 0.07 deg and 5 mm (stds unchanged); gnss-lidar on
  RTK-SLAM stadtgarten_seq2/seq1 y +41.5/+26.4 mm, x -2.7/-4.2 mm, z -5.3/-9.8 mm,
  time offset +64 ms, stds 1.7 to 2.5 times larger. Policy: lidar-lidar floors
  0.5 deg / 0.03 m with default thresholds; gnss-lidar (and gnss-imu) estimated at
  std <= 2 cm, control and translation floor 0.11 m. Optional schema fields:
  `CheckOptions.lidar_lidar_deskew`, `gnss_lidar_deskew`; `CheckPairRecord.deskew`
  also takes `constant_velocity` (`slac.calibration_check/v0.1` unchanged). No
  real bag we hold newly becomes judgeable (Autoware: one concatenated cloud in
  `base_link`, `gnss-lidar` with `--frame-map` is `inconclusive` on 36 s; Koide:
  one cloud, no GNSS). See the tutorial section "lidar-lidar and gnss-lidar on
  rigid scans".
- **`calibrex check` wires `camera-focal`**: the deployed focal lengths (Kalibr
  camchain or `CameraInfo`) are judged against the gyro as a scale
  `s = f_estimate / f_candidate`, `|s - 1|` against `max(k * std, floor)` with the
  new `--focal-scale-floor` (default 0.005). It consumes this run's `camera-imu`
  estimate (skipped `missing_dependency` when that is absent, failed or looser
  than 0.5 deg), is cached under the camera model and the `camera-imu` estimate,
  and equals `calibrex camera-imu focal` bit for bit on Hilti exp21 cam0-cam4.
  On exp21 only cam1 (fx, fy) and cam0 (fy) are judgeable; +3 % known-bad
  flips cam1 to `fail`, +1 % to `warn`; the other cameras stay `inconclusive`.
  Optional schema fields: `CheckPairRecord.focal_scale` (new `CheckFocalScale`),
  `CheckOptions.focal_scale_floor`; `slac.calibration_check/v0.1` unchanged
  otherwise. `run_camera_focal_check` splits out from the standalone runner
  (`max_seconds` option).
- **`calibrex check` runs `imu-lidar` on clouds without per-point time** as rigid
  scans (no deskew) instead of skipping them `unsupported_sensor`.
  `--imu-lidar-deskew auto|gyro|none` (default `auto`: the per-point time field
  when the cloud has one, so the gyro path is unchanged and bit-identical; rigid
  scans only when it is absent or when `none` is forced). Measured against the
  gyro deskew on RTK-SLAM construction_seq1 and Hilti exp21, the rigid rotation
  differs by at most 0.52 deg (yaw), the reported std grows 3.5 to 10 times, the
  lever arm becomes unobservable and the time offset absorbs about half a sweep
  (+48 to +51 ms). Policy: rotation floor 1.5 deg for rigid scans
  (`--rigid-scan-rotation-floor-deg`), axes estimated only at std <= 0.3 deg, a
  1.5 deg known-bad control. Optional schema fields:
  `CheckPairRecord.deskew`, `CheckOptions.imu_lidar_deskew` and
  `rigid_scan_rotation_floor_deg` (`slac.calibration_check/v0.1` unchanged);
  `ImuLidarRunOptions.deskew` and `ImuLidarTranslationOptions.deskew`; the
  rotation/translation artifacts record `options.deskew: none` only in rigid mode.
  Koide indoor_easy_01/02 (a depth-camera cloud) now give `pass (partial: roll,
  pitch, yaw only)`. See the tutorial section "Clouds without per-point time".
- README: a new top with a `calibrex check` hero image (`docs/assets/readme-check-hero.svg`,
  generated from the committed demo summary by `tools/generate_readme_check_hero.py`,
  light and dark aware, digest-bound to its sources), a feature strip, and a shorter nav.
- **`calibrex check` on real third-party bags with `/tf_static`** (Autoware
  all-sensors-bag1, Koide indoor_easy, Aqua beach_pond MCAP; see the tutorial's
  "Real bags with /tf_static"). Fixes: ROS 2 `CameraInfo`/`Image` payloads with
  CDR end padding no longer fail; PointCloud2 topics that are not raw point
  arrays (Draco) are ignored (`ignored_reason`, optional field on topic
  records); pairs whose two frames coincide are skipped with the new reason code
  `degenerate_frames` (enum widening, `slac.calibration_check/v0.1` unchanged).

## 0.5.0 - 2026-10-02

Highlights:

- **`calibrex check <bag>`**: a zero-config audit of the calibration deployed on a
  bag. It reads `/tf_static` (or `--tf`), infers topic roles, runs native
  estimators for 10 sensor pairs including GNSS and vehicle pairs, and reports
  per-axis verdicts with partial coverage and detection power. It caches
  estimators, checks rig closure, writes a self-contained HTML report (`--html`),
  and `calibrex convert kitti-raw` turns KITTI raw drives into bags.
- Pre-registered SOTA audits and a leaderboard: imu-lidar and lidar-vehicle are
  supported, lidar-lidar is refuted.
- Native methods for all 10 target sensor pairs.
- A browser calibration page.
- Camera-IMU static-window exclusion and a camera bound.

- **`calibrex check`, Phase D: rig closure, HTML report, the 1 degree demo.** The
  estimates of a run are now also checked against each other: parallel estimates of
  one frame pair and cycles (a fundamental cycle basis) are judged with the pairs'
  rule (`max(sigma_k * std, floor)`, first-order std assuming independent members,
  axes unobserved by any member left unchecked); composed pairs (`gnss-imu`) are
  excluded as derived. New optional `closures` field in `slac.calibration_check/v0.1`;
  a failing loop raises `overall_verdict` to at least `warn`, never `fail`. New
  `calibrex check --html PATH` and `calibrex render check.json --html PATH` write a
  self-contained HTML report. `tools/check_tf_injection_demo.py` and
  `docs/assets/calibrex_check_demo/` reproduce "a +1 degree yaw in the deployed tf
  fails" on the pooled KITTI development drives; the README gains a `calibrex check`
  section.

- **`calibrex check`, Phase C2: `gnss-lidar` and `gnss-imu` from the bag's
  NavSatFix topic.** New `calibrex.data.navsatfix_track` turns a `NavSatFix` topic
  into the GNSS track the lever-arm estimator uses (header-stamp time, ellipsoidal
  altitude, usable = `status >= 0` and `sqrt(trace(covariance)) <= 0.15 m`, which
  reproduces RTK-SLAM's `blt_std` exactly; unknown covariance keeps only GBAS
  fixes with a recorded fallback std). `gnss-lidar` judges the antenna lever arm in
  the LiDAR frame (x, y, z; the antenna's rotation axes are listed as unchecked)
  and is cached like `imu-lidar` (the fit does not use the candidate). `gnss-imu`
  composes this run's `gnss-lidar` and `imu-lidar` evidence as `calibrex gnss-imu
  compose` does, and is skipped (`missing_dependency`) when either is missing or
  failed. New `--gnss-max-duration-s` gives the GNSS pairs their own span (they
  need minutes of RTK-fixed windows); `calibrex gnss-lidar rtk-slam` gains
  `--max-seconds`. `slac.calibration_check/v0.1` gains the optional
  `options.gnss_max_duration_s`. On RTK-SLAM the bag path reproduces the `rtk.txt`
  path to 0.2 mm (same windows); against the CAD antenna offset the pairs judge only
  the axes with std under 1 cm (y on stadtgarten seq2, x on seq1), a +15 cm
  error on an observable axis fails and z is never checked. See
  `docs/tutorials/calibrex_check.md`.

- **`calibrex check`, Phase C1: vehicle pairs, validated on KITTI raw.** With
  `--vehicle-frame`, `lidar-vehicle`, `imu-vehicle`, `ins-lidar` and
  `lidar-wheel_odometry` now run their native estimators on bag data and are
  judged by the existing verdict rule (unobservable axes, such as roll without
  turns, are listed as unchecked). LiDAR odometry comes from `PointCloud2` scans
  (one pass shared by the three LiDAR pairs; a stream gap over 1 s starts a new
  segment, so several recordings can share a bag), the INS from a
  `nav_msgs/Odometry` topic, the wheel from an Odometry or `TwistStamped` topic;
  `imu-vehicle` uses the INS's body-frame velocity, not the raw IMU. New
  `--topic-kind TOPIC=wheel|ins` classifies odometry and twist topics; twist
  topics now fill the `ins`/`wheel` slots. `slac.calibration_check/v0.1` gains
  the optional `options.topic_kinds`.
  New `calibrex convert kitti-raw DRIVE... --output BAG [--calib-dir DIR]` writes
  a KITTI raw drive as a rosbag2 (`/velodyne_points`, `/oxts/{imu,fix,odometry,twist}`,
  transient-local `/tf_static` from `calib_imu_to_velo.txt`, `base_link` = the OXTS
  frame, no invented per-point time) with a provenance sidecar, using a new
  ROS-free rosbag2 sqlite3 writer (`calibrex.data.rosbag2_writer`) and CDR
  encoders (`ros_cdr_writer`, shared with the tests) and a `TwistStamped` decoder.
  On the development drives the bag run reproduces the KITTI-text CLIs exactly for
  lidar-vehicle, imu-vehicle and ins-lidar (wheel: differs only by the declared
  body-frame wheel proxy); +1 and +3 deg yaw on the Velodyne mount flips the
  pooled lidar-vehicle and lidar-wheel verdicts to `fail`, and is not seen on
  single drives whose yaw is unobservable. See
  `docs/tutorials/calibrex_check.md`.

- **`calibrex check` is 3x faster on a first run and instant on a re-check.**
  The `imu-lidar` and `camera-imu` estimates do not depend on the candidate, so
  their artifacts are cached (`--cache-dir`, default
  `$XDG_CACHE_HOME/calibrex/check`; `--no-cache` disables it) under a key of the
  bag digest, the estimator, every estimator option, the camera intrinsics and
  the calibrex version and source revision. A hit rewrites only the
  candidate-dependent reference fields; `lidar-lidar`, which starts at the
  candidate, is not cached. `slac.calibration_check/v0.1` gains optional
  `pairs[].evidence_from_cache` and `evidence[].from_cache`. Re-judging RTK-SLAM
  construction_seq1 with a +1 and +3 deg yaw error took 87 and 86 minutes and now
  takes 3 seconds, with identical verdicts and per-axis numbers. The IMU-LiDAR
  odometry passes are faster with bit-identical output (Hilti 40 s: 25 to 8.5 min;
  RTK-SLAM 180 s with lever arm: 94 to 28 min): voxel downsampling packs the
  voxel key into one int64 instead of a row-wise `np.unique`; the gyro rotation
  model is evaluated once per scan from a shared start orientation; the rosbag2
  SQLite reader filters by topic in SQL (the IMU topic of a long bag no longer
  drags every point cloud through Python), and the check reads the IMU once and
  keeps decoded scans in memory across passes (`--scan-memory-mb`). Defaults of
  every estimator are unchanged, so benchmark artifacts reproduce.

- **`calibrex check`, Phase B (verdicts).** Without `--plan`, `calibrex check`
  now runs the native estimator of `imu-lidar` (rotation, plus the lever arm
  unless `--no-imu-lidar-translation`), `lidar-lidar` (map registration started
  at the candidate) and `camera-imu` (needs OpenCV and intrinsics from a Kalibr
  camchain or a `CameraInfo` topic) and judges the deployed transform against it.
  Per estimated axis: `tolerance = max(k * std, floor)` (`--sigma-k` 3,
  `--rotation-floor-deg` 0.5, `--translation-floor-m` 0.02), `pass` within one
  tolerance, `fail` beyond two, `warn` between; unobservable axes are listed as
  unchecked, and a pair with no judgeable axis, or whose estimator failed its own
  held-out check, is `inconclusive`. Each axis records a detection-power
  self-test (`detectable_error = tolerance + |delta|`, and whether a 1 deg error
  would be flagged) without re-solving. The overall verdict is the worst pair
  (`fail > warn > inconclusive > pass`; `inconclusive` when nothing ran). Each
  estimator's schema'd artifact is written to `<output stem>_evidence/` and
  referenced by relative path and SHA-256. New options: `--pairs`,
  `--max-duration-s`, `--camera`, `--evidence-dir`, `--fail-on`,
  `--acceleration-unit`, `--detection-probe-deg`. `slac.calibration_check/v0.1`
  gains optional fields only (`options`, `evidence_dir`, per-pair `axes`,
  `unchecked_axes`, `time_offset`, `evidence`, ...) and new reason codes; plan-only
  artifacts stay valid. LiDAR support is generalised from Livox to any
  `PointCloud2` with a per-point time field (offset seconds, absolute seconds as
  Hesai writes it, or absolute nanoseconds), and the IMU-LiDAR and LiDAR-LiDAR
  estimators accept a time window. Real-data validation (Hilti, NTU VIRAL,
  RTK-SLAM, including known-bad yaw perturbations) is in
  `docs/tutorials/calibrex_check.md`. A pair with any unchecked attempted axis is
  marked `coverage: partial` (table: `pass (partial: roll only)`, summary
  `partial_pairs`, CLI warning), and an axis whose estimator known-bad control was
  not detected is left unchecked (`control_not_detected`).
- **`calibrex check`, Phase A (plan only).** `calibrex check BAG [--tf FILE]
  [--vehicle-frame FRAME] [--frame-map TOPIC=FRAME] [--plan]` audits the
  calibration deployed on a robot. It reads candidate extrinsics from the bag's
  `/tf_static` and from `--tf` files (URDF fixed joints, a small
  `slac.check_frames/v0.1` YAML, Kalibr camchain-imucam, RTK-SLAM `calib.yaml`,
  Hilti `lidar_calibration.yaml`), classifies sensor topics (LiDAR, IMU, camera,
  GNSS, odometry, twist), maps them to tree frames, and lists the pairs that can
  be checked, with a machine-readable reason for each skipped pair. No solver
  runs yet: every runnable pair is `planned`. New schemas
  `slac.calibration_check/v0.1` and `slac.check_frames/v0.1`. The rosbag2
  reader now decodes `tf2_msgs/msg/TFMessage` and `sensor_msgs/msg/NavSatFix`
  and can list a bag's topics and read one topic's first messages without a
  full scan. Vehicle pairs (`lidar-vehicle`, `imu-vehicle`) are opt-in: without
  `--vehicle-frame` they are skipped with `no_vehicle_frame`, because they assume
  ground-vehicle motion. See `docs/tutorials/calibrex_check.md`.
- **README refresh.** The README now covers all ten target sensor pairs, the
  pre-registered SOTA audits and leaderboard standings, and current honest
  verdicts; the duplicated quickstart and gallery GIF are removed and the
  solid-state section is condensed.
- **Camera-IMU evidence settings.** Near-static windows (accumulated rotation
  under 1.0 deg) are left out of the fit and the jackknife, and the rotation
  observability bound is 0.3 deg for cameras instead of the 0.1 deg inherited
  from IMU-LiDAR (`ImuLidarRunOptions.min_window_rotation_deg`, default 0 and
  so unchanged for IMU-LiDAR; `CameraImuRunOptions` defaults; CLI
  `--min-window-rotation-deg` and `--observable-rotation-std-deg`). The
  exclusion count, its reason, and the threshold are recorded in the artifact
  options; no schema changed.
  - Hilti 2022 development (exp21, exp07): verdicts go from 10
    `inconclusive` to 5 `pass`, 1 `warn`, 4 `inconclusive`. exp07 changes come
    from the threshold alone. The jackknife is not removed by the exclusion,
    accuracy against Kalibr is not systematically better, and cam2-cam4 on
    exp07 and cam3 yaw on exp21 stay unconstrained.
- **Camera-IMU std diagnosis.** `tools/diagnose_camera_imu_std.py` shows the
  reported std that keeps every Hilti verdict `inconclusive` is the window
  jackknife, not the analytic std: the tracked camera rotation carries a
  per-window scale of about 1-8 % on the forward cameras, and leaving a
  window out moves the fit by a similar amount. The 0.1 deg bound is
  inherited from IMU-LiDAR and is tighter than image-tracked rotations
  support (`docs/benchmarks/hilti_camera_imu.md`).
- **A pre-registered `lidar-lidar` audit is refuted** on three held-back
  NTU VIRAL recordings (`ntu-lidar-lidar-v1`, 2/4 gates).
  - Calibrex's fitted extrinsic is 0.49 deg and 7.7 cm from the rounded
    design value (gates pass), but it does not beat concurrent scan-to-scan
    (paired CI low −0.005) and the cross-recording consistency of
    0.3 deg / 3 cm is violated (1.14 on average).
  - Development was tnp_01 only; the claim was about accuracy against the
    design value and reproducibility, not a margin over a baseline.
  - Tools: `tools/score_ntu_lidar_lidar.py`,
    `tools/build_ntu_lidar_lidar_audit.py`.
- **A pre-registered `lidar-vehicle` audit is supported** on eight unseen
  KITTI drives (`kitti-lidar-vehicle-v1`, 3/3 gates).
  - Calibrex's motion-only rotation fits held-out LiDAR motion better than
    KITTI's calib_imu_to_velo (paired CI low 0.051, 9 of 9 blocks).
  - Its pitch and yaw agree with the OXTS-motion vehicle frame within
    0.21 deg on average (threshold 0.5).
  - Tools: `tools/score_kitti_lidar_vehicle.py`,
    `tools/build_kitti_lidar_vehicle_audit.py`.
- `calibrex camera-imu focal` checks a camera's focal lengths against the
  gyro, without a target (`slac.camera_focal_scale/v0.1`).
  - Per-axis camera/gyro rate ratios give `fx`/`fy` estimates. The
    optical-axis ratio is the control.
  - On Hilti 2022 exp21 the forward cameras agree with Kalibr within 0.5 %
    (cam1 `pass`); the side cameras scatter by 1-2 %.
- **Test fix:** the synthetic rotating-camera test rendered almost black
  frames (the texture offset was inverted). It now renders the full view.
- `calibrex lidar-wheel` calibrates a LiDAR against wheel odometry
  (`slac.lidar_wheel_odometry/v0.1`).
  - It estimates the LiDAR-to-vehicle rotation, the wheel-speed scale, the
    clock offset (started by a yaw-rate scan), and the lever.
  - `trajectory` takes any TUM trajectory and a wheel CSV. `kitti` uses the
    OXTS speed and yaw rate as a stand-in for wheels, since KITTI has none.
  - On KITTI dev drives the rotation matches lidar-vehicle within 0.004 deg.
    The verdict is `warn`: the held-out blocks prefer a different speed
    scale, and the 74 ms offset describes the OXTS velocity output.
- `calibrex imu-vehicle kitti` estimates `R_vehicle_imu` for the KITTI OXTS
  unit from its body-frame velocity and rates with the non-holonomic solver.
  It includes a closure reference against LiDAR-vehicle composed with
  `calib_imu_to_velo`; the closure agrees within 0.1 deg.
- **Fix:** the OXTS-motion vehicle frame used KITTI's level-frame
  `vf, vl, vu` / `wf, wl, wu` as if they were body-frame. They are now
  rotated into the body frame with each packet's roll and pitch.
  - LiDAR-vehicle pitch and yaw now agree with that frame within 0.06 and
    0.15 deg.
  - The earlier "HDL-64 odometry drift" explanation of a 0.5 deg pitch
    disagreement was wrong.
- `calibrex gnss-imu compose` composes GNSS-LiDAR and IMU-LiDAR artifacts
  into the GNSS antenna position in the IMU frame
  (`slac.gnss_imu_lever_arm/v0.1`).
  - Inputs are digest-pinned, std is propagated, and axes inherit
    unobservable inputs.
  - On RTK-SLAM the estimated axes agree with CAD within 1 cm (both
    recordings `inconclusive`).
- `calibrex lidar-vehicle kitti` estimates `R_vehicle_velodyne` from the
  vehicle's non-holonomic motion, as `slac.vehicle_frame_rotation/v0.1`
  (`calibrex.solvers.vehicle_frame_solver`).
  - It is compared both with KITTI's OXTS frame and with a vehicle frame
    derived from the OXTS velocities by the same solver.
  - On five development drives, pitch and yaw are estimated (0.04 and 0.05
    deg std) and roll is not (too few turns).
  - Yaw is 0.15 deg from the OXTS-motion vehicle frame. Pitch is 0.5 deg from
    both references, which points at the LiDAR odometry's vertical drift.
  - No claim (`docs/benchmarks/kitti_lidar_vehicle.md`).
- `calibrex lidar-lidar ros2` calibrates `T_reference_target` for two LiDARs
  in one ROS 2 bag, as `slac.lidar_lidar_extrinsic/v0.1`.
  - Target scans are registered to local maps from the reference LiDAR's
    odometry and solved jointly by robust point-to-plane Gauss-Newton, with
    held-out blocks, a block jackknife, and 1 deg / 5 cm controls.
  - On NTU VIRAL tnp_01 (development), five of six DoFs are estimated
    (roll's std 0.108 deg is just over 0.1). The held-out data reject the
    rounded design extrinsic (delta chi-square 48,901), about 0.5 deg of
    pitch and 5 cm in y and z away from the estimate.
  - Motion-based hand-eye could not recover the translation there, which is
    documented (`docs/benchmarks/ntu_viral_lidar_lidar.md`).
- The trajectory hand-eye solver also starts from the rotation that aligns
  the motion rotation axes, so arbitrary mountings converge. Its jackknife
  and reference differences use small rotations about the parent axes.
- Camera-IMU tracking now equalizes contrast (CLAHE) and tracks up to 600
  features at quality 0.001.
  - On the Hilti exp07 corridor, failed frame pairs dropped from 577 to 16
    of 1321 (cam0), and the forward cameras' std from 0.4-0.9 to 0.1-0.24 deg.
  - The side-looking cameras did not improve, and two got worse on exp21.
  - `tools/analyze_camera_imu_consistency.py` separates a rig-level
    IMU-frame offset from per-camera errors. On exp21 it finds about -0.4 deg
    about the IMU vertical axis, shared by all cameras; the cause is not
    identified.
- `calibrex camera-imu rotation`: targetless camera-IMU rotation, clock
  offset, and gyro bias from ROS 2 image and IMU topics (needs
  `calibrex[opencv]`).
  - Camera rotations come from tracked features, and are aligned with the
    gyro by the IMU-LiDAR rotation evidence.
  - The rotation artifact gains `sensor_modality` (`lidar`, `camera`, or
    `trajectory`; default `lidar`).
- On Hilti 2022 development recordings (exp21, exp07), the clock offset
  agrees with Kalibr within 0.35 ms, and the rotation is 0.0-0.7 deg from
  Kalibr on exp21.
  - Every camera is `inconclusive`: the std exceeds 0.1 deg.
  - A negative bias about the camera y axis appears on all five cameras,
    unexplained.
  - Tracking fails often in the exp07 corridor.
  - No camera-IMU SOTA claim is made (`docs/benchmarks/hilti_camera_imu.md`).
- Rotation jackknife spreads and reference differences are now small
  rotations about the sensor axes rather than Euler-angle differences, which
  broke near a +-90 deg pitch.
- Browser calibration at <https://rsasaki0109.github.io/Calibrex/app/>.
  - Upload an IMU CSV and a sensor trajectory (TUM), or generate a synthetic
    hand-held example. The page calibrates the IMU rotation, clock offset,
    gyro bias, and lever arm, with held-out evidence.
  - Calibrex, NumPy, and SciPy run under Pyodide, so no data leaves the page.
  - The page offers the schema-valid artifacts for download.
  - The docs workflow builds the wheel it installs
    (`tools/build_browser_wheel.py`).
- `calibrex imu-lidar trajectory` runs the same computation from files
  (`calibrex.evaluation.imu_trajectory`, `calibrex.data.imu_trajectory`,
  `calibrex.data.imu_trajectory_synthetic`).
  - The rotation and lever-arm artifact builders are now shared helpers
    (`rotation_artifact_from_evaluation`,
    `translation_artifact_from_evaluation`).
- `accelerometer_lever_arm/v0.2`, now the default, fits gravity per 2 s
  segment instead of per 10 s window, because the odometry's tilt drifts
  within a window.
  - Segments need at least 10 scans, and the segment sensitivity uses 3 s and
    4 s refits.
  - It was revised on spent recordings only, after six hypotheses were
    tested; the negative results are documented.
- A second pre-registered `imu-lidar` extrinsic audit
  (`mid360-imu-lidar-extrinsic-vs-li-init-v2`) is **supported** on the last
  unseen RTK-SLAM recording, `construction_seq2`.
  - Calibrex beats LI-Init by 4.13 mm mean span RMS (CI low 2.79 mm), and
    LI-Init's lever arm by 4.11 mm (CI low 2.87 mm).
  - The lever arm is (27.0, 17.9, -33.0) ± (5.6, 4.6, 7.9) mm; x is 16 mm
    from the design value.
  - `tools/build_mid360_imu_lidar_extrinsic_audit.py --round v2` rebuilds it.
    The scoring tool pins the first round's nuisances.
- A pre-registered `imu-lidar` extrinsic claim (rotation, lever arm, and
  clock offset against LI-Init) is **refuted** on two unseen RTK-SLAM
  recordings (`stadtgarten_seq1`, `construction_seq1`).
  - Calibrex's lever-arm run was `inconclusive` on both: x's segment-duration
    sensitivity is 11.0 and 13.3 mm, above the 10 mm bound. The
    pre-registration counts that as no estimate.
  - The rotation passes on both and replicates seq2 to within 0.05 deg.
  - Candidates are scored on one common odometry per recording
    (`score_imu_lidar_translation_candidate`, `tools/score_mid360_imu_lidar_extrinsics.py`,
    `tools/build_mid360_imu_lidar_extrinsic_audit.py`).
  - The leaderboard lists the refuted claim next to the supported rotation
    claim.
- `calibrex imu-lidar livox-translation` estimates the IMU lever arm
  (translation of `T_lidar_imu`) from the accelerometer, on top of a
  digest-pinned rotation artifact, as `slac.imu_lidar_translation/v0.1`.
  - Segment double integration against LiDAR odometry is linear in the lever
    arm. Gravity and the accelerometer bias are fitted per window and the
    velocity per segment.
  - The reported std includes a segment-duration sensitivity alongside the
    analytic and jackknife std, and held-out windows must detect a 20 mm
    shift of each axis.
  - MID360 hand-held (RTK-SLAM seq2): `pass`, at (21.3, 24.3, -38.5)
    ± (6.1, 5.2, 5.3) mm against the design value (11.0, 23.3, -44.1) mm.
    The vehicle recording is `inconclusive`: its planar motion leaves every
    axis unobservable.
- The first supported SOTA claim: `imu-lidar` rotation and clock offset of
  the Livox MID360 built-in IMU, against LI-Init, on two recordings from two
  dataset families (`docs/benchmarks/mid360_imu_lidar.md`).
  - LI-Init (GPL-2.0) runs only in a pinned container
    (`tools/external/li_init/`). `calibrex.importers.li_init` imports its
    result as an external-run artifact.
  - The claim, metric, and thresholds were pre-registered
    (`docs/benchmarks/mid360_imu_lidar_preregistration.yaml`) before any
    LI-Init score was computed. The audit protocol pins that file's SHA-256.
  - On RTK-SLAM seq2, the mean held-out span residual is 0.0123 rad/s for
    Calibrex and 0.148 rad/s for LI-Init (paired improvement 95 % CI
    0.090-0.185). On the driving recording, LI-Init returned no result.
  - `score_imu_lidar_candidate` scores a fixed rotation and clock offset,
    refitting only the gyro bias. Its `holdout_spans_s` scores every
    candidate on the same held-out time spans. The first audit build paired
    windows by each method's own segmentation, which shifted LI-Init's
    held-out windows and made the audit `refuted`. The thresholds were not
    changed by the fix.
- The MID360 hand-held IMU-LiDAR rotation now passes. On RTK-SLAM seq2,
  roll, pitch, and yaw are within 0.16 deg of the design value (std 0.02-0.03
  deg), the clock offset is 9.8 ms, and held-out windows detect every 1 deg
  control.
  - The cause of the earlier 1.4 deg yaw discrepancy was LiDAR-only
    constant-velocity deskewing, which cannot follow fast hand-held rotation,
    not coning as first reported. Odometry now accepts a `rotation_model`,
    and the evaluation deskews with gyro rotations mapped by the current
    estimate.
  - It iterates until every axis moves less than 0.25 std. It refuses a pass
    while the last pass still moves an axis by more than its std.
  - It records a `deskew_feedback_ratio`: the share of a deliberate 2 deg
    deskew error that survives in the estimate.
  - Rotation increments are compared with on-manifold gyro preintegration
    (cumulative orientations and first-order bias Jacobians); mean rates
    remain as the `mean_rate` model.
- Added IMU-LiDAR rotation, clock-offset, and gyro-bias evidence
  (`slac.imu_lidar_rotation/v0.1`, `calibrex imu-lidar livox`).
  - Mean LiDAR odometry rates between consecutive scans are aligned with
    interval means of the gyro (a cumulative integral, so evaluation is
    vectorized).
  - A Procrustes initialization handles arbitrary mountings.
  - Yaw-only vehicle motion leaves yaw unobservable, and the artifact reports
    it as such.
  - Livox ROS 2 bags are read without ROS through explicit stream profiles:
    point-time field and encoding (offset seconds or absolute nanoseconds)
    and acceleration units (m/s^2 or g). The MID360 design `T_lidar_imu` is
    the reference.
- LiDAR odometry can motion-compensate each sweep from per-point capture
  times under a constant-velocity model, re-registering once with the refined
  motion. On RTK-SLAM MID360 data the one-second distance error against RTK
  drops from 1.94 cm to 1.53 cm (p90 5.89 cm to 3.95 cm).
  The RTK-SLAM GNSS-LiDAR results were re-run with deskewing:
  - x now agrees across both sequences (5.1 and 5.2 cm).
  - The ~42 ms clock offset reported before was an artefact of undeskewed
    sweeps: they are stamped at sweep start, but their geometry sits
    mid-sweep.
- Odometry windowing for multi-sensor evaluations moved to
  `calibrex.evaluation.odometry_windows`.
- Added GNSS antenna lever-arm and clock-offset evidence against LiDAR
  odometry (`slac.gnss_lidar_lever_arm/v0.1`, `calibrex gnss-lidar rtk-slam`)
  with an RTK-SLAM dataset reader (RTK `rtk.txt`, Livox rosbag2 scans, CAD
  reference from `calib.yaml`).
  - The solver profiles out one ENU alignment per odometry window by weighted
    Procrustes, so the outer robust fit has four unknowns.
  - GNSS epochs are used as observed and the odometry is interpolated to them,
    with a blend-dependent pose variance. Interpolating the noisy GNSS track
    instead biased the clock offset toward mid-epoch values.
  - LiDAR odometry gains an incremental scan-to-local-map mode
    (`local_map_scans`) for sparse Livox scanners; on MID360 it cut the
    one-second distance error against RTK from 7.3 cm to 2.2 cm.
- Added INS/GNSS-LiDAR trajectory hand-eye calibration evidence
  (`slac.ins_lidar_hand_eye/v0.1`, `calibrex ins-lidar kitti`).
  - LiDAR motions come from a new vectorized point-to-plane scan-to-scan
    odometry that never uses the reference as a prior.
  - A joint solver estimates `T_ins_lidar`, the clock offset, and a reference
    trajectory scale. On planar vehicle motion it recovers yaw from the
    translation equations, which a rotation-first method cannot do.
  - Every DoF is classified as estimated, prior, or unobservable using the
    larger of its analytic and block-jackknife std. The model rejects a pass
    verdict while any extrinsic DoF is unobservable.
  - Held-out time blocks must detect each estimated DoF's known-bad shift
    (delta chi-square).
  - Same-rig KITTI drives can be pooled, and reference gaps are never
    interpolated.
  - Real-data findings are in `docs/benchmarks/kitti_ins_lidar.md`, including
    that pairing KITTI `sync` frames by directory position misaligns Velodyne
    and OXTS by hundreds of milliseconds after a dropped frame; frames are now
    paired by file index and timestamp.
- Generalized the SOTA claim audit to any sensor pair.
  `slac.sota_audit_protocol/v0.1` scopes a claim by modalities (camera, LiDAR,
  IMU, GNSS, INS, radar, RGB-D, wheel odometry, vehicle, robot arm), estimated
  quantities, and a method category, and `calibrex sota audit` evaluates it.
  Camera-LiDAR audits now run through the same engine, and their published
  schemas are unchanged. `calibrex sota leaderboard` builds
  `slac.sota_leaderboard/v0.1` standings per sensor pair from digest-pinned
  audit results, keeps refuted claims counted beside supported ones, and lists
  target pairs without a claim. The first leaderboard
  (`docs/benchmarks/sota_leaderboard.md`) records that no pair has a supported
  claim yet.
- Declared every artifact kind once in `calibrex.core.schema_registry`.
  Validation, `calibrex schema`, and auto-detection previously drew on seven
  hand-maintained tables that had drifted: `slac.remote_archive_selection/v0.1`
  and `slac.camera_lidar_confidence_calibration/v0.1` were accepted by their
  models but failed auto-detection, and `continuous-time-imu-intrinsics` was
  missing from `ValidationKind`. Accepted versions are now derived from each
  model's `schema_version` annotation. `calibrex schema all` also writes
  `schemas/schema_ledger.json` (kinds, current and accepted versions, schema
  digests, retired versions, and migrations), and `calibrex schema ledger`
  prints it. Retired versions fail with the retiring release and a remedy;
  `slac.doctor/v0.1` (replaced by `slac.environment_readiness/v0.1`) is the
  first, since its workflows lack the now-required `template_path`.
- Added `tools/check_schema_compat.py` and a CI job that fail when a released
  schema version is no longer readable, is narrowed without a version bump, or
  a released artifact no longer validates. Decided narrowings are recorded in
  `tools/schema_compat_acknowledged.yaml`.
- `slac.external_calibration_run/v0.1` keeps requiring a sha256
  `container_digest`, accepted as a v0.1 fix: a mutable image tag cannot
  reproduce a container run. External runs recorded with a tag must be
  re-imported with the resolved digest.
- The published `slac.result/v0.1` schema accepts an empty legacy provenance
  map again, matching the model and the omitted-map case. Such results remain
  blocked from production admission by `calibrex validate`.
- Fixed the A2D2 Pandey dataset manifest, which had violated
  `slac.dataset_manifest/v0.1` since v0.4.1. A new test validates every
  committed `slac.*` artifact; the digest-bound raw-replay stub config remains
  a tracked known violation.
- `calibrex doctor` now reports Python path isolation: ROS site-packages on
  `sys.path`, dependencies resolved from outside the active environment, and
  imported ROS modules. `core_ros_independent` is measured instead of always
  `true`, and a shadowed environment downgrades readiness to `warn`.
- pytest no longer autoloads third-party plugins, so the suite runs in a shell
  where ROS is sourced (ROS pytest plugins previously aborted collection). CI
  adds a ROS Jazzy job that runs the tests and `calibrex doctor` with
  `setup.bash` sourced, plus static and runtime checks that the package never
  imports ROS modules.
- Pinned ruff and mypy exactly and moved the strict mypy gate to
  `tools/typecheck-requirements.txt`, shared by CI and developers.

- Added independent evidence for imported Camera--IMU calibrations.
  `calibrex external-run evaluate-camera-imu` evaluates a Kalibr (or other
  external-run) `T_cam_imu` and `timeshift_cam_imu` on a separate
  `slac.camera_imu_motion_recording/v0.1`. It uses a middle-block temporal
  holdout, a train-only gyro bias, excitation gates, and eight signed
  rotation/time known-bad controls. The result is
  `slac.external_camera_imu_evidence/v0.1` with
  pass/warn/fail/inconclusive/blocked status and digest-bound provenance. A
  recording whose bytes match a declared fitting input is blocked.
  `calibrex external-run synthesize-camera-imu-fixture` writes the
  nonphysical Kalibr fixture used to prove the external-run contract with a
  Kalibr producer.

- Switched the public installation path to the verified GitHub Release wheel
  and removed the automatic PyPI publishing workflow following the maintainer
  decision to skip PyPI distribution.

- Added the continuous-time trajectory foundation. `slac.continuous_time_trajectory/v0.1`
  is a typed, schema-valid trajectory contract that declares the interpolation
  model, knot-domain validity (queries outside the domain are rejected, never
  clamped), and clock/capture-time semantics, and converts losslessly to the
  existing piecewise linear-slerp adapter. Analytic SE(3) manifold operations
  (`se3_manifold`) provide right-trivialized exp/log, the SO(3)/SE(3) left
  Jacobians and adjoint, the point-transform Jacobian, and two-knot screw
  interpolation Jacobians, all verified against finite differences. A sparse
  Gauss--Newton/Levenberg--Marquardt fitter (`continuous_time_sparse`)
  assembles the normal equations as a block-banded system in which every
  factor touches at most its two bracketing knots, estimating all knots on the
  manifold from body-frame point measurements and pose measurements (poses
  become four anchored point factors using only the analytic point Jacobian).
  `calibrex trajectory build-contract` and `calibrex trajectory fit` expose the
  workflow, and the fit result validates against
  `slac.continuous_time_trajectory_fit/v0.1` with provenance pinned to both
  input digests.

- Added schema-valid empirical SE(3) uncertainty evidence
  (`slac.empirical_se3_uncertainty/v0.1`). `calibrex camera-lidar
  empirical-uncertainty` refits the native probabilistic multi-frame refiner on
  deterministic contiguous temporal-block subsamples that never split a block
  across the train/holdout boundary, reports tangent-space intervals with
  Sidak-corrected joint family coverage against injected truth, retains weak or
  unobservable directions instead of dropping them, and applies an
  overconfident interval control that must be refuted before the
  PASS/WARN/FAIL policy can certify the intervals. Without a declared reference
  (`--stability-only`) the artifact reports the empirical spread with an
  INCONCLUSIVE policy. Every resample refit is retained as a digest-bound
  schema-valid result, and the artifact validates against its generated schema.

## 0.4.1 - 2026-08-11

- Promoted the no-download KITTI-shaped Camera-LiDAR evidence path into a
  strict five-minute demo. Its deterministic, generator-bound 300x300 fixture
  now detects 16/24 mandatory ±1 deg / ±0.10 m perturbation cases, passes all
  six falsification-policy gates without relaxing thresholds, and remains
  explicitly synthetic rather than a real-sensor accuracy claim. Added direct
  generator parity and strict end-to-end tests plus README commands that
  validate the result and verify its digest-bound provenance bundle.
- Added an external-only I2PNet KITTI-large adapter with pinned MIT source,
  checkpoint/archive/input digests, an explicit non-reference initial-transform
  contract, a Windows-safe pure-Torch neighbor backend, deterministic
  frame-derived NumPy/Torch sampling, enforced deterministic CUDA execution,
  and schema-valid diagnostic correspondence and aggregate pose outputs. The
  first two-frame KITTI raw development audit rejects the diffuse cost-volume
  correspondences
  (mean reliability `0.000032333` below the fixed `0.01` gate), while the
  intended direct pose output improves the fixed `10 deg / 1.37049 m`
  perturbation to `0.04847 deg / 0.08211 m`. Added a frozen, schema-valid pose
  initializer protocol and subprocess runner that binds the problem, adapter,
  checkpoint, frames, perturbations, failure policy, and every generated
  initial/manifest/pose digest. Its disjoint three-frame, four-perturbation
  development matrix completes all trials but records 0/4 strict hits at the
  prespecified `<0.5 deg` and `<0.2 m` gate; mean error changes from
  `8.1046 deg / 0.40395 m` to `6.7043 deg / 0.37994 m`. This remains
  initializer-development evidence and does not alter a release or SOTA gate.
- Added a digest-bound pose-initializer failure-analysis artifact and external
  I2PNet analyzer. It replays all 12 frame corrections, reproduces benchmark
  metrics, and decomposes each required/predicted correction into signed
  parallel gain and orthogonal leakage. The frozen single-axis rotation gains
  are `x=0.002904`, `y=0.957952`, and `z=-0.000960`; translation gains are
  `x=1.374653`, `y=-0.023960`, and `z=0.949139`. This identifies a strong
  development-set axis anisotropy without claiming a causal network defect.
- Added a schema-valid, development-only Camera-LiDAR provider-support
  comparison and CLI. It requires every confidence calibration to share the
  exact problem, threshold grid, split seeds, support gates, and excluded
  evaluation datasets; ranks a support-first diagnostic point while keeping
  rejected calibrations runtime-ineligible; and records complete source
  digests and provider identity. The first KITTI-360 drive 0002 comparison
  rejects adapter v0.3, v0.4, and v0.5 because every candidate has zero
  worst-split geometric holdout inliers. Adapter v0.5 is retained only as the
  diagnostic leader, not as a runtime lock or SOTA result. A second disjoint
  development capture on drive 0003 also rejects the outdoor and indoor
  SuperGlue checkpoints: outdoor retains 10/1,049 geometric inliers at its
  diagnostic point and indoor retains 0/276, while both have zero worst-split
  holdout support. Diagnostic tie-breaking now prefers frame and accepted
  support when every geometric metric is tied, instead of reporting an empty
  highest-threshold operating point.
- Completed the Camera-LiDAR v0.7 `integrated-r5-numba-v03-v02` release audit
  without retuning the two evaluation sequences. At the frozen
  `(0.5 deg, 0.5 m)` six-DoF cell, KITTI raw 0018 reached 85/200 strict hits
  and failed its 88% gate, while KITTI-360 drive 0000 reached 161/200 and
  passed its 76.5% gate; both had zero D2D execution failures. The paired
  five-seed falsification packs retain all 1,000 cases per dataset. Raw records
  529 failure-aware refinement failures, while KITTI-360 rolls back all 1,000
  unsupported candidates. Both raw-recomputed 1,007-artifact bundles verify,
  and the digest-frozen learned-targetless SOTA audit returns `refuted` with
  external comparable baselines and independent-rig evidence still missing.
- Added schema-valid post-hoc Camera-LiDAR correspondence-quality diagnostics
  for initializer, candidate, and selected poses. The final reports expose the
  provider bottleneck directly: 112/2,834 accepted correspondences and 7/25
  supported frames on raw, versus 33/2,269 and 2/25 on KITTI-360. These reports
  record that holdout was not used for selection and are provenance-bound to
  the exact refinement result and correspondence artifact.
- Added the Phase 3 provider-neutral probabilistic 2D--3D correspondence
  artifact and an optional Apache-2.0 OpenCV PnP-RANSAC adapter with
  confidence filtering, covariance-aware diagnostics, tests, and complete
  provider/input provenance. The correspondence contract now distinguishes
  dataset-license provenance from the provider's code/model license; ablation
  evidence is marked unverified when the dataset license or source digests
  are undeclared.
- Added a D2D-initialized native multi-frame probabilistic pose refiner with
  covariance/outlier/reliability weighting, robust loss, bounded SE(3)
  correction, disjoint frame holdout, complete candidate traces, and explicit
  uncertainty ablation switches.
- Added the v0.2 integrated falsification contract around that refiner. A
  candidate is selected from fit evidence only and rolls back to the D2D
  initializer on non-convergence, negligible improvement, correspondence
  support loss, non-finite evidence, or correction-bound proximity. Rejected
  candidates and holdout outcomes remain in the evidence, complete trace
  directories must share one solver/backend identity, and reusable run IDs no
  longer embed the obsolete `v06` release label.
- Added explicit D2D candidate-trace initialization for probabilistic
  multi-frame refinement and its paired ablations. The trace must pin the same
  problem digest and frame pair; results record its ID, digest, solver status,
  and hit decision. Runs without a trace declare `problem_initial_transform`
  and are not mislabeled as solved-D2D initialization evidence. The generic
  method ID is now
  `probabilistic_multiframe_refinement/v0.3`; the v0.2 and old v0.1 IDs remain
  accepted for artifact compatibility. Generated result and benchmark method
  provenance now record the same v0.3 implementation version.
- Added a paired probabilistic-refinement ablation benchmark that executes the
  full estimator and three one-factor removals over fixed frame-split seeds,
  retains every schema-valid result, and reports failure-aware paired
  bootstrap comparisons with both input digests. Results and ablations now
  include translation distance and quaternion-geodesic rotation error to the
  problem's digest-pinned reference, making the planned cm/deg gates directly
  auditable alongside held-out reprojection.
- Added a ROS-independent joint camera--LiDAR continuous-time solver boundary
  using per-point firing times, supplied body twists, probabilistic image
  residuals, bounded extrinsic/clock refinement, disjoint holdout evidence,
  and an explicit weak-motion rejection gate.
- Added rolling-shutter row-time deskew plus fixed-clock, no-per-point-time,
  no-rolling-shutter, and no-covariance temporal ablations. The paired
  multi-split benchmark retains every result, failure, source digest, and
  bootstrap comparison. Optional frozen extrinsic/time references now produce
  initial/final rotation, translation, and clock errors in each result and
  paired ablation, making injected-recovery gates directly auditable.
  Continuous-time problems separately declare their dataset license; absent
  license provenance marks the ablation benchmark unverified.
- Added a ROS-independent, provenance-pinned piecewise SE(3) body-trajectory
  adapter with linear translation, shortest-arc quaternion interpolation, and
  strict no-extrapolation behavior. Continuous-time camera--LiDAR refinement
  now uses this non-constant motion model when supplied while retaining the
  constant-twist input as a backward-compatible fallback. A CLI adapter
  attaches existing schema-valid `T_world_body` trajectory artifacts while
  preserving both input digests and frame semantics.
- Added a digest-frozen Camera-LiDAR SOTA audit protocol/result and CLI. It
  evaluates every numerical/integrity gate, dataset-family and independent-rig
  coverage, and can emit only supported, refuted, or incomplete verdicts;
  schema validation prevents an unsupported success label.
- Extended benchmark distributions with p90 and p95 and made mean, median,
  p90, p95, maximum, and failure rate individually addressable by frozen SOTA
  audit requirements, so tail failures cannot be hidden behind a mean. Paired
  bootstrap improvement confidence bounds can also be frozen against an
  explicit reference/candidate method pair, while runtime and peak-memory
  mean/p95 statistics are directly auditable. Evidence whose benchmark
  provenance declares `data_verified: false` is rejected, and result
  validation now enforces the semantics of all three verdicts rather than
  guarding only `supported`. Optional smoke requirements cannot inflate the
  achieved dataset-family or independent-rig coverage counts.
- Added a schema-valid Camera-LiDAR D2D research pipeline with frozen
  Fibonacci-sphere protocols, complete candidate traces, Bull's Eye artifacts,
  an external FeatDepth provider adapter, rectified KITTI problem construction,
  deterministic parallel execution, and a 200/200-hit KITTI raw 0018
  rotation-only reproduction gate.
- Replaced the D2D z-buffer's global stable sort with an exactly equivalent
  linear reduction and added an optional fused Numba CPU projection adapter.
  Six-DoF v0.3 traces pin the backend and dependency version, refuse mixed
  backend resume or mixed full-trace falsification input, and preserve the
  NumPy default; full 325-evaluation and official-frame regressions retain
  identical numerical results.
- Completed the frozen KITTI raw 0018 rotation matrix at `1, 2, 10, 20 deg`
  with 200 deterministic directions per level. The `1, 2, 10 deg` levels
  recovered 200/200; `20 deg` recovered 178/200 with zero computational
  failures, while its `17.0067 deg` p90 and `28.4920 deg` p95 retain the
  direction-dependent capture failures that its `0.2425 deg` median would
  otherwise hide. All new traces and aggregate artifacts are schema- and
  digest-validated.
- Added the KITTI-360 half of the D2D reproduction path: a pinned external
  MiDaS v3.1 provider, official MEI fisheye projection, official
  pose/azimuth-based Velodyne motion compensation with a schema-valid
  provenance manifest, KITTI-360 problem construction, and CLI support. The
  frozen `10 deg` gate completed at 200/200 hits with zero failures (mean
  `0.3812 deg`, maximum `0.4676 deg`), matching the paper's 100% target.
- Completed the frozen KITTI-360 rotation matrix at `1, 2, 10, 20 deg` with
  200 deterministic directions per level. The `1, 2, 10 deg` levels recovered
  200/200; `20 deg` recovered 184/200 with zero computational failures. Its
  `0.3882 deg` median contrasts with a `28.4409 deg` p95 and `45.4505 deg`
  maximum, retaining the direction-dependent capture failures. All new trace,
  benchmark, and Bull's Eye artifacts are schema- and digest-validated.
- Added a digest-pinned A2D2 MiDaS/provider and pre-registered camera-view D2D
  smoke path. The frozen five-direction `10 deg` run completed 5/5 trials but
  failed recovery at 0/5 hits; the two-frame/pre-registration limitation and
  negative result are retained rather than weakening the gate.
- Added digest-checked resumable Camera-LiDAR trial execution plus a bounded
  six-DoF D2D solver, paired Fibonacci rotation/translation protocols,
  schema-valid candidate traces, benchmark aggregation, and CLI commands. The
  first real KITTI-360 `(0.5 deg, 0.5 m)` smoke converged to
  `0.2505 deg / 2.08e-17 m` and passed the frozen hit gate.
- Completed the first full six-DoF matrix cell on KITTI raw 0018 at the frozen
  `(0.5 deg, 0.25 m)` perturbation. All 200 trials converged with zero
  computational failures, but only 84/200 passed the strict
  `<0.5 deg / <0.20 m` accuracy gate. Rotation error was `0.5230 deg` mean,
  `0.5239 deg` median, and `0.7228 deg` p95; translation error was `0.1447 m`
  mean, `0.1421 m` median, and `0.1803 m` p95. The negative 42% hit-rate result,
  all trace identities/digests, and schema-valid aggregate provenance are
  retained without weakening the gate.
- Corrected the UniCalib adapter's default provenance URL to the official WACV
  2026 `han-15/UniCalib` repository and locked it with a unit assertion. Updated
  the SOTA evidence review for TLC-Calib's May 2026 code/data release while
  retaining its non-commercial, external-process-only license boundary.

- Added a digest-locked fixed-frame KITTI raw Camera-LiDAR input artifact and
  recovery benchmark comparing a scalar-reference Pandey I2I optimizer with a
  vectorized, safety-gated deterministic coarse-to-fine path. On the fixed
  official KITTI raw 0005 input, all paired accuracy metrics were identical
  while mean trial runtime improved by 1.719x.
## 0.4.0 - 2026-07-29

- Extended `calibrex doctor` from an environment check into an optional
  dataset-readiness workflow with type inference, quality and degeneracy
  diagnostics, compatible-workflow suggestions, a versioned
  `slac.doctor/v0.1` artifact, generation provenance, and validation support.
- Added local and GitHub-hosted Calibration CI with `calibrex ci`, a composite
  action, Step Summary rendering, enforced falsification and protocol gates,
  digest-bound inputs and custom policies, provenance-bound SVGs, and the
  versioned `slac.calibration_ci/v0.1` decision artifact.
- Added the schema-versioned `slac.external_calibration_run/v0.1` contract for
  digest-bound subprocess, container, precomputed, and imported calibration
  results; migrated the Koide adapter to it and added a ROS-free Kalibr
  camchain importer with explicit frame/time conventions and failure states.
- Added a reproducible full-scale KITTI raw Camera-LiDAR falsification runner
  with a frozen reference/known-bad protocol, selected-frame and raw-input
  SHA-256 provenance, independently verified evidence bundles, an honest
  PASS/FAIL/INCONCLUSIVE decision artifact, and an opt-in official-data test.
- Prepared the public launch surface with a concrete five-minute quickstart,
  an approachable documentation home, a repository social preview, citation
  metadata, and corrected issue links.
- Added guarded PyPI trusted publishing after a GitHub release is published,
  including tag-to-package-version checks and clean wheel smoke tests.
- Replaced the abbreviated license notice with the complete Apache-2.0 text and
  added a distributable `NOTICE`.
- Restored the Calibrex product, Python distribution/package, CLI, and source
  path. Existing `slac.*` schema IDs, `slac_version`, `slac_native`, and
  versioned protocol/policy IDs remain the stable legacy wire namespace.
- Added provenance-bound SVG evidence cards via
  `calibrex render --format evidence-card` and schema-source comparison tables
  via `calibrex compare --format evidence-table`, with deterministic README
  artifacts and drift tests.
- Reworked the README around a concise evidence-first story, generated visual
  summaries, an honest accepted-vs-known-bad comparison, public-data demos, and
  a compact calibration coverage matrix.

## 0.3.0

- Trajectory artifact (`slac.trajectory/v0.1`, schema #18) with ground-truth-free
  quality gates (kinematic health, interpolation clamp fraction, cross-segment
  drift proxy) on motion-compensated online runs.
- Two-pass and native-deskew KISS-ICP odometry modes at rosbag conversion;
  native deskew cut Indoor02 identity selftest translation error ~9.5 cm → ~4.3 cm
  (~55%) with modest rotation regression (~1.2° → ~2.0°); two-pass odometry
  FAILed its own trajectory gate (cross-segment RMSE 0.366 m).
- Anchored temporal time-offset estimation (`time_offset_anchor: initial`),
  dual-mode adapted/anchored provenance, joint extrinsic/temporal separability
  evidence row, and validation-only `inject_time_offset_s` (anchored mode
  tracked +50 ms injection exactly on selftest; adapted mode absorbed it).
- LiDAR-IMU rotation evidence (`lidar_imu` family): `sensor_msgs/Imu` CDR
  decoding, `--imu-topic` bag conversion with OS1 IMU restamp, symmetric
  rotation-rate smoothing (holdout RMSE ~8 deg/s → ~3.6 deg/s, passing 5 deg/s
  gate), per-axis observability, known-bad rotation probes, and supporting-only
  gravity consistency (4.42° on Indoor02).

## 0.2.0

- Pure-Python rosbag2/MCAP reader (sqlite3 + MCAP, CDR PointCloud2 and Odometry).
- Odometry-aware online motion compensation for rosbag2 replays with world-frame
  source maps, odometry-extrapolation gate, and clock-domain restamping.
- KISS-ICP rig-frame odometry path and TIERS Indoor02 real-data validation
  (identity self-consistency control: ~10× extrinsic error reduction with motion
  compensation vs static control).
- Per-point deskew via `sensors.<name>.point_time_field` on rosbag2 online runs.
- Camera-LiDAR projection promoted to ADR-0004-protocol evidence rows with
  family-aware assessment policy gates and KITTI committed-sample demonstration.
- Temporal time-offset perturbation probes and 1D holdout-RMSE estimator on
  motion-compensated online runs (synthetic ±5 ms recovery; honest real-data
  degeneracy reporting).
- Project rename Calibrex → slac (ADR 0006).

## 0.1.0-alpha.1

- Project bootstrap.
- Typed config and result models.
- CLI skeleton with `doctor`, `schema`, `validate`, `verify`, `assess`, `evidence`, `init`, `inspect`, `calibrate`, `evaluate`, `render`, `visualize`, `compare`, `report`, and `export`.
- Frame graph and SE3 utilities.
- Dataset manifest schema, filesystem dataset adapter, timestamp normalization, and MCAP adapter boundary.
- HTML report, report sidecars, and PASS/WARN/FAIL quality aggregation.
- Metric registry, deterministic holdout splitting, and threshold-based metric grading profiles.
- RGB-D Open3D SLAC adapter boundary and example config.
- Backend-neutral graph problem compiler and `slac compile`.
- Public dataset catalog, TUM RGB-D config, KITTI raw config, and direct TUM downloader helper.
- KITTI raw loader and fixed-mounted LiDAR prior factor.
- KITTI `calib_velo_to_cam.txt` importer for fixed-LiDAR initial transforms.
- Autonomous-driving starter profile with Radar and Autoware export surface.
- Public Livox Horizon-Horizon PCD evidence demo for rigidly mounted solid-state LiDAR pairs.
- LiDAR pair evidence metrics, known-bad perturbation cases, holdout point-to-plane summaries, and evidence protocol metadata.
- Machine-readable `summary.json`, `metrics.json`, `observability.json`, `degeneracy.json`, and `evidence.json` report sidecars.
- Machine-readable `comparison.json` artifacts with metric, transform, evidence summary, and evidence protocol compatibility comparisons.
- Static JSON schemas for config, result, comparison, dataset manifest, and report sidecars.
- Bulk schema generation via `slac schema all --output-dir schemas`, with schema drift tests and CI smoke coverage.
- 3D rig viewer artifact for reference, candidate, and estimated extrinsic comparison.
- Falsification assessment framework: `slac assess` applying a policy to evidence, policy artifacts, `--enforce` exit codes, per-DoF known-bad challenge summaries, a fixed support denominator, mandatory challenge verification, and assessment recomputation verification inside evidence bundles.
- Evidence bundle verification: `slac verify`, an `evidence-bundle-verification` schema, a `--require-raw-recomputed` raw recomputation gate, a verified-raw-inputs requirement, and structured verification claims.
- `slac render` and `slac evidence` commands for producing report and evidence artifacts from an existing result without recomputing metrics.
- Protocol and policy JSON schemas, and `compare --enforce-compatible` protocol compatibility enforcement.
- README GIF gallery generation (`generate_calibration_evidence_gif.py --readme-gallery`) with a provenance manifest and schema-validated visual modes.
- Local release smoke helper, wheel smoke coverage in CI, and a draft GitHub release workflow.
- Configurable public-dataset frame sampling: `DatasetConfig.sample_limit` and `slac inspect --sample-limit`.
- N-way `slac report-compare` command with labeled results, protocol-compatibility gating, metric-family rankings, and a `report_comparison` schema (16 schemas total).
- Pure-Python rosbag1 (v2.0) reader with PointCloud2 decoding (`none`/`bz2`, optional `lz4` extra), `slac inspect --type rosbag1`, and the TIERS LidarsCali example config.
- Native LiDAR point-to-plane solver backend (`solver.backend: native_lidar_point_to_plane`) wired into `slac calibrate`, with real rank/condition-number/weak-DoF observability replacing the `uncomputed_alpha_backend` stub and point-to-plane metrics in results.
