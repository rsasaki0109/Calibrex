# SOTA Work Handoff

This page hands the pre-registered SOTA audit work to the next agent. It covers:

- where each sensor pair stands;
- how an audit is run, and the rules that keep it honest;
- which data is already spent;
- the pitfalls met so far;
- the next candidates.

Read [`AGENTS.md`](https://github.com/rsasaki0109/Calibrex/blob/main/AGENTS.md) and the
[SOTA leaderboard](benchmarks/sota_leaderboard.md) first. State as of 2026-10-22;
the INS-LiDAR audit evaluated merged `main` at `e2ac3e20`.

## Goal and the one rule

Calibrex may call a method state of the art only for a claim that a frozen,
pre-registered audit marks `supported`. Five things follow from that:

- **Evaluation data is never seen first.** The claim, the evaluation data, the scoring
  code and the thresholds are committed before any method reads that data.
- **Each evaluation runs once,** from the frozen code. A refuted or incomplete audit is
  published like a supported one, and stays on the leaderboard.
- **Verdicts are never changed.** An analysis after the verdict is labelled *post hoc*,
  and it may only explain the verdict.
- **Data is spent once a method has read it.** It can be used for development, never
  again for evaluation.
- **The reference is not ground truth.** A dataset's own calibration (KITTI, Kalibr,
  LI-Init) is the reference, and pages say so.

## Standings

| Pair | Standing | Audits | Page |
| --- | --- | --- | --- |
| `imu-lidar` | supported | rotation+clock (4/4), full extrinsic v2 (4/4); full extrinsic v1 refuted (0/8) | [MID360](benchmarks/mid360_imu_lidar.md) |
| `lidar-vehicle` | supported | KITTI motion-only rotation, 8 drives (3/3) | [KITTI lidar-vehicle](benchmarks/kitti_lidar_vehicle.md) |
| `camera-imu` | supported | Hilti v3 per-axis verdicts (7/7); v1 (6/7) and v2 (5/7) refuted | [Hilti camera-IMU](benchmarks/hilti_camera_imu.md) |
| `camera-lidar` | refuted | KITTI check v1 (5/9), v2 (8/9) | [v2](benchmarks/kitti_camera_lidar_v2_check.md), [v1](benchmarks/kitti_camera_lidar_check.md) |
| `camera-camera` | refuted | Hilti (4/6): accurate, but a 1 deg turn was flagged in only 70 % of trials | [Hilti camera SOTA](benchmarks/hilti_camera_sota.md) |
| `lidar-lidar` | refuted | NTU VIRAL (2/4): accurate, but does not beat scan-to-scan; cross-recording consistency 1.143 against 1.0 | [NTU VIRAL](benchmarks/ntu_viral_lidar_lidar.md) |
| `ins-lidar` | refuted | KITTI + Boreas rotation v1 (6/8); one recording has no observable rotation | [INS-LiDAR rotation](benchmarks/ins_lidar_rotation_v1.md) |
| `camera`, `gnss-imu`, `imu-vehicle`, `lidar-wheel_odometry` | no claim | native methods exist; no audit yet | [estimate on real data](benchmarks/estimate_real_data.md) |

## How an audit is run

The KITTI camera-lidar v2 audit is the most complete worked example (PRs #230 and #231).
Its files are the templates:

| Piece | Example |
| --- | --- |
| Driver: fetch, convert, run every check once, record every command | `tools/kitti_camera_lidar_v2_evaluate.py`, `tools/kitti_camera_lidar_sota_run.py` |
| Partial download of large zip archives (HTTP range, `If-Match` on the pinned ETag, CRC-32 checked) | `tools/kitti_drive_fetch.py`, `tools/kitti_drive_metadata.py` |
| Scorer: frozen metrics, gates and verdict as module constants | `tools/score_kitti_camera_lidar_v2.py` |
| Audit builder: benchmarks, protocol and audit for `calibrex sota audit` | `tools/build_kitti_camera_lidar_v2_audit.py` |
| Pre-registration | `docs/benchmarks/kitti_camera_lidar_v2_preregistration.yaml` |
| Tests | `tests/unit/test_score_kitti_camera_lidar_v2.py`, `test_build_kitti_camera_lidar_v2_audit.py`, `test_kitti_camera_lidar_v2_preregistration.py`, `test_kitti_drive_fetch.py`, `test_kitti_camera_lidar_v2_evaluate.py` |
| Results | `docs/assets/kitti_camera_lidar_v2/evaluation/` (runs, logs, scores, benchmarks, protocol, audit) |

### Steps

1. **Claim.** Write what is claimed, against which reference, and with which gates. An
   honest per-axis claim gates only the axes the method judges and adds a coverage
   gate. This is what turned camera-imu from refuted (v1, v2) to supported (v3).
2. **Spent data.** Before choosing evaluation data, find every recording any method has
   read:
   - grep the repository and its history (`git log --all -S<id>`);
   - for KITTI Odometry, map sequence numbers to raw drives.

   Choose the evaluation set by a rule applied to metadata only: archive file lists,
   sizes, ETags. The `exposure` field of the pre-registration says what was fetched.
3. **Development.** Work only on spent data. Commit the development scores. Any change
   made after looking at results is written down as such.
4. **Thresholds.** Derive them from the development values, by the rules used so far:
   - error gate: 1.5 times the worst development value, rounded up to the next 0.25 deg;
   - coverage floor: the development rate divided by 1.5, rounded down to the next 0.25
     (whole items, rounded down, for counts);
   - miss-rate allowance: the one-sided 95 % upper bound of the development miss rate
     (exact binomial; the rule of three when nothing is missed), rounded up to the next
     0.05.
5. **Freeze.**
   - Commit the code. This is `frozen_code_commit`: everything in `src/` and `tools/`
     that the evaluation uses.
   - Then commit the pre-registration in a docs-only commit. It pins every archive
     (size, ETag) and every calibration file (SHA-256).
   - Add a test that loads the real pre-registration and checks it against the frozen
     tools: claim text, pins, recipe, thresholds.
   - Open a PR and merge it.
6. **Run once** from `main`:
   - first check that `git diff <frozen> HEAD -- src tools` is empty;
   - repeat a run only under the pre-registered rerun policy (a failed download, OOM,
     a killed process), with the same command;
   - keep the failed attempt's record.
7. **Score and publish.**
   - Score with the frozen scorer, build the audit, then run
     `calibrex sota audit protocol.yaml --output audit.yaml`. It exits 0 only for
     `supported`, so exit 2 means refuted, not an error.
   - Regenerate the leaderboard (command below).
   - Update the audit page's results section, `sota_leaderboard.md`, the README SOTA
     table, `docs/index.md` and `CHANGELOG.md`.
   - Open a PR and merge it.

```bash
calibrex sota leaderboard \
  docs/assets/hilti2022_camera_sota/camera_camera/hilti_camera_camera_sota_audit.yaml \
  docs/assets/hilti_camera_imu_sota_audit.yaml \
  docs/assets/hilti2022_camera_sota/camera_imu_v2/hilti_camera_imu_v2_sota_audit.yaml \
  docs/assets/hilti2022_camera_imu_v3/hilti_camera_imu_v3_sota_audit.yaml \
  docs/assets/kitti_camera_lidar/evaluation/kitti_camera_lidar_sota_audit.yaml \
  docs/assets/kitti_camera_lidar_v2/evaluation/kitti_camera_lidar_v2_sota_audit.yaml \
  docs/assets/mid360_imu_lidar_extrinsic_sota_audit.yaml \
  docs/assets/mid360_imu_lidar_extrinsic_v2_sota_audit.yaml \
  docs/assets/mid360_imu_lidar_sota_audit.yaml \
  docs/assets/ntu_viral_lidar_lidar_sota_audit.yaml \
  docs/assets/kitti_lidar_vehicle_sota_audit.yaml \
  docs/assets/ins_lidar_sota_readiness/evaluation/audit/ins_lidar_sota_audit.yaml \
  --output docs/assets/sota_leaderboard.json \
  --target-pair camera-lidar --target-pair lidar-lidar --target-pair imu-lidar \
  --target-pair camera-imu --target-pair camera --target-pair camera-camera \
  --target-pair gnss-imu --target-pair ins-lidar --target-pair lidar-vehicle \
  --target-pair imu-vehicle --target-pair lidar-wheel_odometry \
  --markdown standings.md   # paste its tables into docs/benchmarks/sota_leaderboard.md
```

Checks before every push:

- `.venv/bin/python -m pytest`;
- `ruff check .`;
- `uvx --isolated --with-requirements tools/typecheck-requirements.txt mypy src/calibrex`
  (what CI runs);
- `mkdocs build --strict`.

The repository's own checks and PR template ([`.github/PULL_REQUEST_TEMPLATE.md`](https://github.com/rsasaki0109/Calibrex/blob/main/.github/PULL_REQUEST_TEMPLATE.md)) apply.

## Spent data

A recording listed here has been read by the named pair's methods. It may be used for
development; it is not evaluation data for that pair again. Reading by another pair's
method is disclosed in `exposure`; it does not by itself spend the data for a new pair.
The camera-imu v3 audit did exactly this with the camera-camera runs.

| Dataset | Spent for | Recordings |
| --- | --- | --- |
| KITTI raw | `camera-lidar` | **All** synced drives with at least 125 frames: the 36 of 2011_09_26, 2011_09_30_0018 and the 18 of v2 (2011_09_28, 2011_09_29, 2011_09_30, 2011_10_03). Shorter drives were never eligible. |
| KITTI raw | `lidar-vehicle` | 2011_09_26 0005, 0009, 0014, 0015, 0022 (dev); 0027, 0028, 0035, 0039, 0046, 0051, 0057, 0059 (evaluation) |
| KITTI raw | `ins-lidar`, `lidar-wheel_odometry`, `imu-vehicle` | 2011_09_26 0005, 0009, 0014, 0015, 0022 (tool runs, including the 0015 round trip; no audit) |
| KITTI raw | `ins-lidar` | 2011_09_30_0018 OXTS, read by the generated-motion trajectory hand-eye performance controls; not independent calibration evidence |
| KITTI raw | `ins-lidar` evaluation v1 | 2011_09_29_0071, 2011_09_30_0020, 2011_09_30_0027 (first 600 scans and matching OXTS; alias 06 = 0020, alias 07 = 0027) |
| Boreas | `ins-lidar` development | boreas-2020-11-26-13-58 (INS prefix/calibration inspected and first-600-scan development acquisition; see the [audit plan](research/ins_lidar_sota_plan.md)) |
| Boreas | `ins-lidar` evaluation v1 | boreas-2020-12-01-13-26 (first 600 scans, full gps_post_process.csv and calibration; only the fixed +/-1 s reference window enters fitting) |
| KITTI-360 | camera-LiDAR research | 2013_05_28_drive_0000, 0002 and 0003 (the [research plan](research/camera_lidar_sota_plan.md) records the materialized 0003 development capture; it was omitted from the original handoff) |
| Hilti 2022 | `camera-imu` | exp01-07, exp11, exp14-16, exp18, exp21, exp23 (parts 0-2); exp09 and exp10 never run (size) |
| Hilti 2022 | `camera-camera` | exp05, exp06, exp09, exp10, exp11, exp14, exp15, exp16, exp18, exp23 parts 0-2 (first 5 GB of each), plus development on exp01 and exp21 |
| MID360 | `imu-lidar` | RTK-SLAM stadtgarten_seq1, stadtgarten_seq2, construction_seq1, construction_seq2 (every RTK-SLAM recording) and the Zenodo vehicle recording of the rotation audit |
| NTU VIRAL | `lidar-lidar` | rtp_01, spms_01, tnp_02 (evaluation), tnp_01 (development) |

The v2 camera-lidar runs also read the Velodyne and OXTS of the first 600 frames of its
18 drives. A `lidar-vehicle` or `ins-lidar` audit on those drives must disclose that.

## Pitfalls met so far

- **YAML 1.1 reads unquoted IDs as numbers.**
  - `2011_09_28` becomes the integer 20110928, and `0027` the octal 23. The v2 driver would
    have found no calibration pin and refused to run; a test caught it.
  - Quote every ID, and test the loaded pre-registration's types.
  - `kitti_lidar_vehicle_preregistration.yaml` has unquoted drive IDs. It is binding and
    must not be edited; read its IDs as written.
- **Never edit `src/` while a run imports it.**
  - Run long jobs from a separate `git worktree` with `PYTHONPATH=<worktree>/src`. The
    editable install is a `.pth` path entry, so `PYTHONPATH` wins.
  - Check with `python -c "import calibrex; print(calibrex.__file__)"`.
- **Pickled option objects keep the defaults of the day they were pickled.** When you
  replay saved estimator inputs, pass the current defaults explicitly.
- **Disk.** About 24-29 GB is free. Fetch only what the method reads; for KITTI that is
  the first 600 frames, because `calibrex check` reads at most 48 scans 1 s apart from
  the start. Delete each recording after its run.
- **Zip archives.**
  - A zip64 archive can scatter one stream's frames across the file (2011_09_30_0018:
    835 ranges for 600 frames); `tools/kitti_drive_fetch.py` handles it.
  - Network drops do happen (curl exit 18). The pre-registered rerun policy covers a
    failed fetch.
- **A near miss is still a miss.** v2 failed one gate by 0.003 deg. Thresholds are not
  revisited after a run; a new version with new evaluation data is the only way on.

## Next candidates

### Development restart, 2026-10-27

The [training-only initialization work](research/camera_lidar_training_initialization.md)
reproduces all twelve archived audit verdicts from their stored scores and adds
an opt-in, per-fold training-only initialization mode to the camera–LiDAR profile
probe. It removes the pooled conditional-start dependency; synthetic tests are
not real-data accuracy evidence. All fourteen snapshot metadata files remain,
but their local NPZ arrays are absent in this workspace. Restore and hash-check
them before the fixed fourteen-input paired comparison. No new evaluation data,
audit verdict, threshold or estimator default changes; capture remains 2/55.

Follow-up recovery reacquired the two spent raw archives and matched their member
inventories/CRCs and calibration SHA-256 pins. All fourteen native captures
completed, with matching options/frame metadata, but all NPZ hashes differ.
Native evidence has small objective/control-statistic differences exceeding the
existing `1e-9` parity tolerance; transforms and discrete decisions match under
that rule. The linked development page retains every difference and the failed
download-cap attempt. Inputs now exist under
`/workspace/work/camera-lidar-recovery-20261027/capture/`; resolve numerical parity
before interpreting the initialization ablation. No new SOTA evidence is claimed.

The subsequent same-input code control at `78eeeae6` matches archived/current
pooled profile evidence on all fourteen reacquired inputs at the unchanged
`1e-9` tolerance. Historical cross-capture profile parity still fails on all
fourteen. A separately scoped training-only comparison completes all arms and
changes detected controls from 20/42 to 19/42: only 0005 roll +3's yaw control
is lost, with two training-fold fits in distant yaw basins. Both arms retain
zero detections on 0002, including missed pitch +3. Results and every fold are
linked from the development page. Do not promote or retune thresholds from this
result; investigate fold/basin stability and nuisance ambiguity next. Capture
remains 2/55 and the historical numerical drift remains unresolved.

The [control-seeded baseline ablation](research/camera_lidar_control_seeded_baseline.md)
then removes the targeted pitch fold’s training-search deficit but still misses
pitch +3 (t 0.774 to 0.848). Detected controls rise 19/42 to 26/42 while the worst
active-axis training-fold reference disagreement among detected controls rises
0.350 to 8.165 deg. Reject promotion; investigate correspondence quality rather
than spending another iteration on search or thresholds. All fourteen inputs
and both arms are retained; no fresh evaluation or default change.

The [local boundary tangent ablation](research/camera_lidar_boundary_tangent.md)
uses PCA of depth-discontinuity points rather than the previous azimuth proxy.
All fourteen original-arm parity checks pass; detected controls stay 19/42,
but worst active-axis fold reference disagreement among detected controls rises
0.350 to 6.547 deg. Target pitch +3 still fails (t 1.298). Reject promotion;
local direction penalties alone do not resolve correspondence ambiguity.
All paired outcomes and the interrupted scheduling-only run are retained.

The [bounded nearest-ridge distance ablation](research/camera_lidar_nearest_edge.md)
then replaces the scalar image response with a truncated distance reward.
All fourteen original-arm parity checks pass, but controls decrease 19/42 to
13/42 (eight lost, two gained), with 0002 still 0/21. Worst detected fold
reference disagreement decreases 0.350 to 0.246 deg, while target pitch +3
still fails (t -1.764) despite fitted pitch errors within 0.052 deg.
Reject promotion; inspect actual wrong-pose image matches and temporal persistence
before another objective variant. No new evaluation or default change.

The [fixed-pose match diagnostic](research/camera_lidar_pitch_matches.md)
replays all six pitch folds on all eleven existing prepared frames. Small pitch
errors hid baseline full rotation disagreements of 0.638–4.717 deg. At least one
saved wrong control beats the reference in 5/6 pooled folds and 10/11 frames.
Overlays show displaced points matching different road-side and vehicle contours.
No fitting or threshold change: investigate boundary identity/structure, and
report full rotation error in future ablations. Temporal score persistence alone
cannot reject the repeated wrong-pose advantages observed here.

The [local contour-coherence screen](research/camera_lidar_contour_coherence.md)
links mutual 3D neighbors and rewards their agreement on one image ridge component.
It covers 93.05% of prepared points, but saved wrong controls still beat the
reference in 5/6 pooled folds (frames improve 10/11 to 8/11). Reject integration
before spending a full optimization batch. All six folds and eleven frames are
retained. Component identity alone does not enforce neighboring-match geometry;
direction, separation and collapse remain untested structural hypotheses.

The [neighbor match geometry experiment](research/camera_lidar_match_geometry.md)
adds direction/spacing preservation to the contour graph. Its fixed-pose screen
reduces wrong-control wins from 5/6 to 1/6 folds. The frozen full fourteen-input
paired search retains 19/42 detections (two gained, two lost); 0002 remains 0/21.
Target pitch +3 worst full rotation error decreases 2.803 to 0.354 deg, but its
control t=1.539 remains below 2.5. All-fold worst full rotation error is still
12.067 deg and computation is substantially higher. Retain experimentally, do
not promote. All original-arm parity checks pass; future reports include full
rotation vectors/norms to avoid hiding nuisance-axis errors.

The [geometry failure cross-scoring](research/camera_lidar_geometry_failures.md)
finds explicit training-search deficits in the two lost 0005 controls: roll
fold 1 and yaw fold 5 have roughly 2 deg baseline errors while saved scalar fits
score about 0.092 higher under the geometry training objective. Their own
profiled controls also beat the baseline on training frames. The 0002 pitch miss
has no such deficit among inspected candidates; it remains a held-out ranking
ambiguity. Next test bounded training-only control-seeded baseline restarts under
the geometry objective; recovery is not established by cross-scoring alone.

The [deficit-gated geometry restart experiment](research/camera_lidar_geometry_restarts.md)
recovers both lost 0005 controls: 19/42 to 21/42, with all original nineteen
retained and detected-fold full rotation disagreement unchanged at 0.296 deg.
All fourteen geometry-control parity checks pass. 0002 remains 0/21; its targeted
pitch t stays 1.539. Fifty axis-fold baselines change (two on 0005, 48 on 0002),
and all-fold worst full rotation disagreement worsens 12.067 to 13.568 deg.
Retain experimentally; no native default promotion or SOTA claim. The search
regression is recovered, while difficult-scene correspondence ambiguity remains.

### Camera–LiDAR priority reset (2026-10-28)

The [external comparison reset](research/camera_lidar_external_comparison_reset.md)
supersedes further two-drive local tuning. Freeze the rotation-checker experiment
at `308885cd` (results `6b6379a9`). Official Koide preprocessing and six-DoF
calibration now execute on a three-frame spent-0005 smoke; exit 0, but reference
rotation disagreement worsens 1.503 to 2.227 degrees and point-time warnings
remain. This is execution evidence, not a fair comparison or successful recovery.
Next deliverable: a shared spent-data six-DoF native-versus-Koide comparison,
then required development coverage and learned-baseline execution. The fresh-data
protocol remains a blocked draft; no fresh sensor payload was read. Existing
55-drive, exposure and independent-family requirements remain in force.

The [shared-bundle follow-up](research/camera_lidar_shared_koide_bundle.md) now
adapts the point-time origin while preserving acquisition times, and runs Koide
and existing native Pandey MI on the same preprocessed cloud/image and seed.
Rotation reference disagreements are 0.645 and 1.281 deg respectively (initial
1.503 deg); neither meets the scoped recovery event. This is an optimizer-only
hybrid-pipeline diagnostic, not a D2D or independent end-to-end comparison.
Next resolve cloud/camera reference-time alignment and independent image-depth
input for the D2D release candidate, then expand spent-data coverage. Do not tune
on the single shared bundle. Fresh evaluation data remains unread.

The [motion-aligned stereo D2D execution](research/camera_lidar_aligned_stereo_d2d.md)
now uses OXTS/IMU–LiDAR motion to express a scan at image time without the target
camera–LiDAR reference, and stereo-only depth. On fixed spent-0005 frame 1, D2D
has 1.110 deg / 0.274 m disagreement and Koide 1.374 deg / 0.195 m; both fail the
scoped event. This is an assisted-input track, not an equal-information monocular
comparison or the integrated-r5 release protocol. Next freeze a multi-interval,
multi-drive development matrix; do not tune or repeat single-frame variants.
Monocular depth/provider exposure, 55-drive coverage and learned baselines remain
pending. Fresh evaluation payloads remain unread.

The [frozen assisted-input matrix](research/camera_lidar_assisted_matrix.md)
executed all 12 cases (two spent drives, three fixed temporal positions, two seed
signs) from precommitted plan `6a1ff93d`. D2D recovers 5/12, Koide 0/12; every
execution completes with valid output. D2D improves rotation 12/12 but worsens
translation 10/12. Unequal stereo input and custom OXTS preprocessing prevent a
monocular-superiority claim. Next test joint estimation over the three frames
per drive with unchanged inputs/seeds/gates, keeping this single-frame baseline.
Do not tune thresholds or select successful cases. Development coverage is still
2/55; independent-family/learned-baseline/fresh-audit requirements remain unmet.

The [joint three-frame experiment](research/camera_lidar_joint_assisted.md)
was frozen at `752c1f11` and completed all four drive/seed cases for both methods.
D2D recovers 3/4 and Koide 0/4, with all three native frames supported. However,
the predeclared translation comparison improves over the matching single-frame
median only 1/4 and worsens 3/4: joint fitting did not establish translation
stability. Do not compare 3/4 directly with the earlier 5/12 denominator. Retain
experimentally, freeze settings, and broaden spent-drive coverage; no default
promotion, seed cherry-picking or another two-drive parameter sweep. All existing
55-drive/learned-baseline/independent-family/fresh-audit requirements remain unmet.

The [additional-drive joint experiment](research/camera_lidar_expanded_joint.md)
extends the fixed configuration to spent 2011-09-26 0009 and 0011, three fixed
frames each. A preparation-only timestamp-gap bug was fixed and re-frozen at
`9053ed48` before any calibration; original logs remain. All four new cases
complete with all frames: D2D 1/4, Koide 1/4. D2D improves rotation 4/4 but worsens
translation 4/4. Across four recordings/eight joint cases, D2D is 4/8 with
translation improvement only 1/8; Koide is 1/8. Thus the translation regression
persists outside the original pair. Retain failures and settings; next assess
metric translation constraints together with depth/motion validity, not threshold
relaxation or favorable-seed selection. Coverage remains only 4/55 with unequal
stereo input; no superiority, independent-family or fresh-audit claim.

The [metric stereo translation experiment](research/camera_lidar_stereo_translation.md)
adds opt-in, ROS-independent translation-only point-to-plane refinement, with
D2D rotation fixed and no target reference in the fitter. Plan/source `9d01486a`
ran all eight joint initializers: all hit 15 outer iterations; none is accepted.
Returned pipeline recovery stays 4/8 by exact D2D fallback, and new-stage success
is 0/8. Raw candidates improve translation 5/8 and worsen 3/8, including 0011
positive from 0.136 to 0.340 m. All inner fits succeed and retain 2,000 matches per
frame, so sparse support/rank checks do not establish reliability. Retain this
negative result; do not relax criteria or promote the module. Next diagnose
association stability and stereo-surface validity before further method claims.

### INS-LiDAR after the first audit

The first two-family, two-platform rotation audit is [refuted, 6/8](benchmarks/ins_lidar_rotation_v1.md).
All eight estimated rotation axes agree within 0.286 deg and all eight observed
controls pass, but KITTI 0071 has no observable rotation. Its zero-axis case is
a failed scorable recording and gets the frozen worst-case control fraction of
one. Do not reinterpret `recording_completion=1.0` in the descriptive scorer as
nonempty rotation availability: all four artifacts completed, but only three
provide any calibrated rotation.

Post hoc inspection finds uncertainties of 0.378/0.226/1.225 deg against the
0.1 deg ceiling. The unobserved pitch/yaw differ by 0.558/1.976 deg; promoting
them would not establish reliable calibration. The physical cause is not yet
isolated. Use these spent windows to study informative motion, registration
stability and a declared data-sufficiency policy. A new claim needs new held-out
recordings and a new registration; no v1 gate or verdict may be relaxed.

The [post hoc registration-quality comparison](research/ins_lidar_registration_quality.md)
excludes four motions crossing the two unconverged scan registrations in 0071.
It still observes zero rotation axes; uncertainty changes are small. The new
`exclude_failed_registrations` option remains opt-in (default false), and this
negative development result does not revise the v1 audit. Study errors in the
remaining motions and block stability next; do not treat this filter as a
demonstrated calibration improvement.
The follow-up reproduces all nine native jackknife fits: omitting block 7
(35-40 s) changes the yaw component by 0.811 deg, although block 3 has the
largest rotation residual. Residual ranking alone is not a justified block
rejection policy; no such rule has been adopted.
The subsequent two-arm translation-noise comparison reduces the unobserved
yaw reference difference from 1.976 to 0.239 deg, but still has 1.056 deg
uncertainty and fails the yaw control. Defaults are unchanged. An OXTS-only
development acquisition of all 1059 samples finds the same 16.510 deg heading
range as the first 600; no additional point clouds were acquired. These
reference-only proxies do not decide observability or rule out a benefit from
more samples. The complete 0071 OXTS recording is spent development data.

The published protocol replays the six achieved and two contradicted gates
without sensor data. No calibration rerun is needed to reproduce the verdict.

### 1. camera-lidar v3 (closest to supported)

**What failed in v2.** The far-optimum mechanism. On 2011_09_28_0002, four turned runs
took an optimum from the profile-maxima search start that:

- observed roll and pitch, against the candidate optimum's pitch alone;
- had roll's held-out control at `t` 2.58-2.65, just above the 2.5 bound;
- judged roll 1.45-1.50 deg off the day's calibration.

This is post hoc
([v2 results](benchmarks/kitti_camera_lidar_v2_check.md#results-refuted-8-of-9-gates)).
v1 failed the same way, one notch lower.

**A rule to try.** An optimum from a far start must clear its axes' controls with a
margin well above the bound (for example `t` >= 3.5), or agree with the candidate
optimum on the axes both observe. Weigh coverage: v2 already judges about 80 % of axes.

**Development data.** Every KITTI raw drive (55, all spent). The existing
`tools/kitti_camera_lidar_v3_replay.py` captures seven real checks per drive and
compares both rules on identical saved inputs. The
[2026-10-23 native pilot](benchmarks/kitti_camera_lidar_v3_development.md#native-spent-data-pilot-2026-10-23)
completed 0005 with exact replay parity and no difference between rules. The
[0002 comparison](benchmarks/kitti_camera_lidar_v3_development.md#known-failure-case-2026-10-23)
reduces maximum judged turned-run error from 1.503 to 0.100 deg, but loses the
only counted 3 deg detection (1/1 to 0/1). Preselected yaw +3 and post hoc pitch
+3 native checks both confirm the proposed replay's inconclusive result with
no judged rotation axes. Do not describe this as an overall improvement or
promote 3.5 from these data. **2/55 is incomplete; 53 remain.** The original
rounded-calibration guard failure is retained and its tolerance unchanged.
Inspect the detection/coverage cost, including on spent 0034, and preserve
the all-55 denominator and local NPZ inputs. Check free space before batches.

The [post hoc start trace](benchmarks/kitti_camera_lidar_v3_development.md#why-a-scalar-adoption-margin-is-insufficient-on-these-probes)
reproduces both selected controls under both rules with full evidence parity.
Each has an unobservable candidate-start fit, a rejected one-observed-axis
profile fit, then a two-observed-axis fit. The latter's minimum control t is
2.6211 for pitch +3 versus 2.6493 for yaw +3: a scalar margin retaining the
former also admits the latter's erroneous roll. The pitch fit also has an
erroneous roll; keeping detection alone does not fix accuracy. Develop a
different start/axis evidence rule on cached spent inputs before spending
disk and bandwidth on the remaining batch. Do not simply relax the two-axis
requirement: the earlier 0056 single-axis false estimate motivated that guard.

The [conditional-axis probe](benchmarks/kitti_camera_lidar_v3_development.md#conditional-axis-development-probe)
then ran all 14 cached inputs (42 one-axis fits). It recovers pitch +3 on
0002 with +0.15 deg active-axis reference error and t=3.3036, but a conditional
std of 0.0417 deg hides a 7.8 deg nuisance-sweep span. All 18 fits meeting the
ordinary conditional evidence conditions have active-axis error at most 0.15
deg, yet these separate fits are not joint observability, native verdicts or
a promoted method. The tool and byte-bound batch are available for research;
account for uncertain nuisance axes before adoption. No fresh data was read;
capture progress stays 2/55 and 53 remain.

The [training-only nuisance-profiled controls](benchmarks/kitti_camera_lidar_profile_controls.md)
use committed source `7a6ea534` and all 14 cached inputs. At the same refitted
fold poses, direct/profiled controls detect 25/42 versus 21/42 axes (control
outcomes only, no std/coverage/verdict gate). Four 0002 pitch controls lose
detection, but its roll +3 still passes with profiled t=3.158 and about 1.5 deg
maximum absolute roll reference error among the saved fold fits. This is not
an accurate-calibration guarantee or a promoted method. Probe broader nuisance
basins before setting new gates. All outputs and this counterexample are retained;
the pooled conditional initialization means these are not nested validation folds.

The follow-up opt-in `--multi-start` uses source `24543be7`, fixed before all
14 cached inputs were re-run. Training-only nuisance profiles plus their
individual/combined starts retain the original local optimum on ties. Paired
baseline/direct evidence matches within `1e-9`. The only changed detection is
the known 0002 roll +3 false control: t falls from 3.158 to 1.732 and fails the
unchanged 2.5 threshold. All 0005 flags stay unchanged. Profiled controls now
detect 20/42 instead of 21/42, with zero on 0002 and pitch +3 still missed.
This exposes a selected ambiguity; it does not correct the saved fits, provide
native verdicts, recover useful coverage, or justify promotion. Both batches
and their source capsules are preserved. Capture progress remains 2/55.

The [fixed 2x2 spatial-weight ablation](benchmarks/kitti_camera_lidar_spatial_weights.md)
preserves frame totals and candidate-outside support, but does not restore
0002 pitch detection. Detected multiple-start controls fall from 20/42 to
12/42 (0005: 20 to 12; 0002: zero in both). Do not adopt it or relax thresholds
from this negative result. Source `50c77d37` first failed on all 14 inputs
because it rejected the intentionally retained image margin; that attempt is
preserved. The fix at `67ff8c5d` was frozen before the complete fourteen-input
retry, which binds the original failed report and unchanged inputs/settings.
This is control evidence, not a native coverage or accuracy claim; 53 drives
and the separate pre-registration remain pending.

The [projected-azimuth orientation proxy](benchmarks/kitti_camera_lidar_orientation.md)
was frozen at `c237f78d` before six selected inputs (baseline, pitch +3 and roll
+3 on each cached drive), using two independent workers. It compares image
gradients to a scan-direction proxy, not actual boundary normals. Scalar/proxy
controls detect 8/18 versus 0/18; 0002 pitch stays missed. Do not adopt or tune
this proxy from the result. All six outputs and a separate default scalar-mode
replay are retained; the latter matches previous numerical evidence excluding
provenance within `1e-9`. No new drive, fresh evaluation, SOTA gate or default
changes; capture progress remains 2/55.

**Evaluation candidates.** KITTI-360 (another car, sensor set and calibration) is the
natural choice: sequences 0004-0007, 0009 and 0010, subject to completing the exposure
audit. Sequence 0003 is spent: the research plan already records a development capture.
The original handoff's claim that it was unread was incorrect.

**Access update, 2026-10-23.** The [local recheck](benchmarks/kitti360_camera_lidar_v3_access.md)
verified exact official metadata ranges for 21 archives; the earlier cloud 403
remains historical evidence. Published-pose and explicit raw-OXTS conversion
both passed three-frame checks on spent 0000 and 0002. OXTS uses its own clock
and recorded origin without consuming optimized published poses. Provisional
candidate sensor contents remain unread; exposure auditing is incomplete and
no evaluation has started. This is converter readiness, not accuracy evidence.

**Work needed first.**

- ~~Capture and compare all 55 spent KITTI raw drives using the existing
  `tools/kitti_camera_lidar_v3_replay.py`~~ done 2026-10-31 (see the last entry);
  a selected new rule still needs confirmation via the real driver.
- Complete exposure auditing and independent-platform research requirements;
  KITTI-360 sequences alone do not add dataset families or rigs.
- KITTI-360 has few, long sequences: pre-register several disjoint 60 s segments per
  sequence, chosen by a rule from metadata.

### 2. ins-lidar follow-up

**Audit update, 2026-10-22.** The [first rotation audit](benchmarks/ins_lidar_rotation_v1.md)
is refuted (6/8), as described above. Boreas 2020-12-01 and KITTI 0071/0020/0027
are now spent evaluation recordings, not unread candidates. The
[audit plan](research/ins_lidar_sota_plan.md) retains the independent-Applanix
adapter and failure-aware scoring. A follow-up needs development on spent
recordings, new held-out data and a new pre-registration. Preserve two dataset
families and two platforms; different dates of one rig are not independent
rigs. Clock accuracy was outside the first rotation claim.

The native trajectory hand-eye on KITTI 0005+0009 constrains roll, pitch and the clock
offset, but not yaw or translation ([KITTI INS-LiDAR](benchmarks/kitti_ins_lidar.md)). A
per-axis honest claim, as camera-imu v3, could gate what is judged plus coverage. KITTI
drives of the other days are unread by INS-LiDAR methods; disclose the v2 camera-lidar
reads.

### 3. The refuted pairs

- **camera-camera** needs better 1 deg detection on Hilti before a new round.
- **lidar-lidar** needs cross-recording consistency, and a comparison with scan-to-scan
  that the method wins.

### 4. Lower priority

- **imu-vehicle:** no axis was constrained on KITTI; it needs data with excitation.
- **lidar-wheel_odometry:** needs real wheel odometry.
- **gnss-imu:** only composed from other pairs so far.

## Open work not in this handoff

Draft PRs #202, #205, #206, #208, #211 and #213 (CUDA, B-spline and LiDAR-IMU
performance) are being consolidated by another session. Leave them to it.


### Stereo translation diagnostic follow-up (2026-10-29)

Post-hoc replay of all eight rejected trajectories reproduced 120 saved inner
costs. In each case's last five updates, selected-pair retention averaged only
6.8–16.2%; 40.1–75.5% of previously selected exact pairs were still eligible but
excluded by uniform-index resampling. This establishes sampling churn, not its
causal share of non-convergence or calibration error. The next controlled
spent-data comparison is deterministic point-identity sampling with all other
settings fixed, across all eight cases. Original acceptance remains 0/8.
See [the replay evidence](research/camera_lidar_stereo_translation.md#post-hoc-correspondence-replay-2026-10-29).

### Fixed point-priority comparison (2026-10-29)

[Eight-case controlled comparison](research/camera_lidar_stereo_priority.md), frozen
at `c003dae8`: changing only the sampling rule raises late-update selected-pair
retention to 49.1–97.3%, and eligible-but-unsampled losses fall below 0.3%.
Acceptance rises to 1/8, but that case worsens reference translation by 3.24 cm;
returned-pipeline recovery stays 4/8. Seven still fail convergence. Keep the
uniform default and rejection gates. Sampling stabilization alone is insufficient;
next investigate surface/depth consistency and frozen-rotation sensitivity.

### Reference-assisted consistency controls (2026-10-29)

[Diagnostic controls](research/camera_lidar_stereo_reference_controls.md), frozen
at `84e87282`: substituting reference rotation yields only 1/8 converged. Starting
at the full reference pose drifts 3–13 cm on all four distinct recordings (three
reach the iteration limit). All 12 distinct frames have negative median
LiDAR-minus-stereo z residuals at reference, −0.119 to −0.056 m on gated pairs.
This rules out fixed D2D rotation as a sufficient explanation, but does not isolate
stereo bias. Next audit depth generation, camera/reference coordinate conventions
and temporal alignment before another solver variation. Reference-assisted
controls are not deployable calibration scores; earlier verdicts remain unchanged.

### Stereo projection conversion fix (2026-10-29)

[Projection audit](research/stereo_projection_audit.md) found a concrete frontend
bug: `fx*baseline/disparity` omitted the right-pixel term for unequal camera z
offsets. Corrected full-projection depth conversion is tested with known geometry.
On existing pairs it changes 0002 depth by −3.25 to −3.59 cm at the median, but
other drives by less than 0.34 mm. All residual medians remain negative. Reference
transforms and source stereo timestamp identities agree; physical point timing
has not been independently validated. Median predicted vertical stereo mismatch
is 0.133–0.288 px; next compare proper stereo rectification before more solver
changes. No historical arrays, calibration verdicts or gates were modified.

### Stereo re-rectification comparison (2026-10-29)

[Frozen comparison](research/stereo_rerectification_comparison.md) at `bae08665`:
12 frames, two frontends, eight unchanged D2D initializers. Re-rectification
reduces gated median z residual magnitude in 11/12 frames, but valid pixels drop
in all frames. Both it and the corrected-projection control reject all eight
translation fits for max iterations; returned recovery stays 4/8. Raw candidates
improve in five and worsen in three relative to control. No promotion or verdict
change. Next inspect stereo correspondence consistency and disparity-dependent
residuals, retaining support and downstream calibration as required outcomes.

### Independent stereo round-trip diagnostic (2026-10-29)

[Diagnostic](research/stereo_cycle_diagnostic.md), frozen at `ae87b4f0`: all 12
forward depth arrays reproduced exactly. Of 183,329 projected stereo-supported
points, 90.5% satisfy independent 1 px reverse consistency, 1.6% contradict and
7.9% lack reverse support. Consistent populations have lower absolute-z p90 in
all frames, but all retain negative median residuals. At <8 px disparity,
per-frame consistent median z residuals are −6.71 to −3.53 m without a depth gate;
visibility/timing/reference effects remain unresolved. Next test the frozen
reverse-consistency mask across all eight fits without fitting range cutoffs to
these results. No optimizer rerun, production change or accuracy claim this step.

### Fixed stereo round-trip mask comparison (2026-10-29)

[Eight-case comparison](research/stereo_cycle_filter_comparison.md), frozen at
`0bb89aa4`: the 1 px mask retains 84.4–89.1% of depth pixels and 83.7–88.9% of
surface pixels. All eight fits still reach max iterations, despite 2,000 matches
per frame at every update. Raw candidates improve in five/worsen in three versus
unmasked; 0011-positive worsens from 0.3244 to 0.4546 m. Accepted stage 0/8,
returned recovery 4/8 via exact D2D fallback. Do not promote the mask or relax
convergence. Close this sequence of frontend tweaks: next broaden to other spent
recordings and compare an alternative measurement model before further tuning.
The projection formula fix remains valid, but downstream benefit of later
frontend variants has not been established.

### Broader six-DoF measurement-model comparison (2026-10-29)

[0013/0015 comparison](research/camera_lidar_broader_models.md), frozen at
`fa09ba7e`: two additional spent drives, three metadata-selected frames each,
two unchanged seeds, eight completed/converged solver runs. D2D recovers 1/4,
improves rotation in 4/4 but worsens translation in 4/4. Native Pandey recovers
0/4; its 0015-negative solution has 17.4007-degree reference rotation difference
despite convergence. No defaults/gates promoted. Sparse assisted-input comparison
coverage is now six recordings, not completion of all-55 capture/replay. Next
prioritize independently validated full-pose quality/rejection, preserving both
coverage and false-acceptance evidence; convergence alone is demonstrably unsafe
as an accuracy criterion. No held-out audit or SOTA claim.

### Joint inverse-depth pose estimator (2026-10-29)

[Six-drive/twelve-case experiment](research/stereo_disparity_pose.md), frozen at
`517c4eb6`: a new experimental ROS-independent six-DoF solver uses bilinear
pixel-scaled inverse-depth residuals and fixed LiDAR populations. Synthetic known
planes recover all axes; unobservable/missing-depth tests reject. Real recovery
is **1/12 versus matched-input D2D 5/12**. Eight new fits lose required support;
four pass numerical checks, but three of those fail reference accuracy gates.
Do not promote, relax retention or call SOTA. Next address explicit visibility
and missing-depth boundary continuity, with controlled hole/occlusion tests and
independent full-pose validation. Preserve the negative result and original inputs.

### Continuous support weighting result (2026-10-29)

[Controlled twelve-case test](research/stereo_tapered_support.md), frozen at
`c9f24ea1`: continuous sin/cos measurement/missing-support blend passes synthetic
hole/step continuity and known-plane recovery tests, but real recovery falls to
**0/12** (hard mode 1/12, matched D2D 5/12). Five numerical accepts all fail
accuracy gates; remaining statuses are three lost-support, two budget exhaustion
and two bounds. Do not promote or relax gates. Stop extending this image-support
penalty approximation; next require independently supported correspondences and
actual geometry recovery through controlled holes/occlusions, not continuity alone.
Historical outputs remain unchanged; no fresh data or SOTA claim.

### Surface-registration comparison (2026-10-29)

[Single-frame ICP/D2D comparison](research/stereo_surface_registration.md), frozen
at `464279c5`: existing Chen–Medioni tangent-plane ICP with mutual matches,
trimming and spatial holdout passes five geometric hole/occlusion/overlap tests,
but recovers **0/12** on central real frames versus matched single-frame D2D
**3/12**. Seven ICP fits converge inaccurately; five reach the iteration limit.
All have rank six and 0.0317–0.0712 m holdout plane RMSE. Same-stereo-surface
holdout is not independent calibration validation. No defaults/gates promoted.
Next challenge the shared stereo reconstruction using an independently specified
measurement source, not another association/penalty variant on these same depths.
Historical joint D2D 5/12 remains a separate three-frame result.

### Independent image matcher comparison (2026-10-29)

[ZNCC/SGBM comparison](research/stereo_matcher_comparison.md), frozen at
`ae028fc7`: same six central frames, re-rectification, twelve seeds and D2D
settings. All 24 fits converge; full-pose recovery is SGBM 2/12 versus ZNCC 1/12.
On exact common projected points, ZNCC improves median residual magnitude in 5/6
but worsens absolute p90 in 5/6 and loses substantial depth coverage everywhere.
Do not promote. Negative common-point medians persist in both; causality remains
unresolved. Next establish a published modern stereo/cross-modal model's pinned
code/checkpoint/license/training exposure/runtime before further comparisons.
Do not continue classical matcher/penalty tweaks as a substitute for SOTA evidence.
No new held-out data or historical verdict change.

### Published learned stereo runtime (2026-10-29)

[UniMatch runtime](research/unimatch_runtime.md), protocol frozen at `3a9d60e8`:
official SceneFlow GMStereo-scale2-regrefine3 checkpoint and source commit pinned;
MIT source license and separate checkpoint-license uncertainty documented.
Strict weights-only loading and native-resolution spent 0005/frame-76 CPU smoke
succeeded: 17.24 s, 1242×375 output, all finite. This establishes runtime readiness,
not depth accuracy, calibration recovery or contemporary SOTA. Next freeze and
complete the same six-central-frame/twelve-seed D2D comparison against re-rectified
SGBM (2/12), disclosing RGB versus grayscale inputs and preserving failures.
No new data, thresholds or production defaults changed.

### Learned stereo full-pose comparison (2026-10-29)

[UniMatch/SGBM comparison](research/learned_stereo_comparison.md), frozen at
`2f73c756`: six native-resolution image pairs, twelve matched single-frame D2D
cases, all 24 converged. Recovery remains **2/12 versus 2/12**; UniMatch gains
0009-negative and loses 0002-positive. Translation improves 10/12 (median
0.2771 → 0.2131 m), rotation only 6/12 (median 0.7135 → 0.7565 deg). More valid
depth everywhere does not establish calibration success. All twelve recomputed
SGBM outputs exactly reproduce the prior control. No default promotion or SOTA.
Next pre-register matched three-frame joint learned/SGBM comparison using only
the archived frames; recompute both frontends and preserve all seeds/failures.
Do not compare against historical three-frame 5/12 as if inputs were identical.

### Three-frame joint learned stereo result (2026-10-29)

[Joint comparison](research/joint_learned_stereo.md), frozen at `1948a499`:
18 pairs and 24 completed/converged fits. Matched re-rectified SGBM recovers
**6/12**, UniMatch **5/12**. Translation improves 8/12 (median 0.2190 → 0.1834 m),
rotation only 2/12 (median 0.3096 → 0.3684 deg). UniMatch has more valid depth
in all 18 frames but no full-pose advantage. Six central-frame depth pairs exactly
reproduce the prior experiment. Do not promote, relax the 0.2043 m near miss,
or confuse this matched SGBM 6/12 with historical differently processed 5/12.
Close matcher/aggregation trials. Next use independently generated known camera
and LiDAR geometry to test the same joint solver with exact and perturbed depth,
non-planar surfaces and controlled visibility before choosing another algorithmic
change. Such a control is diagnostic only, not new real-world/SOTA evidence.

### Independent known-geometry control (2026-10-29)

[Results](research/known_geometry_joint_results.md), frozen at `9ffd9aee`: analytic
camera/LiDAR raycasts from distinct origins, five spheres plus slanted plane,
three views, five measurement/visibility conditions. All 15 fits converge but
perturbed recovery is **1/10** (reference-start 5 trials kept separate). Even
exact optical z plus oracle visibility fails both perturbations; truth-start
drift is only 0.035 deg/6.25 mm. Ground truth has higher MI than every returned
perturbed fit, proving a feasible better objective was missed on this scene,
not proving the global optimum equals truth or explaining all real failures.
Constant +0.25 m shift gives identical traces due to histogram shift invariance.
Next test deterministic broader search against unchanged objective under a stated
equal evaluation budget on frozen controls before revisiting real data. Preserve
baseline, all starts, runtime and truth-start drift; no reference-guided seeds.
Stop further stereo replacement as a presumed cure. No production/SOTA change.

### Equal-cap fixed multistart result (2026-10-29)

[Results](research/joint_multistart_results.md), frozen at `b221d707`: same objective,
original global bounds and 800-call ceiling; one start versus five fixed starts
at 160 each. All 15 baselines/75 starts executed. Perturbed accepted recovery
**1/10 → 0/10**: all ten selected candidates exhaust their per-start budgets.
Geometry-only pass count improves 1→5 but must not be called recovery. All-return
reference-start drift worsens to 0.8646 deg despite higher MI; other four match
baseline. Equal ceilings are not equal work (perturbed 193–337 vs 800 actual calls).
Baseline outputs exactly reproduce prior study. Reject promotion/real-data rerun.
Next allocation must preserve converged baseline and explicitly count remaining
exploration/refinement work; retain visibility/drift controls because higher MI
alone is unsafe. No reference-guided starts or after-result tuning.

### Baseline-preserving continuation (2026-10-29)

[Results](research/joint_continuation_results.md), frozen at `420daa35`: unchanged
starts/objective/bounds, full baseline plus four ≤40-call probes, remaining calls
to highest-MI explorer with exact state continuation. Converged higher-MI results
only; otherwise exact baseline fallback. Recovery **1/10 → 5/10**, all positive
seeds pass and all negative seeds fail. All refined explorers converge; total
353–689 ≤800 calls. Six baseline fallbacks/nine replacements, exact baseline
reproduction. Truth-start controls **5/5 → 4/5**: all-return case now converges
wrongly at 0.8560 deg/0.1427 m. Budget issue resolved; MI-only selection remains
unsafe. Do not promote or rerun real data as a validated improvement. Next freeze
reference-free geometric candidate validation on disjoint analytic LiDAR samples,
retaining occlusion/unsupported populations and correct/wrong candidate controls.
No real/SOTA claim; conditions/seeds are dependent, not independent scenes.

### Disjoint geometric validation fails (2026-10-29)

[Results](research/joint_geometric_validation_results.md), frozen at `3249838a`:
40 fixed candidate checks on new angular LiDAR rays, no camera-occlusion filtering,
reference-free per-frame median/p90/support criteria. Accepts 16/20 accurate and
**2/20 inaccurate** poses; rejects four correct biased-depth candidates. Crucially
the all-return truth-start drift (0.8560 deg) PASSES, as does exact-range negative
continuation (0.5731 deg). Shared camera depth/scene means disjoint samples are not
independent sensor evidence; aggregate residuals do not bound pose error. Do not
promote, retune thresholds or call continuation safe. Next compare fitting metric
geometry versus MI using only original fitting samples, unchanged initial poses,
fixed budget, occlusion/bias controls; reserve these points for diagnostics. Freeze
estimator first, keep full-pose labels and baseline; no heldout-guided parameter
selection. Synthetic results remain development evidence, not SOTA.

### Direct metric-depth fitting (2026-10-29)

[Results](research/joint_metric_results.md), frozen at `d6651dd8`: fixed-population
bilinear optical-z soft-L1/TRF fitting, all original samples and unchanged pose
bounds/gates. Recovery **MI 1/10 → metric 6/10**, both signs succeed for visible
exact z, radial depth and holes. All-return occlusion fails 2/2 (~0.752 deg);
+0.25 m depth bias fails 2/2 (~0.222 m). Truth-start controls regress 5/5→3/5.
All 15 metric fits converge/retain support, 111–202 actual residual calls. Frozen
holdout happens to agree for these outputs but retains its known false-accept
limitation. Do not promote, relax gates or claim SOTA. Next isolate physically
motivated non-oracle visibility handling on additional analytic controls, retaining
visible-only recovery and truth-start drift checks; assess depth-bias/translation
confounding separately rather than adding both mechanisms at once. Baseline exactly
reproduced; no core/default changes.

### One-sided metric occlusion control (2026-10-29)

[Results](research/occlusion_pose_results.md), frozen at `a2e22ce5`: three fixed
sphere arrangements, visible/all-return populations, both signs + true starts,
36 fits. Supported positive residual r becomes r/sqrt(1+(r/.2)^2); negative and
unsupported penalties unchanged. Perturbed recovery **8/12 → 12/12**; all-return
**2/6 → 6/6**, visible stays 6/6. True-start controls **4/6 → 6/6**, though nonzero
drift remains. Original visible-only accuracy worsens ~0.14→0.20 deg while still
passing. All asymmetric fits converge/retain support, 131–238 calls. Original
symmetric results exactly reproduce. Positive synthetic result only: no production
or SOTA claim. Next isolate depth-bias versus translation confounding with fixed
visibility handling, zero/positive/negative bias and truth-start controls; test
observability before accepting a fitted nuisance bias. No further saturation tuning.

### Japanese development review (2026-10-29)

[Japanese brief](research/development_brief_ja.md) summarizes evidence through
`064ea6f3`, separating real-data and analytic controls, adoption limits and the
unexecuted depth-bias/translation-identifiability experiment proposal.
[Printable HTML](assets/development_brief_20261029/index.html) and
[three-page PDF](assets/development_brief_20261029/calibrex-development-review-ja.pdf)
are shareable reading copies. This documentation adds no new experiment or accuracy claim.

### Depth bias versus translation (2026-10-29)

[Results](research/depth_bias_pose_results.md), frozen at `9bee5c16`: the frozen
one-sided fit plus 0/+0.25/−0.25 m constant camera-depth bias on the three layouts,
both populations, signed and truth starts. 210 fits plus 84 sensitivity checks,
no failures dropped; zero-bias results reproduce the prior run exactly. Pose-only:
positive bias fails **0/18** (Δt_z ≈ 0.85–0.97 b, 15 failures converge), negative
bias passes 18/18 only because saturation masks it. Local sensitivity at truth
calls every condition distinguishable (e ≈ 0.90–0.94) yet predicts only
0.10–0.18 b translation shift, so it is not a safety check. Joint pose+bias fit:
pose 66/66, bias within 0.05 m only 22/66; every estimate trades β̂ − b ≈ −Δt_z
(4–7 cm), zero-bias translation error grows to 4.4–8.5 cm. Verdict
**not_supported**: do not adopt the bias variable or widen its tolerance.
Positive depth bias remains a blocker before real-data reruns. Next work needs a
new frozen protocol (zero-bias β/t_z offset source, trade-breaking geometry, or
bias detection with wrong-candidate controls). No production or SOTA change.

### Depth-bias response and local confounding (2026-10-29)

[Results](research/depth_bias_response_results.md), frozen at `ae51adee`: fixed
visibility solver, 3 layouts × 2 populations × biases 0,+.25,-.25 m × ± seeds/truth
start =54 fits. Perturbed recovery: **12/12, 0/12, 12/12** respectively; truth-start
6/6,0/6,6/6. +bias median z displacement +0.2236 m; -bias tolerance is influenced by
positive-residual saturation, not estimated correction. All 18 zero-bias outputs
exactly reproduce. Three +bias runs hit max_nfev; other 51 converge, 86–592 calls.
Raw Jacobian bias unexplained fraction .9539–.9842 (no <.05 flags); planar control
correctly exposes degeneracy. Local raw rank does not prove robust/global bias
identifiability. Next freeze experimental seven-parameter pose/bias fitting with
zero correction initialization, separate bias/pose scoring, raw+robust sensitivity
and plane degeneracy controls. Preserve fixed six-parameter baseline, no truth-fed
bias or holdout tuning. Review PDF remains historical; stages A/B completed, C pending.

### Joint pose and depth-bias estimate (2026-10-29)

[Results](research/joint_pose_bias_results.md), frozen at `b58e9d93`: correction
initialized at zero, fixed ±.5 m bound, same asymmetric depth objective, 54 sphere
fits plus 9 plane controls. Perturbed pose-only recovery **24/36→36/36** (+bias
0/12→12/12), but pose+bias recovery only **12/36**, truth-start joint 6/18. All
sphere fits numerically converge/pass local rank; every estimated bias is too low
by .0408–.0668 m, including clean measurements. Only staggered layout meets joint
bias gates. All nine plane fits converge but are correctly flagged degenerate;
none accepted jointly. 8–566 optimizer calls, no failures omitted. Do not promote,
relax bias gate or empirically subtract the observed offset. Next freeze analytic
first-surface depth-sampling diagnostic versus bilinear pixel sampling to isolate
model mismatch, keeping fixed seeds/objective and all bias/occlusion controls.
Oracle sampler is diagnostic only; rank does not imply unbiased estimation/SOTA.

### Analytic depth-sampling control (2026-10-29)

[Results](research/analytic_sampling_results.md), frozen at `b355c938`: same solver
function definitions, only bilinear sampler replaced by known first-surface raycast.
54 sphere cases, archived baseline pinned. Joint recovery **12/36→33/36**, truth-start
**6/18→11/18**. Median bias error .05590 m→.00000948 m; zero-bias max .000341 m,
strongly implicating pixel/bilinear approximation in previous systematic offset.
Do not omit failures: four max_nfev, six negative-bias truth starts converge with
wrong bias (up to .24990 m). 50/54 converge; 8–639 actual calls. Oracle is diagnostic
only, not deployable or real-data evidence. Next separate boundary-aware image-only
sampling from investigation of exact-sampler negative-bias truth-start derivative/
stopping failures; do not change both together or relax gates. Keep all six wrong-bias
truth starts as controls. No production/SOTA claim.

### Analytic truth-start stopping diagnosis (2026-10-29)

[Results](research/analytic_stopping_results.md), frozen at `bf5d96e0`: no new fits;
18 truth-start cases probed at initial/final poses, 3 finite-difference steps and
fixed bias grids. All six negative-bias failures stopped by xtol despite positive
exact bias gradient 233–247; -.01 m bias step lowers cost in all six. Pose probes
at 1e-8 cause >1 m residual jumps for one point with unchanged support, maximum
Jacobian singular values ~1e9 and forward/central disagreement ~1. Bias derivatives
remain smooth and agree with analytic values. First-surface silhouettes cause
real discontinuity even without pixel interpolation; rank is not smooth covariance.
Evidence identifies a local-linearization mismatch, not every internal SciPy cause.
Next freeze separated scalar bias updates versus pose updates under one counted
budget, original objective/samplers and all controls; require actual cost decrease.
Do not use injected bias or mix in boundary-aware sampling/tolerance changes.
No revised recovery, production or SOTA verdict.

### Separated bias/pose blocks fail overall (2026-10-29)

[Results](research/block_bias_results.md), frozen `91e7aecb`, serialization-only fix
`c2409b5c` (initial plan/traceback retained). Same analytic objective/sampler, scalar
bounded search then pose-only TRF, ≤800 shared calls/eight cycles. Perturbed joint
recovery **33/36→0/36**, all 36 capped; geometry+bias alone passes 18/36 but not
accepted. Truth starts **11/18→17/18**; negative truth starts recover bias in 6/6,
only 5/6 converge. 33 regressions/six gains, actual 19–785 calls, no gate relaxation.
Reject full block replacement. Preserve joint baseline; if pursuing scalar rescue,
use residual-based trigger and remaining budget after completed joint fit, retaining
fallback and all regression controls. No truth-label/bias-guided triggers or combining
sampler changes. Monotonic cost is not proof of pose accuracy. No production/SOTA.

### Baseline-preserving scalar rescue (2026-10-30)

[Results](research/scalar_rescue_results.md), frozen `cb8c265c`: unchanged joint
fit first, residual-only ±.01 m probes, fixed-pose bounded bias correction using
remaining shared 800-call budget. All 54 baseline records exactly reproduce;
all output poses identical to baseline. Perturbed joint recovery **33/36→33/36**,
truth starts **11/18→17/18**, six gains/no regressions. Six triggers/adoptions,
bias errors .000218–.006862 m; 11–639 calls. Four previous unconverged cases remain
failures. Staged joint termination + scalar termination does not prove new joint
stationarity. Oracle-only synthetic evidence, no production/SOTA claim.
Next prioritize image-only sampling versus the frozen bilinear baseline, with
full bias/occlusion/truth-start and planar controls; do not tune the four oracle
failures or combine sampler/rescue changes in one unattributable comparison.

### Image-only edge-preserving depth sampling (2026-10-30)

[Results](research/edge_sampling_results.md), frozen `8a9babf4`: nearest-pixel
anchor, retain 2x2 neighbors within .2 m optical z and renormalize bilinear weights.
No scene/oracle information, no scalar rescue; identical joint solver functions.
54 sphere + nine plane cases. Perturbed joint recovery **12/36→32/36**, truth
starts **6/18→10/18**, median bias error **.055900→.002071 m**. All sphere poses
pass pose gates and numerically converge. 26 gains but **two regressions**:
staggered negative-bias truth starts both populations. Six negative truth starts,
four positive-seed and two original positive truth starts still fail bias gate.
Nine planes remain rejected as degenerate; 8–201 calls. Do not promote or relax
thresholds. Next isolate residual-triggered scalar rescue with this image-only
sampler, full 63-case denominator and shared budget; keep all failures and report
staged termination honestly. No real-data or SOTA claim, core/defaults unchanged.

### Scalar rescue with image-only sampling (2026-10-30)

[Results](research/edge_scalar_rescue_results.md), frozen `0b0fa0bb`: unchanged
rescue algorithm, imports switched to image-only sampler/joint fit. All 63
baseline records reproduce exactly, all poses unchanged. Perturbed joint recovery
**32/36→32/36**, truth starts **10/18→16/18**, six gains/zero regressions. Six
negative-bias truth starts recover, including the two earlier regressions. Eight
triggers/adoptions; two positive-seed corrections improve but still fail. Six
sphere failures remain, including two >.05 m near-threshold failures. Nine planes
stay rejected. 8–215 optimizer calls, median bias error .002071→.001974 m.
Staged termination is not new joint stationarity; no production/SOTA claim.
Next prioritize a frozen transfer comparison on existing spent real recordings,
fixed sampler/rescue versus comparable bilinear metric baseline, full drives/seeds,
truth-start drift and rejection/runtime reporting. Real depth lacks synthetic bias
truth. Do not tune the six synthetic failures or spend fresh holdouts first.

### Real-data metric transfer fails (2026-10-30)

[Results](research/real_metric_transfer_results.md), frozen `1ceb2d52`: six spent
KITTI drives × positive/negative/reference starts × bilinear/edge/edge+rescue,
54 fits, three archived restored-SGBM frames per drive. Fixed 4096 file-order
points/frame, 102 input files hash-verified. Perturbed accepted recovery is
**0/12 for all three**. Reference starts **1/6, 3/6, 3/6**; perturbed translation
median worsens .309→.507 m with edge sampling. Rescue triggers/adopts 17/18 but
poses are unchanged, so no recovery gain. All nested edge baselines reproduce;
81–192 / 57–151 / 90–170 calls. Synthetic gains do not transfer; reject adoption.
Historical MI 6/12 is not a matched baseline because objective and point sampling
differ. Next freeze reference-vs-fit cost/support/signed-residual diagnostics and
a matched MI control on the same selected inputs before choosing another optimizer
or frontend. Do not tune synthetic thresholds, infer bias accuracy from unlabeled
real depth, or spend fresh holdouts. Core/defaults unchanged; no SOTA claim.

### Matched-input MI and real cost diagnosis (2026-10-30)

[Results](research/real_metric_diagnosis_results.md), frozen `4a1cf7b8`: 18 MI fits
on the same 4096 file-order points/frame as metric transfer, plus four fixed-state
cost/support probes of all 54 archived metric fits. MI recovery **1/12**, reference
controls **5/6**, all converge. Historical full-cloud MI 6/12 is not this selected
input's baseline: sampling cannot be assumed harmless. At equal fitted bias,
inaccurate fits still beat reference cost on common support in **17/18 bilinear,
13/18 edge, 12/18 rescue** cases. Reference zero-bias support only 1414–1737/12288,
median >1 m residual fraction ~22%. Recomputed costs match archive. Do not infer
one exclusive cause or global minima; no adoption/SOTA claim.
Next freeze full-cloud matched MI/metric control on the same spent recordings,
including both seeds and reference starts, to resolve the input-population
confound before changing the objective or consuming fresh data. No optimizer/
threshold tuning; preserve low-cost inaccurate cases as evidence.

### Full-cloud control rejects the metric path (2026-10-30)

[Results](research/full_cloud_transfer_results.md), frozen `07398b48`: 54 full-cloud
fits, same six drives, three frames and 18 starts; 107739–124044 points/frame.
MI historical result records reproduce exactly **12/12**, perturbed recovery
**6/12**; bilinear and edge+rescue remain **0/12**. Reference controls MI **5/6**,
bilinear **1/6**, edge+rescue **0/6**. Thus thinning harmed MI but does not explain
metric failure. Sixteen scalar adoptions leave poses unchanged. Common-support,
equal-bias wrong-pose cost preference remains 17/18 bilinear and 15/18 rescue.
All budgets/hashes/counts verified; no defaults/adoption/SOTA claim.
Close sampler/scalar tuning. Next independently specify a stereo uncertainty
control (range-dependent depth error from disparity noise) on known geometry,
fixed noise/occlusion conditions and unchanged regression controls, before any
real-data retuning. Keep full-cloud MI as baseline; do not fit noise thresholds
to these spent real cases or consume fresh holdouts as development data.

### Stereo uncertainty normalization rejected (2026-10-30)

[Results](research/stereo_uncertainty_results.md), frozen `2eb1336d`: 342 paired
synthetic fits, bilinear metric baseline versus .1/max(.1,.5*z_measured²/120)
residual scaling before existing saturation/loss. Clean + .5 px disparity noise
(two seeds), all three optical biases, layouts/populations/starts; nine clean
plane controls per method. Clean fixed outputs exactly reproduce 63/63.
Clean perturbed joint **12/36→1/36**, reference **6/18→0/18**; 18 regressions/one
gain. Noisy perturbed recovery **0/72 for both**, reference **0/36→8/36** only.
Noisy perturbed candidate convergence just 2/72. Both reject all nine planes;
8–800 actual calls. Reject candidate, do not tune floor/sigma or adopt.
Next distinct hypothesis: independently specify direct disparity-domain error
rather than measured-depth variance weighting, freeze the same regression controls
before evaluation. No claim of success yet; MI real baseline/core unchanged.

### Direct disparity preserves clean controls, still fails noisy recovery (2026-10-30)

[Results](research/direct_disparity_results.md), frozen `c577773e`: 171 candidate
fits against archived fixed-metric baseline, exact paired depth hashes. Bilinear
sample disparity, residual .2*(d_observed-120/(z+bias)), same robust loss/search.
Clean joint recovery stays **12/36 perturbed, 6/18 reference**, no regressions.
Noisy reference **0/36→24/36**, but noisy perturbed **0/72→0/72**, all candidate
noisy perturbed fits fail convergence/support. Nine planes rejected, 8–800 calls.
No adoption or real transfer. Next freeze reference-versus-failed-pose objective
landscape diagnostics with controlled bias on these same noise realizations before
another residual variant or optimizer change. Do not relax caps/gates or claim
reference-start gains as perturbed recovery. Core/MI defaults unchanged.

### Disparity pose-path diagnosis (2026-10-30)

[Results](research/disparity_landscape_results.md), frozen `21203727`: all 171
archived fits, two fixed-bias eleven-point reference→fit paths, 3762 diagnostic
passes. Depth hashes and fitted endpoint costs reproduce. No new fitting/verdicts.
Noisy perturbed cases at fitted bias: returned fit beats reference in 72/72,
but an interior sample beats BOTH endpoints in **69/72** (34/36 +35/36). Same
counts on common support. At injected bias reference beats fit in **66/72**;
ranking depends strongly on nuisance bias. No global-minimum claim.
Next freeze bounded derivative-free coordinate search of the same disparity
objective, original starts and shared 800-call cap versus archived TRF. No truth
paths as initialization/gates, no sampler/loss changes. Full clean/noisy/bias/
occlusion/reference/plane controls required. No adoption or SOTA claim.

### Derivative-free coordinate search rejected (2026-10-30)

[Results](research/disparity_pattern_results.md), frozen `3c966459`, formatting-only
rerun `da10b9bb` (initial plan/partial output/log retained). All 171 final cases,
paired depth hashes verified, same objective and 800-call limit. Clean perturbed
joint **12/36→0/36**, noisy **0/72→0/72**. Reference controls clean **6/18→16/18**,
noisy **24/36→32/36**; those are not perturbed recoveries. Nine planes rejected,
85–800 calls, monotonic accepted cost. Reject replacement; no defaults changed.
Close residual/weight/search micro-variants. Next review independent geometric
camera/LiDAR correspondences and reusable external adapters for a genuinely
distinct matched comparison; freeze assumptions/evaluation before new fits.
Retain full-cloud MI baseline 6/12 and all negative evidence. No SOTA claim.

### Independent geometry pilot prepared (2026-10-30)

[Preparation](research/stereo_registration_preparation.md), freeze `92a59724`:
review confirms prior SuperGlue/I2PNet correspondence failures; do not rerun their
threshold variants. Native trimmed mutual point-to-point and Chen-Medioni ICP
exist; Open3D absent. Prepared 18 stereo/LiDAR frame pairs, original restored SGBM,
pose-free backprojection and independent .2 m voxel selection, finite <=80 m.
Source 25536–42927 points, target 38351–71126, all NPZ hashes/counts verified.
Manifest/schema committed; arrays under /workspace/work/stereo-registration-20261030.
No fits or new accuracy claim. Next implement/freeze 108-trial runner per
stereo_registration_protocol.md (18 pairs ×3 starts ×2 native ICP variants),
full failure accounting and all-three-frame drive success. Historical joint MI6/12
is not a matched single-frame comparator. No reference-dependent preparation.

### Native stereo geometry registration fails pilot (2026-10-30)

[Results](research/stereo_registration_results.md), freeze `bd9709ee`: 108 fits
(18 frame pairs ×3 starts ×2 native ICP variants), full input/artifact hashes checked.
Perturbed frame recovery point **1/36**, plane **0/36**; all-three-frame drive/start
recovery **0/12 both**. Reference controls **6/18, 2/18 frames**, **1/6, 0/6 drives**.
Converged perturbed **8/36,9/36**, others iteration cap; geometric-only 5/36 each
is not recovered. Reject adoption; no post-result cap or voxel/gate tuning.
Full detailed gzip results under /workspace/work/stereo-registration-pilot-20261030,
summary/schema/hashes committed. Historical joint MI6/12 is contextual only.
Next quantify reference-pose stereo/LiDAR overlap, nearest-target distances and
normal agreement on all18 frames before another registration variant; offline
truth diagnostics only, no truth-filtered fitting. No defaults or SOTA claim.

### Stereo geometry overlap and normal diagnosis (2026-10-30)

[Results](research/stereo_geometry_diagnostic_results.md), frozen `c7ddb8c6`, all18
prepared pairs, reference-pose diagnostics only. Camera-FOV source coverage within
1 m **80.97–93.89%** (median86.15%), but within .2 m median46.97%. Whole-source
within1 m median17.82% reflects partial FOV overlap. Normal pairs (both PCA-valid,
FOV and <=1 m) agree within15 deg only **28.90–63.64%**, median49.22%; target
normal validity median99.31% is not surface accuracy. Explicit counts/reverse
quantiles retained. No new fits/masks or accuracy changes; input hashes verified.
Next specify measurement-only correspondence/surface reliability on independent
controlled geometry (noise/occlusion), verify accuracy and spatial/directional
support before frozen real comparison. Do not tune truth-based masks on these
18 spent frames, equate eigengap with correctness or merely raise ICP iteration caps.

### Measurement-only surface gate fails precision (2026-10-30)

[Results](research/surface_reliability_results.md), frozen `54c0a3e0`: 27 controlled
frame conditions, three analytic layouts/origins, clean +.5px seeds730/731.
5x5 affine disparity fit, RMS<=1px and depth sigma<=.2m; no truth enters selection.
Selected clean joint XYZ/normal precision100%, but noisy median **13.31/13.05%**
(all-valid4.80/4.88%), range7.92–29.33%, failing80% screen in **18/18 noisy**.
Retention5.17–28.58%, occupied4–13/16 cells; all noisy normal spectra pass ratio
screen despite inaccurate normals. Reject gate; no real transfer/ICP rerun.
Next derive/test angular uncertainty from disparity-slope covariance and verify
coverage against exact normals before any new filtering; keep support failures
explicit. Do not substitute planarity/depth uncertainty for normal accuracy or
tune the18 spent real frames. Core/defaults and MI baseline unchanged.

### Angular confidence gate retains no points (2026-10-30)

[Results](research/normal_uncertainty_results.md), freeze `ce9d5e9e`: coefficient
covariance from .5px Gaussian noise on5x5 affine disparity, conservative95% normal
ball→cone; same27 conditions/depth hashes. Added cone<=15deg retains **0/27**
conditions' points, so all noisy support screens fail; empty precision is null.
Prior gate median cone90deg, coverage~99% is uninformative. All-valid coverage
~91–94% reveals curved/boundary model error. Exact-plane20000-sample sanity check
coverage99.465%, median bound35.68deg versus actual9.19deg: bound is loose, so do
not claim intrinsic absence of accurate normals. No ICP/real transfer or defaults.
Next test tighter angular uncertainty on exact planes with coverage AND support,
then separately test model bias on curved/boundary patches. Do not tune thresholds
on spent real frames or treat noise covariance as coverage of model error.

### Tangent normal confidence fails coverage (2026-10-30)

[Results](research/normal_tangent_results.md), freeze `f349514f`, import-only rerun
`febdaa2b` (initial outputs/plan retained; final cases identical).27 exact-plane
scenarios ×20000 draws: **9/27 fail94% coverage**, all frontal slope cases;
minimum78.47%, median98.005%. Old ball min97.90%. Thus first-order normalized
normal covariance is not calibrated across this noise/geometry range.
27 surface controls: clean selections empty; noisy nonempty only near_foreground
frame2 (10/2 points), all support screens fail. No real filter/ICP or defaults.
Close5x5 confidence shortcuts. Next direction: coherent surface estimation with
larger support and explicit variance-versus-curvature/boundary bias control;
freeze independent coverage criteria before fitting or real-data selection.
Do not tune windows/thresholds on18 spent real frames. No SOTA claim.

### Larger coherent surfaces improve but fail all-frame gate (2026-10-30)

[Results](research/coherent_surface_results.md), freeze `f3672551`:108 conditions,
5/11/21 and smallest-passing adaptive support, same27 measurement hashes. Plane
smooths center depth as well as normal; not comparable to prior raw-XYZ precision.
21px noisy median joint precision **80.94/81.69%**, retention **30.65/30.81%**,
but only **5/9 frames per seed** pass gates; adaptive **3/9,4/9**, median79%.
Clean21px median74.65% (2/9 pass): curvature/boundary model bias remains. Angular
coverage63–87% under nominal95% plane bound. No real transfer/ICP or defaults.
Next measurement-model validation: independent spatial patch prediction or
curvature-aware fitting, exact-plane and curved/boundary coverage/support controls.
Do not choose a window using truth or tune RMS/scale on18 spent real frames.

### Disjoint surface prediction passes controlled screen (2026-10-30)

[Results](research/surface_prediction_results.md), freeze `d72a83a1`:54 paired rows,
same27 measurements.21px checkerboard221-fit/220-holdout predictive chi-square
with shared coefficient covariance,99% gate; no truth selection. Exact-plane
10000-draw acceptance98.88%. Noisy joint precision medians **99.92/99.87%**,
retention **23.44/23.86%**, all18/18 noisy and9/9 clean pass quality/support screen.
Archived baseline27/27 exact. Final full-patch estimate reuses validation pixels;
not independent confidence. Some angular coverage<95%, no guarantee or ICP claim.
Next parameterize camera/stereo geometry with exact analytic parity first: current
fx240/fB120 constants must not be applied verbatim to real KITTI. Freeze real
intrinsics/stereo/noise/resampling assumptions before18-frame quality transfer;
no tuning to reference normals, then matched ICP only if justified. No defaults/SOTA.

### Camera-parametric predictive surfaces transfer partly (2026-10-30)

[Results](research/real_surface_prediction_results.md), freeze `22453494`: explicit
camera intrinsics/fB/noise, bitwise parity of all arrays on27 analytic conditions.
36 paired real quality rows on18 frames: baseline **13/18**, predictive **15/18**
pass; median normal agreement **86.12→88.16%**, retention **26.81→24.77%**.
Failed candidate frames0005/38,0005/114,0009/111; no exclusions or relaxed gate.
No ICP rerun per all-frame requirement. Restored-depth fB/z is surrogate disparity,
.5px independence is uncalibrated on real data; no calibration/SOTA claim.
Next fixed spatial distance/normal residual maps for all18, emphasizing3 failures,
compare measurement support/boundaries; diagnose before changing model and never
turn reference-derived maps into fit masks. Matched registration remains pending.

### Spatial maps show interior surface disagreement (2026-10-30)

[Results/maps](research/surface_error_maps_results.md), freeze `9a8327fd`: all18
numeric/PNG maps, unchanged predictive surface masks. Aggregate counts/fractions
reproduce prior report; all hashes and16-cell totals verified. **18/18** interior
normal failure rates exceed boundary-band rates; median **17.83% vs7.27%**.
Distance failure interior higher14/18, median13.69% vs7.72%. Failed0005/38,
0005/114,0009/111 interior normal failures68.8%,42.5%,35.5%. No semantic/dynamic
causal claim. Prior prose distance68.62% belongs to frame114 (corrected), machine
outputs unchanged. Numeric NPZ local, all18 PNGs committed. No fits or mask edits.
Next review prior stereo-consistency trials before choosing an independent
cross-view/temporal check for coherently wrong geometry. No repeat matcher/gate
tuning or truth-derived masks on18 spent frames; freeze distinct comparison first.

### Reverse consistency still leaves three failed surfaces (2026-10-30)

[Results](research/surface_cycle_results.md), freeze `d6bcae4e`: conditional1px
cycle test after predictive surfaces, all18 frames.18 raw restored SGBM and12
historical filtered arrays reproduce exactly. No plane refits or optimizers.
Three failures remain agreement **45.71%,75.50%,78.83%** (<80%), retention93.78%,
95.83%,93.69% of prior selected. Agreement improves13/18 but median88.16→87.95%;
no remedy/adoption. Rejected/missing group noisier, yet accepted surfaces unreliable.
Next audit independent camera-motion availability and temporal overlap before any
temporal consistency design. Pin motion/time/frame/scale assumptions, no reference
camera/LiDAR extrinsic in measurement selection. If unavailable/insufficient overlap,
report it. No reverse-threshold retuning, ICP or fresh-data/SOTA claim.

### Temporal readiness blocked on sparse input (2026-10-30)

[Results](research/temporal_surface_audit_results.md), freeze `be471af7`:12 image-only
pair audits, intervals3.728–11.590s. Only **1/12** meets30-cheirality/6cells screen
(0011/116→174);0002 many essential inliers but zero cheirality, not proof of no overlap.
OXTS available, but local IMU→camera chain uses target LiDAR-camera extrinsic;
monocular scale unknown. No independent metric camera motion in declared input set.
±1neighbor local inventory0/36 complete stereo pairs. No data downloaded or new fits.
Next prerequisite: freeze bounded ±1 stereo acquisition/preparation on SAME six
spent drives (archive/member/timestamp/byte pins) before fetching, then short-gap
support and metric-motion validation. StereoVO shares depth errors, not independent
truth. No target-extrinsic-derived selection, arbitrary scale or fresh holdouts.

### Short-baseline neighbor acquisition and support (2026-10-30)

[Results](research/temporal_neighbors_results.md): scope `550055c0`, exact member
ranges `ce811402`, audit `11f93d00`, histogram reconciliation `79ce9e80`.
72 images/36 stereo neighbors, only same six spent drives,56.85MiB range requests;
archive/CRC/SHA/byte caps verified. First audit aborted on raw-vs-equalized pixels;
initial plan/log retained, all18 centers match after existing equalization.
Short-gap .1031–.1038s support **28/36** (old long-gap1/12, not paired accuracy).
0005/0009/0013/0015 each6/6,0011 4/6,0002 0/6. All failures retained. No metric
motion, temporal masks, ICP or calibration claim. Right images acquired but unused.
Next freeze metric stereoVO with independent controlled-motion/scale validation,
no target extrinsic, all36 pairs/failures and explicit shared-stereo error coupling.
No fresh recordings/holdouts, default changes or SOTA claim.

### Camera-only metric motion feasibility (2026-10-30)

[Results](research/stereo_metric_motion_results.md), freeze `271b9e75`:
18 generated correspondence controls pass (<.2deg/.05m), maxima .04315deg/.005561m.
All36 real pairs evaluated;54 unique depths,18 central arrays exact archive parity,
all hashes verified. Metric camera-only bidirectional PnP support/cycle **34/36**;
0013/106→107 and107→108 fail50% inlier ratios, not cycle. Cycle median .03353deg/
.01547m. No target extrinsic/OXTS/LiDAR used; shared stereo error means no independent
real accuracy claim. Previous eight essential failures remain included.
Next freeze controlled-motion/visibility temporal surface consistency, then real
estimated-motion-only diagnosis. Keep failed motion pairs and missing/out-of-view/
occluded support explicit; reference/LiDAR score only after selection. No temporal
filter/ICP/extrinsic calibration/default/SOTA claim yet.

### Temporal surface verification rejected (2026-10-30)

[Results](research/temporal_surface_check_results.md), freeze `53c39a61`:18 exact
render projection controls pass, min98.34% visible consistency. All18 real centers,
36 estimated-motion checks; both-neighbor1px disparity agreement. Quality/support
**15/18→13/18**, retention median24.77→19.43%. Original failures still agreement
**45.24%,76.02%,76.98%**.0013/107 empty from unsupported motions;0015/74 distance
failure21.79%. All categories retained; no truth in temporal classification.
Close self-consistency mask sequence, no ICP/tolerance retuning. Next reference-
assisted error attribution of estimated motion versus available OXTS/time/camera-
conversion proxy, evaluation-only and uncertainty explicit. Target-extrinsic chain
must never feed selection or be called independent metric truth. No defaults/SOTA.

### Motion proxy and offline replacement leave main failures (2026-10-30)

[Results](research/camera_motion_proxy_results.md): motion freeze `7fb15ad9`,
reference-assisted counterfactual freeze `5bed804c`.36 OXTS camera-time/chain proxy
comparisons: median estimate disagreement **.03515deg/.02631m**, max.10140deg/
.13856m. Common±10ms proxy change max.02690deg/.003180m; same-index max.14515deg/
.08386m, omit lever translation max.04472m. Sensitivity is not uncertainty bound.
18 counterfactual centers preserve masks/gates/unsupported flags; proxy motion uses
target calibration EVALUATION ONLY. Main failures agreement **45.20/76.21/76.54%**
versus45.24/76.02/76.98 estimated motion. Overall13/18→14/18 diagnostic only, original
measurement-only verdict unchanged. No independent motion truth or adoption claim.
Next underlying stereo matching ambiguity/surface reconstruction attribution, review
prior frontend evidence first; no reference-selected masks, SGBM retuning or another
self-consistency gate. LiDAR normal/calibration/timing alternatives remain explicit.

### Stereo appearance does not certify surface correctness (2026-10-30)

[Results](research/stereo_ambiguity_results.md), freeze `1812d061`:18 fixed raw
stereo appearance probes,121636 sampled/120010 common-valid, offsets±1/2/4/8,
zero-mean5x5 SAD, no matcher/mask/depth changes.18 raw depths exact archive parity.
Wrong-normal ambiguity median44.56% vs32.48%; but wrong-group median base cost
LOWER in11/18, including all3 persistent failures. Association is not calibrated
confidence; strata and exclusions explicit. No new selector or recovery claim.
Close stereo-surface confidence variants. Next review a materially distinct direct
RGB–LiDAR provider beyond failed SuperGlue/I2PNet: architecture/checkpoint/license,
training overlap and CPU/GPU readiness, then ONE frozen feasible spent-data pilot.
No fresh holdouts, SGBM/threshold retuning or runtime-as-accuracy/SOTA claim. Preserve
full-cloud MI6/12 and all negative evidence; report concrete prerequisites if blocked.

### Direct-provider official-source readiness review (2026-10-30)

[Review/pilot prerequisites](research/direct_provider_readiness.md),
[machine evidence](assets/direct_provider_review_20261030/review.json): official
CoFiI2P `ed90edf5`, CorrI2P `4f4a7550`, DeepI2P `cd21389d` pinned/hashed and read,
not executed. Conditional priority CoFiI2P correspondence mechanism; all current
unmodified paths blocked by CPU-only runtime/unchecked weights/dependencies.
CoFi/Corr no root license found in pinned trees (terms unresolved, not legal verdict),
Deep MIT but CUDAExtension. KITTI train0–8/val9–10 code does not establish exact
checkpoint/raw-frame isolation. No weight digests or inference/accuracy claim.
CoFi loader prealigns using dataset calibration; MUST bypass with explicit
initializer-only input boundary and tested transform/crop/normal lineage. Its test
model path differs from train/val, so do not claim the entire model requires labels.
Concrete conditional54-invocation spent-data pilot is specified; prerequisites
GPU or validated port, pinned appropriate weight/terms, no-reference adapter,
training provenance. No fresh data, package install or code copied into core.
Guessed2D3D-MATR URL unresolved and retained, not an official candidate claim.

### CoFiI2P initializer-only geometry boundary completed (2026-10-30)

[Contract and limitations](research/cofii2p_input_boundary.md), implementation freeze
`333e368b`, [synthetic verification](assets/cofii2p_input_boundary_20261030/verification.json):
four hash-pinned inputs only (image/cloud/camera/explicit initializer); no reference
or dataset loader. Nonidentity points/normals, IDs, correction composition, odd-size
crop/pixel mapping, BGR values, reference-file mutation and input rejection covered
by four unit tests. Three-point synthetic CLI smoke passes eight planned checks.
No real frame, model inference, accuracy or checkpoint-parity claim. Original point
rows survive; no sampling/pyramid yet. Actual rounded resize/pixel-center affine
differs from official nominal0.5 K scaling; BGR/feature stride also need explicit
checkpoint compatibility review. Caller initializer/normal provenance remains an
upstream obligation, not established by reference-file mutation alone.
Next resolve preprocessing parity and model bridge (sampling/lineage, neighborhoods,
fine-feature pixel coordinates), plus GPU/validated port, suitable pinned weights/
terms and training overlap. Then freeze one54-call spent-data pilot; do not resume
stereo-mask variants or claim SOTA from geometry tests. Core/defaults unchanged.

### CoFiI2P model-shaped CPU geometry and patch audit (2026-10-30)

[Contract/source findings](research/cofii2p_model_bridge.md), freeze `b6d0628e`,
[synthetic result](assets/cofii2p_model_bridge_20261030/result.json): independent
voxel mean+CSR source lineage, count selection, five replacement-sampled levels,
128 KNN and explicit fine-pixel decoder. Default20480/10240/5120/2560/1280 smoke
on4096 synthetic points repeated exactly; all39 selected exhaustive KNN-distance
checks pass. Unique voxels per level4096/3775/2581/1504/811; keep duplicates and
centroid contributor sets. Related tests21 pass; no model call or real frame.
Official eval adds patch row to u/column to v despite (h,w) flatten; decoder requires
row_major or official_eval_legacy explicitly.12/16 positions differ, NOT measured
model error. Freeze convention before any pilot, never select by reference errors.
Open3D voxel ordering/KNN ties/random streams and truth-prealigned voxel grid NOT
numerically reproduced; this is NOT a validated CPU inference port. Local BGR,
crop and pixel-center mapping still lack checkpoint parity. Next prioritize suitable
checkpoint provenance/terms, upstream image preprocessing, GPU/validated inference
runtime and training overlap; then one preregistered54-call spent-data pilot.
Do not continue geometry polishing as a substitute for model evidence. Core/defaults
and historical MI6/12 plus all negative evidence remain unchanged.

### CoFiI2P inference deferred on concrete runtime/checkpoint blockers (2026-10-30)

[Readiness/handoff](research/cofii2p_runtime_readiness.md),
[machine evidence](assets/cofii2p_runtime_review_20261030/review.json): runtime
torch2.4.0+cpu/Python3.12, CUDA unavailable, Open3D absent. Official OneDrive/Baidu
HEADs both proxy CONNECT403 under enforced policy; GitHub releases page no releases.
No checkpoint bytes/digest, external execution, new real data or accuracy result.
DeepI2P KITTI RGB conversion and CoFi nuScenes PIL pipeline support RGB ancestry,
not exact distributed KITTI crop/color/K provenance. Do not call local BGR parity.
Issue26 collaborator acknowledges historical train-only translation suppression and
plans retraining; IMPORTANT pinned ed90edf5 line225 is COMMENTED OUT, so current
source DOES train with translations. Distributed checkpoint training revision/date
still unknown. Issue25 collaborator reports no validation/test distinction; this
does NOT prove overlap with our spent raw frames. Issue7 about3GB inference is
author report, not measured bound. Issue20 coarse-only report does not prove patch
axis issue caused it or permit reference-selected variants. Require GPU workspace,
suitable official checkpoint SHA/terms/training recipe and prepared-image provenance,
then bounded no-reference model smoke and frozen54-call spent-data pilot. User was
asked for available GPU/checkpoint location; do not invent access, bypass proxy,
retrain blindly or keep polishing CPU geometry instead of obtaining model evidence.

### Bounded alternative-provider follow-up closed (2026-10-30)

[Review](research/direct_provider_followup.md),
[machine evidence](assets/direct_provider_followup_20261030/review.json): official
NCLR40de8eea, FreeReg d13a45f5, FFreeReg branch0b397094 pinned,18 source hashes.
NCLR has calibration/PnP components inside representation learning, but README
weights are todo; no ready pretrained unknown-extrinsic provider established.
Root Apache2.0, sparse-convolution dependencies. FreeReg has CPU fallbacks in some
components (do NOT blanket claim CUDA-only), but multiple unpinned large weights/
dependencies and no feasible complete current runtime verified. FFreeReg removes
virtual-camera/diffusion requirements but still DepthPro+YOHO/FCGF weights, explicit
CUDA demo and MinkowskiEngine. Estimated K, scale resampling and point lineage
need frozen policy before calibration use; do NOT relabel a stereo substitution
as Free-FreeReg reproduction. FreeReg/FFreeReg no tracked root license; dependency
terms separate. About42MB source clones include bundled demos, none evaluated.
Zero model calls/weight download/calibration fits; core/defaults unchanged.
Close speculative provider search and adapter expansion. Next useful execution
requires accessible GPU and suitable official checkpoint with SHA/terms/training
and image provenance, as in CoFi handoff. No accuracy/SOTA advance claimed.

### User has NO GPU; CoFiI2P CPU graph now exercised (2026-10-30)

CPU-only is an explicit user constraint, not a pending GPU request. Supersedes the
previous GPU-first next actions. [Protocol/results](research/cofii2p_cpu_probe.md),
freeze `3b15f88a`, seed20261030/two threads: actual external CoFiI2P test-mode model
with random weights and None label arguments completes both2048 and20480 points.
Forward4.13/10.36s, peakRSS735.4/1901.6MiB, init5.40/5.53s. All8 outputs finite/CPU,
repeat tensors exact; same seeded model-state hash across point counts. Random
predicted pairs10/4 are NOT correspondence accuracy, no PnP fit/real frame/checkpoint.
Only two external copied files patched:4 CUDA transfers to input device, unused
Open3D/plot imports removed, native NumPy kernel generation with NPY cache instead
of Open3D PLY, bounded predicted-pair loop. Existing SciPy geometry/order/tie/grid
differences remain; not verified official CPU/GPU/Open3D numerical parity. Source
copy and outputs workspace-only, no external code copied to core; einops0.8.1 tiny
dependency installed, no GPU/Open3D/TorchVision package install. Core/defaults unchanged.
Next accessible official KITTI weight SHA/terms/training+image recipe, then bounded
CPU checkpoint smoke and frozen spent-data pilot. GPU is NOT required for this
exercised graph. Do not re-request GPU or repeat closed MI search/stereo variants.

### CPU checkpoint loader and raw pair lineage exercised (2026-10-30)

[Contract/results/template](research/cofii2p_checkpoint_cpu.md), freeze `5b90e2e6`:
request pins checkpoint/bridge/prepared chain, declares image/layout/source/training/
terms, rejects reference fields. `torch.load(weights_only=True,map_location=cpu)`
plus exact tensor keys/shapes/dtypes/finiteness and strict load; no prefix/wrapper/
dtype guessing, unsafe fallback or random fallback. Native fine_process indices
observed unchanged; exports original-image pixels, LiDAR voxel centroids and CSR
of ALL original contributor IDs. No covariance/confidence/quality/pose invented.
Official checkpoint still absent and official hosts still policy-excluded; no
repeat denial probes or mirror search. Full synthetic raw state196.97MiB saved,
then loaded in separate process at20480 points: init+load5.82s/forward10.21s,
peakRSS1.92GiB, allCPU/finite/exact repeat.4 pairs but only3 distinct voxels/pixels,
all retained; no pose fit/accuracy. Fixture seed matches target initialization;
different-state load mutation verified separately by small-model tests. Synthetic
kind requires explicit --allow-fixture and remains labeled in result.32 related
tests, schemas/digests/original centroid+ID/pixel checks pass. Source/weights outside
core; no learned model or fresh dataset acquired. Next fill learned-request template
with an accessible official KITTI artifact and SHA/source/terms/training+image recipe,
then bounded CPU smoke and preregistered spent-data pilot. CPU-only user constraint
persists; stop adding substitute random-weight experiments as accuracy evidence.

### v3 capture complete on all 55 spent drives (2026-10-31)

[Results](benchmarks/kitti_camera_lidar_v3_development.md#all-55-spent-drives-captured-and-replayed-2026-10-31):
`tools/kitti_camera_lidar_v3_batch.py` fetched, converted, captured (seven real
checks) and replayed all 55 spent KITTI raw drives in one cloud session, with
archive/calibration pins committed first (`4d82d125`). Summary `complete`; the
current-rule replay equals the committed v2 aggregate exactly (52 scored,
125/156 judged axes, 3 deg 122/125, 1 deg 125/125, maxima 0.450/1.503 deg).
The proposed 3.5 margin keeps coverage but flags only **114/125** 3 deg turns and
adds a 1.350 deg yaw error on 0036 while removing the 0002/0057 errors
(turned-run judged axes >0.5 deg: 10 → 5). Against v2's gate values it would
pass on these spent drives (8.8 % 3 deg misses, 1.350 deg), but only post hoc:
3.5 was chosen after reading v2 evaluation drive 0002, and the margin moves a
large error to 0036. Do not adopt it as a default; no threshold change. Four zip64 drives first failed on cut range transfers;
all eight failed attempts are kept. The fetch fix `a7a83db0` retries each range
as a fresh, length/ETag/CRC-checked request. (The interim `--retry-all-errors`
appended partial bodies and was caught by the length check.) NPZ inputs remain
only on the batch machine for 23 drives; the other 32 need re-capture before
replaying a new rule. Next: a different start-selection/axis-evidence rule
developed on cached inputs, then replayed on all 55 drives with coverage, missed
turns and judged-axis errors reported together. KITTI-360 pre-registration stays
pending; no fresh data was read and no SOTA claim is made.
