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

- Capture and compare all 55 spent KITTI raw drives using the existing
  `tools/kitti_camera_lidar_v3_replay.py`, then confirm via the real driver.
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
