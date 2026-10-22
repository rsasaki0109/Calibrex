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
completed 0005 with exact replay parity and no difference between rules. An
initial rounded-calibration guard failure is retained; the retry reconstructs
the converter's quaternion TF edges without relaxing its tolerance. **1/55 is
not a completed development comparison or evidence of improvement.** Capture
the remaining 54, retain failures, then confirm the selected rule through the
real driver. Preserve the local NPZ inputs and check free space before batches.

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
