# imu-lidar check on Snail-Radar (Hesai XT32, two IMUs)

This page reports a pre-registered audit of `calibrex check --pairs imu-lidar` on a
spinning LiDAR, on a hand-held rig, an e-bike and an SUV. The audit asks three things:

- whether check fails the dataset's own calibration;
- whether it judges a rotation axis at all;
- whether it catches the LiDAR turned 3 deg.

Until now imu-lidar was audited only on Livox MID360 recordings
([MID360](mid360_imu_lidar.md)).

The plan is [`snail_lidar_imu_preregistration.yaml`](snail_lidar_imu_preregistration.yaml):

- frozen code `65e107b1`;
- harness `tools/snail_lidar_imu_audit.py`;
- records: [`evaluation/summary.json`](../assets/snail_lidar_imu/evaluation/summary.json),
  with per-sequence `*_score.json` beside it and the development scores in
  `docs/assets/snail_lidar_imu/development/`.

## Data and method

[Snail-Radar](https://snail-radar.github.io/) (Huai et al., IJRR; Open Database License)
records a Hesai Pandar XT32 with a Bynav X36D IMU. From 2023-11-05 on it also records an
Xsens MTi-3.

Each sequence's folder zip is read over HTTP range requests, and the harness takes out only:

- the XT32 scans, thinned to every 4th firing;
- both IMUs;
- `tf_static.launch`.

check runs on the first 300 s with four candidates:

- `tf_static.launch` itself (the reference);
- three copies with the LiDAR turned 3 deg about its own x, y or z axis, which turns it
  against both IMUs.

A case is one (sequence, IMU) pair.

The reference is the dataset's own calibration, and it is not ground truth:

- the X36D-XT32 rotation was fitted by the authors with KISS-ICP on 20230921/data5, so
  that sequence is excluded;
- the MTi-3 transform comes from LI-Init;
- the lever arms are hand-measured, so translation is not scored.

### Development (spent)

Three sequences were used for development: 20230921/data4 (hand-held), 20231019/data2
(e-bike) and 20240113/data5 (SUV, both IMUs), giving four cases.

| Measure | Result |
| --- | --- |
| Reference rotation fails | 0 |
| Reference passes every judged rotation axis | 3 of 4 (data4's yaw warned at 0.81 deg) |
| Reference judges a rotation axis | 4 of 4 |
| Judged turned axes failed | 9 of 9 |
| Cases that detect a turn | 4 of 4 |

The gates are the development rates divided by 1.5, rounded down to 0.25:

- no reference rotation fail;
- a pass fraction of at least 0.5;
- a judged fraction of at least 0.5;
- no missed judged turn;
- a detecting-case fraction of at least 0.5.

A run without every one of the 60 expected cases refutes every claim.

## Result: all three claims refuted

32 unseen sequences were planned: the X36D on all 32 and the MTi-3 on 28, giving 60 cases.
56 cases were scored.

| Claim | Gate | Result | Verdict |
| --- | --- | --- | --- |
| honesty: reference rotation fails | 0 | **3** | refuted |
| honesty: reference passes every judged rotation axis | ≥ 0.5 | 37/60 = 0.62 | (gate met) |
| coverage: a rotation axis judged | ≥ 0.5 | 48/60 = 0.80 | refuted (incomplete) |
| detection: judged turned axes missed | 0 | 0 of 80 | refuted (incomplete) |
| detection: cases that detect a turn | ≥ 0.5 | 48/60 = 0.80 | refuted (incomplete) |

Two causes refute the claims.

**1. Three reference rotation fails, all on the MTi-3 on the morning of 2023-11-05.** On
20231105/data2, data3 and data5, check puts the MTi-3's roll 5.18, 5.09 and 4.86 deg from
`tf_static.launch`, with pitch warned at about -0.7 deg.

The same transform is used on every day. On later days it agrees on roll:

- within 0.63 deg on 2023-12-08 and later (7 passes, 5 warns of 0.50-0.63 deg);
- roll was not judged on 2023-11-05 afternoon or 2023-11-09.

The X36D on the same three sequences passes every judged axis.

The fails could be a different MTi-3 mount that morning, or a check error specific to that
day; this audit cannot tell which. Under the pre-registered rule they are honesty
failures.

**2. Two sequences were never read, so the run is incomplete.** For 20231109/data3 and
20240123/data3, the first OneDrive metadata request (the share's root item, before the
pinned file was looked up) returned an HTTP error. The harness records such a sequence
as missing, and four cases (one X36D, three MTi-3) are absent. By the plan, a missing case
refutes every claim. No data of these two sequences was read.

Coverage and detection met their numeric gates on the 56 cases that ran. They are refuted
only by the incomplete run, and this page does not count them as supported.

### Detail

| | X36D (30 cases) | MTi-3 (26 cases) |
| --- | --- | --- |
| reference pair status | pass 20, warn 7, fail 2, inconclusive 1 | pass 10, warn 6, fail 3, inconclusive 7 |
| reference passes every judged rotation axis | 26 | 11 |
| roll / pitch / yaw judged | 24 / 20 / 8 | 15 / 13 / 0 |

- **X36D pair fails:** both are on translation, with no rotation fail: 20231019/data1 and
  20231105/data4.
- **Turned candidates:** every judged turned axis failed: 39 roll, 33 pitch and 8 yaw.
- **Yaw:** judged only on the hand-held rig and the e-bike.
- **The SUV:** judges roll and/or pitch.
- **Inconclusive cases:** 8, all `no_judgeable_axes`. Seven are MTi-3 and one is the
  X36D on 20231201/data3.
- **Reference errors:** the X36D roll error is negative on every judged case (-0.15 to
  -0.57 deg, a warn only at -0.57), which suggests a small offset in the authors'
  rotation.

### Run log

The frozen harness started once at 10:30 UTC on 2026-11-12. The cloud container restarted
twice during the run, at about 12:36 and 20:44.

After each restart the remaining sequences were run by
[`resume_eval.py`](../assets/snail_lidar_imu/evaluation/resume_eval.py.txt) (kept verbatim as text). The script:

- imports the frozen harness and calls its `run_sequence` and `verdicts` unchanged;
- skips every sequence that already had a score file;
- re-runs from the start the three sequences each restart interrupted, reusing the
  estimator caches they had completed.

No completed sequence was run twice, and no result was looked at before the run finished.
The two missing sequences failed in the resumed run, not because of a restart.

## What follows

- **imu-lidar honesty on Snail-Radar is not supported.**
  - The X36D side passed every judged rotation axis except where the reference itself is
    in question.
  - The MTi-3's 2023-11-05 roll is the open question.
- **The claim row in the [handoff](../sota_handoff.md) stays on MID360.** This audit adds
  a refuted check audit.
- **A later audit could retry against long-outage failures.** Retrying such failures in
  the harness would be a code change, frozen before any new data.
- **The 30 sequences read here are spent for imu-lidar.**
  - 20231109/data3 and 20240123/data3 were not read, but they are named in a refuted plan.
  - A future plan that uses them must disclose that.
