# radar-imu: the check audits on unseen radar-inertial recordings

`calibrex check --pairs radar-imu` judges the rotation of `T_radar_imu` from a 4D radar's
Doppler returns against the gyro and accelerometer (see the
[check tutorial](../tutorials/calibrex_check.md#radar-imu-a-4d-radars-doppler-against-the-imu)).
Two pre-registered audits ran on Doer & Trommer's recordings (TI IWR6843AOP radars, an
ADIS16448 IMU), with the authors' calibration as the candidate and with `T_radar_imu`
turned 3 deg about each radar axis.

| Audit | Data | Honesty | Coverage | Detection |
| --- | --- | --- | --- | --- |
| v2 | JGN 2022, 4 bags × 3 radars, 219–368 s | supported (no fail; pass 3/12) | supported (judged 7/12) | supported (7/12 detect; no turned case exits 0) |
| v1 | IROS 2021, 9 bags, 84–206 s | refuted (pass 1/9) | — | refuted (1/9 detect) |

The method is the same in both (as merged in #369); v2 tested v1's reading that its
recordings were too short.

## v2: longer multi-radar recordings, all three claims supported

- Plan: [`doer_radar_imu_v2_preregistration.yaml`](doer_radar_imu_v2_preregistration.yaml),
  frozen at `086bfd33`.
- Harness: `tools/doer_radar_imu_v2_audit.py`.
- Records: [`evaluation_v2/summary.json`](../assets/doer_radar_imu/evaluation_v2/summary.json)
  and one `*_score.json` per bag beside it.

| Claim | Gate | Result | Verdict |
| --- | --- | --- | --- |
| honesty | the reference never fails; passes on at least 25 % of the 12 cases | no fail; passes on 3 (25 %) | supported |
| coverage | an axis judged on at least 50 % of the cases | 7 of 12 (58 %) | supported |
| detection | no turned case exits 0 under `--fail-on partial`; every judged turned axis fails; at least 50 % of the cases fail a turned candidate | none exits 0; none missed; 7 of 12 (58 %) | supported |

What was judged, and how:
- **Only yaw was judged.** That is the rotation about each radar's z axis; roll and
  pitch stayed unchecked on every case.
- **The 3 deg yaw turn failed wherever yaw was judged.** That is 7 of 7 cases.
- **The reference warned on 4 of the 7 judged cases.** Its yaw was 0.52–1.23 deg off
  the estimate, against tolerances of 0.50–0.65 deg:
  - lab_floor right: 1.00 deg against a 0.54 deg tolerance, just short of a fail;
  - office_floor centre and left;
  - workshop right.

  A warn is a flag, not a fail. These warns pass the plan's honesty claim, but they
  say the yaw tolerance is tight for this rig.
- **Five cases judged nothing.** These are all three radars on basement, and workshop's
  centre and left. On each, every std was above 0.5 deg or a control was not detected.
- **A turn of the wrong axis once failed.** On lab_floor right, the pitch-turned
  candidate failed on yaw: 1.08 deg, where the reference was already at 1.00.

### Per case (the authors' calibration as the candidate)

The estimate minus the authors' rotation, about the radar's x, y and z axes, with its
reported std, in deg.

| Bag | Radar | Windows | roll | pitch | yaw | radar-imu | clock offset |
| --- | --- | ---: | --- | --- | --- | --- | ---: |
| basement | centre | 40 | -0.12 ± 0.71 | -0.93 ± 0.57 | +0.22 ± 0.31 | inconclusive | 10.4 ms |
| basement | left | 36 | +1.42 ± 0.59 | -0.82 ± 0.62 | -0.66 ± 0.30 | inconclusive | 14.8 ms |
| basement | right | 40 | +0.77 ± 0.76 | -0.43 ± 0.89 | +1.02 ± 0.31 | inconclusive | 13.5 ms |
| lab_floor | centre | 39 | +1.02 ± 0.99 | -1.13 ± 0.43 | +0.39 ± 0.20 | pass (yaw) | 9.4 ms |
| lab_floor | left | 39 | -0.66 ± 0.48 | -0.24 ± 0.54 | +0.08 ± 0.15 | pass (yaw) | 11.1 ms |
| lab_floor | right | 39 | -2.90 ± 0.72 | -2.27 ± 0.67 | +1.00 ± 0.18 | warn (yaw) | 12.9 ms |
| office_floor | centre | 40 | +2.26 ± 0.58 | -0.54 ± 1.14 | +0.90 ± 0.22 | warn (yaw) | 9.8 ms |
| office_floor | left | 40 | -0.55 ± 0.44 | -0.70 ± 0.50 | -0.52 ± 0.16 | warn (yaw) | 11.8 ms |
| office_floor | right | 40 | -2.81 ± 0.69 | -0.15 ± 0.57 | +0.25 ± 0.19 | pass (yaw) | 11.4 ms |
| workshop | centre | 34 | +2.11 ± 0.92 | -0.11 ± 0.82 | +0.12 ± 0.28 | inconclusive | 11.4 ms |
| workshop | left | 32 | -0.48 ± 0.57 | +2.15 ± 0.70 | +0.02 ± 0.28 | inconclusive | 13.0 ms |
| workshop | right | 34 | -0.51 ± 0.53 | -0.56 ± 0.85 | +1.23 ± 0.21 | warn (yaw) | 13.3 ms |

- **The clock.** The offset was 9–15 ms on every radar.
- **The lever arm.** It was recovered per radar. The left and right radars come out
  about 15 cm apart in the IMU's y axis.
- **The unjudged roll.** It is up to 2.9 deg (4 std) off the authors' rotation: lab_floor
  and office_floor right, office_floor and workshop centre. That is why roll is not judged
  (its control or its std kept it out), and why only yaw should be read from this pair
  until roll is understood. The [roll/pitch research note](../research/radar_roll_pitch.md) traces it to the
  radar's z velocity: about 2.5 times noisier than x and y, and barely excited.

## v1: shorter recordings, both claims refuted

- Plan: [`doer_radar_imu_preregistration.yaml`](doer_radar_imu_preregistration.yaml),
  frozen at `fb0dcc35`.
- Harness: `tools/doer_radar_imu_audit.py`.
- Records: [`evaluation/summary.json`](../assets/doer_radar_imu/evaluation/summary.json) and
  one `*_score.json` per bag beside it.
- The development run on the ICINS 2021 recordings:
  [`development_summary.json`](../assets/doer_radar_imu/development_summary.json).

### Result: both claims refuted, on coverage

| Claim | Gate | Result | Verdict |
| --- | --- | --- | --- |
| honesty | the reference never fails; passes on at least 25 % of the bags | no fail; passes on 1 of 9 (11 %) | refuted |
| detection | no turned case exits 0 under `--fail-on partial`; every judged turned axis fails; at least 25 % of the bags fail a turned candidate | none exits 0; the one judged turned axis failed; 1 of 9 bags (11 %) | refuted |

- **Nothing wrong was judged.** No wrong verdict was given:
  - the authors' calibration was never failed;
  - no 3 deg turn passed with full coverage;
  - every turn on an axis that was judged failed.
- **But eight of the nine bags judged no axis at all.** These recordings are shorter
  than the development ones (84–206 s against 75–390 s; seven of the nine under 140 s).
  The rotation stds stayed above the 0.5 deg bound, or the held-out 1 deg control was
  not detected.
- The pre-registered fractions need a judged axis on at least a quarter of the bags.
  So both claims fail, as the plan says they do.

### Per bag (the authors' calibration as the candidate)

The estimate's error against the authors' rotation, about the radar's x, y and z axes,
with its reported std. A bag's estimate does not depend on the candidate.

| Bag | Windows | roll | pitch | yaw | radar-imu | clock offset |
| --- | ---: | --- | --- | --- | --- | ---: |
| gym | 10 | +0.14 ± 0.57 | -0.55 ± 1.03 | -0.28 ± 0.61 | inconclusive | 5.3 ms |
| indoor_floor | 23 | +2.35 ± 0.69 | +2.40 ± 1.04 | +0.22 ± 0.33 | inconclusive (yaw control not detected) | 9.3 ms |
| mocap_dark | 15 | +0.22 ± 0.44 | -0.43 ± 0.39 | +0.16 ± 0.26 | **pass** on yaw (rz3: fail on yaw) | 12.3 ms |
| mocap_dark_fast | 8 | -0.10 ± 1.19 | -0.63 ± 0.96 | -0.11 ± 0.64 | inconclusive | 8.6 ms |
| mocap_difficult | 9 | +0.76 ± 0.91 | +0.17 ± 0.87 | +0.33 ± 0.67 | inconclusive | 11.9 ms |
| mocap_easy | 9 | -0.15 ± 0.72 | +0.20 ± 0.46 | -0.44 ± 0.47 | inconclusive (controls not detected) | 12.8 ms |
| mocap_medium | 9 | +0.86 ± 0.46 | -0.66 ± 0.90 | -0.16 ± 0.50 | inconclusive | 12.7 ms |
| outdoor_campus | 12 | +0.77 ± 0.48 | -1.57 ± 0.67 | -1.12 ± 1.31 | inconclusive | 10.9 ms |
| outdoor_street | 21 | +0.04 ± 0.83 | -1.60 ± 0.59 | -0.66 ± 0.42 | inconclusive (yaw control not detected) | 11.2 ms |

All errors and stds are in deg.

- **The clock.** The offset was 5–13 ms on every bag, as in development (8.9–12.5 ms).
- **The lever arm.** It came out 2–7 cm from the authors' (0.06, 0.04, -0.04) m.
- **The unjudged estimates.** They mostly lie within 2.5 std of the authors' rotation.
  - indoor_floor's roll and pitch are 2.3–3.4 std off.
  - outdoor_campus and outdoor_street are 2.3–2.7 std off in pitch.
  - None of these axes was judged, so no verdict rests on them.

### What v1 says, and what it does not

- radar-imu did not hand out a wrong verdict on unseen data. On short recordings it
  says so (`inconclusive`, with the axes unchecked) instead of guessing.
- It is not yet useful on recordings of this length. It needs longer recordings (the
  development bags that were judged ran 170–390 s) or a tighter estimate.
- The claims stay refuted. A new claim needs new unseen data and a new pre-registration.
  The IROS 2021 archive is now spent for `radar-imu`.
