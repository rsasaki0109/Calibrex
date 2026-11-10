# radar-imu: the check audit on unseen radar-inertial recordings

`calibrex check --pairs radar-imu` judges the rotation of `T_radar_imu` from a 4D radar's
Doppler returns against the gyro and accelerometer (see the
[check tutorial](../tutorials/calibrex_check.md#radar-imu-a-4d-radars-doppler-against-the-imu)).
This page reports the one pre-registered run on Doer & Trommer's IROS 2021
radar-thermal-visual-inertial archive (TI IWR6843AOP radar, ADIS16448 IMU, nine bags of
84–206 s), with the authors' calibration as the candidate and with `T_radar_imu` turned
3 deg about each radar axis.

- Plan: [`doer_radar_imu_preregistration.yaml`](doer_radar_imu_preregistration.yaml),
  frozen at `fb0dcc35`.
- Harness: [`tools/doer_radar_imu_audit.py`](../../tools/doer_radar_imu_audit.py).
- Records: [`docs/assets/doer_radar_imu/evaluation/`](../assets/doer_radar_imu/evaluation/).
- The development run on the ICINS 2021 recordings:
  [`development_summary.json`](../assets/doer_radar_imu/development_summary.json).

## Result: both claims refuted, on coverage

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

## Per bag (the authors' calibration as the candidate)

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

## What this says, and what it does not

- radar-imu did not hand out a wrong verdict on unseen data. On short recordings it
  says so (`inconclusive`, with the axes unchecked) instead of guessing.
- It is not yet useful on recordings of this length. It needs longer recordings (the
  development bags that were judged ran 170–390 s) or a tighter estimate.
- The claims stay refuted. A new claim needs new unseen data and a new pre-registration.
  The IROS 2021 archive is now spent for `radar-imu`.
