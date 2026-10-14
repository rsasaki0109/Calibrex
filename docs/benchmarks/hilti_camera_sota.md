# Hilti 2022 camera SOTA audits

Two claims were pre-registered on Hilti 2022 recordings that no Calibrex estimator had
read, in [`hilti_camera_sota_preregistration.yaml`](hilti_camera_sota_preregistration.yaml)
(commit `1c87031`, SHA-256 `ad14d590...0ef440`):

- **`camera-camera`**: a new target pair. The check's targetless camera-camera
  rotation is audited on twelve recordings (the first 5 GB of each).
- **`camera-imu` v2**: v1's claim, recipe, scorer and thresholds, on the five unused
  recordings of at least 200 s. v1 failed only on its shortest recording
  ([v1 audit](hilti_camera_imu.md#pre-registered-audit-refuted)).

**How the audits ran.**

- **Code.** Frozen at `0482db6`. `src/` and `tools/` were identical to it for every
  evaluation run.
- **Recordings.** Chosen from bag metadata only: the header and the index, read with
  HTTP range requests
  ([`candidate_metadata.json`](../assets/hilti2022_camera_sota/candidate_metadata.json)).
  Each bag's SHA-256 is pinned from the dataset's LFS pointer.
- **Execution.** `tools/hilti_camera_sota_evaluate.py` downloaded each recording,
  hashing the bytes as it wrote them, ran the pre-registered commands, and deleted the
  download. Every run is kept, with its manifest, under
  [`docs/assets/hilti2022_camera_sota/`](https://github.com/rsasaki0109/Calibrex/tree/main/docs/assets/hilti2022_camera_sota).
- **Reference.** Hilti's target-based Kalibr calibration (`calib_3`, shared by all
  sixteen sequences). It is a calibration with its own error, not ground truth.

## camera-camera: refuted (4 of 6 gates)

**The claim.**

- On twelve unused recordings, the check reproduces Kalibr's rotation of five gated
  camera pairs within 0.75 deg on every judged axis.
- It scores at least 70 percent of the 60 gated units.
- It flags a 1 deg turn of cam3 (side) and cam1 (forward), each about its own y axis,
  in at least 90 percent of the evaluable trials.
- It names the turned camera in at least half of them.

**The verdict: `refuted`.** All 36 checks ran without an error (12 recordings, three
checks each).

| Requirement | Observed | Threshold | Result |
| --- | --- | --- | :---: |
| accuracy-vs-kalibr (every scored unit) | max 0.327 deg | <= 0.75 deg | pass |
| no-failed-units | 0 failed | 0 | pass |
| scored-units-coverage | 47 of 60 (78 %) | >= 70 % | pass |
| detection | 16 of 23 (70 %) | >= 90 % | **fail** |
| localization | 11 of 23 (48 %) | >= 50 % | **fail** |
| evaluable-trials-coverage | 23 of 24 (96 %) | >= 70 % | pass |

**Accuracy against Kalibr** (largest absolute judged-axis error per gated pair, deg):

| Recording | `cam0-cam1` | `cam0-cam3` | `cam1-cam3` | `cam0-cam4` | `cam1-cam4` |
| --- | ---: | ---: | ---: | ---: | ---: |
| exp05 | 0.09 | 0.08 | 0.27 | inconclusive | 0.08 |
| exp06 | 0.08 | 0.21 | 0.02 | inconclusive | 0.03 |
| exp09 | 0.12 | 0.26 | 0.16 | 0.11 | 0.13 |
| exp10 | 0.12 | 0.28 | 0.13 | 0.06 | 0.11 |
| exp11 | 0.12 | 0.12 | 0.14 | inconclusive | inconclusive |
| exp14 | 0.12 | inconclusive | 0.24 | 0.04 | 0.19 |
| exp15 | 0.13 | 0.14 | 0.24 | 0.14 | 0.33 |
| exp16 | 0.28 | inconclusive | inconclusive | inconclusive | inconclusive |
| exp18 | 0.09 | 0.07 | 0.15 | inconclusive | 0.15 |
| exp23p0 | 0.12 | 0.07 | 0.15 | 0.10 | 0.00 |
| exp23p1 | 0.12 | 0.06 | 0.19 | 0.05 | 0.01 |
| exp23p2 | inconclusive | 0.11 | 0.09 | inconclusive | inconclusive |

- **Every scored unit is within 0.33 deg of Kalibr; the development worst was 0.32.**
  The accuracy holds in scenes unlike development: a cupola, an attic, a basement and a
  theatre.
- **13 of 60 units are inconclusive.** No axis could be judged; none is wrong. The
  cam0–cam4 side pair is the weakest (inconclusive on 7 of 12 recordings).
- **exp16 judged only cam0–cam1.**

**Turning a camera by 1 deg** ("detected" means a gated pair with the camera reads
`warn` or `fail`; "named" means the attribution names that camera):

| Recording | cam3 turned: detected / named | cam1 turned: detected / named |
| --- | --- | --- |
| exp05 | yes / yes | yes / ambiguous |
| exp06 | yes / yes | yes / ambiguous |
| exp09 | yes / yes | yes / yes |
| exp10 | **no** / consistent | yes / ambiguous |
| exp11 | **no** / consistent | yes / yes |
| exp14 | **no** / yes | yes / yes |
| exp15 | **no** / consistent | yes / ambiguous |
| exp16 | not evaluable | **no** / consistent |
| exp18 | yes / unexplained | **no** / consistent |
| exp23p0 | yes / yes | yes / yes |
| exp23p1 | yes / yes | yes / yes |
| exp23p2 | yes / ambiguous | **no** / no group |

**Why the detection gate failed (post-hoc; it does not change the verdict).**
`tools/hilti_camera_camera_posthoc.py` reads the committed runs
([`posthoc.json`](../assets/hilti2022_camera_sota/camera_camera/posthoc.json)). The turn
is about the camera's own y axis, which every gated pair reads as pitch (52 of 52).

- **When the check judged that axis, it caught the turn every time.** Over the 52
  pair-trials whose pair was scored on the baseline, pitch was judged in 24 of the
  injected runs: 14 read `fail` and 10 `warn`. None read `pass`.
- **The misses are unjudged axes, not false passes.** In the other 28, pitch was
  `unchecked`. Its jackknife std was above the check's 0.1 deg bound in 25, and its
  known-bad control failed in 13 (both in 10). The pair's verdict came from its other
  axes: `pass` in 24, `inconclusive` in 4.
- **The unjudged estimates still saw the turn.** In 27 of the 28, the pitch estimate
  moved by 0.63 to 1.26 deg in the turn's direction. The judged ones moved by 0.74 to
  1.22 deg (one by 2.7). The measurement was there; the rule that decides whether an
  axis is judged held it back.
- **In 6 of the 28, the turn hid its own axis.** Pitch was judged on the baseline but
  became `unchecked` once the camera was turned.
  - In all six, the turn raised the axis's jackknife std above 0.1 deg, to 0.12 to
    0.72 deg; in three, the control also failed.
  - The estimator re-centred its rectification on the turned estimate each time (by
    0.48 to 1.37 deg), and that did not bring the axis back.
  - This is a real weakness: a large error can make itself unobservable instead of
    failing.
- **Judging every such axis is not the fix.** On the baseline, 28 of the 47 scored
  gated units left pitch `unchecked`. Three of those estimates were more than 0.5 deg
  from Kalibr, up to 1.5 deg, all on cam4 pairs. In two of them the control moved the
  estimate the wrong way or barely (responses −0.91 and −0.13, against 0.5 to 1.5).
- **Localization fails for the same reason, and because one pair cannot separate its
  two cameras.** In five trials only one pair flagged the turn, and attribution was
  `ambiguous` between that pair's cameras:
  - cam1 on four recordings (only cam0–cam1 flagged), as on exp21 in development;
  - cam3 on exp23p2 (only cam0–cam3 flagged).

**What this means for the check.**

- **Read `unchecked` axes as unchecked.** A camera-camera `pass` on these recordings
  vouches only for the judged axes. Pitch, the turn about the axis the cameras share,
  is often not among them.
- **Next step: keep a turned axis judgeable.**
  - Find out why a 1 deg turn raises pitch's jackknife std.
  - Find out whether an axis noisier than the 0.1 deg bound can still be judged, with a
    tolerance that covers its noise, without the cam4 false alarms above.
  - This is development work. Any new claim needs new recordings: these twelve are now
    spent for camera-camera, and Hilti 2022 has no sequence left that no camera
    estimator has read.

## camera-imu v2: refuted (5 of 7 gates)

**The claim.** v1's claim, restricted to recordings of at least 200 s. It covers the
two forward cameras on five unused recordings: exp15, exp16 and the three parts of
exp23.

- Rotation within 1.0 deg and clock offset within 0.5 ms of Kalibr.
- Constrained on all three axes.
- Consistent across recordings within 0.75 deg.
- Kalibr's cam0-to-cam1 rotation reproduced within 0.5 deg.

**How it ran.** Every bag was downloaded in full and its SHA-256 matched the pinned one.
The ten units ran v1's command unchanged, and v1's scorer scored them unchanged.

**The verdict: `refuted`.**

| Requirement | Observed | Threshold | Result |
| --- | --- | --- | :---: |
| rotation-vs-kalibr (every unit) | max 0.940 deg | <= 1.0 deg | pass |
| time-offset-vs-kalibr (every unit) | max 0.479 ms | <= 0.5 ms | pass |
| constrained (every unit) | 2 of 10 units not constrained | 0 | **fail** |
| no-failed-units | 0 failed | 0 | pass |
| cross-recording-consistency | cam0 0.811, cam1 0.711 deg | <= 0.75 deg | **fail** |
| relative-cam0-cam1 | max 0.461 deg (exp15) | <= 0.5 deg | pass |
| scored-recordings-coverage | 5 of 5 recordings | >= 3 | pass |

| Unit | Rotation to Kalibr | Time error | Constrained | Windows | Axis std (deg) | Policy |
| --- | ---: | ---: | :---: | ---: | --- | --- |
| cam0 exp15 | 0.940 deg | +0.02 ms | **no** | 28 | 0.22 / 0.21 / 0.33 | warn |
| cam0 exp16 | 0.504 deg | -0.37 ms | yes | 22 | 0.15 / 0.20 / 0.21 | pass |
| cam0 exp23p0 | 0.535 deg | -0.18 ms | yes | 31 | 0.06 / 0.09 / 0.14 | pass |
| cam0 exp23p1 | 0.646 deg | -0.18 ms | yes | 32 | 0.07 / 0.08 / 0.21 | pass |
| cam0 exp23p2 | 0.647 deg | -0.48 ms | yes | 35 | 0.07 / 0.13 / 0.17 | pass |
| cam1 exp15 | 0.599 deg | -0.11 ms | yes | 28 | 0.12 / 0.13 / 0.16 | pass |
| cam1 exp16 | 0.289 deg | -0.00 ms | yes | 23 | 0.13 / 0.18 / 0.18 | pass |
| cam1 exp23p0 | 0.501 deg | -0.28 ms | yes | 31 | 0.07 / 0.10 / 0.15 | pass |
| cam1 exp23p1 | 0.406 deg | +0.01 ms | **no** | 35 | 0.07 / 0.14 / 0.33 | inconclusive |
| cam1 exp23p2 | 0.322 deg | -0.00 ms | yes | 36 | 0.12 / 0.21 / 0.19 | pass |

**What failed, and what that says.**

- **Recording length did not make the yaw axis observable.**
  - The two units that are not constrained, cam0 exp15 and cam1 exp23p1, have 28 and
    35 windows, more than any v1 unit that passed.
  - Their yaw jackknife std is 0.328 and 0.327 deg, just over the 0.3 deg bound.
  - Whether yaw is constrained depends on how much the recording turns about it, not
    on the recording's length. v1's conclusion ("pre-register a minimum recording
    length") is refuted with it.
- **The estimates moved more between recordings than in development.**
  - cam0's estimates span 0.81 deg across the five recordings; development spanned
    0.59 deg over six.
  - The pre-registration listed consistency, then 0.68 deg for cam1, as a known risk.
  - The [rig and motion analysis](hilti_camera_imu.md#rig-and-motion-parts-on-exp01-and-exp21)
    found a motion-dependent part of 0.2-0.3 deg between the halves of one recording.
    Across a construction site, an attic and a theatre it is larger.
- **The accuracy gates held, narrowly.**
  - Worst rotation 0.94 deg (gate 1.0) and worst clock error 0.48 ms (gate 0.5).
  - cam0 exp15, the unit that is not constrained, is also the worst rotation.

**What this means.** Twice now, camera-IMU has been accurate to Kalibr within about
1 deg and 0.5 ms on unseen recordings. It has also twice failed to be constrained on
every axis and to repeat within 0.75 deg across recordings.

- A claim it could support would gate observability per axis from the recording's
  own excitation (as the check's verdict already does), not from its length.
- It would also need a consistency bound that covers the motion-dependent part, about
  1 deg.
- That is a new claim, on recordings camera-IMU has not run on. Seven remain:
  - exp05, exp06, exp11, exp14 and exp18 (74 to 152 s);
  - exp09 and exp10, whose full bags (34.1 and 28.3 GB) exceed this environment's
    27 GB disk budget.
  - The camera-camera audit above read the first 5 GB of all seven.
