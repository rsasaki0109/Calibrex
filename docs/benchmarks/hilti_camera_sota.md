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

**Why the detection gate failed (post-hoc; it does not change the verdict).** The turn
is about the camera's own y axis, which every gated pair reads as pitch.

- **When the check judged that axis, it caught the turn every time.** Over the 52
  pair-trials whose pair was scored on the baseline, pitch was judged in 24 of the
  injected runs, and all 24 read `warn` or `fail`. None read `pass`.
- **The misses are unjudged axes, not false passes.** In the other 28, pitch was
  `unchecked`: the check reported that it could not judge that axis, and judged only
  the other axes. On those recordings the turn went unseen, and a pair verdict of
  `pass` covered roll and yaw only.
- **In 6 of the 28, the turn hid its own axis.** Pitch was judged on the baseline but
  became `unchecked` once the camera was turned.
  - Under the turn, the axis's control check rejects it (re-centring did not recover it).
  - This is a real weakness: a large error can make itself unobservable instead of
    failing.
- **Localization fails for the same reason, and because one pair cannot separate its
  two cameras.** When only cam0–cam1 reads cam1's turn, attribution is `ambiguous`
  between cam0 and cam1. This happened on four recordings, as on exp21 in
  development.

**What this means for the check.**

- **Read `unchecked` axes as unchecked.** A camera-camera `pass` on these recordings
  vouches only for the judged axes. Pitch, the turn about the axis the cameras share,
  is often not among them.
- **Next step: keep a turned axis judgeable.** Find out why a 1 deg turn makes pitch
  fail its control check, and fix that.
  - This is development work. Any new claim needs new recordings: these twelve are now
    spent for camera-camera.

## camera-imu v2

Running. This section will be filled in when the five recordings are scored.
