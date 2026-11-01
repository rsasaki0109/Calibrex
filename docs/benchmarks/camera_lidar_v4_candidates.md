# Camera-LiDAR v4: unread-data candidates (metadata survey)

**Status, 2026-11-01: metadata only. No pre-registration exists and no candidate
sensor payload has been read.** KITTI raw and KITTI-360 are spent for
camera-LiDAR ([v3 KITTI-360 page](kitti360_camera_lidar_v3_prep.md)). A v4 check
claim needs recordings from other dataset families. The research plan's final
gate asks for two families and an independent rig anyway. This page records
which candidates exist, what of them this repository has read, and the exposure
policy the owner chose.

## Exposure policy (owner's decision, 2026-11-01)

The owner cannot say whether work outside git ever opened these candidates'
camera images or point clouds. They chose to **use them with that disclosed**,
not to treat them as spent. A pre-registration on them must state two things:
that the repository's history shows them unread (or what it read), and that the
owner could not confirm exposure outside git. The resulting claim is weaker than
one with confirmed exposure, and must say so. KITTI-360's candidates stay
spent, because the owner's answer there included "opened".

## Candidates

| Family | Access from the cloud | Read by this repository | Role |
| --- | --- | --- | --- |
| **Argoverse 2 sensor** | anonymous S3 listing, `s3://argoverse/datasets/av2/sensor/` | nothing; the name appears in no commit | primary candidate |
| **Boreas** | anonymous S3 listing, `boreas.s3.us-west-2` | INS-LiDAR read LiDAR, Applanix and calibration of `2020-11-26-13-58` and `2020-12-01-13-26`, and listed `2020-12-04-14-00`'s calibration. No camera image of any sequence was read | second family |
| A2D2 | anonymous S3 | one public camera-LiDAR pair (2 frames) in the research plan's smoke | **excluded**: its LiDAR is distributed already mapped into each camera's view, so the extrinsic has no independent reference |
| nuScenes, Waymo, PandaSet | registration or authenticated download | nuScenes mini for radar | not reachable from this environment without credentials |

### Argoverse 2 sensor dataset

There are 700 train, 150 validation and 150 test logs. One validation log
(`02678d04-…`, its listing only) has 157 LiDAR sweeps over about 15.6 s, which
are 10 Hz feather files, 151 MB in all. It also has 319 `ring_front_center`
JPEGs at 20 Hz (96 MB), seven ring and two stereo cameras, per-log
`calibration/` and `city_SE3_egovehicle.feather` poses. The LiDAR points are in
the ego-vehicle frame. The camera's ego extrinsic is therefore the
camera-LiDAR calibration a check would judge, and the per-log calibration is
its reference. Logs are about 15 s long, so the check's short-recording frame
spread applies. Reading feather files needs `pyarrow`, an optional adapter
dependency in the same way OpenCV is.

### Boreas

The bucket has 126 top-level prefixes, including calibration-only and object
releases. Sequences run from 2020 to 2025. One sequence (`2020-12-04-14-00`)
lists 24,510 camera images (73.9 GB) and 11,923 LiDAR scans (59.6 GB) over
about 20 minutes. Its `calib/` folder holds `P_camera.txt`,
`T_camera_lidar.txt` and `T_applanix_lidar.txt`. An INS-LiDAR adapter for
Boreas LiDAR and Applanix already exists. A camera path does not.

## Argoverse 2 converter (development, 2026-11-01)

`calibrex convert av2`
([tutorial](../tutorials/calibrex_check.md#converting-an-argoverse-2-sensor-log))
writes one log's camera, one LiDAR in its own sensor frame and the poses. One
validation log is now development data: `02678d04-cc9f-3148-9f95-1ba66347dff9`,
about 247 MB fetched (front centre camera, both LiDARs, calibration and poses).
On it the check passes the vendor calibration with all three rotation axes
judged. With the LiDAR left in the ego frame it judged none, which is why the
converter moves points back into the sensor frame.

## Argoverse 2 split (metadata only, 2026-11-01)

`tools/av2_camera_lidar_v4_split.py` fixed the split from anonymous bucket
listings alone ([split](../assets/camera_lidar_v4/av2_split.json)):

| Role | Logs | Rule | Members pinned |
| --- | ---: | --- | ---: |
| Development | 30 | the first 30 validation logs by log ID | 7.7 GB |
| Evaluation | 150 | every test log | 37.7 GB |
| Reserve, never read | 120 validation, 700 train | everything else | – |

The members a `calibrex convert av2` run reads are the two calibration tables,
the poses, every LiDAR sweep and every `ring_front_center` image. They are
pinned per log by the SHA-256 of their sorted key, size and ETag lines. All 180
chosen logs have every member, with 155–197 sweeps and 319–399 images each.
The one log already read for the converter (`02678d04-…`) is the first
development log. No other payload has been read.

## Gate rule, fixed before the Argoverse 2 development results

This rule was committed before any development check ran on Argoverse 2. The
one log read so far was read only to build the converter. The gates follow from
the rule once the 30 development logs have run. Development values come from
those logs, scored by the frozen v2 scorer under the default rule. KITTI raw and
KITTI-360 are reported beside them and set no gate. The rules are v2's,
unchanged, except one:

- **Judged-axis error, baseline and turned runs.** 1.5 times the worst
  development value, rounded up to the next 0.25°.
- **Judged-axis coverage.** The development rate divided by 1.5, rounded down
  to the next 0.25.
- **Scored logs.** The development rate divided by 1.5, rounded down to a whole
  number of the 150 logs.
- **3° turns missed.** The one-sided 95 % exact binomial upper bound of the
  development miss rate (the rule of three when none is missed), rounded up to
  the next 0.05.
- **1° turns missed.** 0.25.
- **3° turns judged right, failed logs.** None allowed.
- **Vendor calibration flagged (the change).** The binomial allowance above,
  from the development rate, where v2 allowed none. On KITTI-360's spent
  development data the default rule flagged the vendor calibration on 1 of 22
  segments. The cause was a scene-dependent bias lasting the whole minute,
  which no single recording can reveal
  ([cause](kitti360_camera_lidar_v3_prep.md#cause-of-the-0002-s04-flag-post-hoc-spent-data-only)).
  "None allowed" would test the reference's luck on 150 logs, not the check.
  The claim must therefore state its vendor-flag rate.

The secondary claim measures the opt-in warning
(`--camera-lidar-start-agreement-deg 0.5`, replayed on the same inputs). Its
baseline warning rate and its run-level 3° and 1° miss rates get the binomial
allowance, and its judged-axis errors the 1.5× rule. If the development batch
already fails a gate derived from itself, the pre-registration records that.
It is not tuned away.

## First development batch (converter superseded, 2026-11-01)

The first batch ran all 30 development logs from commit `b1f3a20b`, with no
failed attempt ([receipts](../assets/camera_lidar_v4/av2_dev_20261101/gates.json)).
Its converter applied the distributed `k1..k3` and each point's `offset_ns`.
The default rule flagged the vendor calibration on **12 of 28** scored logs.
Judged baseline axes reached 4.5° from it, and turned-run axes 8.75°. The
gates the rule derives from those numbers (6.75°, a 0.6 vendor-flag
allowance) would make any claim empty.

Probes on four development logs changed one choice at a time. Zeroing the
distortion turned 19f53e16 from −4.5° to a pass, and 15ec0778's judged axis
from fail to pass. A sweep snapshot instead of per-point offsets also
turned 19f53e16 to a pass. 0fb7276f (about 2 m/s) kept a 2.75–2.9° roll
error either way. `av2-api` itself projects with `K` alone, and moves a whole
sweep with one ego pose. So the converter was changed to follow it (commit
`0364679d`), not to fit those four logs. The batch is re-run on all 30 logs
with the corrected converter, and the gates come from that run. These first
receipts stay as history.

## Second development batch (av2-api conventions, 2026-11-01)

All 30 development logs ran again from commit `76901adc`, with no failed attempt
([receipts and gates](../assets/camera_lidar_v4/av2_dev_r2_20261101/gates.json)).
The rule committed above gives:

| Requirement | Development (default rule) | Gate by the rule |
| --- | --- | --- |
| Vendor calibration flagged | 7 of 28 scored (first batch: 12) | ≤ 0.45 |
| Worst judged baseline axis | 3.00° (first batch: 4.5°) | ≤ 4.5° |
| Worst judged turned-run axis | 7.12° | ≤ 10.75° |
| Judged baseline axes | 0.75 (first batch: 0.51) | ≥ 0.5 |
| Scored logs | 28 of 30 | ≥ 93 of 150 |
| 3° turns missed / judged right | 5 of 63 / 0 | ≤ 0.2 / none |
| 1° turns missed | 6 of 63 | ≤ 0.25 |

The opt-in warning misses 1 of 61 3° turns and 2 of 61 1° turns by the run
verdict. It warns on 8 of 28 baselines.

Most logs now judge every axis within ±0.5° of the vendor calibration. The
exceptions are of two kinds:

- **Camera-x rotation (vertical tilt).** It reads −3.00° (0fb7276f), −1.75°
  (05fa5048), −1.50° (214e388e), and about −0.6° on two more, with reported
  stds of 0.04–0.09°.
- **Yaw.** It reads +0.6° to +0.8° on three logs, with a std of 0.03°.

The calibrations differ between logs, but two logs with the same calibration
read −1.45° and +0.10°. So the exceptions belong to the log, as on KITTI-360,
not to the calibration. Every one of the large tilt readings has the same sign,
which a scene effect alone would not explain. The cause is open.

### Tilt investigation on 0fb7276f (post hoc, development data)

The default check on this log reads camera-x −3.0°. The diagnostics below
changed nothing in the check:

- **The objective genuinely prefers the tilt.** Along camera-x, the pooled edge
  objective of the saved inputs peaks at −2.5° (score 0.019) and drops to 0.003
  at the vendor calibration. Two clean logs (02678d04, 070bbf42) peak sharply
  at 0°, and 05fa5048 peaks near −1.5°. So the axis is not merely weakly
  constrained.
- **The ego hood is not the cause.** The `up_lidar`'s edge points fall in a band
  around the horizon (image rows about 800–1100 of 2048), far above the hood at
  the bottom of the image.
- **The forward cameras share it.** The same log was converted for five ring
  cameras, each checked against its own vendor calibration:

| Camera | Verdict | Camera-x reading |
| --- | --- | ---: |
| front centre | fail | −3.0° |
| front left | inconclusive (no axis judged) | −3.0° |
| front right | inconclusive (no axis judged) | −3.0° |
| side left | pass | +0.35° |
| side right | pass | +0.15° |

A pitch error of the LiDAR extrinsic would show in the side cameras too, about
its camera-z axis, and it does not. A front-centre calibration error would not
appear in the front-left and front-right cameras, and it does. What remains is
something the forward views share. That is either their common scene (palm
fronds against the sky, whose LiDAR and image silhouettes may disagree
vertically) or a timing or motion effect along the direction of travel, which
displaces points radially from the focus of expansion. These two are not yet
separated.

Splitting the saved inputs (0fb7276f and 05fa5048, against two clean logs)
separates them:

| Subset of edge points | 0fb7276f | 05fa5048 | 02678d04 | 070bbf42 |
| --- | ---: | ---: | ---: | ---: |
| all | −2.50° | −1.50° | 0.00° | 0.00° |
| frames at under 0.5 m/s | −2.75° (2 frames) | – | −0.5° (1 frame) | – |
| above image row 950 (above the optical axis) | +0.25° | +1.50° | 0.00° | +0.25° |
| at or below row 950 | −2.50° | −1.50° | 0.00° | 0.00° |
| depth 40–200 m | −2.75° | −2.75° | −0.25° | 0.00° |

- **Not motion.** Stationary frames carry the same tilt, and a clean log at
  11 m/s carries none.
- **Not a vertical translation error.** Points beyond 40 m carry it as well,
  and a translation error would fade with range.
- **It comes from edges below the optical axis.** Edges above it agree with the
  vendor calibration.

One mechanism fits, though it is not yet tested: near-side edge
quantization by sparse beams. At an object's top silhouette, the near-side
edge point is the last beam that hits the object, which lies below the true
silhouette by up to one beam spacing. At its bottom the bias reverses. A scene
whose below-axis edges are mostly tops (planters, car bodies, kerbs) would
bias tilt one way. Below the horizon the `up_lidar` is a 32-beam VLP-32C with
coarser spacing than KITTI's 64 beams. Beam spacing alone does not yet account
for 3°. A test would localise vertical depth edges between beams, or weight
them by beam spacing, on development logs.

**Follow-up: the vendor calibration is right, and the check finds a false
optimum.** Two further diagnostics settle the question:

- **The mechanism above does not apply.** Its edges come from range jumps
  along each beam, which are object sides, not tops. Keeping only edges with a
  neighbouring beam's edge at a similar azimuth and range kept 90 % of them and
  left every optimum unchanged (0fb7276f −2.50°).
- **The annotated cuboids align at the vendor tilt.** The log's 3D annotations
  were projected with the vendor calibration and with −2.5° tilt. At the
  vendor tilt, the vehicle, pedestrian and bollard boxes sit on their objects
  (the sedan's box spans roof to wheels). At −2.5° every box drops onto the
  road. The annotations come from the same LiDAR, so this is not an
  independent reference. It does show that the LiDAR-to-camera alignment
  under the vendor calibration is visibly right.

So on this scene the edge objective has a **false optimum**. The `up_lidar`'s
edges crowd a band around the horizon, mostly foliage and distant structure.
Moved down about 78 px, they land on the strong ground-level edges below
(kerb lines, car bottoms, building bases). Every start reaches it, so the
opt-in cross-start warning does not catch it on 0fb7276f or 214e388e. It
withholds the tilt only on 05fa5048. The split above does catch it: edges
above the optical axis agree with the vendor calibration, and edges below it
do not. An internal agreement test between those two halves is a candidate
for development. It would ask for the same answer from disjoint parts of the
image, much as the cross-start rule asks it from disjoint starts.

**No v4 claim is frozen.** Gates of 4.5° and a 45 % vendor-flag allowance would
be honest, but they would not support a meaningful check claim. The 150
evaluation logs stay unread.

## Next steps (not yet decided)

1. **Tilt bias.** Find the cause of the negative camera-x readings on
   development logs (0fb7276f: −3.0°) before any freeze. This should happen
   only on development data.
2. **Boreas.** A camera, LiDAR and Applanix converter, and a split with the
   two INS-LiDAR sequences as development data. Boreas is the second dataset
   family the research plan's final gate needs.
3. **Pre-registration.** Freeze a v4 pre-registration only when its derived
   gates support a meaningful claim, with the exposure policy above disclosed.
