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

## Next steps (not yet decided)

1. **Converters.** ~~Argoverse 2~~ done above. Boreas camera, LiDAR and
   Applanix are still to do. Both are development work, validated first on
   development data only.
2. **Split rules.** Argoverse 2: ~~fix from metadata~~ done below. For Boreas,
   the two INS-LiDAR sequences would be development data.
3. **Pre-registration.** Develop on the development split, derive gates by the
   committed v2 rule, and freeze a v4 pre-registration that discloses the
   exposure policy above.
