# GrandTour LiDAR-LiDAR audit (v1)

This is a pre-registered audit of `calibrex lidar-lidar` on a legged robot. The extrinsic
is Hesai XT-32 → Livox MID360 on GrandTour (ANYmal D with the Boxi payload, MIT license).
A second claim covers the MID360 thinned to an L1-like density, as a stand-in for a
Unitree Go2's L1.

**Verdict: `refuted`, 2 of 6 gates.**

## Files

- [Pre-registration](grand_tour_lidar_lidar_preregistration.yaml). Frozen code
  `26145ec`; pushed before any evaluation data was downloaded.
- [Development note](grand_tour_lidar_lidar_dev.md). Mission 2024-11-02-17-43-10 only.
- [Protocol](../assets/grand_tour_lidar_lidar/evaluation/grand_tour_lidar_lidar_sota_protocol.yaml),
  [audit](../assets/grand_tour_lidar_lidar/evaluation/grand_tour_lidar_lidar_sota_audit.yaml),
  [scores](../assets/grand_tour_lidar_lidar/evaluation/scores.json).
- [Run log](../assets/grand_tour_lidar_lidar/evaluation/runs.json):
    - every download matched its pinned SHA-256;
    - every step finished on its first attempt;
    - no rerun was needed.

## Setup

- **Missions.** 2024-11-03-07-52-45, 2024-11-04-13-07-13, 2024-11-11-12-42-47 and
  2024-11-14-13-45-37. The rule picked, on each of the first four dates after
  development, the mission with the smallest Hesai + Livox files.
- **Methods.** Every run starts from the dataset's static tf.

| Method | Target | Mode |
| --- | --- | --- |
| `calibrex_dense` | full MID360 | default (`rigid_nearest`) |
| `calibrex_sparse` | 2,000 random points per scan within 20 m | `reference_trajectory`, 10 scans accumulated, 5 cm voxels |
| `sparse_rigid` | the same thinned target | `rigid_nearest` (reported only) |
| `dataset_tf` | — | the static tf (reported only) |

- **Same-time check.** Single Livox scans at 10, 30, 50, 70 and 90 % of each mission are
  measured against the Hesai scan nearest in time (about 10 ms apart). It uses no
  odometry and no map.

## Result: refuted, 2 of 6 gates

| Gate | Observed (worst mission) | Threshold | Status |
| --- | --- | --- | --- |
| `calibrex_dense` passes on every mission | 3 of 4 (failure rate 0.25) | all 4 | contradicted |
| `calibrex_dense` cross-mission consistency | 1.52 | ≤ 1.0 | contradicted |
| `calibrex_dense` same-time median | 27.4 mm | ≤ 35 mm | achieved |
| `calibrex_sparse` passes on every mission | 2 of 4 (failure rate 0.5) | all 4 | contradicted |
| `calibrex_sparse` rotation to dense | 0.10 deg | ≤ 0.25 deg | achieved |
| `calibrex_sparse` translation to dense | 2.4 cm | ≤ 1 cm | contradicted |

### Per mission

The sparse-to-dense differences are only counted for passing runs.

| Mission | Dense verdict | Dense consistency | Dense same-time | Tf same-time | Sparse verdict | Sparse to dense |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 11-03 | pass | 0.66 | 23.8 mm | 225 mm | pass | 0.10 deg, 1.1 cm |
| 11-04 | pass | **1.52** | 27.4 mm | 224 mm | **inconclusive** (pitch) | 0.18 deg, 4.5 cm |
| 11-11 | pass | 0.67 | 24.9 mm | 89 mm | **inconclusive** (pitch, y) | 0.20 deg, 4.0 cm |
| 11-14 | **warn** | 0.36 | 33.6 mm | 213 mm | pass | 0.03 deg, 2.4 cm |

## What the result shows

### Same-time alignment holds

On every mission, the fitted extrinsic aligns single same-time scans to 24–34 mm. With
the dataset tf the same scans miss by 89–225 mm. The [development
finding](grand_tour_lidar_lidar_dev.md#the-dataset-tf-does-not-align-the-released-clouds)
reproduces on unseen data: the released static tf does not describe the released clouds.

### The reported std is not an honest cross-mission uncertainty

The dense estimates are not consistent across missions:

- **z.** It spreads from 0.103 m to 0.125 m. Each run's jackknife std in z is 1–3 mm.
- **11-04.** It has yaw −89.52 deg and pitch +0.20 deg, against −89.89 to −89.99 deg and
  −0.01 to +0.06 deg on the other missions.

The rig and the tf are the same in every mission. So either the mount moved between
days, or each run is biased by something the block jackknife does not capture. Two
candidates are odometry drift over the map window and scene-dependent plane geometry.
This audit cannot tell which. This is the gate NTU VIRAL failed too.

### The method's own evidence did catch one problem

On 11-14 the dense run warned. A 5 cm shift in y lowered the held-out chi-square instead
of raising it: the held-out blocks preferred a different y than the train blocks. The
verdict policy reported this, and it counts as a failure here, as pre-registered.

### Sparse: honest, but not accurate enough in translation

With an L1-like target, two missions left pitch (and y on 11-11) unobservable. Those runs
correctly reported `inconclusive` rather than a number.

Where the sparse runs passed, rotation stayed within 0.10 deg of dense. Translation did
not: it was 1.1 and 2.4 cm from dense against the 1 cm gate. The 1 cm gate came from a
development mission where every sparse variant was within 6 mm. That mission was easier
than the evaluation missions.

### Accumulation did not beat rigid single scans

`sparse_rigid` (reported only) was as close or closer to dense on three of four missions:

- 11-03: 0.9 cm;
- 11-11: 1.7 cm;
- 11-14: 1.8 cm.

On 11-04 it warned, with pitch unobservable. The pre-registration did not claim a margin,
and none appeared.

## What would have to change

- **Honest uncertainty first.** The reported std should cover the cross-recording
  scatter. Two options:
    - inflate it by a between-recording term learnt on development data;
    - report a per-mission bias bound.
  Until then, a consistency gate of 0.3 deg / 3 cm will keep failing.
- **Sparse targets.** Reaching 1 cm needs either a better-constrained pitch (more
  vertical structure, or motion that tilts the robot) or a looser, honest gate set on
  development data that is representative of the evaluation scenes.

The four evaluation missions are now spent for `lidar-lidar`. No SOTA claim is made.
