# GrandTour LiDAR-LiDAR (development)

This page records the first real-data run of the per-point trajectory mode of
`calibrex lidar-lidar`. That mode was written for a sparse, non-repetitive LiDAR on a
legged robot (see [the Go2 tutorial](../tutorials/unitree_go2_lidar_lidar.md)).

It is **development on spent data**. No claim is pre-registered and no SOTA claim is
made. The dataset's calibration is a reference, not ground truth.

## Data

- **Source:** [GrandTour](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset),
  from ETH Zurich RSL. License MIT on Hugging Face. Recorded on an ANYmal D quadruped with
  the Boxi sensor payload.
- **No public Go2 recording fits.** None was found with the L1 and a second LiDAR.
  [ForEnt](https://zenodo.org/records/18824718) (Go2 + MID360) records only
  `/utlidar/robot_odom` from the L1, not its clouds; its zip listing and bag metadata were
  read by HTTP range.
- **Pairs.** Reference: Hesai XT-32. Targets:
    - Livox MID360, mounted upside-down, non-repetitive: the closest analogue of a Go2 L1;
    - Velodyne VLP-16 on the ANYmal body, as a control pair.
- **Development mission.** It was chosen by a metadata-only rule: among the missions with
  Hesai, Livox and Velodyne points, the smallest `hesai_points.tar` + `livox_points.tar`.
  That is `2024-11-02-17-43-10`: 226 s, 2,259 Hesai and 2,265 Livox scans. The tar SHA-256
  values are in [`input_sha256.txt`](../assets/grand_tour_lidar_lidar/input_sha256.txt).
- **Unread missions.** Every other mission's sensor data is unread. Their
  `metadata/tf.yaml` was read for 13 missions; the static tf is identical in all of them.
- **Conversion.** `tools/grand_tour_zarr_to_rosbag2.py` converts the Zarr groups to a
  ROS-free rosbag2. Hesai per-point `time` is absolute seconds, Livox absolute nanoseconds,
  Velodyne an offset (`--point-time-encoding absolute_s`,
  `--target-point-time-encoding absolute_ns` / `offset_s`).
- **Reference.** `inv(T_box_hesai) T_box_target`, from the groups' `transform` attributes
  (the dataset's static tf, chained through `base` for the Velodyne).

```bash
calibrex lidar-lidar ros2 WORK/bag \
  --reference-topic /hesai_points --point-time-field time --point-time-encoding absolute_s \
  --target-topic /livox_points --target-point-time-field time \
  --target-point-time-encoding absolute_ns \
  --body-transforms T_box_hesai_points.yaml T_box_livox_points.yaml \
  --target-motion reference_trajectory --time-offset-search 0.1 0.01 \
  --registration-query-workers 4 --dataset-family grand_tour --dataset-license MIT \
  --output hesai_livox.yaml
```

## Results

Every run passes. In every run:

- all six DoFs are estimated;
- held-out blocks detect every 1 deg / 5 cm control;
- 225 samples (150 train, 75 held out).

Reported std is the larger of the analytic and the 8-group block jackknife value.

### Hesai → Livox MID360

| Run | roll | pitch | yaw | x | y | z | held-out median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rigid_nearest` | −179.893° ± 0.013 | −0.003° ± 0.018 | −89.889° ± 0.030 | 0.000 m ± 0.003 | 0.003 m ± 0.002 | 0.099 m ± 0.001 | 18.8 mm |
| `reference_trajectory` | −179.888° ± 0.018 | 0.000° ± 0.011 | −89.874° ± 0.026 | 0.001 ± 0.003 | 0.013 ± 0.005 | 0.099 ± 0.001 | 16.3 mm |
| `reference_trajectory`, offset estimated | −179.887° ± 0.017 | 0.001° ± 0.012 | −89.877° ± 0.024 | 0.001 ± 0.003 | 0.012 ± 0.004 | 0.099 ± 0.001 | 16.2 mm |

- **Clock offset.** Estimated at −7 ± 12 ms (jackknife), with the profile minimum inside
  the ±0.1 s window. The Livox and Hesai clocks agree: no offset needs correcting.
- **Per-point placement.** Placing each point along the reference trajectory lowers the
  robust residual sigma from 26.6 mm to 23.5 mm and the held-out median from 18.8 mm to
  16.3 mm. The extrinsic moves by at most 0.02 deg and 1 cm (y).
- **Against the dataset tf.** The tf differs by about 0.5 / 0.9 / 0.4 deg and
  −25 / −26 / +22 cm. The held-out chi-square rejects it by 1.5 million.

### Hesai → Velodyne VLP-16 (control)

| Run | roll | pitch | yaw | x | y | z | held-out median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rigid_nearest` | 0.035° ± 0.019 | −0.187° ± 0.015 | 88.667° ± 0.044 | 0.008 m ± 0.003 | 0.695 m ± 0.006 | −0.110 m ± 0.001 | 24.2 mm |
| `reference_trajectory` | 0.029° ± 0.020 | −0.186° ± 0.014 | 88.641° ± 0.031 | 0.004 ± 0.001 | 0.689 ± 0.003 | −0.111 ± 0.001 | 20.6 mm |

The tf differs here by 1.7 deg of yaw and 33–50 cm.

## The dataset tf does not align the released clouds

An odometry-free check, `tools/grand_tour_same_time_alignment.py`, takes single scans
10 ms apart: no map and no trajectory. It moves each target scan by each candidate and
measures it against the Hesai scan.

| Target scan | Estimate: median point-to-plane | Dataset tf: median point-to-plane |
| --- | ---: | ---: |
| Livox 0, 500, 1000, 1500, 2000 | 14–29 mm | 131–254 mm |
| Velodyne 0, 500, 1000, 1500, 2000 | 17–33 mm | 198–278 mm |

[Livox](../assets/grand_tour_lidar_lidar/same_time_livox.json),
[Velodyne](../assets/grand_tour_lidar_lidar/same_time_velodyne.json).

What the check does and does not show:

- **The fitted extrinsics align the released clouds; the tf does not.** The tf misses by a
  decimetre or more in every scan checked.
- **It is not one wrong Hesai entry.** The Livox→Velodyne relation implied by the two
  estimates is Hesai-free. It still differs from the tf by 1.6 deg and 0.77 m.
- **The cause is not found.** The Hesai clouds are sensor-centred: the ring elevation
  angles are constant to 0.004–0.009 deg about the origin. The MID360 driver config in
  `leggedrobotics/grand_tour_box` applies no extrinsic.
- **It is not a clock offset.** The estimated offset is −7 ± 12 ms.
- **Open.** Whether the released point frames or the tf entries differ from what their
  names say remains open. It is a question for the dataset's maintainers.

So the dataset tf cannot serve as a reference for a lidar-lidar audit on GrandTour until
this is resolved. An audit would need either:

- the maintainers' answer;
- a reference that the same-time check accepts.

## What this does not show

- That the per-point mode helps a sparse L1. The MID360 here gives about 12,000 points per
  scan, ten times an L1, so accumulation was not needed (`--target-accumulate-scans 1`).
  That gain is shown only on synthetic data.
- Cross-mission reproducibility, the gate the NTU VIRAL audit failed. It needs more
  missions, read only under a pre-registration.

## Artifacts

[`docs/assets/grand_tour_lidar_lidar/`](../assets/grand_tour_lidar_lidar/):

- the five lidar-lidar artifacts (schema-valid, with provenance);
- the conversion summaries;
- the same-time checks;
- the input SHA-256 values.
