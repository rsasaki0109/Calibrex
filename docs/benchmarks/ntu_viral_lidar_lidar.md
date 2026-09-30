# NTU VIRAL LiDAR-LiDAR Extrinsic

The first LiDAR-LiDAR calibration in Calibrex. It estimates `T_reference_target`
between the two Ouster OS1-16 units of the [NTU VIRAL](https://ntu-aris.github.io/ntu_viral_dataset/)
UAV: one horizontal (`/os1_cloud_node1/points`) and one mounted on its side
(`/os1_cloud_node2/points`). The dataset's license is CC BY-NC-SA 4.0.

```bash
calibrex lidar-lidar ros2 tnp_01_rosbag2 \
  --reference-topic /os1_cloud_node1/points --target-topic /os1_cloud_node2/points \
  --point-time-field t \
  --body-transforms tnp_01/lidar_horz.yaml tnp_01/lidar_vert.yaml \
  --dataset-family ntu_viral --dataset-license "CC BY-NC-SA 4.0" \
  --output tnp_01.yaml
```

`--body-transforms` reads the dataset's `T_Body2Lidar` files (OpenCV YAML). Their
composition `inv(T_body_horz) T_body_vert` serves two purposes: it is the
initial value, and it is the reference that the result is compared against
after the fit. These are design values (-0.55 m, 90 deg turns), not a
measurement.

## Method

The two LiDARs see only a small overlap at any instant, and a 16-beam LiDAR
turned on its side gives a weak odometry. The method therefore registers
target scans to reference maps, instead of matching the two sensors' motions.

1. The reference LiDAR's odometry places its scans in one frame.
2. Every 10th target scan (1 Hz) becomes a sample. It must land on the local
   map built from the 11 reference scans around its time, through `P X`:
   `P` is the reference pose and `X = T_reference_target`.
3. All samples are solved jointly for one `X`, by Gauss-Newton on
   point-to-plane residuals with Huber weights (`calibrex.solvers.lidar_lidar_map_solver`).
4. Evidence, as for the other pairs:
   - samples are grouped in 10-second blocks, and every third block is held
     out;
   - an 8-group block jackknife;
   - known-bad shifts of 1 deg per rotation axis and 5 cm per translation
     axis must raise the held-out chi-square by at least 9;
   - the reference transform is scored on the same held-out blocks.

Rotation std and reference differences are small rotations about the
reference LiDAR's axes. This keeps them meaningful for the side-mounted unit,
whose Euler angles sit at (90, 0, 180) deg.

## Result on tnp_01 (development recording)

Verdict: `inconclusive`. Five of the six DoFs are estimated; roll's
jackknife std of 0.108 deg is just above the 0.1 deg bound.

| DoF | Estimate | Reported std (analytic / jackknife) | Difference to design | 1 deg / 5 cm held-out control |
| --- | ---: | --- | ---: | ---: |
| roll | 89.981 deg | 0.108 (0.008 / 0.108) deg | +0.02 deg | Δχ² 3,509 |
| pitch | -0.460 deg | 0.016 (0.002 / 0.016) deg | +0.46 deg | Δχ² 95,379 |
| yaw | 179.843 deg | 0.019 (0.003 / 0.019) deg | -0.16 deg | Δχ² 111,112 |
| x | -0.490 m | 7.2 mm | +9.8 mm | Δχ² 7,176 |
| y | -0.024 m | 6.8 mm | -53.6 mm | Δχ² 3,097 |
| z | 0.042 m | 8.6 mm | +47.2 mm | Δχ² 13,756 |

- **Samples.** 578 samples: 388 train and 190 held out, with 197,000 and
  95,000 correspondences.
- **Residuals.** The robust residual sigma is 86 mm, and the held-out median
  residual is 61 mm. The per-point noise is dominated by the 16-beam scans
  and the odometry drift over a map window.
- **The design value is rejected.** Replacing the estimate with it raises
  the held-out chi-square by 48,901: the data point about 0.5 deg of pitch
  and 5 cm in y and z away from it. The design value is rounded, so a real
  difference of this size is plausible, but no independent measurement
  exists.
- **The solver hit its 30-iteration cap**, with the last updates still
  moving. The jackknife std covers that.
- **The first 300 s agree.** Run alone, they give differences to design
  within 0.03 deg of these for pitch and yaw, 0.13 deg for roll (the least
  constrained axis), and 1.6 cm for translation.

[Artifact](../assets/ntu_viral_lidar_lidar/tnp_01.yaml).

## What did not work

- **Motion-based hand-eye.** `A X = X B` on the two odometries was tried on
  the first 300 s.
  - The rotation came within about 1 deg of design, but the translation was
    unobservable (x off by 0.5 m, std 0.3 m).
  - The per-motion residuals (0.75 deg, 9 cm median) show that the side-mounted
    16-beam odometry is too weak for this.
  - Gyro-aided deskewing and larger local maps did not help.
- **The hand-eye solver changes stay.** It now starts from the axis-alignment
  rotation as well as the yaw starts, so arbitrary mountings converge. Its
  jackknife and reference differences now use small rotations about the
  parent axes.

## Next

- Use more sequences.
- Estimate the LiDAR-to-LiDAR clock offset.
- Pre-register an audit on the held-back recordings (rtp_01, spms_01,
  tnp_02) against an external baseline.

No SOTA claim is made for lidar-lidar.
