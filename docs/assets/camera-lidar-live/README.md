# Interactive real Camera–LiDAR projection

The browser's Camera–LiDAR tab projects **real KITTI raw** Velodyne returns onto
the corresponding rectified `image_02` photograph. Select frame `0060` or `0090`
from development drive `2011_09_26_drive_0005_sync`. No rendered sensor data is used.

Move roll, pitch and yaw (±5°), or expand the translation controls (±0.20 m).
**Inject +3° pitch** and **Restore vendor** offer reproducible comparisons. The
divider shows the supplied KITTI calibration on the left and the manually
adjusted projection on the right. Every control recomputes projections locally;
it does not play a pre-rendered video or run a calibration optimizer.

The displayed pixel displacement compares **the same points visible in both
projections**, before clipping either overlay to the divider. It measures how
much the manual offsets move the projection, not calibration accuracy. There is
no live pass/fail verdict. Sensor timing, occlusion and imperfect calibration may
leave mismatches even at the supplied vendor setting.

## Sources and attribution

Credit: **A. Geiger, P. Lenz, C. Stiller and R. Urtasun / KITTI**.
[KITTI raw](https://www.cvlibs.net/datasets/kitti/raw_data.php) data and the
derived photographs, sampled points and projection assets remain under
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).
They require attribution and non-commercial use. Calibrex's source-code license
does not relicense these data.

[download-receipt.json](download-receipt.json) records the official archive URLs,
member names, byte counts and SHA-256 hashes. `source/` preserves the original
photographs and date-level calibration files. The two compressed point subsets
preserve original Velodyne returns and their indices in the original scans.

## Reproduce and inspect

Run `PYTHONPATH=src python tools/build_camera_lidar_demo.py` from the repository
root. The generator uses the committed photographs, calibration files and
point subsets; no network or ROS installation is needed. `--prepare-from DIR`
can rebuild the subsets from the original downloaded scans and receipt.

Points are transformed by `R_rect_00 × T_cam0_velo`, then translated by
`K⁻¹ × P_rect_02[:,3]` to enter rectified camera 2. This includes the projection
matrix's camera baseline; omitting it would misalign the overlay. Manual offsets
apply `Rz(yaw) × Ry(pitch) × Rx(roll)` to these camera points, followed by XYZ
translation in metres. Intrinsics and photograph resolution stay fixed.

For display, returns have camera depth 3–60 m, original Velodyne Z above −1.6 m,
and a generous projected margin around the photograph. Up to 6,000 points are
sampled evenly by original scan index. Colours represent camera Z, clipped to
5–40 m. This is a display subset, not an input to an accuracy evaluation.

The [generated index](../../app/samples/camera_lidar_demo.json) binds original and
derived inputs, generator, images and point assets to SHA-256. Per-frame assets
also contain independently generated SciPy projection references for JavaScript
verification. Images and point assets load only when their scene is selected.

The earlier recorded Camera–LiDAR check GIF remains available separately; its
saved endpoint verdicts are not applied to arbitrary slider positions. Broader
board/edge/mutual-information **method comparison** remains a separate demo task.
