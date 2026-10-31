# Frame-paired stereo geometry registration pilot

2026-10-30. This is a distinct measurement/correspondence construction following
failed metric residual and local-search variants. Review found that virtual-LiDAR
SuperGlue and I2PNet correspondence exports already failed geometric support
(camera_lidar_sota_plan.md); do not rerun their confidence/checkpoint variations.
Open3D is not installed. Native trimmed mutual point-to-point and Chen-Medioni
point-to-plane ICP are available and can serve as classical geometric pilots.
Their existence does not prove camera/LiDAR registration quality or SOTA.

## Input preparation (execute first)

Use the same six spent drives and their three archived SGBM frames, restored to
original camera rays. Backproject finite optical depths in (.1,80] m using recorded
intrinsics. Independently retain the first point in each .2 m voxel for camera and
LiDAR clouds, with finite XYZ and radial distance <=80 m. The source is in LiDAR
coordinates and the target in camera coordinates. No reference or initial transform
is used for point construction, filtering, voxel selection or pair association.
No cross-frame concatenation: each target is paired only with its matching cloud.
Pin all source/input hashes before generating 18 NPZ pairs. Store full XYZ arrays
outside the repository and commit their paths/hashes/counts. No calibration is
performed by the preparation tool.

## Frozen next pilot

For each of the 18 pairs, run both native ICP variants with positive/negative
original starts and a labeled reference-start control: 108 single-frame fits.
Keep their public defaults, except explicitly disable point-to-point multi-start
diagnostics to avoid extra restarts; keep effective diagnostics and spatial 20%
holdout. Native settings are 50 iterations, 1 m correspondence gate, .8 trimming,
mutual matches, .5 m spatial holdout voxels, seed0. Point-to-plane normal count12
and eigengap .02 stay defaults. Preserve returned failure/degeneracy/holdout
indicators. Frame pairs never match across time. Reference enters scoring and the
labeled controls only. Voxel size and depth gates are not retuned after outcomes.

Report every frame and start, rotation <.5 deg/translation <.20 m jointly with
solver convergence, source/target coverage, train and holdout errors and rejection.
For 12 drive-level perturbed starts, require all three frame fits to pass to label
the drive/start successful; also report individual 36-frame denominator. Reference
controls are 18 frames and six drive groups. Do not average transforms or select
the best frame using truth. Compare variants within the same prepared pairs.
Historical three-frame MI6/12 is a contextual baseline, not this per-frame pilot's
paired numerical comparator. A fair shared multi-frame estimator comparison would
follow only if this geometry construction shows usable support and accuracy.

Input preparation and protocol are the deliverable of this stage. Bind the pilot
runner and native solver hashes in a separate executable plan before fitting;
no fits may precede that runner freeze. Preserve every failed case. No production,
fresh-data or SOTA claim, and no learned checkpoint is required.
