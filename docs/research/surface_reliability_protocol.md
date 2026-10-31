# Measurement-only surface reliability pilot

Frozen before execution, 2026-10-30. This is an offline quality test, not fitting
or real-data tuning. Reuse three analytic sphere layouts with three camera origins,
clean and .5 px Gaussian disparity noise (seeds730/731 +100*frame):27 frames.
These layouts are spent controlled development data; raycast truth is independent
of measurement noise. No LiDAR selection or reference pose enters the candidate.

Candidate fits affine disparity to each full 5x5 patch by fixed least squares.
Accept finite positive patches with disparity-plane RMS <=1 px and first-order
depth sigma=.5*z_plane^2/120 <=.2 m. Assumed noise=.5 px even for clean controls.
This favors near surfaces. No alternate thresholds, edge tuning or real frames.
Predict normal analytically from affine disparity and intrinsics. Selection and
normals use measurements only; no clean depths or analytic surface labels enter.

Score every interior pixel before voxelization. Analytic normal is exact sphere
normal or the background plane normal, determined at the clean ray intersection.
Position error uses the original noisy pixel XYZ (not the plane-smoothed estimate).
A correct surface sample requires XYZ error <=.2 m AND unsigned normal error
<=15 degrees. Compare all valid patches with selected patches on identical data.
Record counts/fractions, individual position/normal precision, occupied 4x4 image
cells, and normal-matrix singular values. These spectra describe directions, not
full six-DoF observability or covariance. Include all27 frames; no frame exclusion.
This evaluates surface reliability, not correct cross-modal correspondences or ICP.

Necessary continuation screen on all18 noisy frames: selected joint precision
>=80%, retention >=10% of interior domain, occupied cells>=8/16, and normal singular
ratio >=.05 in every frame. Also report paired precision changes. These fixed
screen gates are development criteria, not derived deployment safety guarantees.
If any fail, do not transfer the filter to real data or run ICP using it. Keep
empty selections explicit and do not tune thresholds after outcomes. Pin code,
protocol and source hashes before evaluation; retain schema/provenance. No SOTA.
