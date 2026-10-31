# Evaluation-only OXTS motion attribution

Pre-registered2026-10-30. All36 archived camera-only metric motion estimates,
including two failed internal screens. No motion refits or new selection masks.
Build an evaluation proxy, explicitly NOT independent motion ground truth:
T_camera2_imu = shift(P_rect_02) * R_rect_00 * T_cam0_velo * T_velo_imu.
Require camera2_velo to match the existing calibration reference within1e-8.
This consumes the target extrinsic only after estimation, never in selection.

Use all available pinned OXTS packets within±2 indices of each18 centers, absolute
IMU timestamps and local Mercator world poses. Interpolate translation linearly
and orientation by SLERP at original camera timestamps; no extrapolation. All
relative world origins use the same first available packet per drive. Relative
proxy target-from-source = C_I * inverse(W_I_second) * W_I_first * inverse(C_I).
Report rotation and translation disagreement and both translation magnitudes.

Fixed diagnostic variants: common camera clock shift -10,0,+10ms, same-index OXTS
instead of time interpolation, and omitted camera/IMU lever arm (rotation retained).
Record proxy-change magnitudes and estimate disagreements. These are sensitivity
scenarios, NOT measured uncertainty bounds or a search for a correcting timestamp.
Do not pick a favorable variant or modify any output based on reference agreement.

All36 pairs remain denominator. Test interpolation at integer nanosecond stamps,
no-extrapolation, transform direction and rotational lever-arm effect. Pin packet,
time, calibration and archive/source hashes before running. OXTS/target calibration
errors and world-pose interpolation remain limitations. No Gaussian covariance,
independent calibration accuracy, ICP adoption or SOTA claim follows. No downloads.
