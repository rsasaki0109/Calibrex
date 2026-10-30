# Stereo geometry registration preparation

Prepared 2026-10-30 from freeze `92a59724`.
[Protocol](stereo_registration_protocol.md),
[preparation plan](../assets/stereo_registration_20261030/preparation-plan.json),
[18-pair manifest](../assets/stereo_registration_20261030/prepared.json),
[schema](../assets/stereo_registration_20261030/prepared.schema.json).

Review of camera_lidar_sota_plan.md found existing SuperGlue correspondence
failure on two development captures and I2PNet diffuse correspondence rejection;
its direct-pose gate also failed 0/4. Repeating those provider/threshold variants
would duplicate known failures. Existing native point-to-point and point-to-plane
ICP offer a distinct geometric correspondence pilot without a learned checkpoint.
Open3D is currently absent; no dependency installation or external run occurred.

All 18 original frame pairs were backprojected/voxelized without pose labels.
Prepared source clouds contain 25536–42927 points, targets 38351–71126 points.
Every NPZ hash, finite XYZ shape and manifest count was checked; the manifest
validates against the committed schema. Arrays are retained under
/workspace/work/stereo-registration-20261030, source inputs are pinned separately.
No clouds from different times were concatenated. These counts establish input
availability, not geometric overlap, valid correspondences or successful recovery.

Two focused tests, repository Ruff and strict documentation build pass. Tests
check camera ray backprojection/missing-depth rejection and deterministic voxel
selection. Core/defaults remain unchanged. No calibration or SOTA result is added.

Next implement and freeze the runner for the protocol's 108 single-frame ICP
trials. The executable plan must bind the native implementations, prepared array
hashes, starts and options before fitting. Retain per-frame failures and all-three-
frames drive success; no best-frame selection, pose averaging or truth-based cloud
filtering. The shared three-frame MI6/12 score remains historical context because
this pilot evaluates each frame independently.
