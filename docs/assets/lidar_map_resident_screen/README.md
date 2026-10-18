# Rejected resident LiDAR-map integration

All nine fresh workers and all three predeclared cyclic comparisons are retained. The candidate passed complete calibration and evidence comparisons but failed the incremental CUDA speed gate in every pair. Production source is unchanged.

The workload replays six seconds of the spent public GLIM bag. The same topic supplies both sensors, with fixed 0.01 m target-only Gaussian noise (seed 20261018). It is an execution control with no independent dual-LiDAR accuracy evidence. Every arm keeps map fitting, four jackknives, covariance, holdout and six known-bad controls on CPU; only reference odometry changes.

The prior runtime is the complete current source capsule already published in ../resident_scan_validation. Its raw archive and manifest hashes are bound here. candidate.source-delta.tar.gz replaces exactly three modules; the complete candidate.sources-manifest.json verifies all 396 reconstructed modules. This retains complete measured source identity without duplicating unchanged code.

correctness/ binds the 100-test final candidate qualification, its exact test sources and launcher. LF/CRLF working-file forms are checked explicitly. The experimental runtime commit is retained in the local experiment branch and reconstructed by the source delta; it is not part of the production change.

From the repository, run `python -m tools.audit_lidar_map_resident_screen docs/assets/lidar_map_resident_screen --check`. No GPU or raw input is required. The frozen launcher records every measured parameter and input transform. Further timing must use separate source copies and a new output directory.
