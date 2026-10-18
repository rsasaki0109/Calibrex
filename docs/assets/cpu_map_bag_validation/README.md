# Whole-bag CPU map validation

All 18 fresh workers and 90 full fits are retained, with complete covariance, held-out scoring, six controls, raw measurements, original logs, source and input bindings, and original summaries. The two densities are separate workloads.

Observed whole-worker medians are 17.645548 to 14.132126 seconds (1.249x) at 0.2 m map voxels and 32.449422 to 19.234374 seconds (1.687x) at 0.03 m. Both include bag streams and common CUDA odometry. The earlier map-only ratios are not whole-bag ratios; these results are not pooled or multiplied with them.

The same spent public GLIM topic supplies reference and noisy target scans. This is an execution control, not independent two-LiDAR accuracy, temporal validation or SOTA evidence. The host was unreserved; every worker uses one BLAS thread and an initially empty independent CUDA compiler cache.

Run `python -m tools.audit_cpu_map_bag docs/assets/cpu_map_bag_validation --check` from the repository. Replay needs no GPU or raw bag. All physical/evidence bits, five fits per worker, prepared sample identities, kernel identities, source/input bindings, timing scopes and medians are checked. Frozen helper definitions are compared as syntax trees and are never executed from the archive.

The complete prior source is the already published resident-scan validation capsule. Apply the already published CPU-map lazy delta, then this archive's current delta (bag-option provenance only). Dependency hashes and both complete 396-module manifests are checked. No raw scans, coordinate arrays or compiler binaries are redistributed.
