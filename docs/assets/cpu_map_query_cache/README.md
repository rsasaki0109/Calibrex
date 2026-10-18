# CPU map normal and query-cache development evidence

Both complete cohorts are retained: 18 fresh workers, 90 fits, all covariance and evidence, raw measurements, launchers, supervisor logs and original decisions. The first gate misses 2x (1.987x); its comparator also flagged only creation-time metadata. The follow-up's lazy eight-worker arm observes 2.138x versus prior CPU, while its certificate candidate is slower and fails its gates. No screen is relabeled as passing all original gates.

The runtime change retains native reached-point normals and an explicit CPU query budget. The additional certificate implementation is experimental source evidence only. This post-hoc engineering choice is distinct from a pre-registered accuracy or SOTA claim.

The spent public KITTI copied-sensor workload supplies no independent two-LiDAR accuracy or temporal validation; raw windows are not motion compensated. Reference odometry and bag decoding are outside wall time. The host was unreserved, with one BLAS thread.

Run `python -m tools.audit_cpu_map_query_cache docs/assets/cpu_map_query_cache --check` from the repository. The CPU-only auditor verifies original hashes, source reconstruction, work, complete fit/evidence bit parity, both medians, original metadata errors and decisions. No GPU or raw data is needed for this replay.

The full prior source is `../resident_scan_validation/current.sources.tar.gz`, pinned by the index. Apply the lazy delta to that source, then the current delta to reconstruct the rejected experiment. All three complete 396-module manifests are checked. Original frozen launchers use repository-relative data paths and independent output directories; copy a launcher to the repository root for a new separately named reproduction. Raw scans and prepared coordinate arrays are identified by hashes and are not redistributed.
