# Resident scan-validation comparison

All six fresh workers from three predeclared, alternating complete comparisons. Both arms use resident CUDA odometry and CUDA spline plane fitting. Only the captured scan validation differs. Startup, empty-cache compilation, six seconds of public GLIM input, eight odometry passes, four full fits, covariance, holdout, 32 signed controls, saving, import verification and placement hashing enter wall time.

Every result and full-fit float bit and all placement digests agree. Only the standard created_at/command/git_commit provenance and two new validation work counters are normalized. All other work counters remain exact.

The source capsules retain the complete measured module bytes. The correctness receipt binds 67 actual-device/option tests; its working-file source hashes are checked against measured sources with explicit per-file LF/CRLF equivalence.

Replay from the repository using `python -m tools.audit_resident_scan_validation docs/assets/resident_scan_validation --check`. No CuPy or GPU is needed. The frozen auditor is auditor.py.txt; the report uses audit.schema.json.

The host was unreserved. These are development/performance observations on one spent public sequence and explicit positive grids; defaults, other devices, a combined CPU speed ratio and calibration accuracy are not established. The calibration verdict remains INCONCLUSIVE.
