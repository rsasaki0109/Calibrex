# Timed Camera-LiDAR evaluation reuse

All six alternating fresh-process runs are retained. The complete prepared-input
workflow changes from a median 136.523854 to 33.203555 s
(4.111724x), and the artifact-backed API from 134.120710
to 30.923065 s (4.337239x). The host was unreserved;
the [benchmark report](../../benchmarks/continuous_camera_lidar_evaluation_reuse.md)
retains every timing and its variation.

The input contains 9,216 selected original KITTI LiDAR points from nine scans
and original OXTS velocities of those scans. Pinhole image means, signed
per-point times, fixed noise and extrinsic transforms are generated controls.
These are performance/parity inputs, not independently measured image matches,
clock truth or accuracy evidence. The original 500-evaluation bound and every
solver option remain unchanged; all runs converge after 341 candidate evaluations.

All complete results and objective/acceptance traces match exactly, excluding
only the two declared source/run provenance fields (`git_commit`, `created_at`).
There are 2,462,720 point-projection requests per run. Actual motion compensation
computations fall from 2,462,720 to 556,032. Point counts, confidence/projection
gates, covariance loss, time sensitivity and disjoint holdout remain intact.

Whole-worker timing includes startup/imports, actual Git prechecks, problem/model
validation, object construction, the complete solve, counters, result packaging,
saving and imported-source hashing. Input generation, source freezing, compression
and external audits are untimed. Both variants use one BLAS thread and identical
dependencies. Only `calibrex/solvers/continuous_time_camera_lidar_solver.py`
changes in the complete runtime manifests. Each worker retains actual full and
abbreviated Git identities as well as imported module hashes.

`problem.json` is a public-schema-valid complete input. Compressed source points
and the original kinematics control preserve the exact generation inputs; their
uncompressed hashes are checked against the original generation receipt. The
audit verifies original points/IDs/distributions, OXTS velocities/timestamps,
declared point times, source capsules, every run, all results and work counts.
`index.json` binds every other byte, including validation receipts and the
[initial failed provenance check](failed-provenance-v1/README.md), which is
excluded from successful timing statistics. No failed run is relabeled as warmup.

KITTI source data and derived numeric controls are attributed to Andreas Geiger,
Philip Lenz, Christoph Stiller and Raquel Urtasun, KITTI, under
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/).
The [official raw-data page](https://www.cvlibs.net/datasets/kitti/raw_data.php)
and original source URL remain in the motion input's provenance. Source-code
capsules and generator code retain the project's Apache-2.0 license.

Audit without calibration, a GPU or another download:

```bash
python -m tools.benchmark_continuous_camera_lidar_reuse --audit-only docs/assets/continuous_camera_lidar_evaluation_reuse
```
