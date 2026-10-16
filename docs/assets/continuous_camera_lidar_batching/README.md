# Timed Camera-LiDAR ordered-batch comparison

This archive contains all six fresh-process runs, complete prepared problems,
source/helper/schema pins, results and receipts for the linked report.
`index.json` binds every publication byte other than itself.

The nine scans and 9,216 original points of KITTI
`2011_09_30_drive_0018_sync` are unchanged from the previous evaluation-reuse
control. Original point distributions and OXTS velocities are preserved in
`source-points.json.gz` and `motion-control.json.gz`. `input-provenance.json`
and `control-generator.py.txt` declare their lineage, point times, image means,
camera/extrinsics and fixed noise. Generated image matches and point times
make this a performance/parity control, not independent accuracy evidence.

KITTI attribution: Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel
Urtasun. Public KITTI data and its derived control retain CC BY-NC-SA 3.0:
https://www.cvlibs.net/datasets/kitti/ and
https://creativecommons.org/licenses/by-nc-sa/3.0/.
Calibrex source snapshots and measurement code retain Apache-2.0.

The protocol uses benchmark schema v0.2. Observed scalar/batch work is checked
against an independent replay of the unchanged trace and two-clock cache.
All 556,032 deskewed points and 2,462,720 projections remain in each run.
Only top-level result provenance created_at/git_commit are excluded from
numerical comparisons; actual full/short Git IDs and source SHA are checked
separately. Prior v0.1 archives remain independently auditable.

Run `python -m tools.benchmark_continuous_camera_lidar_reuse --audit-only
docs/assets/continuous_camera_lidar_batching` from the repository root.
Frozen worker paths in the historical protocol describe the original run;
the audit uses the portable capsules and needs no original path or download.
