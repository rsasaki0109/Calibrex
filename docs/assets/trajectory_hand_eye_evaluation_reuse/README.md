# Complete prepared-input INS-LiDAR evaluation reuse

This archive retains all six complete fresh-process runs, versioned prepared
problem, generator, original public OXTS bytes, full source capsules/manifests,
measurement code/schema, public result schema, complete residual/optimizer
histories, counters and validation receipts. index.json binds every publication
byte other than itself.

The reference uses all 2,762 OXTS poses of KITTI
2011_09_30_drive_0018_sync. Sensor poses 10-2751 and their 788 relative motions
are generated using the declared extrinsic, clock, scale and seeded noise.
No raw LiDAR registration is performed. This is a performance/numerical-parity
control, not independent calibration accuracy or raw-pipeline evidence.

KITTI attribution: Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel
Urtasun. Original data and its derived control retain CC BY-NC-SA 3.0:
https://www.cvlibs.net/datasets/kitti/ and
https://creativecommons.org/licenses/by-nc-sa/3.0/.
Calibrex source snapshots and measurement code retain Apache-2.0.

raw-oxts.tar.gz contains all original OXTS packet/timestamp bytes and the
original download provenance. The audit reconstructs every reference pose
and generated sensor motion. history.json.gz retains every normalized
evaluation's parameters/residual digest and every least-squares result's
parameters, termination counters and complete residual/Jacobian digests.
Real full and short Git identities are verified separately before comparison.

interrupted-v1 retains an earlier incomplete cohort and its audit correction;
none of its timing is included in the final six-run comparison.

Audit from the repository root without calibration or raw downloads:
python -m tools.benchmark_trajectory_hand_eye_reuse --audit-only
docs/assets/trajectory_hand_eye_evaluation_reuse
