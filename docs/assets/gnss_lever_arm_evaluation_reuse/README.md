# Complete prepared-input GNSS–LiDAR evaluation reuse

This archive retains the selected six fresh-process runs, prepared problem,
generator, original public OXTS packets/timestamps and download provenance,
complete core source capsules/manifests, measurement code and schemas,
official results, full residual/optimizer histories and validation receipts.
`index.json` binds every publication byte other than itself.

All 2,762 OXTS poses of KITTI `2011_09_30_drive_0018_sync` are used. The antenna
track and 29 odometry windows are generated with the declared transform,
lever arm, clock and seeded noise. There is no raw LiDAR registration or
independent accuracy evidence. This is a prepared-input performance/parity
control; its unchanged policy verdict is inconclusive for the lever arm.

The final whole-process median is 13.871298 → 4.896686 seconds (2.832793x).
The complete prepared evaluation phase is 11.028193 → 2.085993 seconds
(5.286783x). All eight default jackknife fits and four held-out controls remain.
See [the benchmark](../../benchmarks/gnss_lever_arm_evaluation_reuse.md) for
timing boundaries, all pairs, work counters and limitations.

`history.json.gz` retains every requested residual's fit, window IDs, pair and
epoch counts, exact parameters and full residual digest. It also retains every
least-squares solution, active mask, termination fields and full residual/
Jacobian digests. The audit replays both bounded caches and checks actual
interpolation, every queried epoch and SVD work. Real full and abbreviated Git
identities and imported source bytes are independently verified before comparing
all six complete results/histories.

Raw input/source hashes and archived numerical payloads are exact. Only offline
regeneration from OXTS allows platform math roundoff: `rtol=0`, `2e-8` m for
translation and `1e-13` for rotation-matrix elements. Times and IDs stay exact.

The `qa` directory includes the two failed endpoint fixture attempts and their
repairs. They concerned single-axis angle array shape and the artificial
fixture's GNSS gap bound; the final regression and complete reuse suite pass.
No measured core or protocol was repaired after the selected six-run cohort.

KITTI attribution: Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel
Urtasun. Original data and generated derivatives retain CC BY-NC-SA 3.0:
[dataset](https://www.cvlibs.net/datasets/kitti/) and
[license](https://creativecommons.org/licenses/by-nc-sa/3.0/).
Calibrex source snapshots and measurement code retain Apache-2.0.

Audit offline from the repository root:

```bash
python -m tools.benchmark_gnss_lever_arm_reuse \
  --audit-only docs/assets/gnss_lever_arm_evaluation_reuse
```
