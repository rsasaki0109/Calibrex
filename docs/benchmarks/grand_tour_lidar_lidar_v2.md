# GrandTour LiDAR–LiDAR audit v2

**Verdict — `refuted` (4 of 5 required gates).**

This pre-registered audit covers the Hesai XT-32 and Livox MID360 on four
previously unread GrandTour missions. The dense claim covers same-time scan
alignment and the reproducibility of judged rotation with an empirical
uncertainty floor. The sparse claim covers held-out controls on estimated
axes and the fraction of axes estimated. The sparse scan stream emulates an
L1-like point budget using MID360 data. These are consistency and observability
claims on one rig.

- [Pre-registration](grand_tour_lidar_lidar_v2_preregistration.yaml)
- [Frozen audit](../assets/grand_tour_lidar_lidar/v2_evaluation/grand-tour-lidar-lidar-v2-audit.yaml)
- [Audit protocol](../assets/grand_tour_lidar_lidar/v2_evaluation/grand-tour-lidar-lidar-v2-protocol.yaml)
- [Per-mission scores](../assets/grand_tour_lidar_lidar/v2_evaluation/scores.json)
- [Run log](../assets/grand_tour_lidar_lidar/v2_evaluation/runs.json)
- [Task A post hoc development analysis](../research/lidar_lidar_rotation_scatter.md)
- [GrandTour v1 audit](grand_tour_lidar_lidar.md)

## Frozen method and inputs

The code was frozen at `2471ae6659c0c3d9ec75aca3843f4b20d94685bd`. The pre-registration
commit `1eee59f23c498bf74f87401400fc0869c0e4af92` was pushed and verified before any evaluation
archive download. All missions were selected by date and pinned archive sizes
and SHA-256s from GrandTour revision `af6b6dd6f91696f8ec8c98134e195514834911b2` (MIT). The run verified
12 of the 12 archive pins and attempted 8
of the 8 planned fits.

Dense uses nearest-scan registration with map half window zero. The refit with
half window five contributes to each DoF's reported uncertainty. For the
between-mission rotation check, the audit applies the all-day Task A floor
of 0.16°, 0.03° and 0.40° about the Hesai x, y and z axes. It combines the
floored uncertainty of each fit with the propagated uncertainty of the
other missions' chordal mean.

Sparse uses the Unitree Go2 L1 preset with a Livox stream thinned to at most
2,000 points per scan within 20 m. The controls are 1° or 5 cm perturbations.
A sparse run may be inconclusive on individual axes. Each dropped axis
contributes zero to its six-axis coverage denominator.

## Results

| Gate | Observed | Threshold | Status |
| --- | ---: | ---: | --- |
| `dense-complete` | 0 | 0 | achieved |
| `dense-same-time` | 0.02841 | 0.055 | achieved |
| `dense-rotation-reproducibility` | 0.1872 | 1.5 | achieved |
| `sparse-honesty` | 0.75 | 0 | contradicted |
| `sparse-axis-coverage` | 1 | 0.5 | achieved |

| Mission | Dense fit | Same-time median (mm) | Rotation ratio | Sparse fit | Estimated sparse axes |
| --- | --- | ---: | ---: | --- | ---: |
| `2024-10-01-11-29-55` | pass | 25.01 | 0.187 | missing | — |
| `2024-11-15-12-06-03` | pass | 25.48 | 0.187 | missing | — |
| `2024-11-18-17-31-36` | pass | 17.73 | 0.073 | pass | 6/6 |
| `2024-12-03-13-26-40` | pass | 28.41 | 0.078 | missing | — |

## Execution exceptions

- `2024-10-01-11-29-55:calibrex_sparse` — exit code -9
- `2024-11-15-12-06-03:calibrex_sparse` — exit code -9
- `2024-12-03-13-26-40:calibrex_sparse` — exit code -9

A dash denotes evidence unavailable under the frozen scoring rule. A missing
dense rotation prevents evaluating the peer mean. Same-time pose alignment
uses raw scans nearest in time. The reported uncertainty floor is an
empirical reproducibility allowance and its coverage is specific to this
audited claim.
