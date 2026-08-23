# Solid-state LiDAR public-data benchmark

This is the primary evaluation path when no hardware capture is available.
It uses public recordings only; no new sensor data or surveyed rig is
required.

Public recordings usually do not publish the independently surveyed
source/target extrinsic or an absolute clock reference. Therefore this
benchmark evaluates reproducible, ground-truth-free evidence. It does not
certify absolute extrinsic error, absolute time-offset error, or universal
SOTA performance.

## What the public gate can claim

The public-data gate can measure:

- temporal holdout RMSE under a frozen capture window, solver budget, and split;
- improvement or regression between declared solver variants;
- convergence, point-to-plane observability rank, correspondence support, and
  failure categories;
- sensitivity to declared known-bad timing or extrinsic perturbations;
- stability across datasets, holdout boundaries, and deterministic sampling
  seeds;
- whether point-time and deskew metadata are actually decoded and consumed.

It must not report those results as an absolute accuracy claim. In particular,
a trajectory, a map, a published calibration, or an identity control is not
the same thing as an independently surveyed extrinsic.

## Recommended public fixtures

| Fixture | Role | What it tests | Limitation |
|---|---|---|---|
| [AgRob Modular-e](https://zenodo.org/doi/10.5281/zenodo.8083431) | real Livox MID-70 ↔ RS-LiDAR pair | motion, deskew, adaptive support, temporal holdout | trajectory-only reference; no surveyed extrinsic |
| [TIERS LidarsCali](https://github.com/TIERS/tiers-lidars-dataset) | VLP-16 ↔ Livox Horizon/Avia multimodal bag | point time, ROS 1 replay, clock profile, holdout | no published absolute pair transform/clock truth for this protocol |
| AIST GLIM `glim_versatile` | Livox Avia duplicate-topic identity control | identity semantics, bounded replay, known-bad controls | not a second-sensor metrology reference |
| Livox Horizon-Horizon PCD sample | small smoke fixture | schema, geometry, report/GIF workflow | one pair; not an independent temporal or absolute-truth gate |

The checked cross-dataset declaration keeps these roles explicit with
`reference_mode`, `absolute_extrinsic_ground_truth`,
`independent_temporal_holdout`, and `identity_control` fields.

## 1. Validate the frozen declaration

Start from the repository root:

```bash
calibrex validate \
  examples/public_datasets/solid_state_cross_dataset_benchmark.yaml \
  --kind solid-state-cross-dataset-benchmark-config
```

Do not tune thresholds or select a favorable split after inspecting the
holdout result. Change the declaration only as a versioned protocol change.

## 2. Obtain only the public inputs you need

The small Livox PCD fixture can be downloaded through the Calibrex catalog:

```bash
python tools/download_public_dataset.py \
  livox_horizon_horizon_pcd_sample \
  --output-dir data/public
calibrex public-datasets show livox_horizon_horizon_pcd_sample --json
```

The larger fixtures are intentionally not downloaded automatically. Use the
official links in their manifests and retain the downloaded archive/file under
`data/public/`:

- `examples/public_datasets/agrob_modular_e/manifest.yaml` records the Zenodo
  archive, license, size, and checksum metadata;
- `examples/public_datasets/tiers_livox_lidars_cali/manifest.yaml` records the
  TIERS repository, SharePoint download, bag path, and expected size/checksum;
- `examples/public_datasets/glim_versatile/manifest.yaml` records the AIST
  Zenodo bag and trajectory archive metadata.

Before a long replay, inspect the downloaded source and validate its manifest.
The source archive itself is an input; it is not an accuracy reference.

## 3. Run the reproducible public comparison

The default checked run uses two temporal boundaries and two deterministic
sampling seeds per dataset (four scored replicates). It keeps the variants,
capture windows, temporal split, and solver budget paired:

```bash
python tools/run_solid_state_benchmark_replicates.py \
  examples/public_datasets/solid_state_cross_dataset_benchmark.yaml \
  --output-root outputs/solid_state_benchmark_v02_replicates \
  --output-spec examples/public_datasets/solid_state_cross_dataset_benchmark_v02.yaml \
  --split-id middle_holdout \
  --split-id late_holdout \
  --seed 0 \
  --seed 17

python tools/run_solid_state_cross_dataset_benchmark.py \
  examples/public_datasets/solid_state_cross_dataset_benchmark_v02.yaml \
  --output outputs/solid_state_cross_dataset_benchmark_v02.yaml \
  --markdown-output outputs/solid_state_cross_dataset_benchmark_v02.md \
  --html-output outputs/solid_state_cross_dataset_benchmark_v02.html

calibrex validate outputs/solid_state_cross_dataset_benchmark_v02.yaml \
  --kind solid-state-cross-dataset-benchmark
```

For a stronger public-only stress test, add `early_holdout` and seed `42`
after the four-replicate run is reproducible. Treat that as a new declared
benchmark version, not as an untracked tuning pass. The checked stronger run
is v0.3: it appends the existing v0.2 pilot and evaluates all three holdout
boundaries with seeds `0`, `17`, and `42`:

```bash
python tools/run_solid_state_benchmark_replicates.py \
  examples/public_datasets/solid_state_cross_dataset_benchmark.yaml \
  --output-root outputs/solid_state_benchmark_v03_replicates \
  --output-spec examples/public_datasets/solid_state_cross_dataset_benchmark_v03.yaml \
  --append-spec examples/public_datasets/solid_state_cross_dataset_benchmark_v02.yaml \
  --split-id early_holdout \
  --split-id middle_holdout \
  --split-id late_holdout \
  --seed 0 \
  --seed 17 \
  --seed 42

python tools/run_solid_state_cross_dataset_benchmark.py \
  examples/public_datasets/solid_state_cross_dataset_benchmark_v03.yaml \
  --output outputs/solid_state_cross_dataset_benchmark_v03.yaml \
  --markdown-output outputs/solid_state_cross_dataset_benchmark_v03.md \
  --html-output outputs/solid_state_cross_dataset_benchmark_v03.html

calibrex validate outputs/solid_state_cross_dataset_benchmark_v03.yaml \
  --kind solid-state-cross-dataset-benchmark
```

## 4. Read the result honestly

The report should retain, at minimum:

- per-dataset/per-replicate train and holdout RMSE;
- split and sampling seed;
- result/config/manifest SHA-256 values;
- observability rank and holdout correspondence count;
- convergence/failure category and missing-result status;
- adaptive-vs-uniform deltas and bootstrap intervals;
- the dataset reference label and its limitation.

The checked v0.3 public run is available as a compact
[provenance-bound report](../assets/solid-state-cross-dataset-benchmark-v03.md):
adaptive wins 23/27 scored replicates across all three datasets, with mean
holdout improvement **47.70%** and bootstrap 95% CI **[34.43, 59.99]%**.
AgRob wins are mixed at 5/9 with mean **−1.34%** and CI **[−10.16, 6.68]%**;
TIERS and GLIM each favor adaptive in 9/9 replicates. This is comparative
temporal-holdout evidence, not proof that adaptive has lower absolute
calibration error on every solid-state LiDAR.

## 5. Add a solver change safely

For every solver improvement:

1. freeze the public declaration and all dataset manifests;
2. run the same four-replicate baseline and candidate variants;
3. preserve failed or non-converged runs in the denominator;
4. inspect the known-bad and identity-control behavior;
5. compare the aggregate only after the per-dataset rows are reviewed;
6. update the protocol version if a split, budget, metric, or threshold changes.

The current default remains `adaptive_mad` with the uniform correspondence
fallback. A separate public AgRob isolation probe using adaptive voxelization
with MAD rejection disabled lost all 9 paired conditions (mean improvement
**−22.42%** versus `uniform_none`) and was not adopted. This is a diagnostic
decision, not a new holdout-tuned protocol variant.

The v0.4 candidate was selected from train-only diagnostics on AgRob. It fixes
`continuous_time_outlier_mad_scale=2.5`; the early, middle, and late seed-0
train probes all reduced train RMSE while retaining rank 6. The candidate is
declared in
`examples/public_datasets/solid_state_cross_dataset_benchmark_v04.yaml`, and
the selection table is recorded in
[the train-only note](../assets/solid-state-cross-dataset-benchmark-v04-train-selection.md).
The independent 3×3×3 v0.4 temporal-holdout matrix is now complete in the
[v0.4 benchmark report](../assets/solid-state-cross-dataset-benchmark-v04.md):
adaptive wins **25/27** replicates with mean improvement **57.20%** and
bootstrap 95% CI **[45.69, 67.43]%**. AgRob improves to 7/9 adaptive wins,
while TIERS and GLIM remain 9/9. The report retains nine `max_iterations`
variant failures under the declared `require_converged: false` policy, so the
result is a candidate-evidence upgrade with a follow-up budget-sensitivity
item, not an absolute accuracy or universal SOTA claim.

Never use the holdout score to choose a seed, crop, motion window, adaptive
fallback, or threshold and then call that same score an independent result.

## Physical gate status

The optional `solid_state_metrology_evaluation` packet remains in the schema
as a future path for users who later obtain independent spatial/temporal
measurements. It is not required for this public-data-only project and must
remain `planned` or `inconclusive` when those measurements are unavailable.
