# Solid-state LiDAR public benchmark v0.3

This report summarizes the checked public-data-only run. It is a
ground-truth-free temporal-holdout comparison, not an absolute extrinsic or
clock-accuracy result.

- Schema: `slac.solid_state_cross_dataset_benchmark/v0.2`
- Tool: `tools/run_solid_state_cross_dataset_benchmark.py` (`0.4.0`)
- Spec: `examples/public_datasets/solid_state_cross_dataset_benchmark_v03.yaml`
- Spec SHA-256: `4eca1e7668ddca4fdcb206daaf9d5e99870779ec0292dcc41d2ce64506531267`
- Provenance sources: `142`

## Summary

- Scored datasets: **3/3**
- Scored replicates: **27/27**
- Adaptive wins: **23/27** (`0.852`)
- Mean holdout improvement: **47.70%**
- Median holdout improvement: **65.94%**
- Bootstrap 95% CI: **[34.43, 59.99]%**

| Dataset | Replicates | Adaptive win rate | Mean improvement (95% CI) | Interpretation |
|---|---:|---:|---:|---|
| AgRob Modular-e Livox MID-70 ↔ RS-LiDAR | 9 | 0.556 | −1.34% (−10.16, 6.68) | mixed; retained counterexample |
| TIERS LidarsCali VLP-16 ↔ Livox | 9 | 1.000 | +78.76% (+76.00, 81.73) | adaptive wins all |
| AIST GLIM identity control | 9 | 1.000 | +65.67% (+64.18, 66.95) | adaptive wins all |

AgRob is not hidden or treated as a failure of the dataset: its result is
simply mixed across temporal boundaries and deterministic seeds. The result
supports the declared paired temporal-holdout protocol, not a universal SOTA
claim. The public fixtures do not provide independently surveyed extrinsic or
absolute clock ground truth.

## Reproduction

Run the four-replicate v0.2 pilot first, then extend it with the three-way,
three-seed v0.3 declaration:

```powershell
python tools/run_solid_state_benchmark_replicates.py `
  examples/public_datasets/solid_state_cross_dataset_benchmark.yaml `
  --output-root outputs/solid_state_benchmark_v02_replicates `
  --output-spec examples/public_datasets/solid_state_cross_dataset_benchmark_v02.yaml `
  --split-id middle_holdout --split-id late_holdout `
  --seed 0 --seed 17

python tools/run_solid_state_benchmark_replicates.py `
  examples/public_datasets/solid_state_cross_dataset_benchmark.yaml `
  --output-root outputs/solid_state_benchmark_v03_replicates `
  --output-spec examples/public_datasets/solid_state_cross_dataset_benchmark_v03.yaml `
  --append-spec examples/public_datasets/solid_state_cross_dataset_benchmark_v02.yaml `
  --split-id early_holdout --split-id middle_holdout --split-id late_holdout `
  --seed 0 --seed 17 --seed 42

python tools/run_solid_state_cross_dataset_benchmark.py `
  examples/public_datasets/solid_state_cross_dataset_benchmark_v03.yaml `
  --output outputs/solid_state_cross_dataset_benchmark_v03.yaml `
  --markdown-output outputs/solid_state_cross_dataset_benchmark_v03.md `
  --html-output outputs/solid_state_cross_dataset_benchmark_v03.html
```
