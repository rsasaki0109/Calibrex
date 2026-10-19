# Resident probabilistic refinement evidence

[comparison-v1.tar.gz](comparison-v1.tar.gz) (2,306,023 bytes)
contains all six fresh workers, 48 complete refinement artifacts, raw benchmark
trials/aggregates, measurements/logs, deduplicated frozen source snapshots,
launcher and fixture provenance. Earlier pilot protocols, all attempt records,
available summaries and launcher sources remain separate under `pilots/`.
Raw KITTI coordinates and generated correspondence arrays are not redistributed.

SHA256: `5f22b84de3fa0ebcfd378fd99c78a9c214101b35a708b7aec0d78c0a6e7cd702`.

| Implementation | Whole worker median | Artifact-backed API median |
| --- | ---: | ---: |
| Prior NumPy | 144.719 s | 138.427 s |
| NumPy with validated-input reuse | 123.017 s | 116.300 s |
| Resident CUDA with pose batches | 42.552 s | 36.796 s |

The whole-worker ratios are 3.401x against
prior NumPy and 2.891x against input-reusing
NumPy. Every quality/result field matches exactly, including complete traces,
acceptance, selected poses, holdout and bootstrap confidence intervals. Each
CUDA worker executes 2,568 logical pose evaluations in 236 kernel launches,
with no native fallback. The public-geometry control uses nine previously used
KITTI scans and generated image means; it provides execution/parity evidence,
not independent calibration accuracy. The host was unreserved.

Prior source: `88d5602f47bb906c0f0cb6419a2c24afbd29ecfe`.
Measured resident source: `348daf695e1e987d1773bd5b1f965c740781f0b5`.
The complete runtime modules are verified against their manifests in every
worker. The archive stores each unique module payload once, so replay requires
no Git history, GPU, CuPy, raw dataset, or code execution from the archive.

From the checkout, run the CPU-only audit:

```bash
python tools/audit_probabilistic_refinement_ablation.py \
  docs/assets/probabilistic_cuda_refinement/comparison-v1.tar.gz
```

Add `--verify-git-sources` in a checkout containing both pinned commits to
independently compare every source payload with Git. That check passed when
this archive was published. The audit verifies the complete inventory, input
and output bindings, source manifests/kernel identity, every complete result,
quality aggregate, retained worker/order and both timing ratios. It never
extracts or executes archived sources.

Only separately checked execution lineage, source commit, timestamps,
measured durations and previously verified output hashes are normalized.
No warmup or failed timing is removed from the final six-worker cohort.
See [the implementation and reproduction protocol](../../benchmarks/probabilistic_cuda_refinement.md).
