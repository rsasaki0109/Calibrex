# Public GPU spatial-query candidate measurements

The public GLIM Livox input SHA-256 is
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Measurement source commit:
`e47280c6689ea90a034e97f5289f13259d76440e`. The bound generator captures the sixth
initial odometry registration using the documented dense policy.

All 18 configurations retain exact distance/index parity. At leaf size 16,
the 17-neighbor query median is 15.470796 ms with sorted
insertion and 9.672925 ms with a max heap (1.599392x).
This is an exploratory warmed query-kernel observation, not a complete
calibration speedup. Runtime adapters remain unchanged.

Five GPU-event batches of ten launches follow the recorded fixed order. Capture,
CPU projection, tree construction, uploads, first compilation, PCA, registration
systems, pose updates, fitting and evaluation are excluded. This uses one
captured registration and does not reserve host load or provide extrinsic truth.

The index binds every timing sample, actual shader, generator, schema, publisher,
imported-source manifest and source archive. Local captured point arrays are
not redistributed; their hash and original bag hash are retained. Reproduce
them with the public bag and the bound generator.

```bash
python -m tools.benchmark_cuda_spatial_candidates --output outputs/query-candidates
python -m tools.audit_cuda_spatial_candidates docs/assets/cuda_spatial_candidates
```

See [scope, interpretation and reproduction](../../benchmarks/cuda_spatial_candidates.md).
