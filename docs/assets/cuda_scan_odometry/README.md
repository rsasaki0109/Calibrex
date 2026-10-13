# Optional CUDA LiDAR scan odometry

Scan-to-map odometry keeps target neighbourhood PCA, point transforms, exact 
spatial queries, residuals, Jacobians and Huber systems on GPU. A fused 
projection/query/system kernel returns reduced quality totals and the 6x6 
system once per ordinary iteration. CPU retains scan decoding, range/voxel 
preparation, local-map construction, captured SciPy topology, six-dimensional 
updates, convergence and time-gap/deskew policies. Roundoff-sensitive 
queries, thin eigenspaces and close stopping decisions use native repairs.

CPU is the default. All scalar registration repairs and actual GPU query/system 
calls are recorded. The optional adapter uses FP64 and disables FMA in its 
query/system/covariance kernels; eigensolver arithmetic can differ.

The final source is `e75cb5545a0f54dbc922aa96834e522ae488edaf`. Each final workload 
uses 120 actual temporal scans from the public 
[AIST GLIM versatile recording](https://doi.org/10.5281/zenodo.6864654), 
a three-scan local map and a fixed 
12-second limit. All finite decoded rows are retained before the declared 
native range/voxel preprocessing. Each CPU/GPU pair uses identical inputs 
and options. This is an execution comparison, not independent trajectory 
accuracy or a two-sensor calibration benchmark.

## Complete fresh-process results

| Workload | Voxel | Complete process median | Complete replay API median |
|---|---:|---:|---:|
| dense | 0.03 m | **64.164142 → 21.092158 s (3.04x)** | **58.755552 → 16.283616 s (3.61x)** |
| standard | 0.1 m | **21.198051 → 15.866211 s (1.34x)** | **16.316816 → 8.112583 s (2.01x)** |

Each workload has five fresh CPU/CUDA process pairs with reversed order every 
other pair. Separate compiler caches start empty. Every first compilation, 
slow trial and unconverged registration is included. Process wall time includes 
imports, full stream decode, preprocessing, GPU setup/compilation, all 
registrations, input/source hashes, schema creation and saves. Parent integrity 
guards and benchmark aggregation are outside that timing. API timing includes 
decoding and GPU startup/preparation; it excludes final whole-file hashes/saves.

| Workload | Pair | CPU process s | CUDA process s |
|---|---:|---:|---:|
| dense | 1 | 54.348928 | 37.351310 |
| dense | 2 | 93.927549 | 17.372953 |
| dense | 3 | 64.164142 | 21.092158 |
| dense | 4 | 65.505675 | 25.739689 |
| dense | 5 | 51.840765 | 20.057819 |
| standard | 1 | 18.075286 | 24.368189 |
| standard | 2 | 27.641023 | 16.180802 |
| standard | 3 | 43.820594 | 15.866211 |
| standard | 4 | 21.198051 | 12.118008 |
| standard | 5 | 19.072051 | 12.886276 |

These are local descriptive timings on a shared host. CPU variation, cold 
startup and different point densities materially affect the speedup. The 
dense result must not be presented as a guarantee for default CLI settings.

## Geometry policy and exact comparison scope

The final CPU and CUDA runs both explicitly use `geometry_quantum_m=1e-9`: 
coordinates are canonicalized to a 1 nm grid during scan/local-map preparation. 
This makes spatial-tree tie decisions repeatable when preceding poses differ 
at floating-point roundoff. It is a numerical geometry policy, not an assertion 
of sensor accuracy. Its default is zero, which retains native floating geometry. 
Final speed comparisons use the same policy on both methods; they are not 
comparisons against an unquantized CPU trace.

All 120 poses and every registration transform, convergence decision, iteration 
count, correspondence count, RMSE and minimum Hessian eigenvalue are compared. 
Counts, booleans and iterations match exactly; every float uses abs `1e-9` / 
rel `1e-6`. Only verified execution identities differ. Full input-row hashes, 
timestamps and options match per pair. Unconverged steps remain in the result 
and timing; agreement does not turn them into successful registrations.

- dense: 115/119 converged steps; actual device work `{"gpu_fused_system_calls": 1267, "gpu_query_calls": 1386, "gpu_query_rows": 26122351, "scalar_normal_repairs": 1, "scalar_query_repairs": 1, "scalar_registration_repairs": 2}`.
- standard: 111/119 converged steps; actual device work `{"gpu_fused_system_calls": 1101, "gpu_query_calls": 1220, "gpu_query_rows": 8723977, "scalar_normal_repairs": 0, "scalar_query_repairs": 0, "scalar_registration_repairs": 0}`.

## Earlier attempts and rejected dense run

The preceding implementation (`a6f91e5318105181d6603530d5b2b53ea338c95a`) 
used a 0.1 m voxel and unquantized geometry. Its five complete process pairs 
were **25.013210 → 16.477359 s (1.52x)**. All results matched, 
but the whole-process improvement was too small for the intended workload. 
The `earlier.*` files preserve all ten results, sources and aggregates separately.

The first fused dense attempt (`8cc845cfdcfd41dac481caf429be31e4ab62448a`) 
ran five complete pairs with a 0.03 m voxel and unquantized geometry. Tiny 
preceding pose differences changed spatial-tree choices: later counts, 
iterations and poses diverged. It failed full parity and is not a successful 
speed result. All ten `rejected.*.measurement.json` files, measured sources, 
raw definition and full comparison errors are retained. The original tool 
also omitted a required failed-trial reason during aggregation; the final 
tool records it. No successful aggregate is invented for that rejected run.

The `earlier-attempts.json` file separately retains four exploratory paired 
replays, including failed boundary/eigenspace attempts and the successful 
canonicalization pilot. They ran with uncommitted working-tree source; recorded 
source hashes remain, but exact source snapshots were not archived. They are 
not pooled with the final committed-source comparisons.

## Use and reproduce

Install the optional `calibrex[cuda]` extra for the matching CUDA runtime.

```python
from calibrex.solvers.scan_to_scan_odometry import IncrementalScanOdometry, ScanOdometryOptions

options = ScanOdometryOptions(
    voxel_size_m=0.03, local_map_scans=3, geometry_quantum_m=1e-9,
)
odometry = IncrementalScanOdometry(options, backend="cuda")
# Feed each actual scan and its timestamp through odometry.add(...).
```

`register_point_to_plane`, `run_scan_to_scan_odometry` and `odometry_from_loader` 
also accept `backend="cuda"`. CPU caches passed into direct CUDA registration 
are rejected explicitly; the adapter captures its own target snapshot. CUDA 
supports 2–16 normal neighbours and a bounded retained-input estimate. Temporary 
device scratch/pool allocations are additional.

`calibrex lidar-lidar ros2 ... --odometry-backend cuda` selects GPU reference 
odometry. It can be combined with `--registration-backend cuda` for map 
calibration. Default CLI preprocessing and zero quantum remain unchanged; 
custom dense/canonicalized options are available through the Python API. 
Artifacts record the execution identity and full odometry options.

```console
python tools/benchmark_cuda_scan_odometry.py --bag <PUBLIC_BAG> --output-dir <NEW_OUTPUT> --seconds 12 --repetitions 5 --voxel-size .03 --geometry-quantum-m 1e-9
```

Repeat with `--voxel-size .1` for the standard workload. Outputs must be new 
or empty. The tool freezes source bytes, input bytes and HEAD for the replay.

`index.json` binds every published result/schema/source/definition/aggregate 
by SHA-256. Measured source archives match their declared Git blobs exactly. 
Measurements are unchanged copies of original worker files. Only 
`provenance.command` is made portable in published benchmark definitions and 
aggregates; original-file hashes and that changed field are recorded.
