# Public FeatDepth CPU/CUDA inference evidence

The external FeatDepth runner now supports FP32 batches and bounded parallel PNG
loading, and reads/hashes the checkpoint once from the same bytes that PyTorch
decodes. Its checkpoint reference is reused in the input/provider manifests.
`--device cuda` already existed; the new batch/prefetch options and checkpoint
reuse improve that execution path. CPU remains the default. The large CPU/GPU
ratios below describe moving depth inference to GPU, rather than the incremental
effect of batching alone.

The measured implementation commit is
`0b67865a52fbc17651ee02e96905a83bb54e416e`. Runner SHA256:
`df13e2f39b9711a4112cc0d830b52b8fa2181f8e8028a5b97526c2306ae4b552`.
Actual measured bytes match the committed LF source. The pinned external
FeatDepth commit is `550420b3fb51a027549716b74c6fbce41651d3a5`; checkpoint SHA256
is `559f168c762d01af7c745b32ced8930a12423a0e9b62af791f7d894bd902af27`.
Images, weights, LiDAR scans and generated NumPy arrays remain external.

## Workload and execution

KITTI raw `2011_09_30_drive_0018_sync/image_02`, the same 25 uniformly selected
frames as the public six-DoF demo, 1226 x 370 source images, 1024 x 320 network
inputs. Raw inverse-depth/disparity is returned to each original image grid.
The i7-9750H / GTX 1660 Ti 6 GiB / 32 GiB Windows host was shared with unrelated
user work. Environment: Python 3.12.10, NumPy 2.3.4, Pillow 12.3.0,
PyTorch 2.11.0+cu128, CUDA runtime 12.8. GPU FP32 settings retain the existing
cuDNN defaults, including `benchmark=false`. Training overlap with KITTI raw
is recorded as possible; no online refinement is performed.

```powershell
python tools/benchmark_featdepth_provider.py `
  E:/datasets/calibrex_camera_lidar_v06/kitti_raw/2011_09_30/2011_09_30_drive_0018_sync `
  --featdepth-repository E:/datasets/calibrex_camera_lidar_v06/providers/FeatDepth-git `
  --checkpoint E:/datasets/calibrex_camera_lidar_v06/providers/checkpoints/fm_depth.pth `
  --output-dir outputs/public-kitti-featdepth-batching-v3 --cpu-threads 2 --repetitions 3
```

All nine subprocesses are fresh, with no separate warmup. Order is CPU, serial
CUDA, batched CUDA on odd repetitions and reversed on even repetitions. OS file
caches persist across runs. Every image/checkpoint/depth reference is verified
after each subprocess. Startup, model setup, inference, depth export, primary
manifest and optional timing-manifest writes are included in subprocess wall;
parent verification/comparison and benchmark aggregation are excluded.

| Repetition | CPU, 2 threads (s) | Serial CUDA (s) | CUDA, batch 2 + 2 PNG workers (s) |
| --- | ---: | ---: | ---: |
| 1 | 117.347018 | 13.176117 | 12.458591 |
| 2 | 123.940307 | 13.372894 | 13.078349 |
| 3 | 132.531623 | 22.416443 | 18.169170 |
| Median | 123.940307 | 13.372894 | 13.078349 |

The CPU-2-thread / batched-CUDA wall median ratio is **9.48x**. Within the
already accelerated CUDA path, the wall median ratio is only **1.02x** here.
Inference-plus-export medians are **113.640095 / 2.730953 / 2.345925 s** for
CPU / serial CUDA / batched CUDA: **48.44x** CPU-to-batched-CUDA and **1.16x**
serial-to-batched-CUDA. Short clips are dominated by startup/model preparation.
The prediction-batch timer is a subset of inference/export and must not be added
again. Main-entry timing excludes interpreter/import startup and the timing
manifest write; subprocess wall includes both.

A separate one-repetition execution omits `--cpu-threads`, giving PyTorch's
default six CPU threads on this host. It measures **89.481593 / 18.528891 /
17.824680 s** for CPU / serial CUDA / batched CUDA, an observed **5.02x**
CPU-to-batched-CUDA ratio. See `default_cpu.definition.json` and
`default_cpu.benchmark.json`. This single pair has different concurrent load;
do not mix it into the three-repetition median or infer CPU-thread scaling.

Three repeated batched-CUDA runs have identical per-frame depth-file hashes.
Compared with the same-repetition CPU output, all predictions are finite;
maximum absolute disparity difference is **6.139279e-6**, maximum relative
difference **3.037236e-5** (denominator floor 1e-12), and pixel-weighted RMSE
**1.436477e-7**. Those differences do not alone establish calibration equality.

The batched run records peak Torch GPU allocated/reserved memory of
**675,414,528 / 1,027,604,480 bytes**, versus serial CUDA
**406,979,072 / 599,785,472 bytes**. These process counters exclude host memory
and other GPU users. At batch 2/workers 2, at most six image tasks are queued
alongside the current two-image batch. Decode buffers, outputs and model memory
are additional. Checkpoint loading retains one approximately 630 MiB byte
buffer alongside deserialized weights, then releases it.

## Six fixed initializations

`validation.definition.json` and `validation.benchmark.json` record twenty-four
fresh six-DoF estimates: two-thread CPU, default six-thread CPU, serial CUDA and
batched CUDA depth for each frozen x+/x-/y+/y-/z+/z- problem. Initial rotation and
translation perturbations remain +/-0.5 degrees and +/-0.5 m on the same axis.
Bounds remain +/-2 degrees / +/-1 m and the evaluation cap is 800. Twenty fit
frames resolve histogram ranges; the same five holdout frames never enter
fitting or range selection. The D2D solver/backend is unchanged.

All twenty-four estimates converge. Every GPU-depth estimate has exactly the same
final translation/quaternion components and evaluation count as the same-version
CPU-depth estimate. Full numerical payloads are **not** exactly equal: learned
disparity differences change some histogram counts, objective/trace scores and
resolved ranges. Final holdout MI differs only in y-positive, by
**6.2001394e-7 nat** for both GPU-depth modes. Final poses and reference errors
remain identical across the four depth-generation modes in all six cases.

The initial eighteen trials compare two-thread CPU with both GPU modes. Six
default-six-thread CPU-depth estimates were added after detecting thread-count
differences in saved depth bytes. Their maximum absolute disparity difference
from two-thread CPU is 1.192093e-7, RMSE 1.820104e-10. Their complete calibration
numerical payloads match the two-thread CPU anchor in all six cases. Priors,
split, bounds and stopping rules are unchanged; no model or holdout tuning was
performed for this supplementary check.

| Case | Evaluations | Rotation reference error (deg) | Translation reference error (m) | CPU final holdout MI (nat) | GPU holdout MI absolute difference (nat) |
| --- | ---: | ---: | ---: | ---: | ---: |
| x-positive | 229 | 0.390923 | 0.160078 | 1.293494542 | 0 |
| x-negative | 313 | 0.410874 | 0.104583 | 1.287191168 | 0 |
| y-positive | 277 | 0.527379 | 0.167705 | 1.294100526 | 6.2001394e-7 |
| y-negative | 241 | 0.266233 | 0.091001 | 1.281462561 | 0 |
| z-positive | 265 | 0.258640 | 0.087052 | 1.283677673 | 0 |
| z-negative | 241 | 0.285129 | 0.089486 | 1.281674508 | 0 |

Prior provider artifacts used another PyTorch version, so the comparison anchor
is newly generated CPU FP32 depth from the same environment as the GPU modes.
The six historical priors are preserved exactly in their transform fields.
Reference values are never optimized against. The y-positive reference rotation
error remains greater than its initial 0.5 degrees. Convergence and unchanged
poses do not establish calibration accuracy: covariance/observability are
unknown, scans are not motion compensated, and training overlap is possible.
All twenty-four full result traces and their problem templates are published.

## Published artifacts and provenance

`inference.definition.json` and `inference.benchmark.json` retain every timing,
command, execution configuration, source/provider/timing-file digest and map
difference. First-repetition provider/timing manifests retain all 25 input/output
references, environment versions, checkpoint identity and execution metadata.

Published paths use `/external/calibrex-public-inputs` and `/workspace/Calibrex`.
Parsed path strings and embedded JSON trial notes are normalized; numerical
payloads, original source/output SHA256 values and timings are retained.
`publication.depth-provider.json` is a schema-valid manifest containing original
and published byte hashes for every evidence artifact. Trial output hashes bind
original local files, not their normalized publications. Published problems are
evidence templates: restore external paths and bind the chosen provider's actual
file hash before executing them. External arrays/checkpoints are not embedded.

The manual validation and publication launchers are archived byte-for-byte as
`validation-launcher.py.txt`, `default-cpu-validation-launcher.py.txt` and
`publication-launcher.py.txt`, with their SHA256
values in the publication manifest. They use the frozen local source-problem
directories and new inference outputs named above; restore those inputs and use
new/empty output directories when reproducing the experiment. The reusable
inference benchmark tool is tracked under `tools/`.

These observations describe this workload and shared host. Repeated frames do
not provide independent accuracy trials, confidence intervals or portable
latency guarantees. See the [usage guide](../../tutorials/public_six_dof_demo.md)
for GPU/CPU commands and timing scope.
