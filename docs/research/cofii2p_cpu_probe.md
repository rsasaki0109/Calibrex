# CPU-only CoFiI2P execution probe

The user confirmed no GPU environment is available. CPU-only operation is now a
constraint, not a pending GPU request. Existing MI search/multistart negatives and
stereo-surface failures remain closed; this probe checks an actual model execution
path rather than repeating those searches.

The [previous geometry bridge](cofii2p_model_bridge.md) has not yet established
full model CPU execution. This bounded experiment imports the pinned external
CoFiI2P model in an isolated source copy, supplies synthetic CPU geometry and random
seeded weights, and executes its real `mode='test'` path with all label inputs
`None`. No learned checkpoint or real frame is used. Success is an execution
result, not official CPU/GPU parity, usable learned correspondences or accuracy.

## Frozen changes and inputs

`tools/cofii2p_cpu_probe.py` accepts the reviewed external source directory, a
geometry-bridge NPZ and a new output directory. It refuses an upstream revision
other than `ed90edf51cca540ee76152d91133c7e79b63256c` or a dirty tracked tree, then
copies it outside Calibrex core. Only two external source files are patched:

- `model/network.py`: four active `.cuda()` calls become transfers to the relevant
  input's device. The unused Open3D import is removed. The predicted-pair threshold
  loop fails explicitly if it passes its score domain without obtaining four
  pairs, rather than looping forever on untrained features. No pairs are invented.
- `model/kpconv/kernel_points.py`: retain native NumPy kernel generation, noise and
  rotation, replacing only Open3D PLY cache I/O with NumPy cache I/O. Remove unused
  matplotlib/Open3D imports. Debug plotting and alternate Open3D helper paths are
  not supported by this probe and are not called.

The existing SciPy bridge supplies points, mean intensity/normals and five KNN
levels. Its Open3D ordering/tie/grid differences remain explicit. A deterministic
random512×160 image is generated locally; prepared real-image parity is not
asserted. Options use the original image size, group normalization and original
model dimensions. The reduced case changes point count only, retaining128
neighbors. Only the small `einops==0.8.1` dependency was installed; existing CPU
torch2.4.0 is used. No TorchVision/Open3D/GPU package installation is needed for
this specific forward path.

## Protocol

Pin seed20261030 and two CPU threads. First run2048/1024/512/256/128 points. Only
if that completes, run the already generated standard bridge with20480/10240/
5120/2560/1280 points, same seed/thread count and code. Each process has a180s
ceiling. If it times out or fails, retain the error; do not adjust weights, seed,
model scores, image or matching rules to manufacture correspondence success.

Record source/driver/input hashes, random state hash, init/forward time, peak Linux
RSS, progress hooks, all eight output shapes, CPU device/finiteness, and exact
repeat-forward equality. The native test path chooses correspondences; no
ground-truth fine coordinates or reference pose are supplied. The source copy
and binary outputs remain under `/workspace/work/cofii2p-cpu-probe-20261030/`.
The [plan](../assets/cofii2p_cpu_probe_20261030/plan.json) freezes the implementation
before execution. No calibration fits, PnP accuracy scoring or real-data selection.

Even if successful, the official checkpoint, its terms/training/image provenance
and numerical parity remain unresolved. A random-weight CPU forward is not a
validated scientific reproduction or a successful calibration provider. It can
remove an assumed GPU requirement for the exercised model graph, not the need
for suitable learned weights.
