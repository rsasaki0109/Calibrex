# PNG luminance array evidence

The formal cohort contains six complete fresh workers on the existing 48-image
public KITTI fixture. Its separate six-worker pilot overlapped validation;
`pilot/measurement-context.json` records that limitation. No worker or warmup
is discarded within either cohort. Report the cohorts separately.

Both controls use the same installed Pillow and NumPy. The rows control includes
conversion into the array required by its consumer. Scope is PNG-to-array reads,
with parent process times recorded separately; calibration is excluded. Raw PNG
images and LiDAR scans are not redistributed. Input hashes are in each original
protocol. Each frozen source capsule contains the 401 measured Python modules.
Launcher bytes are preserved as text. The audit never executes either archive.

```bash
python -m tools.benchmark_png_array --audit docs/assets/png_array_decoding/formal
python -m tools.benchmark_png_array --audit docs/assets/png_array_decoding/pilot
```

`inventory.json` binds the exact archive file set and hashes. `validation.json`
records numerical, calibration and fixed-toolchain verification. See the
[scope and timings](../../benchmarks/png_luminance_decoding.md#array-consumers).
