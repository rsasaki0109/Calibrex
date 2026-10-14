import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import jsonschema

source = Path("outputs/numba-spline-formal-v1")
target = Path("docs/assets/numba_bspline_planes")
summary = json.loads((source / "summary.json").read_bytes())
assert summary["quality_passed"] and len(summary["quality"]) == 6
target.mkdir(parents=True, exist_ok=True)
for name in ("protocol.json", "attempts.json", "summary.json"):
    shutil.copyfile(source / name, target / name)
for worker in sorted(source.glob("[0-9][0-9]-*")):
    out = target / worker.name
    out.mkdir(exist_ok=True)
    for name in ("measurement.json", "worker.log"):
        shutil.copyfile(worker / name, out / name)
for original, name in (
    (Path("tools/benchmark_bspline_planes.py"), "benchmark.py"),
    (Path("tools/audit_bspline_planes.py"), "audit.py"),
    (Path("tools/schemas/bspline_plane_fit_benchmark.schema.json"), "benchmark.schema.json"),
    (Path("tools/schemas/bspline_plane_fit_publication.schema.json"), "publication.schema.json"),
    (Path(__file__), "publish.py"),
):
    shutil.copyfile(original, target / name)
with tarfile.open(target / "sources.tar.gz", "w:gz") as bundle:
    for path in sorted((source / "sources").rglob("*")):
        if path.is_file():
            bundle.add(
                path, arcname=path.relative_to(source / "sources").as_posix(), recursive=False
            )

first = summary["medians"]["first_use"]
warm = summary["medians"]["warm"]
protocol = json.loads((target / "protocol.json").read_bytes())
commit = protocol["provenance"]["source_commit"]
table = (
    "| Scope | NumPy median | Compiled median | Observed speedup |\n"
    "| --- | ---: | ---: | ---: |\n"
    f"| Full fit, first use with empty JIT cache | {first['numpy']:.3f} s | "
    f"{first['numba_cpu']:.3f} s | {summary['speedups']['first_use']:.3f}x |\n"
    f"| Full fit after that worker's own prime | {warm['numpy']:.3f} s | "
    f"{warm['numba_cpu']:.3f} s | {summary['speedups']['warm']:.3f}x |"
)
readme = f"""# Fused CPU B-spline plane evaluation

Three alternating fresh-process NumPy/compiled pairs evaluate the same fixed
public GLIM-derived problem twice per worker. Each full fit includes all LM
iterations, sparse solves and marginal covariance. The first fit includes JIT
import and compilation; the second is the warm measurement.

{table}

All six complete fitted-output comparisons pass the predeclared `rtol=1e-8`,
`atol=1e-9` gates. Status, iterations, residual/parameter counts and calibration
parameter order agree exactly. All fits converge in 22 iterations. The complete
trajectory, six calibration blocks, costs, covariance and residual variance are
retained. This is floating-point agreement, not byte identity.

Measured source: `{commit}`. Both variants use this same frozen source tree and
different explicit backend options. Workers verify every imported Calibrex
source file against the archived manifest. BLAS, OMP and MKL use one thread;
the compiled plane evaluator uses four independent nogil row slices. Resources
were unreserved, so these ratios are observations for this workload.

The fixed input has 28,563 plane returns and 1,550 samples per IMU family. It was
captured locally from a public AIST GLIM Livox run; its extraction helper was
unversioned. Input NPZ and metadata hashes bind the consumed problem, and the
protocol identifies the original bag hash. Extraction and the full bag pipeline
are not reproduced by this cohort. Raw sensor inputs are not redistributed.
This archive provides no independent calibration truth or whole-bag speedup.

The small SO(3) workspaces are allocated once per row slice, rather than once
per return. Plane residuals and their 24 local Jacobian columns share the
rotation evaluation. IMU factors, sparse solving and covariance retain their
native implementations; sampling, iterations and calibration budgets are
unchanged. The default backend remains NumPy.

```bash
python -m tools.audit_bspline_planes docs/assets/numba_bspline_planes
```

The audit validates schemas and hashes, checks executed source bindings and
recomputes every quality comparison, median and reported ratio. See
[usage and reproduction](../../benchmarks/numba_bspline_planes.md).
"""
(target / "README.md").write_text(readme, encoding="utf-8")
page = f"""# Compiled CPU plane evaluation for LiDAR-IMU fitting

The joint cubic B-spline fitter has an optional `numba_cpu` plane evaluator.
It combines pose evaluation, whitened plane residuals and their 24 local
Jacobian columns in a compiled loop, reusing small workspaces within each row
slice. Independent slices can run in a persistent CPU thread pool. The kernel
uses FP64 arithmetic with `fastmath=False` and disables Numba's automatic
parallel pass. NumPy keeps the referenced-edge SO(3) logarithm and near-pi guard.

Install the existing optional dependency and select it explicitly:

```bash
pip install 'calibrex[numba]'
calibrex imu-lidar joint PATH_TO_BAG \\
  --lidar-topic /livox/points --imu-topic /livox/imu --acceleration-unit g \\
  --dataset-family glim_versatile --dataset-license 'upstream dataset terms' \\
  --fit-backend numba_cpu --fit-workers 4 --output calibration.json
```

The Python API accepts `BSplineLidarImuFitOptions(plane_backend="numba_cpu",
plane_workers=4)`. Small inputs use fewer workers to avoid thread overhead.
The fitter releases its pool when it finishes or fails. NumPy remains the
default and imports neither Numba nor CuPy. Bag artifacts retain the selected
backend, requested workers and actual runtime/version/arithmetic identity in
their existing schema-valid `options` field.

## Fixed public-input fit comparison

{table}

Three alternating fresh-process pairs use the same committed, frozen source
tree. Every worker first primes its own backend and then repeats the complete
fit. Each compiled worker begins with an empty private JIT cache. The fit
includes residuals, analytic Jacobians, LM, sparse solves and final covariance.
First-use and warm times are separate; compilation must be amortized for the
normal-run improvement. Host resources were unreserved.

All six comparisons preserve status and 22 iterations exactly and pass fixed
`rtol=1e-8`, `atol=1e-9` checks for all fitted arrays and numerical evidence.
Independent tests check every local control axis against finite differences,
knots, repeated/unsorted timestamps, large epochs, zero/small/near-pi rotations,
threaded repeatability and visible control edits. Both native and compiled
paths recover known calibration on synthetic data and pass the synthetic
rosbag CLI test with schema validation and held-out evidence.

The measured input is a locally saved public GLIM-derived fitting problem,
with 28,563 plane rows and 1,550 rows per IMU family. The extraction helper was
unversioned; the input hashes identify the exact consumed problem. These
numbers cover this fixed fitting stage, excluding bag decoding, odometry,
mapping and association. They establish neither a whole-bag speedup nor
independent extrinsic accuracy. The original sensor arrays remain external.
The [complete archive](../assets/numba_bspline_planes/README.md) contains the
protocol, all results, source snapshot, hashes and replayable audit.

```bash
python -m tools.benchmark_bspline_planes \\
  --problem outputs/glim-joint-analytic-profile-v1/problem-01 \\
  --workers 4 --repeats 3 --output outputs/compiled-plane-fit-comparison
python -m tools.audit_bspline_planes
```

The benchmark requires committed runtime sources and external `.npz`/`.json`
input pairs in its documented loader format. It hashes the inputs, verifies
actual frozen imports, saves complete fitted results and rejects numerical or
decision drift. It can also accept repeated `--problem` arguments to measure
several complete fits in each first-use/warm batch.
"""
Path("docs/benchmarks/numba_bspline_planes.md").write_text(page, encoding="utf-8")
files = {
    p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(target.rglob("*"))
    if p.is_file() and p.name != "index.json"
}
index = {
    "schema_version": "slac.bspline_plane_fit_publication/v0.1",
    "files": files,
    "provenance": {
        "generator": "publish.py",
        "source_commit": commit,
        "protocol_sha256": files["protocol.json"],
        "publisher_sha256": files["publish.py"],
        "scope": (
            "complete fixed captured public GLIM-derived fits; extraction unversioned; "
            "no whole-bag timing or independent accuracy claim"
        ),
    },
}
jsonschema.validate(index, json.loads((target / "publication.schema.json").read_bytes()))
(target / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "published_files": len(files),
            "first_use_speedup": summary["speedups"]["first_use"],
            "warm_speedup": summary["speedups"]["warm"],
        }
    )
)
