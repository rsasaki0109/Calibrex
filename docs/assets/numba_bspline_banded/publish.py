from pathlib import Path
import argparse, hashlib, json, shutil, tarfile
import jsonschema

parser=argparse.ArgumentParser()
parser.add_argument('--source',default='outputs/numba-banded-spline-formal-v1')
parser.add_argument('--output',default='docs/assets/numba_bspline_banded')
args=parser.parse_args()
source=Path(args.source);target=Path(args.output)
summary=json.loads((source/'summary.json').read_bytes())
assert summary['quality_passed'] and len(summary['quality'])==6
target.mkdir(parents=True,exist_ok=True)
for name in ('protocol.json','attempts.json','summary.json'):
    shutil.copyfile(source/name,target/name)
for folder in sorted(source.glob('[0-9][0-9]-*')):
    out=target/folder.name;out.mkdir(exist_ok=True)
    for name in ('measurement.json','worker.log'):shutil.copyfile(folder/name,out/name)
for origin,name in ((Path('tools/benchmark_bspline_planes.py'),'benchmark.py'),
                    (Path('tools/audit_bspline_planes.py'),'audit.py'),
                    (Path('tools/schemas/bspline_plane_fit_benchmark.schema.json'),'benchmark.schema.json'),
                    (Path('tools/schemas/bspline_plane_fit_publication.schema.json'),'publication.schema.json'),
                    (Path(__file__),'publish.py')):
    shutil.copyfile(origin,target/name)
with tarfile.open(target/'sources.tar.gz','w:gz') as bundle:
    bundle.add('LICENSE',arcname='LICENSE',recursive=False)
    for path in sorted((source/'sources').rglob('*')):
        if path.is_file():bundle.add(path,arcname=path.relative_to(source/'sources').as_posix(),recursive=False)
protocol=json.loads((target/'protocol.json').read_bytes())
assert hashlib.sha256((target/'benchmark.py').read_bytes()).hexdigest()==protocol['provenance']['launcher_sha256']
commit=protocol['provenance']['source_commit']
table='| Complete fit scope | NumPy median | Structured compiled median | Observed ratio |\n| --- | ---: | ---: | ---: |\n'
for phase,label in (('first_use','First use, empty JIT cache'),('warm',"After the worker's own first fit")):
    values=summary['medians'][phase]
    table+=f"| {label} | {values['numpy']:.3f} s | {values['numba_cpu']:.3f} s | {summary['speedups'][phase]:.3f}x |\n"
readme=f'''# Structured CPU spline normal systems

Three alternating fresh-process pairs use the same frozen runtime v0.3 source
(`{commit}`) and the same captured public GLIM-derived fitting problem.
Every worker performs a complete first fit and then a complete warm fit.
All residuals, LM iterations, sparse solves and final marginal covariance are
included. NumPy remains the default; `numba_cpu` selects the compiled path.

{table}
All six complete fitted-output comparisons pass the predeclared `rtol=1e-8`,
`atol=1e-9` gates. Status, 22 iterations, dimensions and parameter order agree
exactly. Every trajectory, calibration block, covariance, cost and residual
variance is retained. Equality is numerical, rather than byte identity.

Plane normal products are accumulated in seven control blocks, using the
existing four-worker pool. IMU products use a serial compiled kernel and
preserve control/calibration cross terms. Aligned IMU samples share states,
control derivatives and clock probes within each linearization. The final
covariance uses the complete assembled Jacobian and original Schur calculation.
FP64 arithmetic, `fastmath=false`, samples and iteration budgets are unchanged.

The fixed problem has 28,563 plane returns and 1,550 rows per IMU family.
Its extraction helper was unversioned. Input hashes identify the exact consumed
NPZ/metadata; the protocol names the original public bag hash. The raw arrays
are external. This cohort measures fixed fitting, excluding bag decoding,
odometry, mapping, association and evidence controls. It provides neither
whole-bag acceleration nor independent extrinsic accuracy evidence.

Workers verify imported source bytes and use one BLAS/OMP/MKL thread.
Each compiled worker starts with an empty private Numba cache. Host resources
were unreserved and timings vary substantially between pairs. The archive
retains every attempt, including that variation. These observations must not
be multiplied by ratios from earlier cohorts to infer an incremental gain.

```bash
python -m tools.audit_bspline_planes docs/assets/numba_bspline_banded
```

The CPU-only audit verifies schemas, byte/source bindings, backend versions,
complete decisions/numeric arrays and every median/ratio.
See [usage and limits](../../benchmarks/numba_bspline_planes.md).
'''
(target/'README.md').write_text(readme,encoding='utf-8')
files={p.relative_to(target).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted(target.rglob('*')) if p.is_file() and p!=target/'index.json'}
index={'schema_version':'slac.bspline_plane_fit_publication/v0.1','files':files,
       'provenance':{'generator':'publish.py','source_commit':commit,'protocol_sha256':files['protocol.json'],
                     'publisher_sha256':files['publish.py'],'scope':'complete fixed captured public GLIM fits; private empty-cache first use and warm; extraction unversioned; no whole-bag or independent accuracy claim'}}
jsonschema.validate(index,json.loads((target/'publication.schema.json').read_bytes()))
(target/'index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'published_files':len(files),'speedups':summary['speedups']}))
