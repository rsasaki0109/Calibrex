# Native TUM window processes

The complete public TUM fr1/xyz development workload takes
**66.494 → 37.735 seconds** at the median of
two alternating pairs, a **1.762x** improvement over the
depth-frontend implementation. This is one incremental whole-worker comparison;
ratios from earlier benchmark pages are not multiplied together.

Three independent windows are evaluated with a bounded process count. The caller
reuses the already validated primary problem and evaluates its complete window;
up to two child processes evaluate the other windows. The parent restores native
factors, aggregates windows in their configured order, and computes all cross-window
transfers and the final report.

## Enable it

Set the following option on the native TUM factor:

```yaml
pipeline:
  factors:
    tum_rgbd_joint_point_to_plane:
      options:
        window_workers: 3
```

The default remains `1`, preserving serial execution for existing callers and
custom in-process instrumentation. The effective count is bounded by the requested
count, available logical CPUs, replication windows and a maximum of three. A single
window uses the caller. The option adds no dependency and needs no caller-side
`if __name__ == "__main__"` guard on Windows.

## Complete timings

| Scope | Prior seconds | Current seconds | Prior/current |
| --- | ---: | ---: | ---: |
| Complete worker, including all child startup and transport | 66.494 | 37.735 | 1.762x |
| Complete calibration API latency | 63.375 | 34.845 | 1.819x |
| Summed optimizer time across processes | 18.699 | 23.563 | 0.794x |
| Summed paired reassociation time | 18.511 | 21.773 | 0.850x |
| Summed ray reassociation time | 10.744 | 12.452 | 0.863x |
| Summed rematch, curvature and multistart time | 3.205 | 3.325 | 0.964x |
| Summed time for all 75 depth reads | 0.632 | 0.765 | 0.826x |

The first two rows are wall-clock latency. Other rows sum timers from the parent
and children; those timers overlap and are not latency speedups. Each raw process
record is checked against its own elapsed time before totals are formed. Whole
worker time includes the parent process startup, imports, child startup, input
transport, evaluation, native factor restoration, trace writing and aggregation.

| Fresh process run | Whole seconds |
| --- | ---: |
| 01-current | 36.693 |
| 01-prior | 66.237 |
| 02-prior | 66.752 |
| 02-current | 38.777 |

The execution order is current/prior/prior/current. This is a small development
comparison on an unreserved host; it does not provide a confidence interval or a
hardware-independent speed guarantee. BLAS/OpenMP thread variables are pinned to
one in both variants. Child helpers inherit the environment and default unset
BLAS/OpenMP thread limits to one, without changing the caller's environment.

## Work and equality

Every complete run retains the same public inputs, three windows, 20-iteration
optimizer limit, sampling limits, local rotation updates, all ablations, grouped
holdouts, known-bad probes, reassociation, numerical curvature and multistart.

- All 75 decoded 640 × 480 depth images are hashed in full, including invalid and
  out-of-range pixels; all 75 ordered point sample arrays are hashed too.
- All 24 optimizer results, 168 full Jacobians and 1,624 residual arrays match
  exactly. All 1,792 original graph array evaluations remain accepted.
- All 405 paired reassociation states, 102 ray reassociation states and 991 shared
  holdout objectives match exactly, in their original order within each window.
- The original adapter's spatial/pair factor factory call counts match. Native
  factor restoration adds transport work, included in whole-worker time; these
  adapter counters do not count restoration in the transport module.
- Complete schema-valid results match after removing only run IDs, creation
  dates, independently verified Git IDs and the explicit window execution policy
  and worker option. Calibration grades, failed controls, warnings, ranks, pose
  histories and correspondence lineage remain in the comparison.

Only event arrival across windows is canonicalized. Events within each window are
never reordered, deduplicated or dropped. The published aggregate histories are
rebuilt from raw parent and child histories during every audit. Parent/child PIDs,
source imports, actual Git identity, wire fingerprints, solve timing and work counts
are checked separately. Child failures propagate with the window ID and stderr;
cancellation terminates only children owned by that call.

The private transport reconstructs native depth, centered ray, XYZ, paired and
Gaussian-prior callbacks from explicit measurements and variable ownership. It
rejects custom callbacks and altered ownership. Stored SE3 dataclass state crosses
the process boundary without a second quaternion normalization; prior whitening
matrices retain their original floating-point bits. Temporary pickle files are
internal process messages, not a user-supplied result format.

## Validation and provenance

The [validation record](../assets/tum_window_processes/validation.json) binds
Windows/Python 3.12 and Linux/Python 3.10 regression checks to the exact measured
source capsule. Tests cover byte-exact factor transport, independent process use
from a library caller, missing windows, a primary outside the replication list,
failure reporting, owned child cleanup and evidence tampering. Strict mypy, Ruff,
strict MkDocs and all 129 public schemas plus their ledger are checked.

The archived mypy check targeted Windows. Linux CI subsequently found that its
type definitions do not expose the Windows-only `subprocess.CREATE_NO_WINDOW`
constant. The [portability follow-up](../assets/tum_window_portability/validation.json)
uses an explicit `sys.platform` branch and checks both Linux and Windows type
definitions, plus 408 regression cases on each OS. Launch flags on those platforms
are unchanged. CI now checks both type targets. This follow-up adds no timing
claim and leaves the measured source capsule and all recorded timings intact.

The [initial protocol](../assets/tum_window_processes/initial-redirector-run/protocol.json)
completed four computations but failed its parent/child process ownership audit:
the Windows venv redirector inserted another process. Those
[attempts](../assets/tum_window_processes/initial-redirector-run/attempts.json), source
capsules and complete raw traces are preserved, with the
[rejection log](../assets/tum_window_processes/initial-redirector-run/audit-failure.log).
They do not contribute to the accepted timings above. The repaired launcher uses
CPython's direct-interpreter venv convention; the runtime now verifies both the
worker PID and parent PID before accepting its response. Tests also check that
termination owns the actual interpreter and retains its virtual environment.

- Prior runtime: `6c10b0542d279c6571bc80bf7bc5b3d502361e21`.
- Measured process runtime: `04fcb5d9b4fbb20e3f404d742e77af505b349736`.
- Portability follow-up: `a677482a1ff9248bb3be991d408a426aa73c00a6`.
- [Frozen protocol](../assets/tum_window_processes/protocol.json) and
  [all attempts](../assets/tum_window_processes/attempts.json).
- [Recomputed summary](../assets/tum_window_processes/summary.json) and
  [complete archive index](../assets/tum_window_processes/index.json).
- [Source manifests](../assets/tum_window_processes/sources-current-manifest.json),
  [input inventory](../assets/tum_window_processes/inputs.json), and
  [pinned launcher/auditor](../assets/tum_window_processes/launcher.py.txt).

Re-audit the complete evidence:

```powershell
python tools/benchmark_tum_window_processes.py --audit-only docs/assets/tum_window_processes
```

This is CPU execution work on already used development data. It makes no accuracy
or SOTA claim. The existing calibration failures and ambiguity remain visible.
