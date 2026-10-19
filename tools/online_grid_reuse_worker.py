"""Time the unchanged full worker and individual online calls in frozen source.

Wrappers only bracket the original functions with a monotonic clock. No scoring,
input, result, counter, cache or solver option is substituted.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# The base worker selects the frozen package before importing Calibrex.
import online_edge_cuda_worker as worker


def main() -> None:
    folder = Path(sys.argv[4]).resolve()
    protocol = json.loads((folder.parent / "protocol.json").read_bytes())
    own_sha = worker.sha(Path(__file__))
    if protocol["toolchain_sha256"]["tools/online_grid_reuse_worker.py"] != own_sha:
        raise ValueError("timing wrapper differs from frozen toolchain")
    solve, step = worker.online.LevinsonThrunOnlineSolver.solve, worker.online._evaluate_online_step
    calls, updates = [], []

    def timed_solve(*args, **kwargs):
        start = time.perf_counter()
        result = solve(*args, **kwargs)
        calls.append(time.perf_counter() - start)
        return result

    def timed_step(*args, **kwargs):
        start = time.perf_counter()
        result = step(*args, **kwargs)
        updates.append({"frame_id": result[0].frame_id, "seconds": time.perf_counter() - start})
        return result

    worker.online.LevinsonThrunOnlineSolver.solve = timed_solve
    worker.online._evaluate_online_step = timed_step
    try:
        worker.main()
    finally:
        worker.online.LevinsonThrunOnlineSolver.solve = solve
        worker.online._evaluate_online_step = step
    measurement = json.loads((folder / "measurement.json").read_bytes())
    value = {
        "solve_seconds": calls,
        "updates": updates,
        "measurement_sha256": worker.sha(folder / "measurement.json"),
        "provenance": {
            **measurement["provenance"],
            "timing_wrapper_sha256": own_sha,
            "scope": (
                "Original solve API including first projection JIT and all diagnostics; "
                "each online update includes center, 728 candidates and selection. "
                "The first update is retained. No warmup is removed."
            ),
        },
    }
    (folder / "timing.json").write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
