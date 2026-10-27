"""Fixed fourteen-input development comparison with a same-input legacy-code control."""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy
from pydantic import Field
from tools import kitti_camera_lidar_axis_probe as conditional
from tools import kitti_camera_lidar_profile_probe as profile
from tools import kitti_camera_lidar_v3_replay as replay

from calibrex.core.provenance import git_commit
from calibrex.core.result import StrictModel

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'docs/assets/kitti_camera_lidar_v3/profiled_controls_multistart'
DRIVES = ('2011_09_26_0005', '2011_09_28_0002')


class ComparisonReport(StrictModel):
    schema_version: Literal['calibrex.camera_lidar_initialization_comparison/v0.1'] = (
        'calibrex.camera_lidar_initialization_comparison/v0.1')
    status: Literal['running', 'blocked', 'complete'] = 'running'
    provenance: dict[str, Any]
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: [
        'Reacquired arrays are not byte-identical to the original cache.',
        'Same-input code parity does not resolve historical cross-capture numerical drift.',
        '42 axis controls are correlated diagnostics, not independent recordings.',
        'Detection is not accuracy, native coverage, or SOTA support; reference is not truth.',
        'No fresh evaluation, threshold tuning, or estimator default changes.',
    ])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_module() -> Any:
    """Load only the repository's archived source, never caller-provided Python."""
    path = ARCHIVE / 'profile-source.py.txt'
    name = '_calibrex_archived_profile_control'
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def comparable(report: Any) -> dict[str, Any]:
    """Numerical evidence and preparation settings; exclude source/run identities."""
    return {'status': report.status, 'axes': [a.model_dump(mode='json') for a in report.axes],
            'preparation': {k: report.provenance[k] for k in
                            ('options', 'prepared_frames', 'dropped')}}


def parity_gate(rows: list[dict[str, Any]]) -> bool:
    """Require every fixed input, successful arms and the unchanged 1e-9 comparison."""
    expected = {(drive, run) for drive in DRIVES for run in replay.RUNS}
    return (len(rows) == len(expected)
            and {(r['drive'], r['run']) for r in rows} == expected
            and all(r.get('same_input_code_parity') is True for r in rows))


def compare(captures: Path, output: Path) -> ComparisonReport:
    """Run old/current pooled arms first; only then permit the training-only arm."""
    if (output.resolve().is_relative_to(captures.resolve())
            or captures.resolve().is_relative_to(output.resolve())):
        raise ValueError('output must be separate from captures')
    output.mkdir(parents=True, exist_ok=False)
    report = ComparisonReport(provenance={
        'created_at': datetime.now(timezone.utc).isoformat(), 'git_commit': git_commit(),
        'generator_sha256': sha(Path(__file__)), 'legacy_profile_sha256': sha(
            ARCHIVE / 'profile-source.py.txt'), 'profile_sha256': sha(Path(profile.__file__)),
        'conditional_sha256': sha(Path(conditional.__file__)),
        'estimator_sha256': sha(Path(profile.edge.__file__)),
        'replay_sha256': sha(Path(replay.__file__)), 'platform': platform.platform(),
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'absolute_parity_tolerance': 1e-9, 'relative_parity_tolerance': 0,
        'new_evaluation_started': False, 'new_accuracy_claim': False,
    })

    def write() -> None:
        (output / 'report.json').write_text(report.model_dump_json(indent=2) + '\n')

    (output / 'report.schema.json').write_text(
        json.dumps(ComparisonReport.model_json_schema(), indent=2) + '\n')
    write()
    archived = legacy_module()
    for drive in DRIVES:
        for run in replay.RUNS:
            row: dict[str, Any] = {'drive': drive, 'run': run, 'same_input_code_parity': False,
                                   'arms': {}, 'historical_profile_parity': None}
            report.inputs.append(row)
            snapshot = captures / drive / run
            dest = output / drive / run
            started = time.monotonic()
            try:
                meta, _, _, _, _ = replay.load_snapshot(snapshot)
                if (meta.drive, meta.run) != (drive, run):
                    raise ValueError('snapshot identity differs from the fixed input list')
                row.update(snapshot_sha256=sha(snapshot / 'snapshot.json'),
                           arrays_sha256=meta.arrays_sha256)
                previous = conditional.probe_snapshot(snapshot, dest / 'conditional')
                if previous.status != 'complete':
                    raise ValueError('conditional probe did not complete')
                results = {}
                for name, module in [('archived_pooled', archived), ('current_pooled', profile)]:
                    tick = time.monotonic()
                    result = module.profile_snapshot(
                        snapshot, dest / 'conditional/report.json', dest / name, multi_start=True)
                    results[name] = result
                    row['arms'][name] = {'elapsed_s': time.monotonic() - tick,
                                        'report_sha256': sha(dest / name / 'report.json')}
                row['same_input_code_parity'] = (
                    all(r.status == 'complete' for r in results.values())
                    and replay._same(comparable(results['archived_pooled']),
                                     comparable(results['current_pooled'])))
                historical = profile.ProfileReport.model_validate_json(
                    (ARCHIVE / drive / run / 'report.json').read_bytes())
                row['historical_profile_parity'] = replay._same(
                    comparable(historical), comparable(results['current_pooled']))
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            row['pooled_elapsed_s'] = time.monotonic() - started
            write()
            print(drive, run, 'code parity', row['same_input_code_parity'], flush=True)
    if not parity_gate(report.inputs):
        report.status = 'blocked'
        write()
        return report
    for row in report.inputs:
        drive, run = row['drive'], row['run']
        dest = output / drive / run / 'training_only'
        started = time.monotonic()
        try:
            result = profile.profile_snapshot(captures / drive / run, None, dest,
                                              multi_start=True, training_only_start=True)
            row['arms']['training_only'] = {'status': result.status,
                                            'report_sha256': sha(dest / 'report.json')}
        except Exception as exc:
            row['arms']['training_only'] = {'status': 'failed',
                                            'error': f'{type(exc).__name__}: {exc}'}
        row['arms']['training_only']['elapsed_s'] = time.monotonic() - started
        write()
        print(drive, run, 'training', row['arms']['training_only']['status'], flush=True)
    report.status = ('complete' if all(r['arms']['training_only']['status'] == 'complete'
                                     for r in report.inputs) else 'blocked')
    write()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if compare(args.captures, args.output).status == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
