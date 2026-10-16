"""Cold-process checks for selective loading and unchanged public objects."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_solver_package_loads_no_solver_modules() -> None:
    _run("""
        import sys
        import calibrex.solvers as solvers
        assert not any(name.startswith('calibrex.solvers.') for name in sys.modules)
        assert set(solvers.__all__) <= set(dir(solvers))
        assert not hasattr(solvers, 'missing_solver')
    """)


def test_single_refiner_import_does_not_load_unrelated_solvers() -> None:
    _run("""
        import sys
        from calibrex.solvers import ProbabilisticCameraLidarRefiner
        from calibrex.solvers.probabilistic_camera_lidar_refiner import (
            ProbabilisticCameraLidarRefiner as Original,
        )
        assert ProbabilisticCameraLidarRefiner is Original
        assert 'calibrex.solvers.continuous_time_lidar_pair_solver' not in sys.modules
        assert 'calibrex.solvers.radar_ego_velocity_solver' not in sys.modules
    """)


def test_all_solver_exports_are_original_objects_and_support_star_import() -> None:
    _run("""
        import ast
        from importlib import import_module
        from pathlib import Path
        import calibrex.solvers as solvers
        tree = ast.parse(Path(solvers.__file__).read_text())
        checked = next(node for node in tree.body if isinstance(node, ast.If))
        expected = {
            alias.name: node.module for node in checked.body
            if isinstance(node, ast.ImportFrom) for alias in node.names
        }
        assert set(solvers.__all__) == set(expected)
        namespace = {}
        exec('from calibrex.solvers import *', namespace)
        for name, module in expected.items():
            original = getattr(import_module(module), name)
            assert getattr(solvers, name) is original
            assert namespace[name] is original
            assert vars(solvers)[name] is original
    """)


def test_solver_submodule_imports_still_work() -> None:
    _run("""
        from importlib import import_module
        from calibrex.solvers import scan_to_scan_odometry, numba_continuous_time_adapter
        assert scan_to_scan_odometry is import_module('calibrex.solvers.scan_to_scan_odometry')
        assert numba_continuous_time_adapter is import_module(
            'calibrex.solvers.numba_continuous_time_adapter'
        )
    """)


@pytest.mark.parametrize("operation", ["kinds", "entry", "detect"])
def test_schema_lookup_does_not_import_unrelated_artifact_models(operation: str) -> None:
    _run(f"""
        import sys
        import calibrex.core.schema_registry as registry
        if {operation!r} == 'kinds':
            assert 'probabilistic-refinement-result' in registry.validatable_kinds()
        elif {operation!r} == 'entry':
            entry = registry.schema_entry('probabilistic-refinement-result')
            assert entry.model is not None
        else:
            assert registry.kind_for_schema_version(
                'slac.probabilistic_refinement_result/v0.1'
            ) == 'probabilistic-refinement-result'
        assert 'calibrex.calibration_ci' not in sys.modules
        assert 'calibrex.core.sota_audit' not in sys.modules
        assert registry._ALL_ENTRIES is None
    """)


def test_schema_registry_tuple_and_entries_keep_identity_under_concurrent_loading() -> None:
    _run("""
        from concurrent.futures import ThreadPoolExecutor
        import calibrex.core.schema_registry as registry
        with ThreadPoolExecutor(max_workers=8) as pool:
            entries = list(pool.map(
                registry.schema_entry, ['probabilistic-refinement-result'] * 16
            ))
            tuples = list(pool.map(lambda _: registry.schema_entries(), range(16)))
        assert all(entry is entries[0] for entry in entries)
        assert all(value is tuples[0] for value in tuples)
        assert registry.SCHEMA_REGISTRY is tuples[0]
        assert next(e for e in tuples[0] if e.kind == entries[0].kind) is entries[0]
        assert 'SCHEMA_REGISTRY' in dir(registry)
        assert not hasattr(registry, 'missing_schema')
    """)


def test_versions_still_come_from_models_when_namespace_changes() -> None:
    _run("""
        from typing import Literal
        import calibrex.core.schema_registry as registry
        entry = registry.schema_entry('trajectory')
        field = entry.model.model_fields['schema_version']
        field.annotation = Literal['slac.new_trajectory_namespace/v9']
        assert registry.kind_for_schema_version(
            'slac.new_trajectory_namespace/v9'
        ) == 'trajectory'
    """)


def test_legacy_schema_model_exports_preserve_identity() -> None:
    _run("""
        from calibrex.core.schema_registry import CalibrationConfig, config_json_schema
        from calibrex.core.config import CalibrationConfig as Original, config_json_schema as schema
        assert CalibrationConfig is Original
        assert config_json_schema is schema
    """)


def test_cli_import_does_not_load_calibration_pipelines() -> None:
    _run("""
        import sys
        from typing import get_type_hints
        from calibrex.cli.main import main
        assert get_type_hints(main)['return'] is int
        assert 'calibrex.pipelines.online' not in sys.modules
        assert 'calibrex.calibration_ci' not in sys.modules
        assert 'calibrex.evaluation.camera_lidar_sota_audit' not in sys.modules
        assert 'calibrex.solvers' not in sys.modules
    """)


def test_schema_kind_choices_are_complete_without_loading_models() -> None:
    _run("""
        import calibrex.core.schema_registry as registry
        kinds = registry.schema_kinds()
        assert registry._ALL_ENTRIES is None
        assert 'koide-runner' in kinds
        assert 'koide-runner' not in registry.validatable_kinds()
        assert kinds == tuple(entry.kind for entry in registry.schema_entries())
    """)
