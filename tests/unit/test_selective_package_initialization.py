"""Cold-process public API contracts frozen from commit 08452aa1."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = {
    "calibrex": {
        "CalibrationConfig": "calibrex.core.config",
        "load_config": "calibrex.core.config",
        "CalibrationResult": "calibrex.core.result",
        "load_result": "calibrex.core.result",
        "SolidStateCaptureWindow": "calibrex.core.solid_state",
        "SolidStateEvaluationConfig": "calibrex.core.solid_state",
        "SolidStateLidarCalibrationContext": "calibrex.core.solid_state",
        "SolidStateLidarProfile": "calibrex.core.solid_state",
    },
    "calibrex.data": {
        "A2D2LidarDataset": "calibrex.data.a2d2",
        "read_a2d2_lidar_boundary_points": "calibrex.data.a2d2",
        "read_a2d2_lidar_points_reflectivity": "calibrex.data.a2d2",
        "DatasetReader": "calibrex.data.base",
        "StreamSummary": "calibrex.data.base",
        "TimestampedRecord": "calibrex.data.base",
        "ETHZ_HAND_EYE_COMMIT": "calibrex.data.ethz_hand_eye",
        "ETHZ_ROBOT_ARM_REAL_ARCHIVE": "calibrex.data.ethz_hand_eye",
        "ETHZ_ROBOT_ARM_REAL_SHA256": "calibrex.data.ethz_hand_eye",
        "ETHZ_ROBOT_ARM_REAL_URL": "calibrex.data.ethz_hand_eye",
        "AlignedHandEyePosePair": "calibrex.data.ethz_hand_eye",
        "HandEyeMotionDataset": "calibrex.data.ethz_hand_eye",
        "RelativeMotionPair": "calibrex.data.ethz_hand_eye",
        "TimestampedPose": "calibrex.data.ethz_hand_eye",
        "read_ethz_robot_arm_hand_eye_motions": "calibrex.data.ethz_hand_eye",
        "DatasetInspection": "calibrex.data.inspect",
        "inspect_dataset": "calibrex.data.inspect",
        "KITTIRawDataset": "calibrex.data.kitti",
        "LuminanceImage": "calibrex.data.kitti",
        "read_png_luminance": "calibrex.data.kitti",
        "KITTI_RAW_0005_BENCHMARK_FRAME_IDS": "calibrex.data.kitti_benchmark",
        "KITTIBenchmarkInputManifest": "calibrex.data.kitti_benchmark",
        "build_kitti_raw_0005_benchmark_input": "calibrex.data.kitti_benchmark",
        "load_kitti_benchmark_input": "calibrex.data.kitti_benchmark",
        "verify_kitti_benchmark_input": "calibrex.data.kitti_benchmark",
        "LivoxPCDDataset": "calibrex.data.livox",
        "DatasetManifest": "calibrex.data.manifest",
        "StreamManifest": "calibrex.data.manifest",
        "load_manifest": "calibrex.data.manifest",
        "NuScenesDataset": "calibrex.data.nuscenes",
        "PublicDatasetCatalog": "calibrex.data.public_datasets",
        "PublicDatasetEntry": "calibrex.data.public_datasets",
        "TUMDepthImage": "calibrex.data.tum_rgbd",
        "TUMDepthIntrinsics": "calibrex.data.tum_rgbd",
        "TUMDepthPoseEntry": "calibrex.data.tum_rgbd",
        "TUMRGBDDataset": "calibrex.data.tum_rgbd",
        "associate_depth_groundtruth": "calibrex.data.tum_rgbd",
        "read_tum_depth_png": "calibrex.data.tum_rgbd",
        "sample_tum_depth_points": "calibrex.data.tum_rgbd",
    },
    "calibrex.evaluation": {
        "evaluate_quality": "calibrex.evaluation.metrics",
        "MetricDefinition": "calibrex.evaluation.registry",
        "get_metric_definition": "calibrex.evaluation.registry",
        "list_metric_definitions": "calibrex.evaluation.registry",
        "register_metric": "calibrex.evaluation.registry",
        "MetricThreshold": "calibrex.evaluation.thresholds",
        "thresholds_for_profile": "calibrex.evaluation.thresholds",
    },
}


def run(code: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("module", EXPORTS)
def test_package_import_loads_no_scientific_libraries_or_child_modules(module: str) -> None:
    run(f"""
import sys
from importlib import import_module
package = import_module({module!r})
assert not {{'numpy','scipy','pydantic','yaml'}} & sys.modules.keys()
assert not any(n.startswith({module!r} + '.') for n in sys.modules)
assert set(package.__all__) <= set(dir(package))
assert not hasattr(package, 'missing_public_name')
""")


@pytest.mark.parametrize("module", EXPORTS)
def test_all_public_exports_preserve_original_objects_and_star_import(module: str) -> None:
    run(f"""
from importlib import import_module
package = import_module({module!r})
expected = {EXPORTS!r}[{module!r}]
assert set(package.__all__) == set(expected)
namespace = {{}}
exec('from ' + {module!r} + ' import *', namespace)
for name, owner in expected.items():
    original = getattr(import_module(owner), name)
    assert getattr(package, name) is original
    assert namespace[name] is original
    assert vars(package)[name] is original
""")


@pytest.mark.parametrize(
    "module,name",
    [
        ("calibrex", "CalibrationConfig"),
        ("calibrex.data", "TimestampedRecord"),
        ("calibrex.evaluation", "MetricDefinition"),
    ],
)
def test_concurrent_first_access_preserves_identity_and_pickling(module: str, name: str) -> None:
    run(f"""
import pickle
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
package = import_module({module!r})
with ThreadPoolExecutor(max_workers=8) as pool:
    values = list(pool.map(lambda _: getattr(package, {name!r}), range(32)))
original = getattr(import_module({EXPORTS[module][name]!r}), {name!r})
assert all(v is original for v in values)
assert pickle.loads(pickle.dumps(original)) is original
""")


def test_light_dataset_export_keeps_adapters_and_numeric_libraries_unloaded() -> None:
    run("""
import sys
from calibrex.data import TimestampedRecord
assert TimestampedRecord('sensor', 1).timestamp_ns == 1
assert not {'numpy','scipy','pydantic'} & sys.modules.keys()
assert 'calibrex.data.inspect' not in sys.modules
assert 'calibrex.data.nuscenes' not in sys.modules
""")


def test_metric_registration_loads_builtins_before_plugin_and_preserves_duplicates() -> None:
    run("""
from calibrex.evaluation import MetricDefinition, get_metric_definition
from calibrex.evaluation import register_metric, list_metric_definitions
assert get_metric_definition('schema_validation') is not None
item = MetricDefinition('selective_package_test', 'test metric')
assert register_metric(item) is item
assert get_metric_definition(item.name) is item
assert item in list_metric_definitions()
try:
    register_metric(item)
except ValueError:
    pass
else:
    raise AssertionError('duplicate registration must fail')
""")


def test_wheel_import_does_not_load_unrelated_dataset_and_evidence_pipelines() -> None:
    run("""
import sys
from importlib import import_module
from calibrex.evaluation import lidar_wheel
from calibrex.data import kitti, inspect
assert lidar_wheel is import_module('calibrex.evaluation.lidar_wheel')
assert kitti is import_module('calibrex.data.kitti')
assert inspect is import_module('calibrex.data.inspect')
assert 'calibrex.evaluation.metrics' not in sys.modules
assert 'calibrex.evaluation.evidence_summary' not in sys.modules
assert 'calibrex.evaluation.camera_lidar_sota_audit' not in sys.modules
""")
