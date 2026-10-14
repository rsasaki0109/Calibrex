"""CPU-only regression checks for complete recorded calibration evidence."""
import math
from pathlib import Path

from tools.audit_actual_icp_archive import audit as audit_actual
from tools.audit_cuda_point_icp_archive import audit as audit_icp
from tools.audit_icp_neighbor_experiments import audit as audit_neighbor_experiments
from tools.audit_initial_projection_archive import _aggregate_equal
from tools.audit_initial_projection_archive import audit as audit_projection


def test_rejected_projection_archive_preserves_all_complete_models():
    report = audit_projection(Path("docs/assets/initial_lidar_projection_reuse"))
    assert report["formal_trials"] == 14
    assert report["complete_models"] == 17


def test_icp_archive_preserves_complete_results_and_failed_workers():
    report = audit_icp(Path("docs/assets/cuda_point_to_point_icp"))
    assert report["complete_models"] == 22
    assert report["retained_failures"] == 2


def test_only_derived_standard_deviation_accepts_one_ulp_runtime_rounding():
    value = 4.913989134574603
    neighbor = math.nextafter(value, 0.)
    original = {"distribution": {"standard_deviation": value}}
    assert _aggregate_equal({"distribution": {"standard_deviation": neighbor}}, original)
    assert not _aggregate_equal({"distribution": {"standard_deviation": math.nextafter(
        neighbor, 0.)}}, original)
    assert not _aggregate_equal({"distribution": {"mean": neighbor}},
                                {"distribution": {"mean": value}})


def test_actual_frame_archive_preserves_all_full_results_and_evaluation():
    report = audit_actual(Path("docs/assets/cuda_point_to_point_icp/actual_frames"))
    assert report["complete_models"] == 14
    assert report["cohorts"] == 5
def test_rejected_neighbor_experiments_preserve_complete_calibration_controls():
    report = audit_neighbor_experiments(Path("docs/assets/icp_neighbor_experiments"))
    assert report["complete_workers"] == 36
    assert report["full_result_equal"] is True
