"""CPU-only regression checks for complete recorded calibration evidence."""
from pathlib import Path

from tools.audit_cuda_point_icp_archive import audit as audit_icp
from tools.audit_initial_projection_archive import audit as audit_projection


def test_rejected_projection_archive_preserves_all_complete_models():
    report = audit_projection(Path("docs/assets/initial_lidar_projection_reuse"))
    assert report["formal_trials"] == 14
    assert report["complete_models"] == 17


def test_icp_archive_preserves_complete_results_and_failed_workers():
    report = audit_icp(Path("docs/assets/cuda_point_to_point_icp"))
    assert report["complete_models"] == 22
    assert report["retained_failures"] == 2
