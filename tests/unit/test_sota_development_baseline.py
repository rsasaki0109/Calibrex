"""Archived claim replay is distinct from sensor-data evaluation."""
import jsonschema
from tools.sota_development_baseline import ROOT, BaselineReport, build_report, sha

from calibrex.core.sota_audit import SotaAuditResult


def test_archived_verdicts_and_snapshot_bindings_are_reproducible():
    report = build_report()
    assert report.audit_replays
    assert all(row['matches_archive'] for row in report.audit_replays)
    for row in report.audit_replays:
        result = SotaAuditResult.model_validate(row['result'])
        assert result.verdict == row['verdict']
        assert sha(ROOT / row['protocol']) == row['protocol_sha256']
    assert report.camera_lidar_inputs
    for row in report.camera_lidar_inputs:
        path = ROOT / row['snapshot']
        assert sha(path) == row['snapshot_sha256']
        assert row['arrays_present_beside_snapshot'] == (path.parent / 'arrays.npz').is_file()
    assert report.provenance['sensor_data_read'] is False
    assert report.provenance['new_evaluation_started'] is False
    jsonschema.validate(report.model_dump(mode='json'), BaselineReport.model_json_schema())
