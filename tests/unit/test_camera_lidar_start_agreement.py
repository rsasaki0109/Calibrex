import numpy as np
import pytest

from calibrex.evaluation import camera_lidar_edge as edge


def test_agreement_option_is_validated_and_omitted_when_unset():
    assert "start_agreement_deg" not in edge.EdgeAlignmentOptions().as_dict()
    assert (
        edge.EdgeAlignmentOptions(start_agreement_deg=0.5).as_dict()["start_agreement_deg"] == 0.5
    )
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="start_agreement_deg"):
            edge.EdgeAlignmentOptions(start_agreement_deg=bad)


def test_disagreement_compares_other_evidenced_optima_axis_by_axis():
    options = edge.EdgeAlignmentOptions(start_agreement_deg=0.5)
    final = np.array([-1.55, -3.05, -2.25, 0.0, 0.0, 0.0])
    single_pitch = np.array([0.0, -2.85, 0.0])
    # The 0002 pitch +3 trace: roll and yaw disagree beyond 0.5 deg, pitch agrees.
    spread = edge._start_disagreement(final, [single_pitch, final[:3].copy()], options)
    assert set(spread) == {0, 2}
    assert spread[0] == pytest.approx(1.55) and spread[2] == pytest.approx(2.25)
    assert edge._start_disagreement(final, [final[:3].copy()], options) == {}
    assert edge._start_disagreement(final, [single_pitch], edge.EdgeAlignmentOptions()) == {}
    close = final[:3] + np.array([0.5, -0.5, 0.49])
    assert edge._start_disagreement(final, [close], options) == {}
