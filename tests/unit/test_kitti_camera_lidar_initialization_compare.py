"""The same-input bridge cannot hide missing, duplicated, or failed controls."""
from copy import deepcopy

from tools import kitti_camera_lidar_initialization_compare as compare


def test_gate_requires_all_fourteen_successful_distinct_inputs() -> None:
    rows = [{'drive': drive, 'run': run, 'same_input_code_parity': True}
            for drive in compare.DRIVES for run in compare.replay.RUNS]
    assert compare.parity_gate(rows)
    assert not compare.parity_gate(rows[:-1])
    assert not compare.parity_gate([*rows[:-1], rows[0]])
    failed = deepcopy(rows)
    failed[0]['same_input_code_parity'] = False
    assert not compare.parity_gate(failed)


def test_archived_module_is_the_fixed_repository_capsule() -> None:
    archived = compare.legacy_module()
    assert archived.__file__ == str(compare.ARCHIVE / 'profile-source.py.txt')
    assert archived.ProfileFold.model_fields.keys() == (
        compare.profile.ProfileFold.model_fields.keys() - {'training_start_theta'})
