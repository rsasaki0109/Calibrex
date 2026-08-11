from pathlib import Path

from tools.generate_kitti_lidar_camera_demo_fixture import generate_fixture


def test_bundled_fixture_matches_deterministic_generator(tmp_path: Path) -> None:
    generated_root = tmp_path / "fixture"
    generation = generate_fixture(generated_root)
    bundled_root = Path("examples/public_datasets/kitti_lidar_camera_evidence")
    packaged_root = Path("src/calibrex/resources/kitti_lidar_camera_evidence")

    files = generation["files"]
    assert isinstance(files, list)
    assert len(files) == 11
    for record in files:
        assert isinstance(record, dict)
        relative_path = record["path"]
        assert isinstance(relative_path, str)
        assert (generated_root / relative_path).read_bytes() == (
            bundled_root / relative_path
        ).read_bytes()
        assert (generated_root / relative_path).read_bytes() == (
            packaged_root / relative_path
        ).read_bytes()
