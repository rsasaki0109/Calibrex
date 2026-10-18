"""Camera-LiDAR development defaults, available without loading solvers."""

DEFAULT_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
)
DEFAULT_INITIALIZER_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
)
DEFAULT_INITIALIZER_RANSAC_THRESHOLDS_PX: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0)
DEFAULT_INITIALIZER_RANDOM_SEEDS: tuple[int, ...] = tuple(range(5))
