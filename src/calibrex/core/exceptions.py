"""Project-specific exceptions."""


class CalibrexError(Exception):
    """Base exception for user-facing Calibrex failures."""


class ConfigError(CalibrexError):
    """Raised when a calibration config is invalid."""


class ResultError(CalibrexError):
    """Raised when a result file is invalid."""


class FrameGraphError(CalibrexError):
    """Raised when a frame graph is malformed."""


class DatasetError(CalibrexError):
    """Raised when a dataset cannot be inspected or loaded."""
