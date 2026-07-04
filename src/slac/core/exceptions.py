"""Project-specific exceptions."""


class SlacError(Exception):
    """Base exception for user-facing slac failures."""


class ConfigError(SlacError):
    """Raised when a calibration config is invalid."""


class ResultError(SlacError):
    """Raised when a result file is invalid."""


class FrameGraphError(SlacError):
    """Raised when a frame graph is malformed."""


class DatasetError(SlacError):
    """Raised when a dataset cannot be inspected or loaded."""
