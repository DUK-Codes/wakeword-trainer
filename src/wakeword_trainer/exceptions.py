"""Exceptions raised by :mod:`wakeword_trainer`."""

from __future__ import annotations


class WakeWordTrainerError(Exception):
    """Base class for all package-specific errors."""


class ConfigurationError(WakeWordTrainerError):
    """Raised when the supplied configuration is inconsistent or invalid."""


class DatasetError(WakeWordTrainerError):
    """Raised when a dataset is missing, empty, or cannot be read."""


class AudioFormatError(WakeWordTrainerError):
    """Raised when an audio file cannot be decoded."""


class FeatureExtractionError(WakeWordTrainerError):
    """Raised when the mel/embedding ONNX models cannot be run."""


class NegativeDataError(WakeWordTrainerError):
    """Raised when negative data is unavailable or malformed."""


class DownloadError(WakeWordTrainerError):
    """Raised when an optional download fails."""


class ExportError(WakeWordTrainerError):
    """Raised when a model cannot be exported."""


class TTSError(WakeWordTrainerError):
    """Raised by text-to-speech backends."""


class TrainingCancelled(WakeWordTrainerError):
    """Raised when a training run is cancelled through a callback."""
