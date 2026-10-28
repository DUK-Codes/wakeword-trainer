"""wakeword-trainer — train openWakeWord-compatible models from Python.

Quick start::

    import wakeword_trainer as wwt

    wwt.train(
        word="hey computer",
        dataset="data/positives",
        output="models/hey-computer",
        negatives="auto",
    )

The first run without negatives needs either ``download_negatives=True``
(~17 GB) or an explicit negative source. All stages are also available as
separate functions; see :mod:`wakeword_trainer.api`.
"""

from __future__ import annotations

from .api import (
    PRESETS,
    WakeWordTrainer,
    detect,
    evaluate,
    export_model,
    inspect_dataset,
    predict,
    prepare_dataset,
    train,
)
from .audio.features import FeatureExtractor
from .config import (
    AudioConfig,
    AugmentConfig,
    EvalConfig,
    FeatureConfig,
    ModelConfig,
    NegativeConfig,
    TrainConfig,
    TrainerConfig,
)
from .data.dataset import Clip, DatasetReport
from .data.negatives import ACAV_URL, VALIDATION_URL, ensure_acav, ensure_validation_features
from .data.split import DatasetSplit, split_dataset
from .evaluation.evaluator import EvaluationResult, Evaluator
from .exceptions import (
    AudioFormatError,
    ConfigurationError,
    DatasetError,
    DownloadError,
    ExportError,
    FeatureExtractionError,
    NegativeDataError,
    TrainingCancelled,
    TTSError,
    WakeWordTrainerError,
)
from .models.inference import (
    DetectionResult,
    ModelMetadata,
    StreamingDetector,
    WakeWordDetector,
    load_model_metadata,
)
from .training.callbacks import Callback, CancelToken, TrainingEvent
from .training.trainer import Trainer, TrainingResult
from .version import __version__

__all__ = [
    "__version__",
    # high level
    "WakeWordTrainer",
    "train",
    "prepare_dataset",
    "inspect_dataset",
    "evaluate",
    "predict",
    "detect",
    "export_model",
    # configuration
    "TrainerConfig",
    "AudioConfig",
    "AugmentConfig",
    "EvalConfig",
    "FeatureConfig",
    "ModelConfig",
    "NegativeConfig",
    "TrainConfig",
    "PRESETS",
    # data
    "Clip",
    "DatasetReport",
    "DatasetSplit",
    "split_dataset",
    "FeatureExtractor",
    "ACAV_URL",
    "VALIDATION_URL",
    "ensure_acav",
    "ensure_validation_features",
    # training
    "Trainer",
    "TrainingResult",
    "Callback",
    "CancelToken",
    "TrainingEvent",
    # evaluation / inference
    "Evaluator",
    "EvaluationResult",
    "WakeWordDetector",
    "StreamingDetector",
    "DetectionResult",
    "ModelMetadata",
    "load_model_metadata",
    # errors
    "WakeWordTrainerError",
    "ConfigurationError",
    "DatasetError",
    "AudioFormatError",
    "FeatureExtractionError",
    "NegativeDataError",
    "DownloadError",
    "ExportError",
    "TTSError",
    "TrainingCancelled",
]
