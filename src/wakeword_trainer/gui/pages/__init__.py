"""Desktop app pages (CustomTkinter)."""

from __future__ import annotations

from .base import BasePage
from .dataset_page import DatasetPage
from .results_page import ResultsPage
from .settings_page import SettingsPage
from .test_page import TestPage
from .train_page import TrainPage
from .tts_page import TTSPage

__all__ = [
    "BasePage",
    "DatasetPage",
    "ResultsPage",
    "SettingsPage",
    "TestPage",
    "TrainPage",
    "TTSPage",
]
