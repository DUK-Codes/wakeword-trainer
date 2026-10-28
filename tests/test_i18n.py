"""Bilingual string table."""

from __future__ import annotations

import re
from pathlib import Path

from wakeword_trainer.gui.i18n import DEFAULT_LANGUAGE, LANGUAGES, STRINGS, Translator, translate


def test_every_string_has_both_languages():
    for key, entry in STRINGS.items():
        for language in LANGUAGES:
            assert language in entry, f"{key} is missing {language}"
            assert entry[language].strip(), f"{key}/{language} is empty"


def test_translate_and_formatting():
    assert translate("nav_dataset", "ru") == "Датасет"
    assert translate("nav_dataset", "en") == "Dataset"
    assert translate("train_step", "en")  # no format placeholders
    assert translate("results_exported", "en", path="model.onnx") == "Saved: model.onnx"
    assert translate("does_not_exist", "en") == "does_not_exist"


def test_missing_format_arguments_do_not_crash():
    assert translate("results_exported", "en") == "Saved: {path}"


def test_unknown_language_falls_back_to_default():
    assert translate("nav_dataset", "de") == STRINGS["nav_dataset"][DEFAULT_LANGUAGE]


def test_translator_toggle_and_set():
    translator = Translator("ru")
    assert translator.language == "ru"
    assert translator("nav_train") == "Обучение"
    assert translator.toggle() == "en"
    assert translator("nav_train") == "Training"
    translator.set_language("ru")
    assert translator.language == "ru"
    translator.set_language("xx")
    assert translator.language == "ru"


def test_core_translation_keys_used_by_the_gui_exist():
    from wakeword_trainer.gui import i18n

    source_root = Path(i18n.__file__).parent
    pattern = re.compile(r"(?:self\.)?tr\(\s*[\"']([a-z0-9_]+)[\"']")
    missing: dict[str, set[str]] = {}
    for path in source_root.rglob("*.py"):
        if path.name == "i18n.py":
            continue  # its module docstring shows example calls
        text = path.read_text(encoding="utf-8")
        for key in pattern.findall(text):
            if key not in STRINGS:
                missing.setdefault(key, set()).add(path.name)
    assert not missing, f"missing translation keys: {missing}"
