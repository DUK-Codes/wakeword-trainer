"""Minimal RU/EN translation layer for the desktop app.

Every user-visible string lives in :data:`STRINGS`. Pages call
``self.tr("key")`` and implement ``retranslate()`` so switching the language
updates the window without restarting it.
"""

from __future__ import annotations

from typing import Any

LANGUAGES = ("ru", "en")
LANGUAGE_NAMES = {"ru": "Русский", "en": "English"}
DEFAULT_LANGUAGE = "ru"

STRINGS: dict[str, dict[str, str]] = {
    # ---------------------------------------------------------------- window
    "app_title": {
        "ru": "Wake Word Trainer",
        "en": "Wake Word Trainer",
    },
    "app_subtitle": {
        "ru": "Обучение wake word моделей для openWakeWord",
        "en": "Train openWakeWord-compatible wake word models",
    },
    "nav_dataset": {"ru": "Датасет", "en": "Dataset"},
    "nav_settings": {"ru": "Настройки", "en": "Settings"},
    "nav_train": {"ru": "Обучение", "en": "Training"},
    "nav_results": {"ru": "Результаты", "en": "Results"},
    "nav_test": {"ru": "Проверка", "en": "Test"},
    "nav_tts": {"ru": "Озвучка (TTS)", "en": "Speech (TTS)"},
    "language": {"ru": "Язык", "en": "Language"},
    "open_output": {"ru": "Открыть папку", "en": "Open folder"},
    "ready": {"ru": "Готово", "en": "Ready"},
    "error": {"ru": "Ошибка", "en": "Error"},
    "warning": {"ru": "Предупреждение", "en": "Warning"},
    "info": {"ru": "Информация", "en": "Info"},
    "yes": {"ru": "Да", "en": "Yes"},
    "no": {"ru": "Нет", "en": "No"},
    "cancel": {"ru": "Отмена", "en": "Cancel"},
    "close": {"ru": "Закрыть", "en": "Close"},
    "ok": {"ru": "OK", "en": "OK"},
    "browse": {"ru": "Обзор…", "en": "Browse…"},
    "copy": {"ru": "Копировать", "en": "Copy"},
    "copied": {"ru": "Скопировано", "en": "Copied"},
    # --------------------------------------------------------------- dataset
    "dataset_title": {"ru": "Датасет", "en": "Dataset"},
    "dataset_hint": {
        "ru": "Папка с записями вашего wake word. Поддерживаются WAV, FLAC, OGG; MP3/M4A — при наличии ffmpeg.",
        "en": "Folder with your wake word recordings. WAV, FLAC and OGG are supported; MP3/M4A need ffmpeg.",
    },
    "dataset_folder": {"ru": "Папка с примерами", "en": "Recordings folder"},
    "dataset_placeholder": {
        "ru": "Выберите папку с записями…",
        "en": "Choose a recordings folder…",
    },
    "dataset_check": {"ru": "Проверить датасет", "en": "Validate dataset"},
    "dataset_checking": {"ru": "Проверяем файлы…", "en": "Checking files…"},
    "dataset_valid": {"ru": "Файлов пригодно", "en": "Valid files"},
    "dataset_groups": {"ru": "Независимых групп", "en": "Independent groups"},
    "dataset_duration": {"ru": "Общая длительность", "en": "Total audio"},
    "dataset_invalid": {"ru": "Нечитаемых файлов", "en": "Unreadable files"},
    "dataset_duplicates": {"ru": "Дубликатов", "en": "Duplicates"},
    "dataset_no_warnings": {"ru": "Проблем не найдено.", "en": "No problems found."},
    "dataset_empty": {
        "ru": "Сначала выберите и проверьте папку с записями.",
        "en": "Choose and validate a recordings folder first.",
    },
    "dataset_record_title": {"ru": "Запись с микрофона", "en": "Microphone recording"},
    "dataset_record_hint": {
        "ru": "Запишите несколько вариантов слова с разной интонацией, расстояния и шума. Разные сессии повышают качество.",
        "en": "Record several takes with different intonation, distance and background noise. Varied sessions improve quality.",
    },
    "record_start": {"ru": "Начать запись", "en": "Start recording"},
    "record_stop": {"ru": "Остановить и сохранить", "en": "Stop and save"},
    "record_discard": {"ru": "Отменить", "en": "Discard"},
    "recording": {"ru": "Идёт запись…", "en": "Recording…"},
    "record_saved": {"ru": "Сохранено: {name}", "en": "Saved: {name}"},
    "record_need_folder": {
        "ru": "Выберите папку датасета — записи будут сохраняться туда.",
        "en": "Choose a dataset folder first; recordings are saved there.",
    },
    "record_no_device": {
        "ru": "Микрофон не найден. Установите sounddevice и проверьте устройство ввода.",
        "en": "No microphone found. Install sounddevice and check the input device.",
    },
    "record_level": {"ru": "Уровень", "en": "Level"},
    "record_device": {"ru": "Устройство ввода", "en": "Input device"},
    "record_default_device": {"ru": "По умолчанию", "en": "System default"},
    "dataset_refresh": {"ru": "Повторить проверку", "en": "Refresh check"},
    # -------------------------------------------------------------- settings
    "settings_title": {"ru": "Настройки обучения", "en": "Training settings"},
    "settings_hint": {
        "ru": "Пресеты задают количество шагов, размер модели и соотношение негативов. Всё можно изменить вручную.",
        "en": "Presets set the step count, model size and negative ratio. Everything can be adjusted manually.",
    },
    "preset": {"ru": "Пресет", "en": "Preset"},
    "preset_quick": {"ru": "Быстрый — проверить идею", "en": "Quick — sanity check"},
    "preset_balanced": {"ru": "Сбалансированный — для реального использования", "en": "Balanced — practical default"},
    "preset_best": {"ru": "Максимальный — лучшее качество", "en": "Best — highest quality"},
    "preset_quick_desc": {
        "ru": "≈4 000 шагов, маленькая модель. Минуты на CPU, подходит для черновика.",
        "en": "≈4,000 steps, small model. Minutes on CPU; good for a draft.",
    },
    "preset_balanced_desc": {
        "ru": "≈15 000 шагов, средняя модель. Разумный баланс качества и времени.",
        "en": "≈15,000 steps, medium model. A sensible quality/time balance.",
    },
    "preset_best_desc": {
        "ru": "≈40 000 шагов, крупная модель. Заметно дольше, лучше точность.",
        "en": "≈40,000 steps, larger model. Much longer, better accuracy.",
    },
    "word": {"ru": "Фраза (wake word)", "en": "Wake word phrase"},
    "word_placeholder": {"ru": "например, hey computer", "en": "for example, hey computer"},
    "output_dir": {"ru": "Папка результата", "en": "Output folder"},
    "output_dir_hint": {
        "ru": "Сюда сохранятся чекпоинты, ONNX-модель и отчёт об оценке.",
        "en": "Checkpoints, the ONNX model and the evaluation report are written here.",
    },
    "advanced": {"ru": "Дополнительно", "en": "Advanced"},
    "steps": {"ru": "Шагов обучения", "en": "Training steps"},
    "batch_size": {"ru": "Размер батча", "en": "Batch size"},
    "layer_dim": {"ru": "Размер слоя", "en": "Layer size"},
    "n_blocks": {"ru": "Блоков FC", "en": "FC blocks"},
    "clip_seconds": {"ru": "Длина окна, с", "en": "Window length, s"},
    "learning_rate": {"ru": "Learning rate", "en": "Learning rate"},
    "device": {"ru": "Устройство", "en": "Device"},
    "device_auto": {"ru": "Авто", "en": "Auto"},
    "seed": {"ru": "Seed", "en": "Seed"},
    "augment_enabled": {"ru": "Аугментации", "en": "Augmentation"},
    "augment_hint": {
        "ru": "Случайные громкость, шум, питч, скорость; при указании — реверберация и фон.",
        "en": "Random gain, noise, pitch and speed; reverb and background when supplied.",
    },
    "background_dir": {"ru": "Папка с фоновыми шумами", "en": "Background noise folder"},
    "rir_dir": {"ru": "Папка с RIR (реверберация)", "en": "RIR folder (reverb)"},
    "negatives_source": {"ru": "Негативы (не-wake-word данные)", "en": "Negatives (non-wake-word data)"},
    "negatives_auto": {"ru": "ACAV100M (Hugging Face)", "en": "ACAV100M (Hugging Face)"},
    "negatives_npy": {"ru": "Файл признаков .npy", "en": "Feature file (.npy)"},
    "negatives_audio": {"ru": "Папка с аудио", "en": "Audio folder"},
    "negatives_none": {"ru": "Без негативов (только тест)", "en": "None (smoke test only)"},
    "negatives_path": {"ru": "Путь к негативам", "en": "Negatives path"},
    "negatives_download": {
        "ru": "Разрешить скачивание ACAV100M (~17 ГБ, один раз)",
        "en": "Allow ACAV100M download (~17 GB, one time)",
    },
    "negatives_download_hint": {
        "ru": "Если файл уже скачан, он будет использован из кэша.",
        "en": "If the file is already cached, it is reused.",
    },
    "negatives_none_warning": {
        "ru": "Без негативов модель будет срабатывать на любую речь. Используйте только для проверки пайплайна.",
        "en": "Without negatives the model fires on any speech. Use only to test the pipeline.",
    },
    "target_fp": {"ru": "Целевой FP в час", "en": "Target FP per hour"},
    "val_every": {"ru": "Валидация каждые N шагов", "en": "Validate every N steps"},
    "checkpoint_every": {"ru": "Чекпоинт каждые N шагов", "en": "Checkpoint every N steps"},
    "early_stop": {
        "ru": "Ранняя остановка (N валидаций без улучшения)",
        "en": "Early stop (N validations without improvement)",
    },
    "early_stop_off": {"ru": "выключено", "en": "off"},
    "reset_preset": {"ru": "Сбросить к пресету", "en": "Reset to preset"},
    "settings_invalid": {"ru": "Проверьте настройки: {error}", "en": "Check your settings: {error}"},
    "export_tflite": {"ru": "Экспортировать также TFLite", "en": "Also export TFLite"},
    "export_tflite_hint": {
        "ru": 'Требует дополнительную установку: pip install "wakeword-trainer[tflite]".',
        "en": 'Requires the optional extra: pip install "wakeword-trainer[tflite]".',
    },
    # --------------------------------------------------------------- training
    "train_title": {"ru": "Обучение", "en": "Training"},
    "train_start": {"ru": "Начать обучение", "en": "Start training"},
    "train_stop": {"ru": "Остановить", "en": "Stop"},
    "train_stopping": {"ru": "Останавливаем…", "en": "Stopping…"},
    "train_idle": {
        "ru": "Проверьте датасет и настройки, затем запустите обучение.",
        "en": "Validate the dataset and settings, then start training.",
    },
    "train_running": {"ru": "Обучение…", "en": "Training…"},
    "train_preparing": {"ru": "Подготовка данных", "en": "Preparing data"},
    "train_step": {"ru": "Шаг", "en": "Step"},
    "train_loss": {"ru": "Loss", "en": "Loss"},
    "train_lr": {"ru": "Learning rate", "en": "Learning rate"},
    "train_recall": {"ru": "Recall", "en": "Recall"},
    "train_fp": {"ru": "FP в час", "en": "FP / hour"},
    "train_threshold": {"ru": "Порог", "en": "Threshold"},
    "train_elapsed": {"ru": "Прошло", "en": "Elapsed"},
    "train_eta": {"ru": "Осталось", "en": "Remaining"},
    "train_log": {"ru": "Журнал", "en": "Log"},
    "train_curve": {"ru": "Кривая обучения", "en": "Training curve"},
    "train_done": {"ru": "Обучение завершено", "en": "Training finished"},
    "train_failed": {"ru": "Обучение прервано ошибкой", "en": "Training failed"},
    "train_cancelled": {"ru": "Обучение остановлено пользователем", "en": "Training stopped by the user"},
    "train_output_missing": {
        "ru": "Сначала укажите папку результата.",
        "en": "Set an output folder first.",
    },
    "train_need_dataset": {"ru": "Сначала укажите папку с датасетом.", "en": "Set the dataset folder first."},
    "train_need_word": {"ru": "Укажите фразу wake word.", "en": "Enter the wake word phrase."},
    "train_already_running": {"ru": "Обучение уже запущено.", "en": "Training is already running."},
    "train_first_run_download": {
        "ru": "Первый запуск может скачать фичи негативов (~17 ГБ). Это займёт время.",
        "en": "The first run may download negative features (~17 GB). This takes a while.",
    },
    "train_python_missing": {
        "ru": "Не удалось запустить процесс обучения: {error}",
        "en": "Could not start the training process: {error}",
    },
    "train_resume": {"ru": "Продолжить с последнего чекпоинта", "en": "Resume from the last checkpoint"},
    # ---------------------------------------------------------------- results
    "results_title": {"ru": "Результаты", "en": "Results"},
    "results_none": {
        "ru": "Пока нет результатов. Запустите обучение.",
        "en": "No results yet. Start a training run.",
    },
    "results_model": {"ru": "Модель", "en": "Model"},
    "results_metrics": {"ru": "Метрики на отложенной выборке", "en": "Held-out metrics"},
    "results_threshold": {"ru": "Рекомендуемый порог", "en": "Recommended threshold"},
    "results_recall": {"ru": "Recall", "en": "Recall"},
    "results_frr": {"ru": "FRR (пропуски)", "en": "FRR (misses)"},
    "results_fp": {"ru": "Ложных срабатываний в час", "en": "False positives per hour"},
    "results_auc": {"ru": "AUC", "en": "AUC"},
    "results_neg_hours": {"ru": "Часов негативного аудио", "en": "Hours of negative audio"},
    "results_positives": {"ru": "Отложенных примеров", "en": "Held-out clips"},
    "results_export": {"ru": "Экспорт модели", "en": "Export model"},
    "results_export_onnx": {"ru": "Сохранить ONNX как…", "en": "Save ONNX as…"},
    "results_export_tflite": {"ru": "Сохранить TFLite как…", "en": "Save TFLite as…"},
    "results_report": {"ru": "Отчёт", "en": "Report"},
    "results_open_report": {"ru": "Открыть отчёт (Markdown)", "en": "Open report (Markdown)"},
    "results_open_dir": {"ru": "Открыть папку результатов", "en": "Open results folder"},
    "results_warnings": {"ru": "Предупреждения", "en": "Warnings"},
    "results_no_warnings": {"ru": "Предупреждений нет.", "en": "No warnings."},
    "results_exported": {"ru": "Сохранено: {path}", "en": "Saved: {path}"},
    "results_export_failed": {"ru": "Не удалось экспортировать: {error}", "en": "Export failed: {error}"},
    "results_threshold_sweep": {"ru": "Порог и ошибки", "en": "Threshold and errors"},
    "results_scores": {"ru": "Распределение оценок", "en": "Score distribution"},
    "results_curves": {"ru": "Кривые обучения", "en": "Training curves"},
    "results_per_clip": {"ru": "Оценки по файлам", "en": "Per-file scores"},
    "results_table_path": {"ru": "Файл", "en": "File"},
    "results_table_group": {"ru": "Группа", "en": "Group"},
    "results_table_score": {"ru": "Оценка", "en": "Score"},
    "results_table_detected": {"ru": "Детект", "en": "Detected"},
    # ------------------------------------------------------------------- test
    "test_title": {"ru": "Проверка модели", "en": "Model test"},
    "test_hint": {
        "ru": "Скажите фразу в микрофон: индикатор показывает уверенность модели в реальном времени.",
        "en": "Say the phrase into the microphone; the meter shows live model confidence.",
    },
    "test_model": {"ru": "Модель ONNX", "en": "ONNX model"},
    "test_choose_model": {"ru": "Выбрать модель…", "en": "Choose model…"},
    "test_threshold": {"ru": "Порог", "en": "Threshold"},
    "test_mic_start": {"ru": "Включить микрофон", "en": "Start microphone"},
    "test_mic_stop": {"ru": "Выключить микрофон", "en": "Stop microphone"},
    "test_score": {"ru": "Уверенность", "en": "Confidence"},
    "test_detected": {"ru": "СРАБОТАЛО", "en": "DETECTED"},
    "test_waiting": {"ru": "Ждём фразу…", "en": "Waiting for the phrase…"},
    "test_file": {"ru": "Проверить файл", "en": "Test a file"},
    "test_file_score": {"ru": "Оценка файла: {score:.3f}", "en": "File score: {score:.3f}"},
    "test_no_model": {"ru": "Выберите ONNX-модель.", "en": "Choose an ONNX model."},
    "test_no_device": {
        "ru": "Микрофон недоступен. Установите sounddevice.",
        "en": "Microphone unavailable. Install sounddevice.",
    },
    "test_model_loaded": {"ru": "Модель загружена: {word}", "en": "Model loaded: {word}"},
    "test_load_failed": {"ru": "Не удалось загрузить модель: {error}", "en": "Could not load the model: {error}"},
    # -------------------------------------------------------------------- tts
    "tts_title": {"ru": "Синтез примеров (TTS)", "en": "Synthesize samples (TTS)"},
    "tts_hint": {
        "ru": "Дополнительные примеры из синтезированной речи. TTS-движки не входят в пакет и устанавливаются отдельно.",
        "en": "Extra samples from synthesized speech. TTS engines are not bundled and must be installed separately.",
    },
    "tts_backend": {"ru": "Движок", "en": "Engine"},
    "tts_voice": {"ru": "Голос", "en": "Voice"},
    "tts_language": {"ru": "Язык", "en": "Language"},
    "tts_text": {"ru": "Фразы (по одной в строке)", "en": "Phrases (one per line)"},
    "tts_download": {"ru": "Скачать голос", "en": "Download voice"},
    "tts_downloading": {"ru": "Скачиваем голос…", "en": "Downloading voice…"},
    "tts_downloaded": {"ru": "Голос готов: {path}", "en": "Voice ready: {path}"},
    "tts_generate": {"ru": "Сгенерировать в датасет", "en": "Generate into dataset"},
    "tts_generating": {"ru": "Синтез {done}/{total}…", "en": "Synthesizing {done}/{total}…"},
    "tts_done": {"ru": "Готово: {count} файл(ов) в {folder}", "en": "Done: {count} file(s) in {folder}"},
    "tts_not_installed": {
        "ru": "Движок не установлен: {error}",
        "en": "Engine is not installed: {error}",
    },
    "tts_install_plugins": {"ru": "Открыть папку плагинов", "en": "Open plugin folder"},
    "tts_install_template": {"ru": "Создать шаблон плагина", "en": "Create plugin template"},
    "tts_template_created": {"ru": "Шаблон создан: {path}", "en": "Template created: {path}"},
    "tts_plugin_errors": {"ru": "Ошибки плагинов", "en": "Plugin errors"},
    "tts_no_voices": {"ru": "Нет доступных голосов.", "en": "No voices available."},
    "tts_voice_needs_download": {
        "ru": "Голос будет скачан при первом синтезе.",
        "en": "The voice is downloaded on first use.",
    },
    "tts_need_dataset": {
        "ru": "Выберите папку датасета на вкладке «Датасет».",
        "en": "Choose a dataset folder on the Dataset tab.",
    },
    "tts_need_text": {"ru": "Введите хотя бы одну фразу.", "en": "Enter at least one phrase."},
    "tts_license_note": {
        "ru": "Проверьте лицензию голоса и движка перед публикацией сгенерированных данных.",
        "en": "Check the voice and engine license before publishing generated audio.",
    },
}


def translate(key: str, language: str = DEFAULT_LANGUAGE, **kwargs: Any) -> str:
    """Return the localized string, formatted with ``kwargs``."""
    entry = STRINGS.get(key)
    if entry is None:
        return key
    language = language if language in entry else DEFAULT_LANGUAGE
    text = entry[language]
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text


class Translator:
    """Holds the active language and resolves keys."""

    def __init__(self, language: str = DEFAULT_LANGUAGE) -> None:
        self.language = language if language in LANGUAGES else DEFAULT_LANGUAGE

    def __call__(self, key: str, **kwargs: Any) -> str:
        return translate(key, self.language, **kwargs)

    def set_language(self, language: str) -> None:
        if language in LANGUAGES:
            self.language = language

    def toggle(self) -> str:
        self.language = "en" if self.language == "ru" else "ru"
        return self.language
