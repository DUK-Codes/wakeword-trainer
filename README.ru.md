# wakeword-trainer

Обучение **совместимых с openWakeWord** моделей wake word на собственных записях — из Python-скрипта или десктопного приложения. Самостоятельный проект: без зависимости от `openwakeword` и оригинальных ноутбуков.

[English version](README.md)

[![PyPI](https://img.shields.io/badge/PyPI-wakeword--trainer-blue)](https://pypi.org/project/wakeword-trainer/)
[![Release](https://img.shields.io/github/v/release/DUK-Codes/wakeword-trainer)](https://github.com/DUK-Codes/wakeword-trainer/releases/latest)

## Установка

```bash
pip install wakeword-trainer          # библиотека
pip install "wakeword-trainer[gui]"   # + приложение и работа с микрофоном
```

Python 3.10+. Модели признаков (mel + embedding) идут в комплекте.

Если не хотите ставить Python — скачайте `wakeword-trainer-<версия>-windows-x64.exe` из [Releases](https://github.com/DUK-Codes/wakeword-trainer/releases/latest).

Если PyPI недоступен, установка из исходников:

```bash
pip install "wakeword-trainer[gui] @ git+https://github.com/DUK-Codes/wakeword-trainer.git"
```

## Быстрый старт

```python
import wakeword_trainer as wwt

result = wwt.train(
    word="hey computer",
    dataset="data/hey-computer",      # папка с вашими записями
    output="models/hey-computer",
    preset="balanced",                # quick | balanced | best
    negatives="auto",                 # или .npy / папка с негативным аудио
    download_negatives=True,          # однократная загрузка ~17 ГБ (опционально)
)

print(result.best_metrics)
print(wwt.predict(result.onnx_path, "test.wav"))
```

Свои негативы (без загрузки):

```python
wwt.train(
    "hey computer",
    "data/hey-computer",
    "models/hey-computer",
    negatives="data/negative_audio",
)
```

Проверка данных до обучения:

```python
report = wwt.inspect_dataset("data/hey-computer")
print(report.summary())
```

## Десктопное приложение

```bash
wakeword-trainer-gui
```

Проверка датасета, запись с микрофона, прогресс и графики обучения, метрики на отложенной выборке, проверка на живом микрофоне, TTS-примеры, экспорт ONNX/TFLite — в одном окне, RU/EN.

Готовый Windows `.exe` (Python не нужен) прикреплён к [Releases](https://github.com/DUK-Codes/wakeword-trainer/releases). Сама библиотека распространяется отдельно через pip.

## Что внутри

- **Работа с данными** — рекурсивный обход, валидация, поиск дубликатов, group-aware сплит train/val/test (аугментированные копии не утекают).
- **Честная оценка** — recall, FRR, ложные срабатывания в час, AUC, перебор порогов, отчёты JSON/Markdown/PNG на отложенных клипах.
- **Обучение** — пресеты, аугментации, расписание доли негативов, усреднение чекпоинтов, resume, живые события JSONL.
- **Экспорт** — ONNX с метаданными и проверкой в рантайме; TFLite опционально.
- **Инференс** — `predict`, `detect`, `WakeWordDetector`, потоковый `StreamingDetector`.

## Данные кратко

50–300 коротких записей точной фразы — с тех микрофонов и в тех помещениях, где она будет использоваться; 1.5–2.5 с на файл; WAV/FLAC/OGG (MP3/M4A — с `ffmpeg`). Для рабочей модели нужны негативы.

## Документация

| | |
| --- | --- |
| [Установка](docs/installation.md) | extras, ffmpeg, кэши |
| [Python API](docs/python-api.md) | все функции, конфиг, колбэки |
| [GUI](docs/gui.md) | постраничный обзор |
| [Датасет](docs/dataset.md) | запись, группы, сплиты |
| [Негативы](docs/negatives.md) | auto / `.npy` / папка аудио |
| [TTS](docs/tts.md) | Piper, Coqui, Silero, плагины |
| [Экспорт](docs/export.md) | контракт ONNX, интеграция с openWakeWord |
| [Решение проблем](docs/troubleshooting.md) | частые ошибки |
| [Архитектура](docs/architecture.md) | модули и конвейер |

## Разработка

```bash
git clone https://github.com/DUK-Codes/wakeword-trainer.git
cd wakeword-trainer
pip install -e ".[gui,dev]"
ruff check src tests && pytest -q
```

## Лицензия

Apache-2.0 — см. [LICENSE](LICENSE) и [NOTICE](NOTICE).
