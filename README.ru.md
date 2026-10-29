# wakeword-trainer

Обучение **совместимых с openWakeWord** моделей wake word на собственных записях — из Python-скрипта или десктопного приложения. Самостоятельный проект: без зависимости от `openwakeword` и оригинальных ноутбуков.

[English version](https://github.com/DUK-Codes/wakeword-trainer/blob/main/README.md)

[![PyPI](https://img.shields.io/badge/PyPI-wakeword--trainer-blue)](https://pypi.org/project/wakeword-trainer/)
[![Release](https://img.shields.io/github/v/release/DUK-Codes/wakeword-trainer)](https://github.com/DUK-Codes/wakeword-trainer/releases/latest)

## Установка

```bash
pip install wakeword-trainer          # библиотека
pip install "wakeword-trainer[gui]"   # + приложение и работа с микрофоном
```

Python 3.10+. Модели признаков (mel + embedding) входят в пакет.

Если не хотите ставить Python — скачайте установщик
`wakeword-trainer-<версия>-windows-x64-setup.exe` (или портативный
`...-portable.zip`) из [Releases](https://github.com/DUK-Codes/wakeword-trainer/releases/latest).
В установщике можно отметить движок озвучки (Piper); по умолчанию он не ставится,
и его можно добавить позже на странице **«Компоненты»**.

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

Всё работает на вашей машине: модели признаков идут в комплекте, обучение выполняет PyTorch, а датасеты, модели и голоса остаются в выбранных вами папках. Аккаунт и API-ключ не нужны.

Готовый установщик и портативный архив для Windows (Python не нужен) прикреплены к [Releases](https://github.com/DUK-Codes/wakeword-trainer/releases). Сама библиотека распространяется отдельно через pip.

### Дополнительные компоненты

Базовая сборка намеренно компактная, остальное добавляется по необходимости:

- **Движок озвучки** (Piper) — галочка в установщике или кнопка на странице
  **«Компоненты»**; Coqui XTTS доступен только вместе с Python-пакетом.
- **Негативы для обучения** — скачиваются автоматически при обучении или заранее
  на той же странице (валидационный набор ~185 МБ, ACAV100M ~17 ГБ).
- **Голоса** — загружаются со страницы «Озвучка», когда понадобятся.

Подробнее: [docs/components.md](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/components.md).

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
| [Установка](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/installation.md) | extras, установщик, ffmpeg, кэши |
| [Python API](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/python-api.md) | все функции, конфиг, колбэки |
| [GUI](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/gui.md) | постраничный обзор |
| [Компоненты](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/components.md) | движки, голоса, негативы |
| [Датасет](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/dataset.md) | запись, группы, сплиты |
| [Негативы](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/negatives.md) | auto / `.npy` / папка аудио |
| [TTS](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/tts.md) | Piper, Coqui, плагины |
| [Экспорт](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/export.md) | контракт ONNX, интеграция с openWakeWord |
| [Решение проблем](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/troubleshooting.md) | частые ошибки |
| [Архитектура](https://github.com/DUK-Codes/wakeword-trainer/blob/main/docs/architecture.md) | модули и конвейер |

## Разработка

```bash
git clone https://github.com/DUK-Codes/wakeword-trainer.git
cd wakeword-trainer
pip install -e ".[gui,dev]"
ruff check src tests && pytest -q
```

## Лицензия

Apache-2.0 — см. [LICENSE](https://github.com/DUK-Codes/wakeword-trainer/blob/main/LICENSE) и [NOTICE](https://github.com/DUK-Codes/wakeword-trainer/blob/main/NOTICE).
