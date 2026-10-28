# Negatives (non-wake-word audio)

Negatives are examples of everything the model should **not** trigger on: speech, music, TV, noise, silence, other wake words. Without them the classifier learns "this is my phrase" versus "this is nothing", which fails immediately on real audio.

Three sources are supported, all through `negatives=`.

---

## 1. `negatives="auto"` — openWakeWord feature sets

Uses the pre-computed features published with openWakeWord:

| File | Size | Role |
| --- | --- | --- |
| `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | ~17 GB | Training negatives |
| `validation_set_features.npy` | ~185 MB | Held-out negative pool for FP/hour estimates |

The first run with `download_negatives=True` downloads them into `<cache>/downloads` (resumable, size-verified). Later runs reuse the files.

```python
wwt.train(
    "hey computer",
    "data/hey-computer",
    "models/hey-computer",
    negatives="auto",
    download_negatives=True,
)
```

If you already have the files, point at them directly and skip the download:

```python
wwt.train(
    ...,
    negatives=r"D:\features\openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
)
```

Notes:

- the ACAV file is memory-mapped, so it does not count fully against RAM;
- the download is one-time per machine/user cache;
- if the validation file is absent, the false-positive rate is estimated on ACAV windows instead (the report says so);
- ACAV windows have a fixed frame count; with a non-2 s `clip_seconds` the pool trims to the needed frames or raises a clear error if it cannot.

---

## 2. `negatives="features.npy"` — your own embeddings

Any `.npy` array with one of these layouts:

| Shape | Meaning | Sampled as |
| --- | --- | --- |
| `(N, frames, 96)` | N independent clips | one clip per sample, 2 s accounting |
| `(time, 96)` | one long continuous embedding stream | sliding windows, 80 ms hop |

`frames` must be the number of embedding frames produced by your `clip_seconds`. For the default 2 s window it is **16**; for 1 s it is 3. Compute it with:

```python
from wakeword_trainer.audio.features import num_embedding_frames

frames = num_embedding_frames(int(16_000 * clip_seconds))
```

You can build such a file yourself with `FeatureExtractor.embeddings()` over long recordings. This is the recommended fully-offline option.

---

## 3. `negatives="path/to/audio_folder"` — raw audio

Point at a folder with music, podcasts, TV, cafe noise, silence, or speech that is not the wake word:

```python
wwt.train(
    ...,
    negatives="data/negative_audio",
    max_audio_hours=5.0,
)
```

What happens:

1. all supported audio files under the folder are discovered recursively;
2. each file is loaded as 16 kHz mono;
3. non-overlapping (hop = window length) windows are embedded with the bundled feature models;
4. the resulting features are cached in `<cache>/features`, keyed by the file list, sizes, mtimes, clip length and frame count;
5. only `max_audio_hours` (default 5 h) of audio is used — enough for most personal models.

The first run pays the embedding cost; later runs load the cache instantly. Re-record or edit a file and the fingerprint changes, so the cache is rebuilt automatically.

Recommendations:

- 30 minutes to a few hours of varied audio is enough to start;
- include **speech** that is not the phrase, otherwise the model may trigger on any voice;
- include typical background noise of the target environment;
- do not reuse your positive recordings here.

---

## `negatives="none"`

For smoke tests and pipeline debugging only:

```python
wwt.train(..., negatives="none", steps=100)
```

The model will have no idea what to reject, so recall on the training set may look great while everyday audio triggers it. Never ship such a model.

---

## Which one should I use?

| Situation | Recommendation |
| --- | --- |
| First project, internet available | `auto` + `download_negatives=True` |
| Offline machine / slow disk | Folder of audio (5–60 min) |
| You already have embeddings from another pipeline | `.npy` file |
| Quick pipeline sanity check | `none` (never ship) |
| Personal assistant used in a noisy home | Folder of **your** room noise + music + TV |

Combining sources is not supported in one run; precompute a combined `.npy` if you need that.

---

## Cache layout

```
<cache>/
  downloads/
    openwakeword_features_ACAV100M_2000_hrs_16bit.npy
    validation_set_features.npy
  features/
    negatives_<fingerprint>.npy
  tts/
```

`<cache>` defaults to:

- Windows: `%LOCALAPPDATA%\wakeword-trainer\Cache`
- Linux: `~/.cache/wakeword-trainer`
- macOS: `~/Library/Caches/wakeword-trainer`

Override with `WAKEWORD_TRAINER_CACHE_DIR`. To free space, delete the folder — it is rebuilt on demand (the ACAV download has to be repeated).

---

## How negatives are sampled during training

- the negative ratio grows from `train.negative_ratio_start` (4:1) to `train.negative_ratio_end` (preset-dependent) over the run;
- each batch contains positives and negatives around that ratio;
- validation uses a fixed, evenly spaced sample of the negative pool (`evaluation.max_negative_windows` for reports);
- false positives per hour are computed as `false_positives / negative_hours`, where hours come from pool size × hop.

Very small negative pools produce extremely noisy FP/hour estimates; the evaluation report warns when there is less than 0.5 h. Treat such numbers as relative, not absolute.
