# Dataset guide

The dataset is the most important part of a wake word model. A modest amount of **varied, realistic** audio beats a large collection of near-identical takes.

## Format and layout

- Audio: WAV, FLAC, OGG/Opus, AIFF/AU; MP3/M4A/AAC with `ffmpeg`.
- Any sample rate and channel count — files are converted to **16 kHz mono int16**.
- Any folder structure; discovery is recursive.
- One phrase per dataset. Keep other phrases in separate folders.
- Recommended length: 1.5–2.5 s per file. The training window is 2 s by default (`clip_seconds`), shorter clips are padded, longer ones are cropped.

Example:

```
data/hey-computer/
  speaker-anna/
    session-01/
      take-01.wav
      take-02.wav
  speaker-boris/
    kitchen.wav
    car.wav
    kitchen_aug1.wav     # augmented copy of kitchen.wav
```

## How much data?

| Recordings | What to expect |
| --- | --- |
| < 50 | Model may work on the same speaker/mic only; metrics are very noisy |
| 50–150 | Usable personal model; record from several rooms and distances |
| 300+ | Noticeably more robust to speakers, mics and background noise |
| 1000+ | Diminishing returns unless variation grows too |

Add **hard negatives** as negative audio if the phrase is short or common in your language.

## Recording tips

- Use the exact phrase you will say in production, with the same intonation.
- Record with the microphones that will actually be used (laptop, headset, phone, smart speaker).
- Vary: distance (0.3–3 m), room, background noise, speaking speed, volume, emotional tone.
- Include some quiet and some noisy takes; include a few with music/TV in the background.
- Avoid clipping. Aim for peaks around −12…−3 dBFS.
- Leave natural silence around the phrase; the trainer trims silence itself.
- Do **not** duplicate the same take by copying the file with a new name — use real takes. If you must create augmented copies, name them like `take-01_aug1.wav` so they are grouped correctly (see below).

## Grouping and leakage

The splitter must never put two versions of the same recording into different splits, or the evaluation becomes optimistic.

The default grouping (`group_by="stem"`) strips common augmentation suffixes and file extensions:

```
hello.wav  hello_aug1.wav  hello_aug2.wav  hello_noise.wav  hello_v2.wav
```

all belong to the group **`hello`**. The same applies to names with `_augment`, `_reverb`, `_pitch`, `_speed`, `_gain`, `_copy`, `_variant`, `_var`, `_v<digits>`, separated by `_`, `-` or `.`.

If your files do not follow this convention, pass a custom function:

```python
report = wwt.inspect_dataset(
    "data/hey-computer",
    group_by=lambda path: path.parent.name,   # one group per speaker folder
)
```

`group_by="file"` disables grouping (every file is independent) — only use it when you are sure there are no copies.

## Scanning and validation

`wwt.inspect_dataset(folder)` reports:

- valid files, total duration, sample rates, channels;
- files that cannot be decoded;
- duplicate pairs found by content hash;
- groups and their sizes;
- warnings: very short clips, digital silence, inconsistent formats, too few groups.

A file is rejected if it is shorter than `min_clip_seconds` (0.25 s by default), longer than `max_clip_seconds` (60 s), empty or undecodable. Rejections include a human-readable reason.

## Splitting

`split_dataset(clips, val_ratio=0.12, test_ratio=0.18, seed=42)`:

- groups are assigned greedily to hit the target ratios, never splitting a group;
- with fewer than **3 groups** or **6 clips**, everything goes to training and a warning is emitted — evaluation cannot be honest in that case;
- the split is deterministic for a given seed;
- validation is used for model selection, test for the final report.

`wwt.evaluate(..., split="test")` recreates the same split with the same seed. If you add or remove recordings, the split changes — re-run training or evaluate with `split="all"` for a quick sanity check.

## Augmentation

Augmentation is applied **on the fly to training clips only** (not to validation/test):

| Augmentation | Default probability | Range |
| --- | --- | --- |
| Gain | 0.5 | −6…+6 dB |
| White noise | 0.5 | 5…30 dB SNR |
| Pitch shift | 0.3 | ±2 semitones |
| Speed | 0.2 | 0.9×…1.1× |
| Reverb | 0.2 | needs `rir_dir` |
| Background mix | 0.3 | needs `background_dir`, 5…20 dB SNR |

Point `augment.background_dir` at a folder of music, TV, cafe noise, etc., and `augment.rir_dir` at impulse responses, for realistic augmentation. Without those folders the corresponding augmentation is skipped.

Synthetic augmentation does **not** replace real recordings — it helps the model generalize, but cannot teach it your actual microphone.

## Dataset checklist

Before training:

- [ ] 50+ real takes of the exact phrase
- [ ] several speakers (if the model is not personal) and several rooms
- [ ] no clipped or silent files (`inspect_dataset` warnings)
- [ ] no accidental duplicates
- [ ] at least 3 independent groups (more is better)
- [ ] negative audio or the downloaded negative features available
- [ ] the phrase is not already a common word in the assistant's environment

After training:

- [ ] evaluate on held-out clips
- [ ] test with the GUI on the target device, live
- [ ] record 10–20 new takes later and re-check — this is the only honest generalization test
