# Audio preprocessing and targets

`contact/prepare.py` extracts features and teacher labels; `contact/targets.py`
constructs activity, silence and affect targets. Settings are in
`configs/preprocessing.json`. No human ratings are used in these steps.

For CONTACT, pass `--protocols data/audio_preprocessing.json` to use the released
VAD annotations and recording settings. For new audio, omit this option to run
Silero VAD.

## Audio and Mimi features

Use two aligned speaker channels with pauses retained. Audio is resampled to
24 kHz for Mimi and 16 kHz for VAD and the affect teacher, without amplitude
normalization. Track durations must agree within 20 ms.

Mimi produces 512-dimensional continuous features at 12.5 Hz per speaker using
`encoder -> encoder_transformer -> downsample`. Features are stored as float16
and passed to the predictor as float32. Model revisions are pinned in the config.

## VAD and future activity

Silero VAD runs independently on each channel at 16 kHz, with threshold 0.5,
minimum speech duration 100 ms and minimum silence duration 50 ms. Activity is
represented on a binary 50-Hz grid.

Future activity uses four successive bins of 0.2, 0.4, 0.6 and 0.8 seconds.
A bin is active when speech occupies at least half its duration. The eight
binary values across both speakers define a 256-class target. Scoring excludes
frames without the required future context.

## Future joint silence

The silence target is the fraction of the next 4 seconds in which neither
speaker is active, quantized into ten equal-width bins. Targets are sampled
once per second, starting at 4 seconds, and require a complete future interval.
Prediction uses the last available causal state before each target interval.

## Arousal and valence

VAD defines response units. Within each unit, the affect teacher processes
3-second windows with a 1-second hop. Windows require at least 1.5 seconds of
audio and 60% speech activity to exclude mostly silent segments. Response
segmentation settings are stored with the preprocessing configuration.

VoxProfile supplies continuous arousal, valence and dominance pseudo-labels.
Only arousal and valence are used by the predictor; dominance is retained in
the cache but does not contribute to scoring.

Each scalar defines a Gaussian soft target over eight bins, with standard
deviation 0.75 bin widths. The head predicts separate arousal and valence
distributions; their outer product gives the joint likelihood. Prediction
uses context available before the target window. Soft-target KL and ordinal
CDF losses are implemented in `contact/objectives.py`.

## Cache files

Each successfully prepared recording contains:

```text
mimi_ch0.npy    [T, 512], float16
mimi_ch1.npy    [T, 512], float16
vad.npy        [2, T_vad], binary 50-Hz activity
affect.json    response/window IDs, speaker, timestamps and continuous A/V/D
cache.json     recording ID, audio hashes, model revisions and file checksums
```

Activity and silence targets are generated from the cached VAD during scoring.
Preparation checks that batched teacher predictions agree with single-example
predictions before saving a cache.
