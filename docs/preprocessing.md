# Features and predictive targets

The implementation is split between `contact/prepare.py` (audio and teacher
inference), `contact/targets.py` (target construction), `contact/readout.py`
(soft targets and aggregation), and `contact/objectives.py` (loss functions).
`configs/preprocessing.json` records the fixed settings. Human MOS never enters
feature extraction, pseudo-label construction or checkpoint inference.

For the paper's fixed audio inventory, `data/audio_preprocessing.json` supplies
audio hashes, observed VAD intervals and per-recording response-merge settings.
Pass it through `--protocols` when reproducing the benchmark. The VAD is stored
as runs of active frame indices and decoded without changing boundaries.
These annotations contain neither MOS nor model predictions. The direct VAD
path below remains available for new audio or explicit re-annotation.

## Audio and Mimi features

Each recording has two time-aligned speaker channels. Audio is resampled to
24 kHz for Mimi and 16 kHz for VAD and the affect teacher. Pauses are retained;
there is no per-window amplitude normalization. A/V windows are cut on the
source audio timeline before resampling. Track durations differing by more
than 20 ms are rejected, not automatically repaired.

Mimi uses the pinned `kyutai/mimi` checkpoint. Continuous features follow
`encoder -> encoder_transformer -> downsample`, producing 512 dimensions at
12.5 Hz per speaker; these are not discrete codec tokens. Extraction uses
300-second blocks. Features are stored as float16 and passed to the predictor
as float32.

## VAD and future activity

Silero VAD operates independently on each channel at 16 kHz, with threshold
0.5, minimum speech duration 100 ms and minimum silence duration 50 ms. Its
intervals are mapped to a binary 50-Hz grid. The cleanup pass removes short
speech runs, then fills short silence runs. To match the released scorer,
its nominal 150-ms threshold uses `int(.15 * 50) = 7` frames and a strict
`length < 7` comparison; a trailing open run is left unchanged.

The source annotations used two audio preprocessing profiles: SciPy resampling
with a ceiling-rounded VAD length, and TorchAudio resampling with a floor-rounded
length. Both are implemented; the per-recording protocol records the source
profile. The benchmark command consumes its fixed VAD annotations rather than
silently replacing them with a newly computed timeline.

For each eligible model frame, future occupancy is measured in four successive
bins of width 0.2, 0.4, 0.6 and 0.8 seconds (2 seconds total), beginning after
the current 80-ms model frame. Occupancy at least 0.5 is active. The four bits
from each of two speakers form one 256-class activity target; channel 0 occupies
the four least-significant bits. `activity_targets` also returns the validity
mask. The inference future guard excludes incomplete tail observations.

## Future joint silence

At integer-second anchors, the target is the fraction of the next 4 seconds in
which **neither** speaker is active. It is not the average of two independent
speaker-silence ratios. The fraction is mapped to ten equal-width bins with
`min(9, floor(10 * fraction))`. Anchors start at 4 seconds and require a complete
4-second future. The causal predictor state is selected strictly before the
anchor, accounting for the 40-ms feature-availability margin. Invalid frame
positions have target `-100` and are ignored by the loss.

## Arousal and valence

Speech runs define response units, merging gaps up to 0.2 seconds without
crossing a new partner onset. A second merge uses the recording's declared gap
(0.2 or 0.5 seconds in the benchmark), then limits each unit to 8 seconds. The
new-audio default is 0.2 seconds for both passes. Within each
unit, 3-second windows advance by 1 second. A retained window must contain at
least 1.5 seconds of real audio and at least 60% activity for its speaker.
These windows need not lie on integer recording timestamps.

The pinned dimensional VoxProfile teacher supplies continuous arousal, valence
and dominance values. Only arousal and valence are model targets. Dominance is
retained in the cache but not used in the released score. Teacher outputs are
pseudo-labels, not CONTACT human naturalness ratings.

Each A/V scalar becomes a normalized Gaussian distribution over eight bin
centers, with standard deviation 0.75 bin widths. The model predicts two
eight-class distributions per speaker. Their outer product provides the
64-state likelihood used in scoring. These are factorized soft targets, not a
single hard 64-class label. `contact/objectives.py` supplies soft-target KL and
ordinal CDF penalties. Future A/V is supervision only: prediction reads the
last available representation strictly before that target window starts.

## Cache files

Each successfully prepared recording contains:

```text
mimi_ch0.npy    [T, 512], float16
mimi_ch1.npy    [T, 512], float16
vad.npy        [2, T_vad], binary 50-Hz activity
affect.json    response/window IDs, speaker, timestamps and continuous A/V/D
cache.json     recording ID, audio hashes, model revisions and file checksums
```

Activity and silence labels are generated from VAD on demand; they do not need
another neural-model pass or a separate dataset download. The cache therefore
supports scoring with or without human ratings. A batch-versus-singleton
teacher check guards against padding-related A/V changes.
