# Audio locations

`datasets/` holds downloaded audio. `data/` holds the released split, ratings,
normalization statistics and reference scores. Audio and generated caches are
not committed to Git.

```text
datasets/
  contact/
    recordings.csv
    hh_turn/<recording_id>/stereo.wav
    hh_emotion/<recording_id>/stereo.wav
    hai/<recording_id>/stereo.wav
  seamless_interaction/
    raw/
    recordings.csv
```

## CONTACT evaluation audio

[Audio examples](https://github.com/review-anonymous-git/contact-samples) are
distributed separately. Examples are not the full benchmark. The full audio
download location is not yet configured in this repository; no command here
claims to download all evaluation recordings.

Create the directory layout and an inventory of the 493 paper recordings:

```bash
python -m contact.dataset init
```

Place the corresponding audio at those locations. A stereo file must have one
speaker per channel. Alternatively, edit `recordings.csv`: leave `audio` empty
and fill `speaker_a` and `speaker_b` with two synchronized mono files. Paths are
relative to the CSV, or absolute. Preserve the recording's original channel
assignment. Do not substitute a mixed track, remove pauses, concatenate turns,
or silently trim unequal tracks. Archive filenames and speaker names may differ
from this layout; map them explicitly in the CSV.

The manifest format is:

```csv
recording_id,corpus,audio,speaker_a,speaker_b
example_hh,hh_turn,,hh_turn/example_hh/a.wav,hh_turn/example_hh/b.wav
example_ai,hai,hai/example_ai/stereo.wav,,
```

For a few examples, create a CSV with only those recordings and omit
`--inventory`. For the paper inventory, check that no recording is missing:

```bash
python -m contact.dataset prepare \
  --manifest datasets/contact/recordings.csv \
  --inventory data/splits.csv --protocols data/audio_preprocessing.json --check-only
```

This checks IDs, channel counts and durations before loading GPU models. It
cannot detect swapped speakers or an incorrectly mixed recording from its
header; channel assignment must come from the dataset metadata.

## Preprocessing and scoring

Install the audio dependencies and affect-teacher wrapper as described in the
root README. Then run:

```bash
python -m contact.dataset prepare \
  --manifest datasets/contact/recordings.csv --inventory data/splits.csv \
  --protocols data/audio_preprocessing.json \
  --cache-root cache/contact --output-manifest cache/contact/recordings.csv \
  --device cuda:0 --batch-size 4 --skip-existing

python -m contact.infer \
  --manifest cache/contact/recordings.csv --device cuda:0 \
  --resume-dir outputs/inference_records --output outputs/predictions.csv

python -m contact.evaluate \
  --predictions outputs/predictions.csv --output outputs/rescored
```

The first two commands need no human MOS. The last command joins predictions
to the released ratings. It requires all 493 IDs, not just the audio examples.

For paper reproduction, `data/audio_preprocessing.json` fixes each recording's
observed VAD timeline, response-merge setting and audio checksum. These are
audio-derived target annotations, not model scores or human MOS. Mimi features
and A/V teacher outputs are still computed from the audio. Recomputing VAD with
a different resampler or boundary-rounding rule can change target intervals, so
it is not a byte-equivalent reproduction. For new recordings, omit `--protocols`
to run Silero VAD and the default response-window builder directly.

Preprocessing retains its models across recordings. Inference writes
per-recording results when `--resume-dir` is given. Interrupted jobs can reuse
verified completed records by rerunning the same command. Completed output CSVs
are never overwritten; use a new output filename if a complete CSV already
exists. An existing preprocessing cache must match the audio bytes and pinned
model revisions to be reused.

Both commands support `--num-shards N --shard-index K` with zero-based K.
Keep N and the manifest fixed across workers. Each worker needs a different
output CSV and an assigned GPU. This is independent inference, not DDP; it
does not request or release scheduler allocations. For example, on an already
allocated two-GPU node, launch these in separate terminals:

```bash
CUDA_VISIBLE_DEVICES=0 python -m contact.infer \
  --manifest cache/contact/recordings.csv --device cuda:0 \
  --num-shards 2 --shard-index 0 --resume-dir outputs/inference_records \
  --output outputs/predictions_0.csv

CUDA_VISIBLE_DEVICES=1 python -m contact.infer \
  --manifest cache/contact/recordings.csv --device cuda:0 \
  --num-shards 2 --shard-index 1 --resume-dir outputs/inference_records \
  --output outputs/predictions_1.csv
```

After both finish:

```bash
python -m contact.merge \
  --parts outputs/predictions_0.csv outputs/predictions_1.csv \
  --manifest data/splits.csv --output outputs/predictions.csv
```

The merge checks complete coverage, duplicate IDs, finite scores and matching
checkpoint/code/settings hashes. For sharded preprocessing, each output cache
manifest can instead be scored separately, without sharding it a second time.

## Training audio

See [seamless_interaction/README.md](seamless_interaction/README.md) for the
official source and the training-audio placeholder. No training audio is needed
to evaluate the released checkpoint.
