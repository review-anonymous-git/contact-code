# Seamless Interaction training audio

The training source is the approximately 1,600-hour IPC subset of Seamless
Interaction. It is a subset, not the entire public corpus.

- [Official dataset and download tools](https://github.com/facebookresearch/seamless_interaction)
- [Official Hugging Face dataset](https://huggingface.co/datasets/facebook/seamless-interaction)

Obtain the audio from the original provider under its access requirements and
license (the dataset card specifies CC BY-NC 4.0). This repository neither
redistributes that audio nor bypasses the provider's terms. The official tools
support individual-file and batch downloads; do not download the complete
audiovisual corpus just to evaluate CONTACT.

Keep downloads in `raw/`, preserving the provider's paths. Use `recordings.csv`
to pair the synchronized participant files:

```csv
recording_id,corpus,audio,speaker_a,speaker_b
dialogue_example,seamless,,raw/path/to/participant_a.wav,raw/path/to/participant_b.wav
```

The ID above is a placeholder, not an actual training item. This directory is a
download/preprocessing placeholder, not a claim to reproduce the checkpoint's
complete training history. The released package focuses on checkpoint inference
and evaluation; it does not include a training launcher or the exact training
subset manifest.

The same audio preprocessing can be used to construct Mimi features, VAD and
A/V pseudo-labels from a populated manifest:

```bash
python -m contact.dataset prepare \
  --manifest datasets/seamless_interaction/recordings.csv \
  --cache-root cache/seamless --output-manifest cache/seamless/recordings.csv \
  --device cuda:0 --batch-size 4 --skip-existing
```

`seamless` identifies preprocessing data only: CONTACT's released evaluation
normalizers are for its three evaluation subsets, not a calibrated score for
the training corpus. See [../../docs/preprocessing.md](../../docs/preprocessing.md)
for target construction and the loss functions provided by the package.
