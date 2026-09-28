# Seamless Interaction training audio

The training source is the approximately 1,600-hour IPC subset of Seamless
Interaction.

- [Official dataset and download tools](https://github.com/facebookresearch/seamless_interaction)
- [Official Hugging Face dataset](https://huggingface.co/datasets/facebook/seamless-interaction)

Download audio using the provider's tools and follow the dataset's access and
license terms. The training audio is not needed to evaluate the CONTACT model.

Keep downloads in `raw/`, preserving the provider's paths. Use `recordings.csv`
to pair the synchronized participant files:

```csv
recording_id,corpus,audio,speaker_a,speaker_b
dialogue_example,seamless,,raw/path/to/participant_a.wav,raw/path/to/participant_b.wav
```

Replace the example ID and paths with your downloaded files. This release
provides preprocessing code, not a training launcher or training split manifest.

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
