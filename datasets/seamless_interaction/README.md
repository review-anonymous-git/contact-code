# Seamless Interaction training audio

The training source is the IPC subset of Seamless Interaction.
[training_manifest.csv](training_manifest.csv) lists the 28,038 paired recordings
used for training, totaling 1,635.830933 dialogue hours (1,636 hours rounded).
Each dialogue is counted once, not once per speaker.

- [Official dataset and download tools](https://github.com/facebookresearch/seamless_interaction)
- [Official Hugging Face dataset](https://huggingface.co/datasets/facebook/seamless-interaction)

Download audio using the provider's tools and follow the dataset's access and
license terms. The training audio is not needed to evaluate the CONTACT model.

Keep downloads in `raw/`, preserving the provider's paths. The manifest contains:

- `recording_id`: paired recording ID.
- `upstream_dialogue_id`: dialogue ID from the training metadata.
- `speaker_a`, `speaker_b`: upstream audio paths relative to this directory.
- `duration_sec`: paired recording duration in seconds.
- `corpus`, `audio`: preprocessing fields; `audio` is empty for paired mono tracks.

For example, an upstream `naturalistic/train/...wav` file belongs at
`raw/naturalistic/train/...wav`. The manifest is directly readable by the
preprocessing command; it requires no local path substitution.

The same audio preprocessing can be used to construct Mimi features, VAD and
A/V pseudo-labels from a populated manifest:

```bash
python -m contact.dataset prepare \
  --manifest datasets/seamless_interaction/training_manifest.csv \
  --cache-root cache/seamless --output-manifest cache/seamless/recordings.csv \
  --device cuda:0 --batch-size 4 --skip-existing
```

`seamless` identifies preprocessing data only: CONTACT's released evaluation
normalizers are for its three evaluation subsets, not a calibrated score for
the training corpus. See [../../docs/preprocessing.md](../../docs/preprocessing.md)
for target construction and the loss functions provided by the package.
The release does not include a training launcher or optimization loop.
