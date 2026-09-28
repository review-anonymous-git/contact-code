# Model weights

Download [contact.zip](https://github.com/review-anonymous-git/contact-code/releases/download/v1.0/contact.zip)
from the [v1.0 release](https://github.com/review-anonymous-git/contact-code/releases/tag/v1.0).
Extract the archive and place `contact.pt` in this directory. The weights are
distributed as a Release attachment, not as a Git-tracked file.

`manifest.json` records the download URL and checksums. The top-level size and
SHA-256 describe the extracted `contact.pt`; the `archive` entry describes the ZIP.

The file contains the final predictor's parameters, without optimizer state,
training caches or machine-specific paths. Mimi and the dimensional-affect
teacher are separate pretrained models, listed in `configs/preprocessing.json`.

Verify the download before inference:

```bash
python -m contact.checkpoint --checkpoint checkpoints/contact.pt
```
