# Model weights

Place `contact.pt` here. The weights are distributed separately from Git.
`manifest.json` records the file size and SHA-256 checksum; its download URL
will be filled in when the model archive is uploaded.

The file contains the final predictor's parameters, without optimizer state,
training caches or machine-specific paths. Mimi and the dimensional-affect
teacher are separate pretrained models, listed in `configs/preprocessing.json`.

Verify the download before inference:

```bash
python -m contact.checkpoint --checkpoint checkpoints/contact.pt
```
