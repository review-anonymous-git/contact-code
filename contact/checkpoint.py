"""Checkpoint integrity and loading."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, manifest=None):
    path = Path(path)
    manifest = Path(manifest) if manifest else path.parent / "manifest.json"
    info = json.loads(manifest.read_text())
    if not path.is_file():
        raise FileNotFoundError(f"Place the downloaded {info['filename']} in {path.parent}")
    if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return info


def load_state(path):
    import torch
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Expected CONTACT inference checkpoint format 1")
    return payload["state_dict"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/contact.pt"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    info = verify(args.checkpoint, args.manifest)
    print(f"Verified {args.checkpoint}: {info['bytes']:,} bytes, SHA-256 {info['sha256']}")


if __name__ == "__main__":
    main()
