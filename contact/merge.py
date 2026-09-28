"""Merge inference shards only when they cover the requested inventory exactly."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .checkpoint import sha256
from .dataset import write_csv


def merge(parts, manifest, output):
    with Path(manifest).open(newline="") as source:
        inventory = list(csv.DictReader(source))
    expected = {r["recording_id"]: r["corpus"] for r in inventory}
    if len(expected) != len(inventory):
        raise ValueError("Duplicate inventory IDs")
    records, signature, sources = {}, None, []
    for path in map(Path, parts):
        meta = json.loads(path.with_suffix(".run.json").read_text())
        if meta["prediction_sha256"] != sha256(path):
            raise ValueError(f"Prediction checksum mismatch: {path}")
        if signature is not None and signature != meta["signature"]:
            raise ValueError("Shards used different model weights, code or scoring settings")
        signature = meta["signature"]
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != meta["records"]:
            raise ValueError(f"Shard count mismatch: {path}")
        for row in rows:
            rid = row["recording_id"]
            if rid in records or expected.get(rid) != row["corpus"]:
                raise ValueError(f"Duplicate, extra or wrong-corpus prediction: {rid}")
            columns = ("F_raw", "S_raw") if row["corpus"] == "hh_turn" else ("F_raw", "S_raw", "A_raw")
            if not all(np.isfinite(float(row[c])) for c in columns):
                raise ValueError(f"Missing/nonfinite score: {rid}")
            records[rid] = row
        sources.append(dict(filename=path.name, sha256=meta["prediction_sha256"]))
    missing = expected.keys() - records.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} recordings; first: {sorted(missing)[:5]}")
    ordered = [records[r["recording_id"]] for r in inventory]
    if not ordered:
        raise ValueError("Empty inventory")
    write_csv(output, ordered, list(ordered[0]))
    Path(output).with_suffix(".run.json").write_text(json.dumps(dict(
        signature=signature, prediction_sha256=sha256(output), records=len(records), parts=sources), indent=2) + "\n")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Merged {merge(args.parts, args.manifest, args.output)} recordings to {args.output}")


if __name__ == "__main__":
    main()
