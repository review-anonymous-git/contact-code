"""Prepare a manifest of separate-speaker recordings. No MOS labels are read."""
import argparse
import csv
import json
import os
import re
from pathlib import Path

from .checkpoint import sha256

CORPORA = ("hh_turn", "hh_emotion", "hai", "seamless")


def read_manifest(path):
    with Path(path).open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or not {"recording_id", "corpus", "audio", "speaker_a", "speaker_b"} <= rows[0].keys():
        raise ValueError("Expected CSV columns: recording_id,corpus,audio,speaker_a,speaker_b")
    if len({r["recording_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate recording IDs")
    for r in rows:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", r["recording_id"]):
            raise ValueError(f"Unsafe recording ID: {r['recording_id']!r}")
        if r["corpus"] not in CORPORA:
            raise ValueError(f"Unsupported corpus: {r['corpus']}")
        a, b, stereo = r["speaker_a"], r["speaker_b"], r["audio"]
        if not ((stereo and not a and not b) or (not stereo and a and b)):
            raise ValueError(f"Supply stereo audio OR two speaker tracks: {r['recording_id']}")
    return rows


def tracks_for(row, manifest):
    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (Path(manifest).parent / path).resolve()
    if row["audio"]:
        return [(resolve(row["audio"]), c) for c in (0, 1)]
    tracks = [(resolve(row[key]), None) for key in ("speaker_a", "speaker_b")]
    if tracks[0][0] == tracks[1][0]:
        raise ValueError("Speaker tracks must be different files")
    return tracks


def check_audio(tracks):
    import soundfile as sf
    durations = []
    for path, channel in tracks:
        info = sf.info(path)
        if info.channels != (1 if channel is None else 2) or info.frames <= 0:
            raise ValueError(f"Empty audio or unexpected channel count: {path}")
        durations.append(info.frames / info.samplerate)
    if abs(durations[0] - durations[1]) > .020 + 1e-9:
        raise ValueError("Tracks differ by more than 20 ms; do not silently trim or pad")
    return min(durations)


def verify_inventory(rows, inventory):
    with Path(inventory).open(newline="") as source:
        expected = {r["recording_id"]: r["corpus"] for r in csv.DictReader(source)}
    actual = {r["recording_id"]: r["corpus"] for r in rows}
    if actual != expected:
        raise ValueError(f"Inventory mismatch: {len(expected.keys()-actual.keys())} missing, "
                         f"{len(actual.keys()-expected.keys())} extra; corpus names must also match")


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def init(args):
    with args.inventory.open(newline="") as source:
        inventory = list(csv.DictReader(source))
    rows = []
    for r in inventory:
        folder = args.root.resolve() / r["corpus"] / r["recording_id"]
        folder.mkdir(parents=True, exist_ok=True)
        rows.append(dict(recording_id=r["recording_id"], corpus=r["corpus"],
                         audio=os.path.relpath(folder / "stereo.wav", args.output.resolve().parent),
                         speaker_a="", speaker_b=""))
    write_csv(args.output, rows, ("recording_id", "corpus", "audio", "speaker_a", "speaker_b"))
    print(f"Wrote {len(rows)} recording locations to {args.output}; no audio was downloaded")


def prepare_all(args):
    rows = read_manifest(args.manifest)
    protocols = json.loads(args.protocols.read_text()) if args.protocols else None
    if protocols is not None and any(r["recording_id"] not in protocols for r in rows):
        raise ValueError("A recording is missing from the supplied preprocessing protocols")
    if args.inventory:
        verify_inventory(rows, args.inventory)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    selected = rows[args.shard_index::args.num_shards]
    if not selected:
        raise ValueError("Empty shard")
    seconds = 0.
    for row in selected:
        tracks = tracks_for(row, args.manifest)
        seconds += check_audio(tracks)
        if protocols is not None:
            signatures = [dict(channel=c, sha256=sha256(p)) for p, c in tracks]
            if protocols[row["recording_id"]].get("audio") != signatures:
                raise ValueError(f"Audio differs from the benchmark protocol: {row['recording_id']}")
    print(f"Checked {len(selected)} recordings, {seconds/3600:.2f} dialogue hours", flush=True)
    if args.check_only:
        return
    if args.output_manifest.exists():
        raise FileExistsError(f"Choose a new output manifest: {args.output_manifest}")
    from .prepare import MIMI_REVISION, TEACHER_REVISION, WRAPPER_REVISION, prepare
    from .infer import load_cache
    models, completed = {}, []
    for i, row in enumerate(selected, 1):
        folder = args.cache_root / row["recording_id"]
        tracks = tracks_for(row, args.manifest)
        protocol = protocols[row["recording_id"]] if protocols else dict(vad_profile="scipy-ceil", response_merge_gap=.2)
        if folder.exists() and args.skip_existing:
            meta = load_cache(folder)[0]
            signatures = [dict(channel=c, sha256=sha256(p)) for p, c in tracks]
            if (meta["recording_id"] != row["recording_id"] or meta.get("audio") != signatures
                    or meta.get("preparation_version") != 2 or meta.get("protocol") != protocol
                    or meta.get("mimi", {}).get("revision") != MIMI_REVISION
                    or meta.get("teacher", {}).get("revision") != TEACHER_REVISION
                    or meta.get("teacher", {}).get("wrapper_revision") != WRAPPER_REVISION):
                raise ValueError(f"Existing cache has different inputs/settings: {folder}")
            print(f"[{i}/{len(selected)}] verified cache: {row['recording_id']}", flush=True)
        else:
            prepare(tracks, row["recording_id"], folder, args.teacher_repo,
                    args.device, args.batch_size, args.local_files_only, models, protocol)
        completed.append(dict(recording_id=row["recording_id"], corpus=row["corpus"],
            cache_dir=os.path.relpath(folder.resolve(), args.output_manifest.resolve().parent)))
    write_csv(args.output_manifest, completed, ("recording_id", "corpus", "cache_dir"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("init", help="Create the paper inventory and empty audio locations")
    create.add_argument("--inventory", type=Path, default=Path("data/splits.csv"))
    create.add_argument("--root", type=Path, default=Path("datasets/contact"))
    create.add_argument("--output", type=Path, default=Path("datasets/contact/recordings.csv"))
    create.set_defaults(func=init)
    prep = sub.add_parser("prepare", help="Check audio and build the feature/target cache")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--inventory", type=Path, help="Require exactly these recording IDs and corpora")
    prep.add_argument("--protocols", type=Path, help="Fixed per-recording benchmark preprocessing recipes")
    prep.add_argument("--cache-root", type=Path, default=Path("cache/contact"))
    prep.add_argument("--output-manifest", type=Path, default=Path("cache/contact/recordings.csv"))
    prep.add_argument("--teacher-repo", type=Path, default=Path("third_party/vox-profile-release"))
    prep.add_argument("--device", default="cuda:0")
    prep.add_argument("--batch-size", type=int, default=4)
    prep.add_argument("--local-files-only", action="store_true")
    prep.add_argument("--skip-existing", action="store_true")
    prep.add_argument("--check-only", action="store_true")
    prep.add_argument("--num-shards", type=int, default=1)
    prep.add_argument("--shard-index", type=int, default=0)
    prep.set_defaults(func=prepare_all)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
