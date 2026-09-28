"""Score cached two-channel recordings with the released checkpoint."""
import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import sha256, verify
from .model import load_model
from .readout import affect_gain_gap, factorized_prior, scoring_regions, soft_av
from .targets import activity_targets, select_before, silence_targets


def load_cache(folder):
    folder = Path(folder)
    meta = json.loads((folder / "cache.json").read_text())
    if meta["format_version"] != 1:
        raise ValueError("Unsupported feature-cache format")
    expected = {"mimi_ch0.npy", "mimi_ch1.npy", "vad.npy", "affect.json"}
    if set(meta["sha256"]) != expected:
        raise ValueError("Cache manifest must cover both channels, VAD and affect targets")
    for filename, checksum in meta["sha256"].items():
        if sha256(folder / filename) != checksum:
            raise ValueError(f"Cache checksum mismatch: {filename}")
    channels = [np.load(folder / f"mimi_ch{c}.npy", mmap_mode="r") for c in (0, 1)]
    vad = np.load(folder / "vad.npy")
    windows = json.loads((folder / "affect.json").read_text())
    if (channels[0].shape != channels[1].shape or channels[0].ndim != 2
            or channels[0].shape[1] != 512 or not len(channels[0])):
        raise ValueError("Expected aligned [frames,512] Mimi channels")
    if vad.ndim != 2 or vad.shape[0] != 2 or not np.isin(vad, [0, 1]).all():
        raise ValueError("Expected binary [2,frames] VAD")
    if any(not np.isfinite(x).all() for x in channels):
        raise ValueError("Nonfinite Mimi features")
    if len({w["window_id"] for w in windows}) != len(windows):
        raise ValueError("Duplicate affect targets")
    for w in windows:
        if w["speaker"] not in (0, 1) or not 0 <= w["start"] < w["end"] <= vad.shape[1] / 50 + .02:
            raise ValueError("Invalid affect interval or channel")
        soft_av(w["arousal"], w["valence"])
    return meta, channels, vad, windows


@torch.inference_mode()
def score_arrays(model, channels, vad, windows, prior, device="cpu"):
    n = len(channels[0])
    frame_losses, silence_losses, responses = [], [], []
    used = set()
    for start, lo, hi in scoring_regions(n):
        count = min(250, n - start)
        tensors = []
        for channel in channels:
            values = np.zeros((1, 250, 512), np.float32)
            values[0, :count] = channel[start:start+count]
            tensors.append(torch.as_tensor(values, device=device))
        available = (np.arange(250, dtype=np.float64) + start + 1) / 12.5 + .04
        valid = torch.arange(250, device=device)[None] < count
        prediction = model(*tensors, torch.as_tensor(available[None], device=device), valid)
        labels, valid_labels = activity_targets(vad, start, 250, n)
        loss = F.cross_entropy(prediction["activity"][0],
            torch.as_tensor(labels, device=device), reduction="none").cpu().numpy()
        indices = np.arange(lo-start, hi-start)
        frame_losses.extend(loss[indices[valid_labels[indices]]].tolist())
        quiet = silence_targets(vad, start, 250, count)
        indices = indices[quiet[indices] >= 0]
        if len(indices):
            loss = F.cross_entropy(prediction["silence"][0, indices],
                torch.as_tensor(quiet[indices], device=device), reduction="none")
            silence_losses.extend(loss.cpu().tolist())
        for window in windows:
            if not lo <= window["start"] * 12.5 < hi or window["window_id"] in used:
                continue
            used.add(window["window_id"])
            last = select_before(available, window["start"], count)
            if last is None:
                continue
            q = soft_av(window["arousal"], window["valence"])[2]
            logits = prediction["affect"][0, last, window["speaker"]]
            ce = float(-(torch.as_tensor(q, dtype=torch.float32, device=device) * logits.log_softmax(-1)).sum())
            responses.append(dict(window, valid=True, cross_entropy=ce))
    if not frame_losses or not silence_losses:
        raise ValueError("Recording has no valid activity/silence targets after context and future masking")
    affect = None
    if {w["speaker"] for w in responses} == {0, 1}:
        affect = affect_gain_gap(responses, prior)
    return dict(F_raw=-float(np.mean(frame_losses)), S_raw=-float(np.mean(silence_losses)),
                A_raw=affect, valid_activity_frames=len(frame_losses),
                valid_silence_anchors=len(silence_losses), valid_affect_windows=len(responses))


def standardize(raw, corpus, normalizers):
    stats = normalizers[corpus]
    result = {}
    for component in ("F", "S", "A"):
        value = raw[component + "_raw"]
        s = stats.get(component)
        result[component] = (value - s["mean"]) / s["std"] if value is not None and s else None
    result["timing"] = .45 * result["F"] + .55 * result["S"]
    result["affect"] = .25 * result["A"] + .75 * result["S"] if result["A"] is not None else None
    result["overall"] = (result["timing"] + result["affect"]) / 2 if result["affect"] is not None else None
    return result


def run_signature(checkpoint, model_config, prior, normalizers):
    files = dict(checkpoint=checkpoint, model_config=model_config, prior=prior, normalizers=normalizers)
    for name in ("infer", "model", "targets", "readout"):
        files[name] = Path(__file__).with_name(name + ".py")
    return {name: sha256(path) for name, path in files.items()}


def cache_signature(run, meta, corpus):
    payload = dict(run=run, recording_id=meta["recording_id"], corpus=corpus, inputs=meta["sha256"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_record(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", prefix=".pending-", suffix=".json",
                                     dir=path.parent, delete=False) as dest:
        temporary = Path(dest.name)
        try:
            json.dump(payload, dest, indent=2, allow_nan=False)
            dest.flush()
            os.fsync(dest.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache", type=Path, help="One recording cache")
    source.add_argument("--manifest", type=Path, help="CSV: recording_id,corpus,cache_dir")
    parser.add_argument("--corpus", choices=("hh_turn", "hh_emotion", "hai"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/contact.pt"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.json"))
    parser.add_argument("--prior", type=Path, default=Path("data/affect_prior.json"))
    parser.add_argument("--normalizers", type=Path, default=Path("data/normalizers.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume-dir", type=Path, help="Save and reuse verified per-recording predictions")
    args = parser.parse_args()
    if args.cache and args.corpus is None:
        parser.error("--cache requires --corpus")
    if args.output.exists():
        parser.error("Output exists; choose a new output file")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard-index < num-shards")
    if args.cache:
        rows = [dict(cache_dir=args.cache, corpus=args.corpus)]
    else:
        with args.manifest.open() as source:
            rows = list(csv.DictReader(source))
        if len({r["recording_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate manifest recording IDs")
    if not rows or any(r["corpus"] not in ("hh_turn", "hh_emotion", "hai") for r in rows):
        parser.error("Manifest must contain recordings with supported corpus names")
    rows = rows[args.shard_index::args.num_shards]
    if not rows:
        parser.error("Empty shard")
    verify(args.checkpoint)
    signature = run_signature(args.checkpoint, args.model_config, args.prior, args.normalizers)
    model = None
    prior_data = json.loads(args.prior.read_text())
    prior = factorized_prior(prior_data["training_target_mean"])
    normalizers = json.loads(args.normalizers.read_text())
    results = []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        folder = Path(row["cache_dir"])
        if args.manifest and not folder.is_absolute():
            folder = args.manifest.parent / folder
        meta, channels, vad, windows = load_cache(folder)
        if "recording_id" in row and meta["recording_id"] != row["recording_id"]:
            raise ValueError("Manifest and cache recording IDs differ")
        fingerprint = cache_signature(signature, meta, row["corpus"])
        saved = None
        if args.resume_dir:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", meta["recording_id"]):
                raise ValueError("Unsafe recording ID for resume storage")
            saved = args.resume_dir / (meta["recording_id"] + ".json")
        if saved and saved.exists():
            cached = json.loads(saved.read_text())
            if cached["fingerprint"] != fingerprint:
                raise ValueError(f"Resume inputs, code or checkpoint changed: {saved}")
            result = cached["result"]
            print(f"[{index}/{len(rows)}] verified prediction: {meta['recording_id']}", flush=True)
        else:
            if model is None:
                model = load_model(args.checkpoint, args.model_config, args.device)
            raw = score_arrays(model, channels, vad, windows, prior, args.device)
            result = dict(recording_id=meta["recording_id"], corpus=row["corpus"], **raw,
                          **standardize(raw, row["corpus"], normalizers))
            if saved:
                save_record(saved, dict(fingerprint=fingerprint, result=result))
        results.append(result)
        print(f"[{index}/{len(rows)}] {result['recording_id']}: timing={result['timing']:.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    args.output.with_suffix(".run.json").write_text(json.dumps(dict(
        signature=signature, prediction_sha256=sha256(args.output), records=len(results),
        device=args.device, num_shards=args.num_shards, shard_index=args.shard_index,
        seconds=time.monotonic()-started), indent=2) + "\n")


if __name__ == "__main__":
    main()
