"""Prepare Mimi features, VAD and window-level affect targets from two tracks."""
import argparse
import importlib.metadata
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .checkpoint import sha256
from .targets import affect_windows, clean_vad, vad_from_protocol

MIMI_ID = "kyutai/mimi"
MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"
TEACHER_ID = "tiantiaf/whisper-large-v3-msp-podcast-emotion-dim"
TEACHER_REVISION = "09fe2573a1e636bad9333c408d9bd523f888145a"
WRAPPER_REVISION = "85100e60844a3f324a139e24fb9225aa6d8e45d1"


def read_track(path, channel, rate, start=0., end=None):
    with sf.SoundFile(path) as source:
        expected = 1 if channel is None else 2
        if source.channels != expected:
            raise ValueError(f"Expected {expected} audio channels: {path}")
        lo = round(start * source.samplerate)
        hi = source.frames if end is None else round(end * source.samplerate)
        if not 0 <= lo < hi <= source.frames:
            raise ValueError(f"Audio crop exceeds available samples: {path}")
        source.seek(lo)
        wave = source.read(hi - lo, dtype="float32", always_2d=True)[:, channel or 0]
        source_rate = source.samplerate
    if not np.isfinite(wave).all():
        raise ValueError(f"Nonfinite audio: {path}")
    if source_rate != rate:
        divisor = math.gcd(source_rate, rate)
        wave = resample_poly(wave, rate // divisor, source_rate // divisor).astype(np.float32)
    return wave


@torch.inference_mode()
def encode(model, wave, device):
    parts = []
    block = 300 * 24000
    for start in range(0, len(wave), block):
        x = torch.as_tensor(wave[start:start+block], device=device)[None, None]
        x = model.encoder(x)
        x = model.encoder_transformer(x.transpose(1, 2)).last_hidden_state
        x = model.downsample(x.transpose(1, 2)).squeeze(0).T
        parts.append(x.float().cpu().numpy())
    features = np.concatenate(parts).astype(np.float16)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite Mimi features")
    return features


def load_teacher(repo, device, local_only):
    repo = repo.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if revision != WRAPPER_REVISION:
        raise ValueError(f"VoxProfile must be checked out at {WRAPPER_REVISION}")
    if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=repo, text=True).strip():
        raise ValueError("VoxProfile checkout has modified tracked files")
    sys.path.insert(0, str(repo))
    from src.model.emotion.whisper_emotion_dim import WhisperWrapper
    return WhisperWrapper.from_pretrained(TEACHER_ID, revision=TEACHER_REVISION,
        local_files_only=local_only).to(device).eval().requires_grad_(False)


@torch.inference_mode()
def teacher_values(model, waves, singleton=False):
    tensors = [torch.as_tensor(wave, dtype=torch.float32) for wave in waves]
    output = (model(tensors[0][None]) if singleton else
              model(tensors, length=torch.tensor([len(t) for t in tensors])))
    values = torch.stack([v.detach().float().cpu().reshape(-1) for v in output[:3]], 1).numpy()
    if values.shape != (len(waves), 3) or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Expected finite A/V/D predictions in [0,1]")
    return values


def make_vad(tracks, model, profile="scipy-ceil"):
    from silero_vad import get_speech_timestamps
    if profile == "scipy-ceil":
        wave16 = [read_track(p, c, 16000) for p, c in tracks]
    elif profile == "torchaudio-floor":
        import torchaudio
        wave16 = []
        for path, channel in tracks:
            with sf.SoundFile(path) as source:
                source_rate = source.samplerate
            wave = torch.from_numpy(read_track(path, channel, source_rate))
            if source_rate != 16000:
                wave = torchaudio.functional.resample(wave, source_rate, 16000)
            wave16.append(wave.numpy())
    else:
        raise ValueError(f"Unknown VAD preprocessing profile: {profile}")
    if abs(len(wave16[0]) - len(wave16[1])) > 320:
        raise ValueError("Speaker tracks differ by more than 20 ms; align them before scoring")
    frames = min(map(len, wave16)) / 320
    frames = math.ceil(frames) if profile == "scipy-ceil" else math.floor(frames)
    vad = np.zeros((2, frames), np.float32)
    for channel, wave in enumerate(wave16):
        intervals = get_speech_timestamps(torch.from_numpy(wave), model, sampling_rate=16000,
            threshold=.5, min_speech_duration_ms=100, min_silence_duration_ms=50)
        for segment in intervals:
            vad[channel, segment["start"]//320:segment["end"]//320] = 1
        vad[channel] = clean_vad(vad[channel])
    return vad


def prepare(tracks, recording_id, output, teacher_repo, device, batch_size, local_only, models=None, protocol=None):
    from silero_vad import load_silero_vad
    from transformers import MimiModel

    if output.exists():
        raise FileExistsError(f"Choose a new cache directory: {output}")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Audio preparation requires CUDA for the official affect teacher")
    torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    models = {} if models is None else models
    signatures = [dict(channel=c, sha256=sha256(p)) for p, c in tracks]
    protocol = protocol or dict(vad_profile="scipy-ceil", response_merge_gap=.2)
    if "audio" in protocol and protocol["audio"] != signatures:
        raise ValueError("Audio bytes/channels differ from the benchmark preprocessing protocol")
    if "vad_runs" in protocol:
        vad = vad_from_protocol(protocol)
        lengths = [len(read_track(p, c, 16000)) / 16000 for p, c in tracks]
        if max(abs(length - vad.shape[1]/50) for length in lengths) > .020 + 1e-9:
            raise ValueError("Benchmark VAD timeline does not match audio duration")
    else:
        if "vad" not in models:
            models["vad"] = load_silero_vad().eval()
        vad = make_vad(tracks, models["vad"], protocol["vad_profile"])
    windows = affect_windows(vad, recording_id, protocol["response_merge_gap"])
    print(f"Preparing {recording_id}: {vad.shape[1]/50:.2f} s, {len(windows)} affect windows", flush=True)
    if "mimi" not in models:
        models["mimi"] = MimiModel.from_pretrained(MIMI_ID, revision=MIMI_REVISION,
            local_files_only=local_only).to(device).eval().requires_grad_(False)
    mimi = models["mimi"]
    features = [encode(mimi, read_track(p, c, 24000), device) for p, c in tracks]
    del mimi
    torch.cuda.empty_cache()
    # A sub-frame duration difference cannot supply a paired model frame.
    count = min(map(len, features))
    features = [f[:count] for f in features]
    parity = None
    if windows:
        if "teacher" not in models:
            models["teacher"] = load_teacher(teacher_repo, device, local_only)
        teacher = models["teacher"]
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start+batch_size]
            waves = [read_track(*tracks[w["speaker"]], 16000, w["start"], w["end"]) for w in batch]
            values = teacher_values(teacher, waves)
            if parity is None:
                indices = sorted({0, len(waves)-1, int(np.argmin(list(map(len, waves)))),
                                  int(np.argmax(list(map(len, waves))))})
                singles = np.concatenate([teacher_values(teacher, [waves[i]], True) for i in indices])
                reference = values[indices]
                bins = lambda x: np.minimum(7, np.floor(x[:, :2] * 8)).astype(int)
                passed = bool(np.allclose(singles, reference, atol=1e-4, rtol=1e-4)
                              and np.array_equal(bins(singles), bins(reference)))
                parity = dict(passed=passed, max_absolute_difference=float(np.abs(singles-reference).max()))
                if not passed:
                    raise ValueError(f"Teacher batch/singleton check failed: {parity}")
            for window, (a, v, d) in zip(batch, values):
                window.update(arousal=float(a), valence=float(v), dominance=float(d))
            print(f"Affect windows: {min(start+batch_size, len(windows))}/{len(windows)}", flush=True)
        del teacher
    if signatures != [dict(channel=c, sha256=sha256(p)) for p, c in tracks]:
        raise ValueError("Source audio changed during preparation")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    for channel, feature in enumerate(features):
        np.save(pending / f"mimi_ch{channel}.npy", feature)
    np.save(pending / "vad.npy", vad)
    (pending / "affect.json").write_text(json.dumps(windows, indent=2) + "\n")
    metadata = dict(format_version=1, recording_id=recording_id, audio=signatures,
        mimi=dict(model=MIMI_ID, revision=MIMI_REVISION),
        teacher=dict(model=TEACHER_ID, revision=TEACHER_REVISION, wrapper_revision=WRAPPER_REVISION),
        preparation_version=2, protocol=protocol, batch_parity=parity, versions={p: importlib.metadata.version(p) for p in
            ("torch", "transformers", "silero-vad", "soundfile", "numpy", "scipy")},
        sha256={f: sha256(pending / f) for f in ("mimi_ch0.npy", "mimi_ch1.npy", "vad.npy", "affect.json")})
    (pending / "cache.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pending.rename(output)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path, help="Stereo WAV: one speaker per channel")
    source.add_argument("--tracks", type=Path, nargs=2, metavar=("SPEAKER_A", "SPEAKER_B"))
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-repo", type=Path, default=Path("third_party/vox-profile-release"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--protocols", type=Path, help="Fixed per-recording benchmark preprocessing recipes")
    args = parser.parse_args()
    tracks = [(args.audio, c) for c in (0, 1)] if args.audio else [(p, None) for p in args.tracks]
    if not args.audio and tracks[0][0].resolve() == tracks[1][0].resolve():
        parser.error("Supply two different speaker tracks, not two copies of the mix")
    protocol = json.loads(args.protocols.read_text())[args.recording_id] if args.protocols else None
    prepare(tracks, args.recording_id, args.output, args.teacher_repo, args.device,
            args.batch_size, args.local_files_only, protocol=protocol)


if __name__ == "__main__":
    main()
