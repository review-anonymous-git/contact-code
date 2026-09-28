import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from contact.checkpoint import sha256
from contact.dataset import check_audio, init, read_manifest, tracks_for, verify_inventory, write_csv
from contact.merge import merge


def test_layout_and_inventory(tmp_path):
    source = tmp_path / "split.csv"
    inventory = [dict(recording_id="hh_turn__r1", corpus="hh_turn"),
                 dict(recording_id="hai__r2", corpus="hai")]
    write_csv(source, inventory, ("recording_id", "corpus"))
    output = tmp_path / "audio" / "recordings.csv"
    init(argparse.Namespace(inventory=source, root=tmp_path / "audio", output=output))
    rows = read_manifest(output)
    verify_inventory(rows, source)
    assert tracks_for(rows[0], output) == [(tmp_path / "audio/hh_turn/hh_turn__r1/stereo.wav", c) for c in (0, 1)]
    with pytest.raises(ValueError, match="Inventory"):
        verify_inventory(rows[:1], source)
    with pytest.raises(FileExistsError):
        init(argparse.Namespace(inventory=source, root=tmp_path / "audio", output=output))


@pytest.mark.parametrize("change,reason", [
    ({"recording_id": "../bad"}, "Unsafe"),
    ({"speaker_a": "a.wav"}, "OR"),
    ({"audio": "", "speaker_a": "a.wav"}, "OR"),
    ({"corpus": "not-a-corpus"}, "Unsupported"),
])
def test_reject_bad_manifest(tmp_path, change, reason):
    row = dict(recording_id="r1", corpus="hai", audio="stereo.wav", speaker_a="", speaker_b="")
    row.update(change)
    path = tmp_path / "recordings.csv"
    write_csv(path, [row], list(row))
    with pytest.raises(ValueError, match=reason):
        read_manifest(path)


def test_audio_preflight(tmp_path):
    sf = pytest.importorskip("soundfile")
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, np.zeros(16000), 16000)
    sf.write(b, np.zeros(16000), 16000)
    assert check_audio([(a, None), (b, None)]) == 1
    with pytest.raises(ValueError, match="channel"):
        check_audio([(a, 0), (a, 1)])
    sf.write(b, np.zeros(17000), 16000)
    with pytest.raises(ValueError, match="20 ms"):
        check_audio([(a, None), (b, None)])
    row = dict(audio="", speaker_a="a.wav", speaker_b="./a.wav")
    with pytest.raises(ValueError, match="different"):
        tracks_for(row, tmp_path / "recordings.csv")


def make_part(path, rows, signature=None):
    write_csv(path, rows, list(rows[0]))
    path.with_suffix(".run.json").write_text(json.dumps(dict(
        records=len(rows), signature=signature or {"checkpoint": "abc"}, prediction_sha256=sha256(path))))


def test_merge_requires_full_coverage_and_same_model(tmp_path):
    rows = [dict(recording_id=f"r{i}", corpus="hai", F_raw=-1, S_raw=-2, A_raw=-3) for i in range(2)]
    inventory = tmp_path / "inventory.csv"
    write_csv(inventory, [dict(recording_id=r["recording_id"], corpus="hai") for r in rows], ("recording_id", "corpus"))
    parts = [tmp_path / f"part{i}.csv" for i in range(2)]
    for part, row in zip(parts, rows):
        make_part(part, [row])
    with pytest.raises(ValueError, match="Missing"):
        merge(parts[:1], inventory, tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="Duplicate"):
        merge([parts[0], parts[0]], inventory, tmp_path / "duplicate.csv")
    assert merge(parts[::-1], inventory, tmp_path / "complete.csv") == 2
    meta_path = parts[1].with_suffix(".run.json")
    meta = json.loads(meta_path.read_text())
    meta["signature"]["checkpoint"] = "different"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different model"):
        merge(parts, inventory, tmp_path / "mixed.csv")
    parts[0].write_text(parts[0].read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        merge(parts, inventory, tmp_path / "altered.csv")


def test_resume_fingerprint_uses_inputs_and_settings():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from contact.infer import cache_signature
    meta = dict(recording_id="r1", sha256={"vad.npy": "original"})
    first = cache_signature({"checkpoint": "abc"}, meta, "hai")
    assert first == cache_signature({"checkpoint": "abc"}, meta, "hai")
    assert first != cache_signature({"checkpoint": "other"}, meta, "hai")
    assert first != cache_signature({"checkpoint": "abc"}, meta, "hh_turn")
    meta["sha256"]["vad.npy"] = "changed"
    assert first != cache_signature({"checkpoint": "abc"}, meta, "hai")


def test_atomic_prediction_never_overwrites(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from contact.infer import save_record
    path = tmp_path / "resume/r1.json"
    save_record(path, {"value": 1})
    with pytest.raises(FileExistsError):
        save_record(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}
    assert not list(path.parent.glob(".pending-*"))
