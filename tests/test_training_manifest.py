import csv
import re
from decimal import Decimal
from pathlib import Path, PurePosixPath

from contact.dataset import read_manifest


def test_training_inventory():
    path = Path(__file__).resolve().parents[1] / "datasets/seamless_interaction/training_manifest.csv"
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert read_manifest(path) == rows
    assert len(rows) == len({row["recording_id"] for row in rows}) == 28038
    assert len({(row["speaker_a"], row["speaker_b"]) for row in rows}) == 28038
    assert sum(Decimal(row["duration_sec"]) for row in rows) == Decimal("5888991.36")
    for row in rows:
        assert row["corpus"] == "seamless" and not row["audio"]
        assert row["speaker_a"] != row["speaker_b"]
        assert Decimal(row["duration_sec"]) > 0
        for key in ("speaker_a", "speaker_b"):
            audio = PurePosixPath(row[key])
            assert not audio.is_absolute() and ".." not in audio.parts
            assert audio.parts[:3] == ("raw", "naturalistic", "train")
            source_id = re.fullmatch(r"V(\d+)_S(\d+)_I(\d+)", row["upstream_dialogue_id"])
            track_id = re.fullmatch(r"V(\d+)_S(\d+)_I(\d+)_P\d+[A-Z]*", audio.stem)
            assert source_id and track_id
            assert tuple(map(int, source_id.groups())) == tuple(map(int, track_id.groups()))
