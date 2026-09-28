import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from contact.checkpoint import sha256, verify
from contact.infer import load_cache, score_arrays, standardize
from contact.model import ContactModel


def small_config():
    return dict(qwen=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64, max_position_embeddings=128),
        mimi_dim=512, hidden_dim=32, attention_heads=4, timing_layers=1, affect_layers=1,
        dropout=.1, silence_hidden_dim=16, silence_bins=10)


def test_model_shapes_and_causality():
    torch.manual_seed(42)
    model = ContactModel(small_config()).eval()
    x = torch.randn(1, 18, 512)
    changed = x.clone()
    changed[:, 10:] += 10
    time = (torch.arange(18, dtype=torch.float64)[None]+1)/12.5+.04
    valid = torch.ones(1, 18, dtype=torch.bool)
    with torch.inference_mode():
        original = model(x, x, time, valid)
        modified = model(changed, changed, time, valid)
        projected = model.base.model.mimi_projection(x, x)
        native = model.base.model.backbone(inputs_embeds=projected, output_hidden_states=True, return_dict=True)
        expected = torch.stack(native.hidden_states).mean(0)
        actual = model.base._hidden_from_features(x, x, ("fvad",))["fvad"]
        torch.testing.assert_close(actual, expected)
    assert original["activity"].shape == (1, 18, 256)
    assert original["silence"].shape == (1, 18, 10)
    assert original["affect_factors"].shape == (1, 18, 2, 2, 8)
    for key in original:
        torch.testing.assert_close(original[key][:, :10], modified[key][:, :10], rtol=1e-5, atol=1e-6)
    factors = original["affect_factors"].softmax(-1)
    joint = (factors[..., 0, :, None] * factors[..., 1, None, :]).flatten(-2)
    torch.testing.assert_close(joint, original["affect"].softmax(-1))


class UniformPredictor:
    def __call__(self, ch0, ch1, available, valid):
        return {name: torch.zeros(shape) for name, shape in
            (("activity", (1, 250, 256)), ("silence", (1, 250, 10)), ("affect", (1, 250, 2, 64)))}


def test_recording_readout_and_standardization():
    features = [np.zeros((400, 512), np.float16)] * 2
    vad = np.zeros((2, 1600), np.float32)
    windows = [dict(window_id=f"w{c}", response_id=f"r{c}", speaker=c,
                    start=6.+c*3, end=9.+c*3, arousal=.5, valence=.5) for c in (0, 1)]
    scores = score_arrays(UniformPredictor(), features, vad, windows, np.full(64, 1/64))
    assert scores["F_raw"] == pytest.approx(-np.log(256), abs=1e-6)
    assert scores["S_raw"] == pytest.approx(-np.log(10), abs=1e-6)
    assert scores["A_raw"] == 0
    assert scores["valid_activity_frames"] == 337
    assert scores["valid_affect_windows"] == 2
    stats = {"hai": {key: dict(mean=0, std=1) for key in ("F", "S", "A")}}
    result = standardize(scores, "hai", stats)
    assert result["timing"] == .45*scores["F_raw"]+.55*scores["S_raw"]
    assert result["affect"] == .75*scores["S_raw"]


def test_cache_and_checkpoint_checksums(tmp_path):
    for c in (0, 1):
        np.save(tmp_path/f"mimi_ch{c}.npy", np.zeros((100, 512), np.float16))
    np.save(tmp_path/"vad.npy", np.zeros((2, 400)))
    (tmp_path/"affect.json").write_text("[]")
    files = ("mimi_ch0.npy", "mimi_ch1.npy", "vad.npy", "affect.json")
    meta = dict(format_version=1, recording_id="example", sha256={f: sha256(tmp_path/f) for f in files})
    (tmp_path/"cache.json").write_text(json.dumps(meta))
    assert load_cache(tmp_path)[0]["recording_id"] == "example"
    (tmp_path/"affect.json").write_text("[0]")
    with pytest.raises(ValueError, match="checksum"):
        load_cache(tmp_path)
    weight = tmp_path/"example.pt"
    weight.write_bytes(b"example")
    manifest = dict(filename="example.pt", bytes=weight.stat().st_size, sha256=sha256(weight))
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    assert verify(weight) == manifest
    weight.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        verify(weight)


def test_sharded_cli_resume_and_merge(tmp_path, monkeypatch):
    import sys
    import contact.infer as inference
    from contact.dataset import write_csv
    from contact.merge import merge

    rows = []
    for rid in ("r0", "r1"):
        folder = tmp_path / rid
        folder.mkdir()
        for c in (0, 1):
            np.save(folder / f"mimi_ch{c}.npy", np.zeros((400, 512), np.float16))
        np.save(folder / "vad.npy", np.zeros((2, 1600)))
        windows = [dict(window_id=f"w{c}", response_id=f"response{c}", speaker=c,
                        start=6.+c*3, end=9.+c*3, arousal=.5, valence=.5) for c in (0, 1)]
        (folder / "affect.json").write_text(json.dumps(windows))
        (folder / "cache.json").write_text(json.dumps(dict(format_version=1, recording_id=rid,
            sha256={f: sha256(folder/f) for f in ("mimi_ch0.npy", "mimi_ch1.npy", "vad.npy", "affect.json")})))
        rows.append(dict(recording_id=rid, corpus="hai", cache_dir=rid))
    manifest = tmp_path / "recordings.csv"
    write_csv(manifest, rows, list(rows[0]))
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"test fixture")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(small_config()))
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(dict(training_target_mean=[1/64]*64)))
    normalizers = tmp_path / "normalizers.json"
    normalizers.write_text(json.dumps(dict(hai={k: dict(mean=0, std=1) for k in ("F", "S", "A")})))
    monkeypatch.setattr(inference, "verify", lambda _: None)
    monkeypatch.setattr(inference, "load_model", lambda *args: UniformPredictor())
    common = ["infer", "--manifest", str(manifest), "--checkpoint", str(checkpoint),
              "--model-config", str(config), "--prior", str(prior), "--normalizers", str(normalizers),
              "--resume-dir", str(tmp_path / "resume")]
    parts = []
    for shard in range(2):
        path = tmp_path / f"scores{shard}.csv"
        monkeypatch.setattr(sys, "argv", common + ["--num-shards", "2", "--shard-index", str(shard), "--output", str(path)])
        inference.main()
        parts.append(path)
    assert merge(parts, manifest, tmp_path / "merged.csv") == 2

    def unexpected_load(*args):
        raise AssertionError("Verified cached predictions should not load the model")
    monkeypatch.setattr(inference, "load_model", unexpected_load)
    monkeypatch.setattr(sys, "argv", common + ["--output", str(tmp_path / "resumed.csv")])
    inference.main()
    assert (tmp_path / "resumed.csv").read_text() == (tmp_path / "merged.csv").read_text()
    prior.write_text(json.dumps(dict(training_target_mean=[.5/64]*64)))
    monkeypatch.setattr(sys, "argv", common + ["--output", str(tmp_path / "changed.csv")])
    with pytest.raises(ValueError, match="Resume inputs"):
        inference.main()
