from pathlib import Path
import json

import numpy as np
import pytest

from contact.evaluate import load_inputs, metric_rows

DATA = Path(__file__).resolve().parents[1] / "data"


def test_fixed_release_and_paper_endpoints():
    frame, _ = load_inputs(DATA)
    assert len(frame) == 493
    result = metric_rows(frame)
    ours = result[result.scorer.eq("Ours") & result.split.eq("test")]
    for corpus, dim, target in [("hh_turn", "timing", .364), ("hh_emotion", "affect", .281),
                                ("hai", "timing", .397), ("hai", "affect", .371),
                                ("hai", "overall", .358)]:
        value = ours.loc[ours.corpus.eq(corpus) & ours.dimension.eq(dim)
                         & ours.rater.eq("combined") & ours.metric.eq("rho"), "value"].item()
        assert value == pytest.approx(target, abs=.0005)
    assert np.isfinite(result.value).all()


def test_ablation_is_score_removal():
    frame, _ = load_inputs(DATA)
    np.testing.assert_allclose(frame["T"], .45*frame.F+.55*frame.S)
    np.testing.assert_allclose(frame.E, .25*frame.A+.75*frame.S, equal_nan=True)


def test_inference_csv_replaces_components_without_refitting(tmp_path):
    frame, _ = load_inputs(DATA)
    path = tmp_path / "predictions.csv"
    values = frame[["recording_id", "corpus", "F_raw", "S_raw", "A_raw"]].copy()
    values.F_raw += 1
    values.iloc[::-1].to_csv(path, index=False)
    changed, _ = load_inputs(DATA, path)
    assert (changed.F > frame.F).all()
    np.testing.assert_allclose(changed.E, frame.E, equal_nan=True)
    values.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="all released"):
        load_inputs(DATA, path)


def test_benchmark_preprocessing_inventory():
    from contact.targets import vad_from_protocol
    frame, _ = load_inputs(DATA)
    protocols = json.loads((DATA / "audio_preprocessing.json").read_text())
    assert set(protocols) == set(frame.recording_id)
    for recipe in protocols.values():
        assert recipe["response_merge_gap"] in (.2, .5)
        assert recipe["vad_profile"] in ("scipy-ceil", "torchaudio-floor")
        assert len(recipe["audio"]) == 2
        assert all(set(a) == {"channel", "sha256"} for a in recipe["audio"])
        assert np.isin(vad_from_protocol(recipe), [0, 1]).all()
