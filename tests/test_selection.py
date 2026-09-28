from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contact.select_fusion import load_dev, select_candidate, select_weights


ROOT = Path(__file__).resolve().parents[1]


def test_development_selection():
    dev = load_dev(ROOT / "data")
    grid, weights, normalizers = select_weights(dev)
    assert dev.groupby("corpus").size().to_dict() == {
        "hai": 70, "hh_emotion": 32, "hh_turn": 60}
    assert len(grid) == 42
    assert grid.groupby("branch").selected.sum().eq(1).all()
    assert grid.loc[grid.base_weight.eq(0.5), "eligible"].all()
    assert all(weight in grid.base_weight.values for weight in weights.values())
    assert normalizers["hai"]["F"]["n"] == 70
    dev.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="development recordings only"):
        select_weights(dev)


def test_test_rows_do_not_affect_selection(tmp_path):
    original = load_dev(ROOT / "data")
    splits = pd.read_csv(ROOT / "data/splits.csv")
    splits.to_csv(tmp_path / "splits.csv", index=False)
    test_ids = set(splits.loc[splits.split.eq("test"), "recording_id"])
    for name in ("ratings.csv", "scores.csv"):
        frame = pd.read_csv(ROOT / "data" / name)
        frame.loc[frame.recording_id.isin(test_ids), frame.columns != "recording_id"] = np.nan
        frame.to_csv(tmp_path / name, index=False)
    pd.testing.assert_frame_equal(original, load_dev(tmp_path))


def test_constraints_and_tie_break():
    rows = pd.DataFrame({
        "base_weight": [0.1, 0.2, 0.3, 0.5],
        "hh_pair_accuracy": [0.65, 0.8, 0.8, 0.8],
        "hh_c_index": [0.9, 0.69, 0.70, 0.7],
        "hh_mos_rho": [0.5, 0.3, 0.3, 0.3],
        "hai_mos_rho": [0.8, 0.4, 0.4, 0.2],
    })
    grid, weight = select_candidate(rows, 0.02)
    assert not grid.loc[0, "eligible"]
    assert weight == 0.2
    rows.loc[1, "hh_c_index"] = 0.68
    assert select_candidate(rows, 0.02)[1] == 0.3
    rows.loc[2, "hh_mos_rho"] = 0.29
    assert select_candidate(rows, 0.02)[1] == 0.5
