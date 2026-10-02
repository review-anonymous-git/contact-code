"""Refit fusion weights on released dev data without changing the paper readout."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contact.metrics import discrimination, rho
from contact.readout import fit_normalizers, fuse


GRID = np.round(np.linspace(0, 1, 21), 2)
BRANCHES = {
    "timing": ("hh_turn", "F", "timing_combined", 0.02),
    "affect": ("hh_emotion", "A", "affect_combined", 0.04),
}


def load_dev(data):
    data = Path(data)
    splits = pd.read_csv(data / "splits.csv")
    if splits.recording_id.duplicated().any():
        raise ValueError("Duplicate recording IDs in splits.csv")
    frame = splits.loc[splits.split.eq("dev"),
                       ["recording_id", "corpus", "split", "group_id", "condition"]].copy()
    if frame.empty:
        raise ValueError("No development recordings")
    for name, columns in (
        ("scores.csv", ["F_raw", "S_raw", "A_raw"]),
        ("ratings.csv", ["timing_combined", "affect_combined"]),
    ):
        values = pd.read_csv(data / name, usecols=["recording_id", *columns])
        values = values.loc[values.recording_id.isin(frame.recording_id)]
        frame = frame.merge(values, on="recording_id", how="left", validate="one_to_one")
    return frame.sort_values("recording_id").reset_index(drop=True)


def select_candidate(rows, max_accuracy_loss, max_c_index_loss=0.01):
    rows = rows.copy()
    equal = rows.loc[rows.base_weight.eq(0.5)]
    if len(equal) != 1:
        raise ValueError("Expected one equal-weight reference")
    reference = equal.iloc[0]
    rows["eligible"] = (
        (rows.hh_pair_accuracy >= reference.hh_pair_accuracy - max_accuracy_loss)
        & (rows.hh_c_index >= reference.hh_c_index - max_c_index_loss)
        & (rows.hh_mos_rho >= reference.hh_mos_rho)
    )
    eligible = rows.loc[rows.eligible]
    if eligible.empty:
        raise ValueError("No weight satisfies the development constraints")
    best = eligible.sort_values(["hai_mos_rho", "base_weight"],
                                ascending=[False, True]).iloc[0]
    rows["selected"] = rows.base_weight.eq(best.base_weight)
    return rows, float(best.base_weight)


def select_weights(frame):
    if frame.empty or not frame.split.eq("dev").all():
        raise ValueError("Weight selection accepts development recordings only")
    if frame.recording_id.duplicated().any():
        raise ValueError("Duplicate development recording IDs")
    if set(frame.corpus) != {"hh_turn", "hh_emotion", "hai"}:
        raise ValueError("Selection requires all three development subsets")
    normalizers = fit_normalizers(frame)
    normalized = fuse(frame, normalizers)
    grids, weights = [], {}
    for branch, (corpus, component, target, allowance) in BRANCHES.items():
        hh = normalized.loc[normalized.corpus.eq(corpus)]
        hai = normalized.loc[normalized.corpus.eq("hai")]
        rows = []
        for weight in GRID:
            hh_score = weight * hh[component] + (1 - weight) * hh.S
            hai_score = weight * hai[component] + (1 - weight) * hai.S
            accuracy, c_index = discrimination(hh, hh_score)
            rows.append(dict(branch=branch, base_weight=float(weight),
                             silence_weight=round(1 - float(weight), 2),
                             hh_pair_accuracy=accuracy, hh_c_index=c_index,
                             hh_mos_rho=rho(hh_score, hh[target]),
                             hai_mos_rho=rho(hai_score, hai[target])))
        grid = pd.DataFrame(rows)
        if not np.isfinite(grid.select_dtypes(include="number")).all().all():
            raise ValueError(f"Undefined development metric for {branch}")
        grid, weights[branch] = select_candidate(grid, allowance)
        grids.append(grid)
    return pd.concat(grids, ignore_index=True), weights, normalizers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = load_dev(args.data)
    grid, weights, normalizers = select_weights(frame)
    manifest = {
        "purpose": "development-only refit, not a replay of historical paper-weight selection",
        "hh_rating_protocol": "primary",
        "paper_weights_overwritten": False,
        "selection_split": "dev",
        "development_recordings": frame.groupby("corpus").size().to_dict(),
        "development_input_sha256": hashlib.sha256(
            frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest(),
        "grid": GRID.tolist(),
        "reference_base_weight": 0.5,
        "max_hh_pair_accuracy_loss": {"timing": 0.02, "affect": 0.04},
        "max_hh_c_index_loss": 0.01,
        "min_hh_mos_change": 0.0,
        "objective": "maximize corresponding HAI Combined MOS Spearman correlation",
        "tie_break": "lower base weight",
        "weights": {
            "timing_activity_weight": weights["timing"],
            "timing_silence_weight": round(1 - weights["timing"], 2),
            "affect_av_weight": weights["affect"],
            "affect_silence_weight": round(1 - weights["affect"], 2),
            "overall_timing_weight": 0.5,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    grid.to_csv(args.output / "grid.csv", index=False)
    for name, value in (("selection.json", manifest), ("normalizers.json", normalizers)):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest["weights"], indent=2))
    print(f"Saved development refit to {args.output}; published settings are unchanged.")


if __name__ == "__main__":
    main()
