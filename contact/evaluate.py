"""Reproduce recording-level metrics from the fixed CONTACT score release."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (bootstrap_weights, discrimination, paired_summary, rho,
                      weighted_discrimination, weighted_spearman)
from .readout import fit_normalizers, fuse

CORPORA = ("hh_turn", "hh_emotion", "hai")
VIEWS = ("participant", "supervisor", "combined")
TARGETS = (("hh_turn", "timing"), ("hh_emotion", "affect"),
           ("hai", "timing"), ("hai", "affect"), ("hai", "overall"))
EXPECTED = {("hh_turn", "dev"): 60, ("hh_turn", "test"): 120,
            ("hh_emotion", "dev"): 32, ("hh_emotion", "test"): 68,
            ("hai", "dev"): 70, ("hai", "test"): 143}
BASELINES = {
    "UTMOSv2": ("utmosv2",) * 3,
    "VAP": ("vap",) * 3,
    "DualTurn": ("dualturn",) * 3,
    "Talking Turns": ("talking_turns",) * 3,
    "UniSRM": ("unisrm_overall", "unisrm_affect", "unisrm_overall"),
    "TRACE": ("trace",) * 3,
    "TimingStats": ("timingstats",) * 3,
}
DEFINITIONS = {**BASELINES,
    "Ours": ("T", "E", "O"),
    "w/o A--V": ("T", "S", "no_av_O"),
    "w/o future joint silence": ("F", "A", "no_silence_O"),
    "w/o future voice activity": ("S", "E", "no_activity_O"),
    "w/o timing outputs": ("unavailable", "A", "A"),
}


def score_column(name, dimension):
    return DEFINITIONS[name][("timing", "affect", "overall").index(dimension)]


def load_inputs(directory, predictions=None):
    directory = Path(directory)
    index = json.loads((directory / "release.json").read_text())
    for filename, expected in index["sha256"].items():
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Release checksum mismatch: {filename}")
    split = pd.read_csv(directory / "splits.csv", dtype=str, keep_default_na=False)
    labels = pd.read_csv(directory / "ratings.csv")
    scores = pd.read_csv(directory / "scores.csv")
    for f in (split, labels, scores):
        if f.recording_id.duplicated().any() or set(f.recording_id) != set(split.recording_id):
            raise ValueError("Scores, labels and split must have identical unique IDs")
    if split.groupby(["corpus", "split"]).size().to_dict() != EXPECTED:
        raise ValueError("The released paper inventory is fixed")
    if split.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("A session crosses dev and test")
    people = {s: {p for ids in split.loc[split.split.eq(s), "human_speaker_ids"]
                  for p in ids.split(";") if p} for s in ("dev", "test")}
    if people["dev"] & people["test"]:
        raise ValueError("A human speaker crosses dev and test")
    frame = split.merge(labels, on="recording_id", validate="one_to_one").merge(
        scores, on="recording_id", validate="one_to_one")
    for corpus, dim in TARGETS:
        f = frame.loc[frame.corpus.eq(corpus)]
        p, s, c = (f[f"{dim}_{v}"] for v in VIEWS)
        if not np.isfinite(np.column_stack([p, s, c])).all() or not np.allclose(c, (p + s) / 2, atol=1e-12):
            raise ValueError(f"Invalid primary rating policy: {corpus}/{dim}")
    stats = fit_normalizers(frame)
    recorded = json.loads((directory / "normalizers.json").read_text())
    for corpus, components in stats.items():
        for component, values in components.items():
            for key, value in values.items():
                if not np.isclose(value, recorded[corpus][component][key], atol=1e-12, rtol=1e-12):
                    raise ValueError(f"Dev normalization changed: {corpus}/{component}/{key}")
    if predictions is not None:
        supplied = pd.read_csv(predictions)
        columns = ["F_raw", "S_raw", "A_raw"]
        if supplied.recording_id.duplicated().any() or set(supplied.recording_id) != set(frame.recording_id):
            raise ValueError("Predictions must cover all released recording IDs exactly once")
        supplied = supplied.set_index("recording_id").loc[frame.recording_id]
        if "corpus" in supplied and not np.array_equal(supplied.corpus, frame.corpus):
            raise ValueError("Prediction corpus does not match the released split")
        values = supplied[columns].to_numpy(float)
        if not np.isfinite(values[:, :2]).all() or not np.isfinite(values[~frame.corpus.eq("hh_turn"), 2]).all():
            raise ValueError("Predictions contain missing required score components")
        frame[columns] = values
    # Keep development normalization fixed when evaluating new inference outputs.
    return fuse(frame, recorded), index


def metric_rows(frame):
    rows = []
    for name in DEFINITIONS:
        for corpus, dimension in TARGETS:
            col = score_column(name, dimension)
            if col == "unavailable":
                continue
            for split in ("dev", "test", "all"):
                f = frame.loc[frame.corpus.eq(corpus)]
                if split != "all":
                    f = f.loc[f.split.eq(split)]
                common = dict(scorer=name, corpus=corpus, dimension=dimension, split=split, n=len(f))
                if corpus != "hai":
                    acc, ci = discrimination(f, f[col])
                    for metric, value in (("pair_accuracy", acc), ("c_index", ci)):
                        rows.append(dict(common, metric=metric, rater="-", value=value))
                for view in VIEWS:
                    value = rho(f[col], f[f"{dimension}_{view}"])
                    rows.append(dict(common, metric="rho", rater=view, value=value))
    return pd.DataFrame(rows)


def infer(frame, repeats, seed):
    rng = np.random.default_rng(seed)
    baseline, ablations = [], []
    for corpus in CORPORA:
        f = frame.loc[frame.corpus.eq(corpus) & frame.split.eq("test")].reset_index(drop=True)
        weights = bootstrap_weights(f, rng, repeats)
        cache = {}

        def estimate(name, dim, metric, view):
            col = score_column(name, dim)
            key = (col, dim, metric, view)
            if key not in cache:
                if metric == "rho":
                    cache[key] = weighted_spearman(f[col], f[f"{dim}_{view}"], weights)
                else:
                    pa, ci = weighted_discrimination(f, f[col], weights)
                    cache[(col, dim, "pair_accuracy", "-")] = pa
                    cache[(col, dim, "c_index", "-")] = ci
            return cache[key]

        for _, dim in [x for x in TARGETS if x[0] == corpus]:
            endpoints = [("rho", v) for v in VIEWS]
            if corpus != "hai":
                endpoints = [("pair_accuracy", "-"), ("c_index", "-"), *endpoints]
            for metric, view in endpoints:
                full = estimate("Ours", dim, metric, view)
                best = max(BASELINES, key=lambda n: estimate(n, dim, metric, view)[0])
                common = dict(corpus=corpus, dimension=dim, metric=metric, rater=view,
                              n=len(f), sessions=f.group_id.nunique(), repeats=repeats)
                baseline.append(dict(common, baseline=best,
                                     **paired_summary(full, estimate(best, dim, metric, view))))
                comparisons = []
                if corpus == "hh_turn" and metric in ("pair_accuracy", "c_index"):
                    comparisons.append("w/o future voice activity")
                if metric == "rho" and dim != "overall":
                    comparisons.append("w/o future joint silence")
                if (corpus == "hh_emotion" and metric == "pair_accuracy") or (corpus == "hai" and dim == "affect" and metric == "rho"):
                    comparisons.append("w/o A--V")
                for name in comparisons:
                    ablations.append(dict(common, ablation=name,
                        **paired_summary(full, estimate(name, dim, metric, view))))
        print(f"Bootstrap: {corpus}, {f.group_id.nunique()} sessions, {repeats} paired draws", flush=True)
    return pd.DataFrame(baseline), pd.DataFrame(ablations)


def table_tex(metrics, baseline=None, ablations=None):
    lookup = metrics.set_index(["scorer", "corpus", "dimension", "split", "metric", "rater"]).value
    baseline = pd.DataFrame() if baseline is None else baseline
    ablations = pd.DataFrame() if ablations is None else ablations

    def cell(name, corpus, dim, split, metric, view):
        key = (name, corpus, dim, split, metric, view)
        if key not in lookup:
            return "--"
        x = lookup[key]
        scaled = 100*x if metric == "pair_accuracy" else x
        unit = Decimal("0.1" if metric == "pair_accuracy" else "0.01")
        rounded = Decimal(str(scaled)).quantize(unit, rounding=ROUND_HALF_UP)
        value = str(abs(rounded) if rounded == 0 else rounded)
        marker = ""
        if split == "test":
            tests = baseline if name == "Ours" else ablations
            if not tests.empty:
                mask = tests.corpus.eq(corpus) & tests.dimension.eq(dim) & tests.metric.eq(metric) & tests.rater.eq(view)
                if name != "Ours":
                    mask &= tests.ablation.eq(name)
                if (tests.loc[mask, "significant_gain"] == True).any():
                    marker = r"$^{\dagger}$" if name == "Ours" else r"$^{\ddagger}$"
        return value + marker

    lines = [r"\begin{table*}[t]", r"\centering\scriptsize",
             r"\caption{CONTACT evaluation. Acc.: paired accuracy (\%); C-I: C-index; P/S/C: Spearman correlation with participant, supervisor and combined ratings. Ablations remove scores from one checkpoint."]
    if not baseline.empty:
        lines.append(r"$\dagger$: Ours versus the highest observed baseline; $\ddagger$: targeted Full-minus-ablation contrast. Markers use unadjusted, exploratory two-sided paired session bootstrap tests ($p<.05$).}")
    else:
        lines.append("}")
    for panel in ("hh", "hai"):
        ncols = 20 if panel == "hh" else 18
        lines += [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "r"*ncols + "}", r"\toprule"]
        if panel == "hh":
            keys = [(c, d, s, m, "-") for c, d in TARGETS[:2] for s in ("dev", "test") for m in ("pair_accuracy", "c_index")]
            keys += [(c, d, s, "rho", v) for c, d in TARGETS[:2] for s in ("dev", "test") for v in VIEWS]
            header = [f"{d.title()} {s} {'Acc.' if m == 'pair_accuracy' else 'C-I'}" for c,d,s,m,v in keys[:8]]
            header += [f"{d.title()} {s} {dict(zip(VIEWS, 'PSC'))[v]}" for c,d,s,m,v in keys[8:]]
        else:
            keys = [("hai", d, s, "rho", v) for d in ("timing", "affect", "overall") for s in ("dev", "test") for v in VIEWS]
            header = [f"{d.title()} {s} {dict(zip(VIEWS, 'PSC'))[v]}" for c,d,s,m,v in keys]
        lines += ["Scorer & " + " & ".join(header) + r" \\", r"\midrule"]
        for name in DEFINITIONS:
            if panel == "hh" and name == "w/o timing outputs":
                continue
            if name == "Ours":
                lines.append(r"\midrule")
            values = ["--" if name == "w/o timing outputs" and d != "overall" else cell(name,c,d,s,m,v) for c,d,s,m,v in keys]
            lines.append(name + " & " + " & ".join(values) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\par\medskip"]
    return "\n".join([*lines, r"\end{table*}", ""])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--predictions", type=Path, help="Replace model components with a complete inference CSV")
    args = p.parse_args()
    if args.bootstrap < 0:
        p.error("--bootstrap must be nonnegative")
    frame, release = load_inputs(args.data, args.predictions)
    metrics = metric_rows(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    columns = ["recording_id", "corpus", "split", "F", "S", "A", "T", "E", "O"]
    frame[columns].to_csv(args.output / "components.csv", index=False)
    baseline = ablations = None
    if args.bootstrap:
        baseline, ablations = infer(frame, args.bootstrap, args.seed)
        baseline.to_csv(args.output / "baseline_comparisons.csv", index=False)
        ablations.to_csv(args.output / "paired_ablations.csv", index=False)
    (args.output / "table.tex").write_text(table_tex(metrics, baseline, ablations))
    (args.output / "run.json").write_text(json.dumps(dict(
        release=release["version"], bootstrap=args.bootstrap, seed=args.seed,
        status="Frozen retrospective evaluation split; not a new blind test",
        inference="Unadjusted exploratory paired session bootstrap; no multiplicity correction",
        predictions_sha256=(hashlib.sha256(args.predictions.read_bytes()).hexdigest() if args.predictions else None),
        training_performed=False, audio_inference_performed=False), indent=2) + "\n")
    print(f"Verified {len(frame)} recordings. Wrote {len(metrics)} metric rows to {args.output}")


if __name__ == "__main__":
    main()
