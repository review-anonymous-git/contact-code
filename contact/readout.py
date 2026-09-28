"""Recording-level components and development-set score normalization."""
from collections import defaultdict

import numpy as np


def soft_av(arousal, valence, sigma_bins=0.75):
    """Return two eight-bin targets and their arousal-major outer product."""
    if not np.isfinite(sigma_bins) or sigma_bins <= 0:
        raise ValueError("sigma_bins must be positive and finite")
    distributions = []
    centers = (np.arange(8, dtype=float) + 0.5) / 8
    for value in (arousal, valence):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Arousal and valence must lie in [0, 1]")
        logp = -0.5 * ((float(value) - centers) / (sigma_bins / 8)) ** 2
        p = np.exp(logp - logp.max())
        distributions.append(p / p.sum())
    a, v = distributions
    return a, v, np.outer(a, v).ravel()


def factorized_prior(joint):
    """Use the product of the training-prior marginals, as in paper scoring."""
    p = np.asarray(joint, dtype=float).reshape(8, 8)
    if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Invalid training prior")
    p /= p.sum()
    q = np.outer(p.sum(1), p.sum(0)).ravel()
    return q / q.sum()


def affect_gain_gap(windows, prior):
    """Equal response weighting within each speaker; include both speakers."""
    prior = np.asarray(prior, float)
    if prior.shape != (64,) or not np.isfinite(prior).all() or (prior < 0).any():
        raise ValueError("Expected a finite nonnegative 64-state prior")
    if not np.isclose(prior.sum(), 1):
        raise ValueError("Prior must sum to one")
    responses = defaultdict(list)
    seen = set()
    for w in windows:
        if not w["valid"]:
            continue
        if w["window_id"] in seen:
            raise ValueError("Duplicate affect window")
        seen.add(w["window_id"])
        ch = int(w["speaker"])
        if ch not in (0, 1):
            raise ValueError("Speaker must be 0 or 1")
        q = soft_av(w["arousal"], w["valence"])[2]
        ce = float(w["cross_entropy"])
        if not np.isfinite(ce):
            raise ValueError("Nonfinite cross entropy")
        gain = -(q * np.log(prior.clip(1e-30))).sum() - ce
        responses[ch, w["response_id"]].append(gain)
    means = {ch: [np.mean(v) for (speaker, _), v in responses.items() if speaker == ch]
             for ch in (0, 1)}
    if not all(means.values()):
        raise ValueError("Affective mismatch requires valid responses from both speakers")
    return -abs(float(np.mean(means[0])) - float(np.mean(means[1])))


def scoring_regions(frames, window=250, stride=150, context=38, guard=25):
    """Yield (window start, owned start, owned end), in 12.5-Hz frames."""
    if frames < 0 or stride <= 0 or window - stride < context + guard:
        raise ValueError("Overlap must cover context and future guard")
    owned_until = context
    for start in range(0, frames, stride):
        count = min(window, frames - start)
        lo, hi = max(start + context, owned_until), start + count - guard
        if hi > lo:
            yield start, lo, hi
        owned_until = max(owned_until, hi)
        if start + window >= frames:
            break


def fit_normalizers(frame):
    """Fit population mean/std on dev scores only, separately by subset."""
    stats = {}
    for corpus, group in frame.groupby("corpus", sort=False):
        stats[corpus] = {}
        for component in ("F", "S", "A"):
            values = group.loc[group.split.eq("dev"), component + "_raw"].to_numpy(float)
            if component == "A" and corpus == "hh_turn" and not np.isfinite(values).any():
                continue
            if len(values) < 2 or not np.isfinite(values).all():
                raise ValueError(f"Missing dev components: {corpus}/{component}")
            std = float(np.std(values, ddof=0))
            if std <= 1e-12:
                raise ValueError(f"Degenerate dev component: {corpus}/{component}")
            stats[corpus][component] = dict(mean=float(values.mean()), std=std, n=len(values))
    return stats


def fuse(frame, normalizers, f_weight=0.45, a_weight=0.25):
    if not 0 <= f_weight <= 1 or not 0 <= a_weight <= 1:
        raise ValueError("Fusion weights must lie in [0, 1]")
    result = frame.copy()
    for component in ("F", "S", "A"):
        result[component] = np.nan
        for corpus, stats in normalizers.items():
            if component not in stats:
                continue
            s = stats[component]
            mask = result.corpus.eq(corpus)
            result.loc[mask, component] = (result.loc[mask, component + "_raw"] - s["mean"]) / s["std"]
    result["T"] = f_weight * result.F + (1 - f_weight) * result.S
    result["E"] = a_weight * result.A + (1 - a_weight) * result.S
    result["O"] = (result["T"] + result.E) / 2
    result["no_av_O"] = (result["T"] + result.S) / 2
    result["no_silence_O"] = (result.F + result.A) / 2
    result["no_activity_O"] = (result.S + result.E) / 2
    result["unavailable"] = np.nan
    return result
