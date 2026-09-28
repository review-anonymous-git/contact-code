"""Paired discrimination and session-cluster bootstrap statistics."""
import numpy as np
from scipy.stats import spearmanr


def discrimination(frame, score):
    values = np.asarray(score, float)
    pa, ci = weighted_discrimination(frame, values, np.ones((1, len(frame))))
    return float(pa[0]), float(ci[0])


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Missing values in a reported endpoint")
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def frequency_ranks(values, weights):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("Ranks require finite values")
    _, inverse = np.unique(values, return_inverse=True)
    incidence = np.eye(inverse.max() + 1)[inverse]
    mass = weights @ incidence
    ranks = np.cumsum(mass, axis=1) - 0.5 * mass + 0.5
    return ranks[:, inverse]


def weighted_spearman(x, y, weights):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Paired bootstrap requires a complete common inventory")
    a, b = frequency_ranks(x, weights), frequency_ranks(y, weights)
    mass = weights.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a -= ((weights * a).sum(axis=1) / mass)[:, None]
        b -= ((weights * b).sum(axis=1) / mass)[:, None]
        covariance = (weights * a * b).sum(axis=1)
        denominator = np.sqrt((weights * a * a).sum(axis=1) * (weights * b * b).sum(axis=1))
        return covariance / denominator


def weighted_discrimination(frame, score, weights):
    x = np.asarray(score, float)
    if not np.isfinite(x).all():
        raise ValueError("Discrimination requires complete scores")
    natural = frame.condition.str.lower().isin(("natural", "original")).to_numpy()
    group = frame.group_id.to_numpy()
    win, total = np.zeros(len(weights)), np.zeros(len(weights))
    for gid in np.unique(group):
        ni = np.flatnonzero((group == gid) & natural)
        ei = np.flatnonzero((group == gid) & ~natural)
        if len(ni) != 1 or not len(ei):
            raise ValueError(f"Expected one natural recording and manipulations in {gid}")
        if not np.all(weights[:, ni] == weights[:, ei]):
            raise ValueError("A matched session was split across bootstrap clusters")
        success = (x[ni[0]] > x[ei]) + 0.5 * (x[ni[0]] == x[ei])
        win += weights[:, ni[0]] * success.sum()
        total += weights[:, ni[0]] * len(ei)
    compare = (x[natural, None] > x[None, ~natural]) + 0.5 * (x[natural, None] == x[None, ~natural])
    wn, we = weights[:, natural], weights[:, ~natural]
    with np.errstate(divide="ignore", invalid="ignore"):
        return win / total, ((wn @ compare) * we).sum(axis=1) / (wn.sum(axis=1) * we.sum(axis=1))


def bootstrap_weights(frame, rng, repeats):
    # Connected speaker components equal sessions within each reported subset.
    if frame.groupby("component").group_id.nunique().max() != 1 or frame.groupby("group_id").component.nunique().max() != 1:
        raise ValueError("Session/component equivalence is required for this protocol")
    clusters, inverse = np.unique(frame.component.astype(str), return_inverse=True)
    counts = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=repeats)
    return np.vstack([np.ones(len(frame)), counts[:, inverse].astype(float)])


def paired_summary(full, comparison):
    delta = full - comparison
    boot = delta[1:]
    if not len(boot) or not np.isfinite(boot).all():
        raise ValueError("Undefined bootstrap draws; do not silently discard them")
    low, high = np.quantile(boot, [0.025, 0.975])
    p = min(1.0, 2 * min(1 + np.count_nonzero(boot <= 0),
                         1 + np.count_nonzero(boot >= 0)) / (len(boot) + 1))
    return dict(full=float(full[0]), comparison=float(comparison[0]), difference=float(delta[0]),
                ci95_low=float(low), ci95_high=float(high), p_two_sided=float(p),
                significant_gain=bool(delta[0] > 0 and p < 0.05))
