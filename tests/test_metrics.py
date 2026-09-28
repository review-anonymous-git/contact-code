import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from contact.metrics import discrimination, weighted_discrimination, weighted_spearman


def test_repeated_session_counts_and_c_index():
    f = pd.DataFrame(dict(condition=["natural", "unnatural"]*2,
                          group_id=["a", "a", "b", "b"]))
    x = np.array([9., 8., 5., 7.])
    pa, ci = weighted_discrimination(f, x, np.array([[2, 2, 1, 1]]))
    assert pa[0] == pytest.approx(2/3)
    n, m = np.array([9, 9, 5]), np.array([8, 8, 7])
    assert ci[0] == pytest.approx((n[:, None] > m).mean())
    assert discrimination(f, [1, 1, 2, 2]) == pytest.approx((0.5, 0.5))


def test_frequency_ranks_match_literal_duplication():
    x, y = np.array([2., 2., 3., 1.]), np.array([3., 1., 2., 2.])
    w = np.array([[2, 1, 0, 3], [1, 1, 1, 1]])
    actual = weighted_spearman(x, y, w)
    for i in range(len(w)):
        assert actual[i] == pytest.approx(spearmanr(np.repeat(x, w[i]), np.repeat(y, w[i])).statistic)


def test_incomplete_session_rejected():
    f = pd.DataFrame(dict(condition=["natural", "natural"], group_id=["a", "a"]))
    with pytest.raises(ValueError):
        discrimination(f, [1, 2])
