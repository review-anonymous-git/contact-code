import numpy as np
import pandas as pd
import pytest

from contact.readout import affect_gain_gap, fit_normalizers, scoring_regions, soft_av


def test_soft_targets():
    a, v, q = soft_av(0.56, 0.21)
    assert a.shape == v.shape == (8,)
    np.testing.assert_allclose(q.reshape(8, 8).sum(1), a)
    np.testing.assert_allclose(q.reshape(8, 8).sum(0), v)
    assert q.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        soft_av(-0.1, 0.3)


@pytest.mark.parametrize("frames", [1, 60, 100, 250, 251, 400, 1000])
def test_window_ownership(frames):
    owned = [i for _, lo, hi in scoring_regions(frames) for i in range(lo, hi)]
    assert owned == list(range(38, max(38, frames - 25)))
    assert len(set(owned)) == len(owned)


def test_response_then_speaker_weighting():
    def window(uid, speaker, response, ce):
        return dict(window_id=uid, speaker=speaker, response_id=response,
                    arousal=0.5, valence=0.5, cross_entropy=ce, valid=True)
    windows = [window("a", 0, "r0", 1), window("b", 0, "r0", 3),
               window("c", 0, "r1", 4), window("d", 1, "r2", 2)]
    # Channel 0 CE: mean(mean(1,3),4)=3; channel 1 CE=2.
    assert affect_gain_gap(windows, np.full(64, 1/64)) == pytest.approx(-1)
    with pytest.raises(ValueError, match="Duplicate"):
        affect_gain_gap(windows + [windows[0]], np.full(64, 1/64))
    with pytest.raises(ValueError, match="both speakers"):
        affect_gain_gap(windows[:-1], np.full(64, 1/64))


def test_test_scores_do_not_fit_normalization():
    f = pd.DataFrame(dict(corpus=["hai"]*3, split=["dev", "dev", "test"],
                          F_raw=[1., 2., 3.], S_raw=[2., 3., 4.], A_raw=[3., 4., 5.]))
    expected = fit_normalizers(f)
    f.loc[f.split.eq("test"), ["F_raw", "S_raw", "A_raw"]] = 100000
    assert fit_normalizers(f) == expected
