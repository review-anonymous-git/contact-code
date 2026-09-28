import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("torch")
from contact.prepare import read_track


def test_track_selection_and_source_crop(tmp_path):
    from scipy.signal import resample_poly

    rate = 24000
    wave = np.random.default_rng(42).normal(0, .1, (rate, 2)).astype(np.float32)
    path = tmp_path/"stereo.wav"
    sf.write(path, wave, rate, subtype="FLOAT")
    expected = resample_poly(wave[rate//4:rate//2, 1], 2, 3)
    np.testing.assert_array_equal(read_track(path, 1, 16000, .25, .5), expected)
    with pytest.raises(ValueError, match="Expected 1"):
        read_track(path, None, 16000)
    with pytest.raises(ValueError, match="exceeds"):
        read_track(path, 0, 16000, .5, 2)
