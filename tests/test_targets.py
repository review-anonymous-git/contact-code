import numpy as np
import pytest

from contact.targets import activity_targets, affect_windows, clean_vad, select_before, silence_targets
from contact.targets import vad_from_protocol


def test_activity_bit_order_and_current_frame_exclusion():
    vad = np.zeros((2, 400), np.float32)
    vad[0, :4] = 1
    labels, valid = activity_targets(vad, 0, 10, 10)
    assert labels[0] == 0
    vad[0, 4:14] = 1
    vad[1, 64:104] = 1
    labels, valid = activity_targets(vad, 0, 10, 10)
    assert labels[0] == 1 + 128
    assert valid.all()


def test_silence_is_joint_and_uses_strictly_past_frame():
    vad = np.zeros((2, 700), np.float32)
    quiet = silence_targets(vad, 0, 175, 175)
    times = (np.arange(175)+1)/12.5 + .04
    index = select_before(times, 4., 175)
    assert times[index] < 4
    assert times[index+1] >= 4
    assert quiet[index] == 9
    vad[1] = 1
    occupied = silence_targets(vad, 0, 175, 175)
    assert occupied[index] == 0
    assert (quiet >= 0).sum() == 7


def test_response_windows_do_not_cross_responses():
    vad = np.zeros((2, 900), np.float32)
    vad[0, 100:400] = 1
    vad[1, 450:750] = 1
    windows = affect_windows(vad, "example")
    assert len(windows) == 10
    assert windows[0]["start"] == 2
    assert windows[0]["end"] == 5
    assert windows[4]["end"] == 8
    assert len({w["response_id"] for w in windows}) == 2
    assert all(w["end"]-w["start"] >= 1.5 for w in windows)


def test_vad_cleanup_preserves_open_terminal_run():
    vad = np.r_[np.zeros(8), np.ones(4), np.zeros(8), np.ones(3)]
    expected = np.r_[np.zeros(20), np.ones(3)]
    np.testing.assert_array_equal(clean_vad(vad), expected)


def test_benchmark_vad_rle_and_merge_gap():
    protocol = dict(vad_frames=300, vad_runs=[[(10, 60), (80, 160)], [(200, 270)]])
    vad = vad_from_protocol(protocol)
    assert vad.shape == (2, 300)
    assert vad[0].sum() == 130
    assert len(affect_windows(vad, "r", .5)) > len(affect_windows(vad, "r", .2))
    protocol["vad_runs"][0].append((20, 40))
    with pytest.raises(ValueError, match="overlapping"):
        vad_from_protocol(protocol)
