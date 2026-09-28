"""Targets and timestamps on the 50-Hz VAD and 12.5-Hz model grids."""
import math

import numpy as np


def vad_from_protocol(protocol):
    """Decode the fixed benchmark VAD annotations, without human naturalness labels."""
    n = protocol["vad_frames"]
    if not isinstance(n, int) or n <= 0 or len(protocol["vad_runs"]) != 2:
        raise ValueError("Invalid benchmark VAD shape")
    vad = np.zeros((2, n), np.float32)
    for channel, runs in enumerate(protocol["vad_runs"]):
        previous = 0
        for start, end in runs:
            if not isinstance(start, int) or not isinstance(end, int) or not previous <= start < end <= n:
                raise ValueError("Invalid/overlapping benchmark VAD interval")
            vad[channel, start:end] = 1
            previous = end
    return vad


def activity_targets(vad, start, count, total_frames):
    vad = np.asarray(vad)
    indices = np.arange(count) + start
    # Future occupancy starts immediately after the current 80-ms model frame.
    anchors = (indices + 1) * 4
    valid = (indices < total_frames) & (indices * 4 + 100 <= vad.shape[1])
    labels = np.zeros(count, np.int64)
    cumulative = np.pad(np.cumsum(vad, axis=1, dtype=np.float32), ((0, 0), (1, 0)))
    eligible = np.flatnonzero(valid)
    bit = 0
    for channel in (0, 1):
        offset = 0
        for width in (10, 20, 30, 40):
            lo = anchors[eligible] + offset
            mean = (cumulative[channel, np.minimum(lo + width, vad.shape[1])]
                    - cumulative[channel, np.minimum(lo, vad.shape[1])]) / width
            labels[eligible] += (mean >= .5).astype(np.int64) << bit
            offset += width
            bit += 1
    return labels, valid


def silence_targets(vad, start, count, valid_frames, context=38):
    vad = np.asarray(vad)
    labels = np.full(count, -100, np.int64)
    anchors = np.arange(200, vad.shape[1] - 200 + 1, 50, dtype=np.int64)
    times = anchors / 50
    keep = (times > start / 12.5) & (times <= (start + valid_frames) / 12.5)
    anchors, times = anchors[keep], times[keep]
    available = (np.arange(count) + start + 1) / 12.5 + .04
    indices = np.searchsorted(available, times, side="left") - 1
    keep = (indices >= context) & (indices < valid_frames)
    for anchor, index in zip(anchors[keep], indices[keep]):
        ratio = np.logical_not(vad[:, anchor:anchor+200].astype(bool).any(0)).mean()
        labels[index] = min(9, math.floor(ratio * 10 + 1e-10))
    return labels


def select_before(available, onset, valid_count):
    index = int(np.searchsorted(available[:valid_count], onset, side="left") - 1)
    return index if index >= 0 else None


def active_runs(mask):
    edges = np.diff(np.pad(np.asarray(mask, bool).astype(int), (1, 1)))
    return [(a / 50, b / 50) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def merge_runs(runs, other, gap=.2):
    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= gap + 1e-8 and not any(
                merged[-1][0] < onset <= start for onset, _ in other):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def affect_windows(vad, recording_id, response_merge_gap=.2):
    if not .2 <= response_merge_gap <= .5:
        raise ValueError("Response merge gap must be between .2 and .5 seconds")
    runs = [active_runs(vad[c] > .5) for c in (0, 1)]
    units = [merge_runs(merge_runs(runs[c], runs[1-c]), runs[1-c], response_merge_gap) for c in (0, 1)]
    windows = []
    for response_start, response_end, channel in sorted((s, e, c) for c in (0, 1) for s, e in units[c]):
        response_end = min(response_end, response_start + 8)
        response_id = f"{recording_id}:ch{channel}:{round(response_start * 50)}"
        index = 0
        while response_start + index < response_end - 1e-8:
            start = response_start + index
            end = min(start + 3, response_end)
            if end - start + 1e-8 >= 1.5:
                lo, hi = round(start * 50), min(vad.shape[1], round(end * 50))
                voiced = (vad[channel, lo:hi] > .5).sum() / 50
                if min(1., voiced / (end - start)) + 1e-8 >= .6:
                    windows.append(dict(window_id=f"{recording_id}:ch{channel}:aw{lo}",
                        response_id=response_id, speaker=channel,
                        start=round(start, 6), end=round(end, 6)))
            index += 1
    return windows


def clean_vad(mask):
    result = np.asarray(mask, np.float32).copy()
    for value in (1, 0):
        start = None
        for i in range(len(result)):
            active = result[i] >= .5 if value else result[i] < .5
            if active and start is None:
                start = i
            elif not active and start is not None:
                if i - start < 7:
                    result[start:i] = 1 - value
                start = None
    return result
