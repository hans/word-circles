import sys

import numpy as np
import pytest

from conftest import ROOT
from word_circles.dissim import boundary_frames, distance_curve, smoothed_curve, trough_peaks, with_final_boundary


@pytest.fixture(scope="module")
def segmentor_cls():
    path = ROOT / "third_party" / "prom_wordseg"
    if not (path / "wordseg" / "segment.py").exists():
        pytest.fail("pinned prom-wordseg missing: run `uv run snakemake -j1 third_party/prom_wordseg`")
    sys.path.insert(0, str(path))
    from wordseg.segment import Segmentor

    return Segmentor


def test_curves_and_peaks_match_prom_wordseg_exactly(segmentor_cls):
    rng = np.random.default_rng(0)
    utterances = [rng.standard_normal((n, 32)).cumsum(axis=0) for n in (40, 97, 250)]
    seg = segmentor_cls(distance_type="cosine", prominence=0.75, window_size=4)
    seg.get_distance(utterances)
    seg.moving_average()
    their_peaks, _ = seg.peak_detection()
    for x, d, s, p in zip(utterances, seg.distances, seg.smoothed_distances, their_peaks):
        np.testing.assert_array_equal(distance_curve(x), d)
        np.testing.assert_array_equal(smoothed_curve(distance_curve(x), 4), s)
        np.testing.assert_array_equal(trough_peaks(x, prominence=0.75, window=4), p)


@pytest.mark.parametrize("peaks, n_frames, expected", [
    ([10, 30], 50, [10, 30, 50]),   # final boundary appended at T
    ([10, 49], 50, [10, 49]),       # within one frame of T already: nothing appended
    ([10, 50], 50, [10, 50]),
    ([], 50, [50]),                 # no peak: the end boundary alone
])
def test_final_boundary_rule(peaks, n_frames, expected):
    assert with_final_boundary(np.array(peaks, dtype=int), n_frames).tolist() == expected


def test_single_frame_utterance_gets_one_boundary_at_frame_1():
    assert boundary_frames(np.zeros((1, 4)), prominence=0.75, window=4).tolist() == [1]
