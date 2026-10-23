import math

import numpy as np
import pytest
from concept_analysis.trajectory import detect_boundaries
from scipy.signal import find_peaks

from word_circles.graded import graded_curve, select_window


def trajectory(n_frames, seed):
    """A noisy 2-D trajectory circling a few times, like a projected utterance."""
    phase = np.cumsum(np.random.default_rng(seed).uniform(0.05, 0.6, n_frames))
    return np.stack([np.cos(phase), np.sin(phase)], axis=1) + 0.05 * np.random.default_rng(seed + 1).standard_normal(
        (n_frames, 2))


@pytest.mark.parametrize("window", [1, 2, 3])
@pytest.mark.parametrize("prominence", [0.0, 0.175, 0.349])
def test_detector_peaks_are_peaks_of_the_graded_curve(window, prominence):
    z = trajectory(120, seed=window)
    peaks, _ = find_peaks(graded_curve(z, window), prominence=prominence)
    np.testing.assert_array_equal(peaks, detect_boundaries(z, prominence=prominence, smooth_window_size=window))


def test_graded_curve_is_minus_the_angle_to_the_first_frame_unsmoothed():
    z = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(graded_curve(z, 1), [0.0, -math.pi / 2, -math.pi, -math.pi / 4])


def row(param_id, theta, window, R, n_empty=0):
    return {"param_id": param_id, "prominence": 0.0, "threshold_max_angle": theta, "smooth_window_size": window,
            "R": R, "n_empty": n_empty}


def test_window_comes_from_the_replications_selection_rule():
    rows = [row(0, math.pi / 2, 1, 0.7), row(1, math.pi, 2, 0.9), row(2, math.pi / 2, 3, 0.8, n_empty=1),
            row(3, math.pi / 4, 2, 0.75)]
    assert select_window(rows) == (rows[3], False)


def test_window_falls_back_to_best_thresholded_r_when_every_config_leaves_an_utterance_empty():
    rows = [row(0, math.pi / 2, 1, 0.7, n_empty=2), row(1, math.pi, 2, 0.9, n_empty=1),
            row(2, math.pi / 2, 3, 0.8, n_empty=1)]
    assert select_window(rows) == (rows[2], True)
