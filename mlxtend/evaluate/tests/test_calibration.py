# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest

from mlxtend.evaluate import expected_calibration_error


def test_hand_computed_value():
    y = [0, 1, 1, 1]
    p = [0.1, 0.9, 0.8, 0.3]
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.225)


def test_perfect_confident_predictions():
    assert expected_calibration_error([0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0]) == 0.0


def test_confidently_wrong():
    assert expected_calibration_error([0, 0], [1.0, 1.0]) == pytest.approx(1.0)


def test_calibrated_uncertain_model():
    y = [0, 1, 0, 1]
    p = [0.5, 0.5, 0.5, 0.5]
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_single_bin_equals_overall_gap():
    y = [0, 1, 1, 1]
    p = [0.1, 0.9, 0.8, 0.3]
    expected = abs(np.mean(y) - np.mean(p))
    assert expected_calibration_error(y, p, n_bins=1) == pytest.approx(expected)


def test_probability_on_bin_edge():
    # 0.5 falls in the first (low, high] bin when n_bins=2, 0.0 in the first bin
    y = [1, 0]
    p = [0.5, 0.0]
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.25)


def test_numpy_arrays_and_range():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, size=200)
    p = rng.rand(200)
    ece = expected_calibration_error(y, p)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_well_calibrated_simulation_is_small():
    rng = np.random.RandomState(0)
    p = rng.rand(20000)
    y = (rng.rand(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        expected_calibration_error([0, 1], [0.5])


def test_not_1d():
    with pytest.raises(ValueError, match="1-dimensional"):
        expected_calibration_error([[0, 1]], [[0.5, 0.5]])


def test_probability_out_of_range():
    with pytest.raises(ValueError, match="range"):
        expected_calibration_error([0, 1], [0.5, 1.2])


def test_probability_nan():
    with pytest.raises(ValueError, match="range"):
        expected_calibration_error([0, 1], [0.5, np.nan])


def test_non_binary_labels():
    with pytest.raises(ValueError, match="0 and 1"):
        expected_calibration_error([0, 2], [0.5, 0.5])


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_bad_n_bins(bad):
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error([0, 1], [0.5, 0.5], n_bins=bad)


def test_empty_input():
    with pytest.raises(ValueError, match="empty"):
        expected_calibration_error([], [])
