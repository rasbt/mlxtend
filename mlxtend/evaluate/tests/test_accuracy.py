# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from mlxtend.evaluate import accuracy_score


def test_multiclass_binary():
    y_targ = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [1, 0, 0, 0, 1, 2, 0, 2, 2]

    y_1 = accuracy_score(y_targ, y_pred, method="binary", pos_label=1)
    y_2 = accuracy_score(y_targ, y_pred, method="binary", pos_label=2)
    y_3 = accuracy_score(y_targ, y_pred, method="binary", pos_label=1, normalize=False)

    assert_almost_equal(y_2, float(7) / 9, decimal=4)
    assert_almost_equal(y_1, float(6) / 9, decimal=4)
    assert y_3 == 6


def test_standard():
    y_targ = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 0, 1, 1]

    y = accuracy_score(y_targ, y_pred, method="standard")
    y_1 = accuracy_score(y_targ, y_pred, method="binary", normalize=False)
    assert_almost_equal(y, float(3) / 6, decimal=4)
    assert y_1 == 3


def test_balanced_multiclass():
    y_targ = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 2, 2, 2])

    y = accuracy_score(y_targ, y_pred, method="balanced")
    assert_almost_equal(y, 0.578, decimal=3)


def test_balanced_binary():
    y_targ = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1])

    y = accuracy_score(y_targ, y_pred, method="balanced")
    assert_almost_equal(y, 0.542, decimal=3)


@pytest.mark.parametrize("target_type", [list, tuple, np.asarray])
@pytest.mark.parametrize("prediction_type", [list, tuple, np.asarray])
@pytest.mark.parametrize("labels", [[-1, 4, 12], ["cat", "dog", "bird"]])
def test_balanced_array_like(target_type, prediction_type, labels):
    a, b, c = labels
    y_targ = target_type([a, a, a, b, b, c])
    y_pred = prediction_type([a, b, a, b, c, c])

    # Per-class recalls are 2/3, 1/2, and 1.
    assert_almost_equal(accuracy_score(y_targ, y_pred, method="balanced"), 13 / 18)


@pytest.mark.parametrize("prediction_index", [[10, 20, 30], [30, 20, 10], [40, 50, 60]])
def test_balanced_series_uses_sample_positions(prediction_index):
    y_targ = pd.Series(["cat", "cat", "dog"], index=[10, 20, 30])
    y_pred = pd.Series(["cat", "dog", "dog"], index=prediction_index)

    assert_almost_equal(accuracy_score(y_targ, y_pred, method="balanced"), 0.75)


def test_balanced_single_class():
    assert_almost_equal(
        accuracy_score(["cat", "cat"], ["cat", "dog"], method="balanced"),
        0.5,
    )


def test_balanced_different_lengths():
    with pytest.raises(AttributeError, match="same number of elements"):
        accuracy_score(["cat", "dog"], ["cat"], method="balanced")


def test_average():
    y_targ = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 2, 2, 2])

    y = accuracy_score(y_targ, y_pred, method="average")
    assert_almost_equal(y, float(2) / 3, decimal=4)
