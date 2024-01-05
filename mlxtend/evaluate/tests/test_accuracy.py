# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause
import numpy as np
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


def test_average():
    y_targ = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 2, 2, 2])

    y = accuracy_score(y_targ, y_pred, method="average")
    assert_almost_equal(y, float(2) / 3, decimal=4)
