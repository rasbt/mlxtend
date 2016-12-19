# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.evaluate import lift_score
from numpy.testing import assert_array_equal
import numpy as np

def test_multiclass():
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    x = 1.28
    y = lift_score(y_targ, y_pred, binary=True, positive_label=0)
    assert_array_equal(x, y)

def test_binary_with_numpy():

    y_targ = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0])
    x = 1.25
    y = lift_score(y_targ, y_pred, binary=False, positive_label=0)
    assert_array_equal(x, y)


def test_binary():
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    x = 1.25
    y = lift_score(y_targ, y_pred, binary=False, positive_label=0)
    assert_array_equal(x, y)


def test_multidimension():
    y_targ = [[1, 1, 1, 0, 0, 1],[0, 1, 0, 0, 0, 1]]
    y_pred = [[1, 0, 1, 0, 0, 1]]
    x = 1
    y = lift_score(y_targ, y_pred, binary=False, positive_label=0)
    assert_array_equal(x, y)

