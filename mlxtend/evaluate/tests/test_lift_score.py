# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from mlxtend.evaluate import lift_score
from mlxtend.evaluate.lift_score import support


def test_multiclass():
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3, 4]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3, 0]
    x = 2
    y = lift_score(y_targ, y_pred, binary=True, positive_label=1)
    assert_array_equal(x, y)


def test_multiclass_positive_label_0():
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3, 4]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3, 0]
    x = 1.5
    y = lift_score(y_targ, y_pred, binary=True, positive_label=0)
    assert_array_equal(x, y)


def test_multiclass_with_false_binary():
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    assert_raises(AttributeError, lift_score, y_targ, y_pred, False, 1)


def test_binary_with_numpy():
    y_targ = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0])
    x = 1.25
    y = lift_score(y_targ, y_pred, binary=False, positive_label=1)
    assert_array_equal(x, y)


def test_binary():
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    x = 1.25
    y = lift_score(y_targ, y_pred, binary=False, positive_label=1)
    assert_array_equal(x, y)


def test_multidimension():
    y_targ = [[1, 1, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1]]
    y_pred = [[1, 0, 1, 0, 0, 1]]
    x = 1
    y = lift_score(y_targ, y_pred, binary=False, positive_label=1)
    assert_array_equal(x, y)


def test_support_with_two_parameter():
    y_targ = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0])
    x = 0.4
    y = support(y_targ, y_pred)
    assert_array_equal(x, y)


def test_support_with_one_parameter():
    y_targ = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0])
    x = 0.533333333333333333
    y = support(y_targ)
    assert_array_equal(x, y)
