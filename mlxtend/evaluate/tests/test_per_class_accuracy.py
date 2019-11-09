# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

from mlxtend.evaluate import per_class_accuracy
from numpy.testing import assert_almost_equal


def test_multiclass():
    y_targ = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [1, 0, 0, 0, 1, 2, 0, 2, 2]

    y_1 = per_class_accuracy(y_targ, y_pred, binary=True, pos_label=1)
    y_2 = per_class_accuracy(y_targ, y_pred, binary=True, pos_label=2)

    assert_almost_equal(y_2, float(7)/9, decimal=4)
    assert_almost_equal(y_1, float(6)/9, decimal=4)


def test_binary():
    y_targ = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 0, 1, 1]

    y = per_class_accuracy(y_targ, y_pred)

    assert_almost_equal(y, float(3)/6, decimal=4)
