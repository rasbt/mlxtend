# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.evaluate import confusion_matrix
from numpy.testing import assert_array_equal


def test_multiclass():
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    x = np.array([[2, 1, 0, 0],
                  [1, 2, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    y = confusion_matrix(y_targ, y_pred, binary=False, positive_label=1)
    assert_array_equal(x, y)


def test_binary():
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    x = np.array([[4, 1],
                  [1, 2]])
    y = confusion_matrix(y_targ, y_pred, binary=True, positive_label=1)
    assert_array_equal(x, y)
