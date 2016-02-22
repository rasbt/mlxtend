# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.classifier.base import _BaseClassifier
import numpy as np
from nose.tools import raises


@raises(ValueError)
def test_X_array():
    X = [1, 2, 4]
    y = [2, 3, 4]
    bc = _BaseClassifier()
    bc._check_arrays(X, y)


@raises(ValueError)
def test_X_dim():
    X = np.array([2, 3, 4])
    y = [1, 2, 4]
    bc = _BaseClassifier()
    bc._check_arrays(X, y)


def test_okay():
    X = np.array([[2], [3], [4]])
    y = [1, 2, 4]
    bc = _BaseClassifier()
    bc._check_arrays(X, y)


@raises(ValueError)
def test_wrong_dim():
    X = np.array([[2], [3], [4]])
    y = [1, 2, 4, 5]
    bc = _BaseClassifier()
    bc._check_arrays(X, y)
