# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from nose.tools import raises
from mlxtend.feature_extraction.base  import _BaseFeatureExtractor


X, y = np.array([[1, 2], [4, 5], [3, 9]]), np.array([1, 2, 3])
base = _BaseFeatureExtractor()


def test_X_array_pass():
    base._check_arrays(X=X)


def test_X_y_array_pass():
    base._check_arrays(X=X, y=y)


@raises(ValueError)
def test_1D_X():
    base._check_arrays(X=X[1])


@raises(ValueError)
def test_X_int_y():
    base._check_arrays(X=X, y=y[1])


@raises(ValueError)
def test_X_short_y():
    print(y[1:].shape)
    print(X.shape)
    base._check_arrays(X=X, y=y[1:])
