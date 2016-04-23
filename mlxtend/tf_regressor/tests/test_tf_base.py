# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.tf_regressor.tf_base import _TfBaseRegressor
import numpy as np
from mlxtend.utils import assert_raises


def test_init():
    tfr = _TfBaseRegressor(print_progress=0, random_seed=1)


def test_check_arrays_1():
    X = np.array([1, 2, 3])
    y = np.array([1, 1, 1])
    tfr = _TfBaseRegressor(print_progress=0, random_seed=1)
    assert_raises(ValueError,
                  'X must be a 2D array. Try X[:, numpy.newaxis]',
                  tfr._check_arrays,
                  X)

    assert_raises(ValueError,
                  'X must be a 2D array. Try X[:, numpy.newaxis]',
                  tfr._check_arrays,
                  X, y)


def test_check_arrays_2():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 1])
    tfr = _TfBaseRegressor(print_progress=0, random_seed=1)

    assert_raises(ValueError,
                  'X and y must contain the same number of samples',
                  tfr._check_arrays,
                  X, y)


def test_check_arrays_3():
    X = list([[1], [2], [3]])
    tfr = _TfBaseRegressor(print_progress=0, random_seed=1)

    assert_raises(ValueError,
                  'X must be a numpy array',
                  tfr._check_arrays,
                  X)


def test_check_arrays_4():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    tfr = _TfBaseRegressor(print_progress=0, random_seed=1)
    tfr._check_arrays(X, y)


def test_check_arrays_5():
    X = np.array([[1], [2], [3]])
    y = [1, 2, 3]
    tfr = _TfBaseRegressor(print_progress=0, random_seed=1)

    assert_raises(ValueError,
                  'y must be a numpy array.',
                  tfr._check_arrays,
                  X, y)


def test_check_arrays_6():
    X = np.array([[1], [2], [3]])
    y = X
    tfr = _TfBaseRegressor(print_progress=0, random_seed=1)

    assert_raises(ValueError,
                  'y must be a 1D numpy array.',
                  tfr._check_arrays,
                  X, y)


def test_fit():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    tfr = _TfBaseRegressor(print_progress=0, random_seed=1)
    tfr.fit(X, y)


def test_predict_1():
    X = np.array([[1], [2], [3]])
    tfr = _TfBaseRegressor(print_progress=0, random_seed=1)

    assert_raises(AttributeError,
                  'Model is not fitted, yet.',
                  tfr.predict,
                  X)


def test_predict_2():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    tfr = _TfBaseRegressor(print_progress=0, random_seed=1)

    tfr.fit(X, y)
    tfr.predict(X)


def test_shuffle():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    tfr = _TfBaseRegressor(print_progress=0, random_seed=1)
    X_sh, y_sh = tfr._shuffle(arrays=[X, np.array(y)])
    np.testing.assert_equal(X_sh, np.array([[1], [3], [2]]))
    np.testing.assert_equal(y_sh, np.array([1, 3, 2]))
