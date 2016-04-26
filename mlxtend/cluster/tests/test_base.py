# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.cluster.base import _BaseCluster
import numpy as np
from mlxtend.utils import assert_raises


def test_init():
    cl = _BaseCluster(print_progress=0, random_seed=1)


def test_check_array_1():
    X = np.array([1, 2, 3])
    cl = _BaseCluster(print_progress=0, random_seed=1)
    assert_raises(ValueError,
                  'X must be a 2D array. Try X[:, numpy.newaxis]',
                  cl._check_array,
                  X)


def test_check_array_2():
    X = list([[1], [2], [3]])
    cl = _BaseCluster(print_progress=0, random_seed=1)

    assert_raises(ValueError,
                  'X must be a numpy array',
                  cl._check_array,
                  X)


def test_check_array_3():
    X = np.array([[1], [2], [3]])
    cl = _BaseCluster(print_progress=0, random_seed=1)
    cl._check_array(X)


def test_fit():
    X = np.array([[1], [2], [3]])
    tfr = _BaseCluster(print_progress=0, random_seed=1)
    tfr.fit(X)


def test_predict_1():
    X = np.array([[1], [2], [3]])
    cl = _BaseCluster(print_progress=0, random_seed=1)

    assert_raises(AttributeError,
                  'Model is not fitted, yet.',
                  cl.predict,
                  X)


def test_predict_2():
    X = np.array([[1], [2], [3]])
    cl = _BaseCluster(print_progress=0, random_seed=1)

    cl.fit(X)
    cl.predict(X)


def test_shuffle():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    cl = _BaseCluster(print_progress=0, random_seed=1)
    X_sh, y_sh = cl._shuffle(arrays=[X, np.array(y)])
    np.testing.assert_equal(X_sh, np.array([[1], [3], [2]]))
    np.testing.assert_equal(y_sh, np.array([1, 3, 2]))
