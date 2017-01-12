# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.utils import assert_raises
from mlxtend.utils import check_Xy
import numpy as np
import sys

y = np.array([1, 2, 3, 4])
X = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])


def test_ok():
    check_Xy(X, y)


def test_invalid_type_X():
    expect = "X must be a NumPy array. Found <class 'list'>"
    if (sys.version_info < (3, 0)):
        expect = expect.replace('class', 'type')
    assert_raises(ValueError,
                  expect,
                  check_Xy,
                  [1, 2, 3, 4],
                  y)


def test_invalid_type_y():
    expect = "y must be a NumPy array. Found <class 'list'>"
    if (sys.version_info < (3, 0)):
        expect = expect.replace('class', 'type')
    assert_raises(ValueError,
                  expect,
                  check_Xy,
                  X,
                  [1, 2, 3, 4])


def test_invalid_dtype_X():
    assert_raises(ValueError,
                  'X must be an integer or float array. Found object.',
                  check_Xy,
                  X.astype('object'),
                  y)


def test_invalid_dtype_y():

    if (sys.version_info > (3, 0)):
        expect = ('y must be an integer array. Found <U1. '
                  'Try passing the array as y.astype(np.integer)')
    else:
        expect = ('y must be an integer array. Found |S1. '
                  'Try passing the array as y.astype(np.integer)')
    assert_raises(ValueError,
                  expect,
                  check_Xy,
                  X,
                  np.array(['a', 'b', 'c', 'd']))


def test_invalid_dim_y():
    assert_raises(ValueError,
                  'y must be a 1D array. Found (4, 2)',
                  check_Xy,
                  X,
                  X.astype(np.integer))


def test_invalid_dim_X():
    assert_raises(ValueError,
                  'X must be a 2D array. Found (4,)',
                  check_Xy,
                  y,
                  y)


def test_unequal_length_X():
    assert_raises(ValueError,
                  ('y and X must contain the same number of samples. '
                   'Got y: 4, X: 3'),
                  check_Xy,
                  X[1:],
                  y)


def test_unequal_length_y():
    assert_raises(ValueError,
                  ('y and X must contain the same number of samples. '
                   'Got y: 3, X: 4'),
                  check_Xy,
                  X,
                  y[1:])
