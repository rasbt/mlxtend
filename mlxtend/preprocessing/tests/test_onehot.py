# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.preprocessing import one_hot
from nose.tools import raises


def test_default():
    y = np.array([0, 1, 2, 3, 4, 2])
    expect = np.array([[1., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 1.],
                       [0., 0., 1., 0., 0.]], dtype='float')
    out = one_hot(y)
    np.testing.assert_array_equal(expect, out)


def test_autoguessing():
    y = np.array([0, 4, 0, 4])
    expect = np.array([[1., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 1.],
                       [1., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 1.]], dtype='float')
    out = one_hot(y)
    np.testing.assert_array_equal(expect, out)


def test_list():
    y = [0, 1, 2, 3, 4, 2]
    expect = np.array([[1., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 1.],
                       [0., 0., 1., 0., 0.]], dtype='float')
    out = one_hot(y)
    np.testing.assert_array_equal(expect, out)


@raises(AttributeError)
def test_multidim_list():
    y = [[0, 1, 2, 3, 4, 2]]
    one_hot(y)


@raises(AttributeError)
def test_multidim_array():
    y = np.array([[0], [1], [2], [3], [4], [2]])
    one_hot(y)


def test_oneclass():
    np.testing.assert_array_equal(one_hot([0]),
                                  np.array([[0.]], dtype='float'))


def test_list_morelabels():
    y = [0, 1]
    expect = np.array([[1., 0., 0.],
                       [0., 1., 0.]], dtype='float')
    out = one_hot(y, num_labels=3)
    np.testing.assert_array_equal(expect, out)
