# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import numpy as np

from mlxtend.preprocessing import shuffle_arrays_unison
from mlxtend.preprocessing.shuffle import shuffled_split


def test_shuffle_arrays_unison():
    X1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y1 = np.array([1, 2, 3])

    X2, y2 = shuffle_arrays_unison(arrays=[X1, y1], random_seed=3)

    np.testing.assert_array_equal(X2, np.array([[4, 5, 6], [1, 2, 3], [7, 8, 9]]))
    np.testing.assert_array_equal(y2, np.array([2, 1, 3]))


def test_shuffle_arrays_unison_keeps_rows_aligned():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X2, y2 = shuffle_arrays_unison(arrays=[X, y], random_seed=1)

    np.testing.assert_array_equal(X2, X[y2])
    np.testing.assert_array_equal(np.sort(y2), y)


def test_shuffle_arrays_unison_random_seed_zero():
    x = np.arange(8)
    expected = x[np.random.RandomState(0).permutation(len(x))]

    np.random.seed(123)
    first = shuffle_arrays_unison(arrays=[x], random_seed=0)[0]
    second = shuffle_arrays_unison(arrays=[x], random_seed=0)[0]

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    np.testing.assert_array_equal(x, np.arange(8))


def test_shuffle_arrays_unison_positive_seed_reproducible():
    x = np.arange(8)
    expected = x[np.random.RandomState(42).permutation(len(x))]

    np.random.seed(1)
    first = shuffle_arrays_unison(arrays=[x], random_seed=42)[0]
    np.random.seed(2)
    second = shuffle_arrays_unison(arrays=[x], random_seed=42)[0]

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)


def test_shuffle_arrays_unison_none_uses_global_state():
    x = np.arange(8)

    np.random.seed(7)
    expected = x[np.random.permutation(len(x))]
    np.random.seed(7)
    result = shuffle_arrays_unison(arrays=[x], random_seed=None)[0]

    np.testing.assert_array_equal(result, expected)


def test_shuffled_split_random_seed_zero():
    X = np.arange(16).reshape(8, 2)
    y = np.arange(8)

    np.random.seed(1)
    first = shuffled_split(X, y, random_seed=0)
    np.random.seed(2)
    second = shuffled_split(X, y, random_seed=0)

    expected_y = y[np.random.RandomState(0).permutation(len(y))]
    np.testing.assert_array_equal(first[1], expected_y[:6])
    np.testing.assert_array_equal(first[3], expected_y[6:])
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
