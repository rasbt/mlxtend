# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import sys

import numpy as np

from mlxtend.data import make_multiplexer_dataset
from mlxtend.utils import assert_raises


def test_defaults():
    X, y = make_multiplexer_dataset()

    assert X.shape == (100, 6), X.shape
    assert X.dtype == np.int_
    assert y.shape == (100,), y.shape
    assert y.dtype == np.int_


def test_invalid_address_bits():
    msg_1 = "address_bits must be an integer. Got <class 'float'>."

    # for Python 2.7:
    if sys.version_info[0] == 2:
        msg_1 = msg_1.replace("<class", "<type")

    assert_raises(AttributeError, msg_1, make_multiplexer_dataset, 1.2)

    msg_2 = "Number of address_bits must be greater than 0. Got -1."
    assert_raises(AttributeError, msg_2, make_multiplexer_dataset, -1)


def test_imbalance():
    X, y = make_multiplexer_dataset(
        address_bits=2, sample_size=1000, positive_class_ratio=0.3
    )
    np.bincount(y) == (700, 300), np.bincount(y)

    X, y = make_multiplexer_dataset(
        address_bits=2, sample_size=1000, positive_class_ratio=0.7
    )
    np.bincount(y) == (300, 700), np.bincount(y)


def test_address_bits():
    X, y = make_multiplexer_dataset(address_bits=3, sample_size=100)
    assert X.shape == (100, 11)


def test_class_labels():
    X, y = make_multiplexer_dataset(address_bits=2, sample_size=10, random_seed=0)

    assert np.array_equal(y, np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))
    features = np.array(
        [
            [0, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1],
        ]
    )

    print(X)
    assert np.array_equal(X, features)


def test_class_labels_shuffle():
    X, y = make_multiplexer_dataset(
        address_bits=2, sample_size=10, random_seed=0, shuffle=True
    )

    print(y)
    assert np.array_equal(y, np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1]))
    features = np.array(
        [
            [0, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [1, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [0, 1, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ]
    )

    assert np.array_equal(X, features)
