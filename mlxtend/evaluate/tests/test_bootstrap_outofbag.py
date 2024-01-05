# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np

from mlxtend.evaluate import BootstrapOutOfBag
from mlxtend.utils import assert_raises


def test_defaults():
    oob = BootstrapOutOfBag()
    results = list(oob.split(np.array([1, 2, 3, 4, 5])))
    assert len(results) == 200


def test_splits():
    oob = BootstrapOutOfBag(n_splits=3, random_seed=123)
    results = list(oob.split(np.array([1, 2, 3, 4, 5])))
    assert len(results) == 3
    assert np.array_equal(results[0][0], np.array([2, 4, 2, 1, 3]))
    assert np.array_equal(results[0][1], np.array([0]))
    assert np.array_equal(results[-1][0], np.array([1, 1, 0, 0, 1]))
    assert np.array_equal(results[-1][1], np.array([2, 3, 4]))


def test_invalid_splits():
    assert_raises(
        ValueError, "Number of splits must be greater than 1.", BootstrapOutOfBag, 0
    )


def test_get_n_splits():
    oob = BootstrapOutOfBag(n_splits=3, random_seed=123)
    assert oob.n_splits == 3
