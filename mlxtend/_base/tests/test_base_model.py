# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np

from mlxtend._base import _BaseModel
from mlxtend.utils import assert_raises


class BlankModel(_BaseModel):
    def __init__(self, print_progress=0, random_seed=1):
        self.print_progress = print_progress
        self.random_seed = random_seed


def test_init():
    est = BlankModel(print_progress=0, random_seed=1)
    assert hasattr(est, "print_progress")
    assert hasattr(est, "random_seed")


def test_check_array_1():
    X = np.array([1, 2, 3])
    est = BlankModel()
    assert_raises(
        ValueError,
        "X must be a 2D array. Try X[:, numpy.newaxis]",
        est._check_arrays,
        X,
    )


def test_check_array_2():
    X = list([[1], [2], [3]])
    est = BlankModel(print_progress=0, random_seed=1)

    assert_raises(ValueError, "X must be a numpy array", est._check_arrays, X)


def test_check_array_3():
    X = np.array([[1], [2], [3]])
    est = BlankModel(print_progress=0, random_seed=1)
    est._check_arrays(X)
