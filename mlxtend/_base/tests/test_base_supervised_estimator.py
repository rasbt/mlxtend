# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend._base import _BaseSupervisedEstimator
import numpy as np
from mlxtend.utils import assert_raises


def test_init():
    est = _BaseSupervisedEstimator(print_progress=0, random_seed=1)
    assert hasattr(est, 'print_progress')
    assert hasattr(est, 'random_seed')


def test_fit_1():
    X = np.array([[1], [2], [3]])
    est = _BaseSupervisedEstimator(print_progress=0, random_seed=1)
    assert_raises(TypeError,
                  "fit() missing 1 required positional argument: 'y'",
                  est.fit,
                  X)


def test_fit_2():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    est = _BaseSupervisedEstimator(print_progress=0, random_seed=1)
    est.fit(X=X, y=y)


def test_check_target_array():
    y = np.array([1, 2, 3])
    est = _BaseSupervisedEstimator(print_progress=0, random_seed=1)
    est._check_target_array(y)
