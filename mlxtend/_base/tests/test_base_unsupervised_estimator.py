# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend._base import _BaseUnsupervisedEstimator
import numpy as np


def test_init():
    est = _BaseUnsupervisedEstimator(print_progress=0, random_seed=1)
    assert hasattr(est, 'print_progress')
    assert hasattr(est, 'random_seed')


def test_fit():
    X = np.array([[1], [2], [3]])
    est = _BaseUnsupervisedEstimator(print_progress=0, random_seed=1)
    est.fit(X=X)
