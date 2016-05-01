# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend._base import _BaseUnsupervisedEstimator
import numpy as np
from mlxtend.utils import assert_raises


def test_init():
    est = _BaseUnsupervisedEstimator(print_progress=0, random_seed=1)


def test_fit():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    est = _BaseUnsupervisedEstimator(print_progress=0, random_seed=1)
    est.fit(X=X)
