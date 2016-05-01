# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend._base import _BaseRegressor
import numpy as np
from mlxtend.utils import assert_raises


def test_init():
    reg = _BaseRegressor(print_progress=0, random_seed=1)


def test_float_ok():
    y = np.array([1., 2.])
    reg = _BaseRegressor(print_progress=0, random_seed=1)
    reg._check_target_array(y=y)


def test_float_fail():
    y = np.array([1, 2], dtype=np.int64)
    reg = _BaseRegressor(print_progress=0, random_seed=1)
    assert_raises(AttributeError,
                  'y must be a float array.\nFound int64',
                  reg._check_target_array,
                  y)
