# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np

from mlxtend._base import _BaseModel, _Regressor
from mlxtend.utils import assert_raises


class BlankRegressor(_BaseModel, _Regressor):
    def __init__(self):
        pass

    def _fit(self, X, y, init_params=True):
        pass

    def _predict(self, X):
        pass


def test_float_ok():
    y = np.array([1.0, 2.0])
    reg = BlankRegressor()
    reg._check_target_array(y=y)


def test_float_fail():
    y = np.array([1, 2], dtype=np.int_)
    reg = BlankRegressor()
    assert_raises(
        AttributeError,
        f"y must be a float array.\nFound {str(y.dtype)}",
        reg._check_target_array,
        y,
    )


def test_predict_fail():
    X = np.array([[1], [2], [3]])
    est = BlankRegressor()
    est._is_fitted = False
    assert_raises(AttributeError, "Model is not fitted, yet.", est.predict, X)


def test_predict_pass():
    X = np.array([[1], [2], [3]])
    y = np.array([1.0, 2.0, 3.0])
    est = BlankRegressor()
    est.fit(X, y)
    est.predict(X)


def test_fit_1():
    X = np.array([[1], [2], [3]])
    est = BlankRegressor()
    assert_raises(
        TypeError, "fit() missing 1 required positional argument: 'y'", est.fit, X
    )


def test_fit_2():
    X = np.array([[1], [2], [3]])
    y = np.array([1.0, 2.0, 3.0])
    est = BlankRegressor()
    est.fit(X=X, y=y)
