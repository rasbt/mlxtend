# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mlxtend.regressor import LinearRegression


@pytest.mark.parametrize(
    "values",
    [
        [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
        [[1.0, 2.0, 3.0, 4.0]],
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        [[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, -1.0]],
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_qr_matches_minimum_norm_svd_reference(values, dtype):
    X = np.asarray(values, dtype=dtype)
    y = np.linspace(-1.0, 2.0, len(X)) ** 2
    original_X, original_y = X.copy(), y.copy()
    X.setflags(write=False)
    y.setflags(write=False)
    design = np.column_stack((np.ones(len(X)), X))
    rcond = np.finfo(design.dtype).eps * max(design.shape)
    expected = np.linalg.pinv(design, rcond=rcond) @ y

    model = LinearRegression(method="qr").fit(X, y)
    coefficients = np.concatenate((model.b_, model.w_.ravel()))
    assert model.b_.shape == (1,)
    assert model.w_.shape == (X.shape[1], 1)
    assert_allclose(coefficients, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(model.predict(X), design @ expected, rtol=1e-12, atol=1e-12)
    residual = model.predict(X) - y
    assert_allclose(design.T @ residual, 0.0, atol=1e-11)
    assert_array_equal(X, original_X)
    assert_array_equal(y, original_y)


def test_qr_wide_system_refit_replaces_coefficient_shapes():
    model = LinearRegression(method="qr")
    X1 = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
    model.fit(X1, np.array([1.0, 2.0, 4.0]))
    X2 = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    y2 = np.array([1.0, 2.0])
    assert model.fit(X2, y2) is model
    assert model.w_.shape == (3, 1)
    assert_allclose(model.predict(X2), y2, rtol=1e-12, atol=1e-12)
