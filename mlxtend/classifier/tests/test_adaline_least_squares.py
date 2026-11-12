# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import lstsq

from mlxtend.classifier import Adaline


def _training_data(kind):
    x = np.arange(6, dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    if kind == "duplicate":
        return np.column_stack((x, x)), y
    if kind == "constant":
        return np.column_stack((x, np.ones_like(x))), y
    if kind == "zero":
        return np.column_stack((x, np.zeros_like(x))), y
    if kind == "wide":
        return np.arange(24, dtype=float).reshape(3, 8), np.array([0, 1, 1])
    return np.column_stack((x, x**2)), y


@pytest.mark.parametrize("kind", ["duplicate", "constant", "zero", "wide", "full"])
def test_adaline_analytical_fit_uses_least_squares_solution(kind):
    X, y = _training_data(kind)
    original_X, original_y = X.copy(), y.copy()
    design = np.column_stack((np.ones(len(X)), X))
    targets = 2 * y - 1
    # Pivoted QR gives an independent least-squares reference.
    expected, _, _, _ = lstsq(design, targets, lapack_driver="gelsy")

    model = Adaline().fit(X, y)
    coefficients = np.concatenate((model.b_, model.w_.ravel()))

    assert_allclose(coefficients, expected, atol=1e-12, rtol=1e-12)
    assert_allclose(model._net_input(X), design @ expected, atol=1e-12)
    assert_allclose(design.T @ (design @ coefficients - targets), 0, atol=1e-11)
    assert_array_equal(model.predict(X), (design @ expected >= 0).astype(int))
    assert_array_equal(X, original_X)
    assert_array_equal(y, original_y)


def test_adaline_analytical_fit_does_not_square_the_condition_number():
    x = np.linspace(-1, 1, 20)
    X = np.column_stack((x, x + 1e-8 * np.sin(np.arange(len(x)))))
    y = (np.sin(np.arange(len(x))) > 0).astype(int)
    design = np.column_stack((np.ones(len(X)), X))
    targets = 2 * y - 1
    expected, _, _, _ = lstsq(design, targets, lapack_driver="gelsy")

    model = Adaline().fit(X, y)

    assert_allclose(model._net_input(X), design @ expected, atol=1e-7, rtol=1e-7)
    assert np.isfinite(model.w_).all()
    assert np.isfinite(model.b_).all()
