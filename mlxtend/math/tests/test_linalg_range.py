# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mlxtend.math import vectorspace_orthonormalization


@pytest.mark.parametrize("scale", [1.0, 1e100, 1e200, 1e308, 1e-200, 1e-300])
def test_orthogonal_columns_retain_their_directions(scale):
    expected = np.array([[0.6, -0.8], [0.8, 0.6]])
    values = expected * scale
    original = values.copy()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = vectorspace_orthonormalization(values, eps=0.0)
    assert_allclose(result, expected, rtol=1e-14, atol=1e-14)
    assert_allclose(result.T @ result, np.eye(2), rtol=1e-14, atol=1e-14)
    assert_array_equal(values, original)


def test_basis_with_unrepresentable_unscaled_norm():
    signs = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    )
    values = signs * 1e308
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = vectorspace_orthonormalization(values)
    assert_allclose(result, signs / 2, rtol=1e-14, atol=1e-14)
    assert_allclose(result.T @ result, np.eye(4), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("scale", [1.0, 1e200, 1e308, 1e-200])
def test_projection_and_dependent_columns_at_different_scales(scale):
    values = (
        np.array([[0.5, 1.0, 0.5, 0.0], [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]])
        * scale
    )
    expected = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = vectorspace_orthonormalization(values, eps=0.0)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("eps", [0.0, 1e-250, 1e-200, 1e-100])
def test_small_columns_keep_absolute_norm_threshold(eps):
    values = np.diag([2e-200, 5e-201])
    result = vectorspace_orthonormalization(values, eps=eps)
    expected = np.diag([float(2e-200 > eps), float(5e-201 > eps)])
    assert_array_equal(result, expected)


def test_mixed_column_scales():
    expected = np.array([[0.6, -0.8], [0.8, 0.6]])
    values = expected * np.array([1e200, 1e-200])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = vectorspace_orthonormalization(values, eps=0.0)
    assert_allclose(result, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (3, 2)])
def test_zero_and_empty_columns(shape):
    values = np.zeros(shape)
    assert_array_equal(vectorspace_orthonormalization(values), values)


def test_small_columns_below_default_threshold():
    values = np.eye(2) * np.nextafter(0.0, 1.0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = vectorspace_orthonormalization(values)
    assert_array_equal(result, np.zeros((2, 2)))
