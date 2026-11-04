import numpy as np
import pytest

from mlxtend.preprocessing import MeanCenterer


@pytest.mark.parametrize("container", [list, tuple, np.asarray])
def test_matrix_inputs_keep_sample_and_feature_axes(container):
    values = [[1, 2], [3, 6], [8, 4]]
    expected = np.array([[-3, -2], [-1, 2], [4, 0]], dtype=float)
    data = container(values)
    result = MeanCenterer().fit_transform(data)

    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(data, values)


@pytest.mark.parametrize("container", [list, tuple, np.asarray])
def test_vectors_are_single_feature_samples(container):
    data = container([1, 2, 6])
    result = MeanCenterer().fit_transform(data)
    expected = np.array([[-2], [-1], [3]], dtype=float)
    np.testing.assert_allclose(result, expected)
    assert result.shape == (3, 1)


def test_list_transform_uses_fitted_column_means():
    centerer = MeanCenterer().fit([[1, 2], [3, 6]])
    original_means = centerer.col_means.copy()

    first = centerer.transform([[10, 20]])
    second = centerer.transform([[0, 0], [2, 4]])

    np.testing.assert_allclose(first, [[8, 16]])
    np.testing.assert_allclose(second, [[-2, -4], [0, 0]])
    np.testing.assert_array_equal(centerer.col_means, original_means)
