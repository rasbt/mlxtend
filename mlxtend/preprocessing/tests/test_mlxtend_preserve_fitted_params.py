"""Inference batches must not refit or mutate training-set scaling parameters."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from mlxtend.preprocessing.scaling import standardize


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("rows", [1, 3])
def test_constant_transform_uses_fitted_standard_deviation(as_frame, rows):
    train = np.array([[0.0, 1.0], [2.0, 4.0], [4.0, 7.0]])
    query = np.tile([[6.0, 10.0]], (rows, 1))
    if as_frame:
        train, query = pd.DataFrame(train, columns=["a", "b"]), pd.DataFrame(
            query, columns=["a", "b"]
        )
    _, params = standardize(train, return_params=True)
    saved = deepcopy(params)
    result, returned = standardize(query, params=params, return_params=True)
    expected = (np.asarray(query) - np.asarray(saved["avgs"])) / np.asarray(
        saved["stds"]
    )
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(params["stds"], saved["stds"])
    np.testing.assert_array_equal(params["avgs"], saved["avgs"])
    assert returned is params


def test_chunking_the_query_batch_does_not_change_results():
    train = np.array([[0.0, 1.0], [2.0, 4.0], [4.0, 7.0]])
    query = np.array([[1.0, 2.0], [5.0, 8.0], [7.0, 10.0]])
    _, params = standardize(train, return_params=True)
    saved = deepcopy(params)
    full = standardize(query, params=params)
    rowwise = np.vstack([standardize(row[None, :], params=params) for row in query])
    np.testing.assert_allclose(rowwise, full)
    np.testing.assert_array_equal(params["stds"], saved["stds"])


def test_transform_accepts_read_only_fitted_statistics():
    params = {"avgs": np.array([2.0]), "stds": np.array([3.0])}
    params["stds"].flags.writeable = False
    np.testing.assert_allclose(standardize(np.array([[5.0]]), params=params), [[1.0]])


def test_supplied_params_do_not_require_a_zero_row_label():
    train = pd.DataFrame({"a": [0.0, 2.0, 4.0]})
    _, params = standardize(train, return_params=True)
    result = standardize(pd.DataFrame({"a": [6.0]}, index=["sample"]), params=params)
    assert result.index.tolist() == ["sample"]
    np.testing.assert_allclose(result.values, [[4 / np.std([0.0, 2.0, 4.0])]])


@pytest.mark.parametrize("as_frame", [False, True])
def test_fitting_constant_columns_keeps_existing_contract(as_frame):
    train = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    if as_frame:
        train = pd.DataFrame(train, columns=["a", "b"])
    result, params = standardize(train, return_params=True)
    np.testing.assert_array_equal(np.asarray(result)[:, 0], 0.0)
    assert np.asarray(params["stds"])[0] == 1.0


def test_transform_empty_batch_with_params_is_well_defined():
    params = {"avgs": np.array([2.0]), "stds": np.array([3.0])}
    result = standardize(np.empty((0, 1)), params=params)
    assert result.shape == (0, 1)
    assert params["stds"][0] == 3.0


def test_nonconstant_query_keeps_original_transform_formula():
    params = {"avgs": np.array([2.0, 4.0]), "stds": np.array([3.0, 5.0])}
    query = np.array([[5.0, 9.0], [8.0, 14.0]])
    np.testing.assert_allclose(
        standardize(query, params=params), [[1.0, 1.0], [2.0, 2.0]]
    )
