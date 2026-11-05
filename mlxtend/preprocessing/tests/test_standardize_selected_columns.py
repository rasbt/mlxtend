# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# License: BSD 3 clause

import numpy as np
import pandas as pd
import pytest

from mlxtend.preprocessing import standardize


@pytest.mark.parametrize("container", ["numpy", "pandas"])
@pytest.mark.parametrize(
    "columns", [[1], [1, 0], [0, 1], [2, 1], [1, 2], [2, 0, 1], None]
)
def test_standardize_selected_constant_columns(container, columns):
    values = np.array([[1.0, 7.0, 2.0], [2.0, 7.0, 4.0], [3.0, 7.0, 6.0]])
    original = values.copy()
    data = pd.DataFrame(values) if container == "pandas" else values
    selected = [0, 1, 2] if columns is None else columns
    unit = np.sqrt(1.5)
    expected = np.array([[-unit, 0.0, -unit], [0.0, 0.0, 0.0], [unit, 0.0, unit]])

    with np.errstate(divide="raise", invalid="raise"):
        result, params = standardize(data, columns=columns, return_params=True)

    np.testing.assert_allclose(result, expected[:, selected])
    np.testing.assert_allclose(params["avgs"], np.array([2.0, 7.0, 4.0])[selected])
    np.testing.assert_allclose(
        params["stds"], np.array([np.sqrt(2 / 3), 1.0, np.sqrt(8 / 3)])[selected]
    )
    np.testing.assert_array_equal(data, original)
    if container == "pandas":
        pd.testing.assert_index_equal(result.columns, pd.Index(selected))
        pd.testing.assert_index_equal(result.index, data.index)


@pytest.mark.parametrize("columns", [["constant"], ["constant", "signal"]])
def test_standardize_selected_string_columns(columns):
    data = pd.DataFrame({"signal": [1.0, 2.0, 3.0], "constant": [7.0, 7.0, 7.0]})
    original = data.copy(deep=True)
    result, params = standardize(data, columns=columns, return_params=True)

    np.testing.assert_array_equal(result["constant"], np.zeros(3))
    assert params["stds"]["constant"] == 1.0
    if "signal" in columns:
        np.testing.assert_allclose(result["signal"], [-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    pd.testing.assert_frame_equal(data, original)


def test_standardize_nonconstant_columns_with_readonly_params():
    """An empty constant-column mask must not write to fitted statistics."""
    data = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    params = {"avgs": np.array([4.0, 2.0]), "stds": np.sqrt([8 / 3, 2 / 3])}
    for values in params.values():
        values.setflags(write=False)

    result = standardize(data, columns=[1, 0], params=params)

    expected = np.array([[-np.sqrt(1.5)] * 2, [0.0] * 2, [np.sqrt(1.5)] * 2])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(params["stds"], np.sqrt([8 / 3, 2 / 3]))
