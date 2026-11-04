import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from mlxtend.feature_selection import ColumnSelector


@pytest.mark.parametrize("shape", [(4, 3), (4, 1), (0, 3), (0, 1)])
@pytest.mark.parametrize("drop_axis", [False, True])
@pytest.mark.parametrize("as_frame", [False, True])
def test_default_selects_all_columns(shape, drop_axis, as_frame):
    values = np.arange(np.prod(shape)).reshape(shape)
    data = pd.DataFrame(values) if as_frame else values
    selector = ColumnSelector(drop_axis=drop_axis)

    result = selector.fit_transform(data)

    expected = values[:, 0] if drop_axis and shape[1] == 1 else values
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape
    assert selector.cols is None


def test_default_selects_all_sparse_columns():
    values = csr_matrix([[0, 2, 0], [1, 0, 3]])
    result = ColumnSelector().fit_transform(values)
    assert result.shape == values.shape
    np.testing.assert_array_equal(result.toarray(), values.toarray())


@pytest.mark.parametrize("as_frame", [False, True])
def test_default_selector_in_regression_pipeline(as_frame):
    values = np.array([[0, 1], [1, 0], [2, 3], [4, 2]], dtype=float)
    data = pd.DataFrame(values) if as_frame else values
    target = 2 * values[:, 0] - 3 * values[:, 1]
    pipeline = make_pipeline(ColumnSelector(), LinearRegression())
    pipeline.fit(data, target)
    np.testing.assert_allclose(pipeline.predict(data), target, atol=1e-12)
