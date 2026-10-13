import numpy as np
import pytest

from mlxtend.preprocessing import zscore_outlier_mask


def test_flags_extreme_value():
    X = np.array([[1.0], [2.0], [3.0], [100.0]])
    mask = zscore_outlier_mask(X, threshold=1.5)
    np.testing.assert_array_equal(mask, [True, True, True, False])


def test_constant_column_keeps_all_rows():
    X = np.ones((5, 2))
    assert zscore_outlier_mask(X).all()


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one sample"):
        zscore_outlier_mask(np.empty((0, 2)))


def test_nonpositive_threshold_raises():
    with pytest.raises(ValueError, match="positive"):
        zscore_outlier_mask(np.ones((3, 1)), threshold=0)
