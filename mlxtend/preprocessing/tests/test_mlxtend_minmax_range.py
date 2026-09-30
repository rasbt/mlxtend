"""Changing either end of the target interval must change the affine map."""

import numpy as np
import pandas as pd
import pytest

from mlxtend.preprocessing.scaling import minmax_scaling


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize(
    "low,high", [(0, 255), (-1, 1), (0, 1), (2, 3), (0.25, 1), (0, 0.5)]
)
def test_requested_range_matches_affine_reference(as_frame, low, high):
    x = np.array([[7, 1], [7, 2], [7, 3]])
    source = pd.DataFrame(x, columns=["fixed", "varying"]) if as_frame else x
    saved = source.copy()
    columns = ["fixed", "varying"] if as_frame else [0, 1]
    result = minmax_scaling(source, columns, min_val=low, max_val=high)
    expected = np.array([[low, low], [low, (low + high) / 2], [low, high]])
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
    if as_frame:
        pd.testing.assert_frame_equal(source, saved)
    else:
        np.testing.assert_array_equal(source, saved)


def test_single_numpy_column_keeps_shape():
    result = minmax_scaling(np.array([2, 4, 6]), [0], min_val=-1, max_val=1)
    assert result.shape == (3, 1)
    np.testing.assert_allclose(result[:, 0], [-1, 0, 1])


def test_dataframe_preserves_selected_labels_and_index():
    frame = pd.DataFrame({"keep": [100, 200], "scale": [2, 6]}, index=["a", "b"])
    result = minmax_scaling(frame, ["scale"], max_val=10)
    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"scale": [0.0, 10.0]}, index=["a", "b"])
    )
