# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import numpy as np
import pandas as pd

from mlxtend.preprocessing import minmax_scaling


def test_minmax_scaling_arrayerror():
    try:
        ary = [[1, 2], [3, 4]]
        minmax_scaling(ary, [1, "s2"])
    except AttributeError:
        pass
    else:
        raise AssertionError


def test_pandas_minmax_scaling():
    s1 = pd.Series([1, 2, 3, 4, 5, 6], index=(range(6)))
    s2 = pd.Series([10, 9, 8, 7, 6, 5], index=(range(6)))
    df = pd.DataFrame(s1, columns=["s1"])
    df["s2"] = s2

    df_out1 = minmax_scaling(df, ["s1", "s2"], min_val=0, max_val=1)
    df_out2 = minmax_scaling(df, ["s1", "s2"], min_val=50, max_val=100)

    ary_out1 = np.array(
        [[0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [1.0, 0.0]]
    )

    ary_out2 = np.array(
        [
            [50.0, 100.0],
            [60.0, 90.0],
            [70.0, 80.0],
            [80.0, 70.0],
            [90.0, 60.0],
            [100.0, 50.0],
        ]
    )

    np.testing.assert_allclose(df_out1.values, ary_out1, rtol=1e-03)
    assert (df_out2.values == ary_out2).all()


def test_numpy_minmax_scaling():
    ary = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5]])

    df_out1 = minmax_scaling(ary, [0, 1], min_val=0, max_val=1)
    df_out2 = minmax_scaling(ary, [0, 1], min_val=50, max_val=100)

    ary_out1 = np.array(
        [[0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [1.0, 0.0]]
    )

    ary_out2 = np.array(
        [
            [50.0, 100.0],
            [60.0, 90.0],
            [70.0, 80.0],
            [80.0, 70.0],
            [90.0, 60.0],
            [100.0, 50.0],
        ]
    )

    np.testing.assert_allclose(df_out1, ary_out1, rtol=1e-03)
    assert (df_out2 == ary_out2).all()


def test_minmax_scaling_constant_column_numpy():
    # Regression for #1167: a constant column has zero range so the
    # naive (x - min) / (max - min) computes 0/0 and produces NaN. The
    # sibling `standardize` already collapses constant columns to 0.0;
    # `minmax_scaling` should match that contract instead of returning
    # NaN silently with a RuntimeWarning.
    ary = np.array([[5, 1], [5, 2], [5, 3]])

    out = minmax_scaling(ary, columns=[0, 1], min_val=0, max_val=1)

    expected = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]])
    assert not np.isnan(out).any(), (
        "minmax_scaling produced NaN for a constant column; "
        "expected the column to be flattened to min_val (0.0)."
    )
    np.testing.assert_allclose(out, expected, rtol=1e-03)


def test_minmax_scaling_constant_column_pandas():
    # Same as above, exercised through the pandas DataFrame branch so the
    # `.loc`-based indexing path is covered too.
    df = pd.DataFrame({"const": [5, 5, 5], "var": [1, 2, 3]})

    out = minmax_scaling(df, columns=["const", "var"], min_val=0, max_val=1)

    expected = np.array([[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]])
    assert not np.isnan(out.values).any()
    np.testing.assert_allclose(out.values, expected, rtol=1e-03)


def test_minmax_scaling_constant_column_custom_range():
    # When `min_val` is non-zero, a flattened constant column should
    # settle at `min_val` (the lower bound of the requested range),
    # because every value is at the column minimum.
    ary = np.array([[7, 1], [7, 2], [7, 3]])

    out = minmax_scaling(ary, columns=[0, 1], min_val=50, max_val=100)

    assert not np.isnan(out).any()
    expected = np.array([[50.0, 50.0], [50.0, 75.0], [50.0, 100.0]])
    np.testing.assert_allclose(out, expected, rtol=1e-03)
