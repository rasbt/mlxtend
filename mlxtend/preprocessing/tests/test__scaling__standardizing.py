# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import numpy as np
import pandas as pd

from mlxtend.preprocessing import standardize


def test_pandas_standardize():
    s1 = pd.Series([1, 2, 3, 4, 5, 6], index=(range(6)))
    s2 = pd.Series([10, 9, 8, 7, 6, 5], index=(range(6)))
    df = pd.DataFrame(s1, columns=["s1"])
    df["s2"] = s2

    df_out1 = standardize(df, ["s1", "s2"])
    ary_out1 = np.array(
        [
            [-1.46385, 1.46385],
            [-0.87831, 0.87831],
            [-0.29277, 0.29277],
            [0.29277, -0.29277],
            [0.87831, -0.87831],
            [1.46385, -1.46385],
        ]
    )
    np.testing.assert_allclose(df_out1.values, ary_out1, rtol=1e-03)


def test_numpy_standardize():
    ary = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5]])

    ary_actu = standardize(ary, columns=[0, 1])
    ary_expc = np.array(
        [
            [-1.46385, 1.46385],
            [-0.87831, 0.87831],
            [-0.29277, 0.29277],
            [0.29277, -0.29277],
            [0.87831, -0.87831],
            [1.46385, -1.46385],
        ]
    )

    np.testing.assert_allclose(ary_actu, ary_expc, rtol=1e-03)


def test_numpy_single_feat():
    ary = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5]])

    ary_actu = standardize(ary, [1])
    ary_expc = np.array(
        [[1.46385], [0.87831], [0.29277], [-0.29277], [-0.87831], [-1.46385]]
    )

    np.testing.assert_allclose(ary_actu, ary_expc, rtol=1e-03)


def test_numpy_inplace():
    ary = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5]])

    standardize(ary, [1])

    ary = ary_expc = np.array(
        [
            [1, 1.46385],
            [2, 0.87831],
            [3, 0.29277],
            [4, -0.29277],
            [5, -0.87831],
            [6, -1.46385],
        ]
    )

    np.testing.assert_allclose(ary, ary_expc, rtol=1e-03)


def test_numpy_single_dim():
    ary = np.array([1, 2, 3, 4, 5, 6])

    ary_actu = standardize(ary, [0])
    ary_expc = np.array(
        [[-1.46385], [-0.87831], [-0.29277], [0.29277], [0.87831], [1.46385]]
    )

    np.testing.assert_allclose(ary_actu, ary_expc, rtol=1e-03)


def test_zero_division_pandas():
    s1 = pd.Series([0, 0, 0, 0, 0, 0], index=(range(6)))
    s2 = pd.Series([10, 9, 8, 7, 6, 5], index=(range(6)))
    df = pd.DataFrame(s1, columns=["s1"])
    df["s2"] = s2

    df_out1 = standardize(df, ["s1", "s2"])
    ary_out1 = np.array(
        [
            [0.0, 1.46385],
            [0.0, 0.87831],
            [0.0, 0.29277],
            [0.0, -0.29277],
            [0.0, -0.87831],
            [0.0, -1.46385],
        ]
    )
    np.testing.assert_allclose(df_out1.values, ary_out1, rtol=1e-03)


def test_zero_division_numpy():
    ary = np.array([[0, 10], [0, 9], [0, 8], [0, 7], [0, 6], [0, 5]])

    ary_actu = standardize(ary, columns=[0, 1])
    ary_expc = np.array(
        [
            [0.0, 1.46385],
            [0.0, 0.87831],
            [0.0, 0.29277],
            [0.0, -0.29277],
            [0.0, -0.87831],
            [0.0, -1.46385],
        ]
    )

    np.testing.assert_allclose(ary_actu, ary_expc, rtol=1e-03)


def test_standardize_all_columns_ndarray():
    ary = np.array([[0, 10], [0, 9], [0, 8], [0, 7], [0, 6], [0, 5]])

    ary_actu = standardize(ary, columns=None)
    ary_expc = np.array(
        [
            [0.0, 1.46385],
            [0.0, 0.87831],
            [0.0, 0.29277],
            [0.0, -0.29277],
            [0.0, -0.87831],
            [0.0, -1.46385],
        ]
    )

    np.testing.assert_allclose(ary_actu, ary_expc, rtol=1e-03)


def test_standardize_all_columns_pandas():
    s1 = pd.Series([1, 2, 3, 4, 5, 6], index=(range(6)))
    s2 = pd.Series([10, 9, 8, 7, 6, 5], index=(range(6)))
    df = pd.DataFrame(s1, columns=["s1"])
    df["s2"] = s2

    df_out1 = standardize(df, columns=None)
    ary_out1 = np.array(
        [
            [-1.46385, 1.46385],
            [-0.87831, 0.87831],
            [-0.29277, 0.29277],
            [0.29277, -0.29277],
            [0.87831, -0.87831],
            [1.46385, -1.46385],
        ]
    )
    np.testing.assert_allclose(df_out1.values, ary_out1, rtol=1e-03)


def test_standardize_constant_column_numpy_issue_1058():
    # Regression test for #1058: a constant column was being mapped to
    # `-mean(column)` instead of `0.0`. The "Notes" docstring promises that
    # constant columns are set to 0.0.
    ary = np.array([[0, 1, 2, 5], [1, 2, 3, 5], [3, 1, 2, 5]], dtype=float)
    ary_actu = standardize(ary)
    # The 4th column is the constant one and must be all-zero.
    np.testing.assert_allclose(ary_actu[:, 3], np.zeros(3))
    # Sanity check: the non-constant columns stay close to z-score scale
    # (mean 0, std 1 with ddof=0). We only assert mean ~= 0 to avoid
    # depending on the exact std implementation.
    np.testing.assert_allclose(ary_actu[:, :3].mean(axis=0), np.zeros(3), atol=1e-9)


def test_standardize_constant_column_pandas_issue_1058():
    # Same regression as the numpy variant, exercised through the pandas
    # branch of standardize().
    df = pd.DataFrame(
        {"x": [0.0, 1.0, 3.0], "y": [1.0, 2.0, 1.0], "k": [5.0, 5.0, 5.0]}
    )
    df_actu = standardize(df, ["x", "y", "k"])
    np.testing.assert_allclose(df_actu["k"].values, np.zeros(3))


def test_standardize_constant_column_returns_unit_std_param_issue_1058():
    # The contract from the "Notes" docstring is that the std for a constant
    # column ends up as 1.0 in the returned `params` dict, so the column can
    # be reused by a subsequent call without dividing by zero.
    ary = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    out, params = standardize(ary, return_params=True)
    np.testing.assert_allclose(out[:, 0], np.zeros(3))
    assert params["stds"][0] == 1.0
