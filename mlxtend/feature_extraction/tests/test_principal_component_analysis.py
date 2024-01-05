# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from mlxtend.data import iris_data
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from mlxtend.preprocessing import standardize
from mlxtend.utils import assert_raises

X, y = iris_data()
X_std = standardize(X)


def test_default_components():
    pca = PCA()
    res = pca.fit(X_std).transform(X_std)
    assert res.shape[1] == 4


def test_whitening():
    pca = PCA(n_components=2)
    res = pca.fit(X_std).transform(X_std)
    diagonals_sum = np.sum(np.diagonal(np.cov(res.T)))
    assert round(diagonals_sum, 1) == 3.9, diagonals_sum

    pca = PCA(n_components=2, whitening=True)
    res = pca.fit(X_std).transform(X_std)
    diagonals_sum = np.sum(np.diagonal(np.cov(res.T)))
    assert round(diagonals_sum, 1) == 2.0, diagonals_sum


def test_default_2components():
    pca = PCA(n_components=2)
    res = pca.fit(X_std).transform(X_std)
    assert res.shape[1] == 2


def test_eigen_vs_svd():
    pca = PCA(n_components=2, solver="eigen")
    eigen_res = pca.fit(X_std).transform(X_std)
    pca = PCA(n_components=2, solver="svd")
    svd_res = pca.fit(X_std).transform(X_std)
    assert_allclose(np.absolute(eigen_res), np.absolute(svd_res), atol=0.0001)


def test_default_components_zero():
    assert_raises(AttributeError, "n_components must be > 1 or None", PCA, 0)


def test_evals():
    pca = PCA(n_components=2, solver="eigen")
    pca.fit(X_std)

    expected = [2.93035378, 0.92740362, 0.14834223, 0.02074601]
    assert_almost_equal(pca.e_vals_, expected, decimal=5)

    pca = PCA(n_components=2, solver="svd")
    pca.fit(X_std)
    assert_almost_equal(pca.e_vals_, expected, decimal=5)


def test_loadings():
    expect = np.array(
        [
            [0.9, -0.4, -0.3, 0.0],
            [-0.5, -0.9, 0.1, -0.0],
            [1.0, -0.0, 0.1, -0.1],
            [1.0, -0.1, 0.2, 0.1],
        ]
    )

    pca = PCA(solver="eigen")
    pca.fit(X_std)
    assert_almost_equal(pca.loadings_, expect, decimal=1)

    expect = np.array(
        [
            [-0.9, -0.4, 0.3, 0.0],
            [0.4, -0.9, -0.1, -0.0],
            [-1.0, -0.0, -0.1, -0.1],
            [-1.0, -0.1, -0.2, 0.1],
        ]
    )

    pca = PCA(solver="svd")
    pca.fit(X_std)
    assert_almost_equal(pca.loadings_, expect, decimal=1)


def test_fail_array_dimension():
    pca = PCA(n_components=2)
    assert_raises(
        ValueError, "X must be a 2D array. Try X[:, numpy.newaxis]", pca.fit, X_std[1]
    )


def test_fail_array_dimension_2():
    pca = PCA(n_components=2)
    assert_raises(
        ValueError,
        "X must be a 2D array. Try X[:, numpy.newaxis]",
        pca.transform,
        X_std[1],
    )


def test_variance_explained_ratio():
    pca = PCA()
    pca.fit(X_std)
    assert_almost_equal(np.sum(pca.e_vals_normalized_), 1.0)
    assert np.sum(pca.e_vals_normalized_ < 0.0) == 0


def test_pca_on_uncentered_data():
    pca1 = PCA(solver="svd")
    pca1.fit(X)

    pca2 = PCA(solver="eigen")
    pca2.fit(X)
    assert_almost_equal(pca1.e_vals_normalized_, pca2.e_vals_normalized_)
