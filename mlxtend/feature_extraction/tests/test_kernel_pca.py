# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA

from mlxtend.feature_extraction import RBFKernelPCA as KPCA

X1, y1 = make_moons(n_samples=50, random_state=1)


def test_default_components():
    pca = KPCA()
    pca.fit(X1)
    assert pca.X_projected_.shape == X1.shape


def test_default_2components():
    pca = KPCA(n_components=2)
    pca.fit(X1)
    assert pca.X_projected_.shape == (X1.shape[0], 2)


def test_default_0components():
    with pytest.raises(AttributeError):
        pca = KPCA(n_components=0)
        pca.fit(X1)


def test_proj():
    pca = KPCA(n_components=2)
    pca.fit(X1[:2])
    exp = np.array([[-0.71, -0.71], [0.71, -0.71]])
    assert_almost_equal(pca.X_projected_, exp, decimal=2)


def test_reproj_1():
    pca = KPCA(n_components=2)
    pca.fit(X1)
    exp = pca.transform(X1)
    assert_almost_equal(pca.X_projected_, exp, decimal=2)


def test_reproj_2():
    pca = KPCA(n_components=2)
    pca.fit(X1)
    exp = pca.transform(X1[1, None])
    assert_almost_equal(pca.X_projected_[1, None], exp, decimal=2)


def test_fail_array_fit():
    pca = KPCA(n_components=2)
    with pytest.raises(ValueError):
        pca.fit(X1[1])


def test_fail_array_transform():
    pca = KPCA(n_components=2)
    pca.fit(X1)
    with pytest.raises(ValueError):
        pca.transform(X1[1])


def test_transform_reuses_fitted_components():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 3.0], [4.0, 1.0]])

    for n in [None, 1, 2, 3]:
        pca = KPCA(gamma=0.2, n_components=n).fit(X)
        projected = pca.transform(X)
        expected_components = 2 if n is None else min(n, 2)
        assert pca.n_components_ == expected_components
        assert projected.shape == pca.X_projected_.shape
        assert_allclose(projected, pca.X_projected_, rtol=1e-7, atol=1e-9)


def test_transform_held_out_matches_centered_kernel_pca():
    rng = np.random.RandomState(0)
    X_train = rng.randn(20, 4)
    X_test = rng.randn(5, 4)
    gamma = 0.15
    n_components = 2

    pca = KPCA(gamma=gamma, n_components=n_components).fit(X_train)
    reference = KernelPCA(
        kernel="rbf",
        gamma=gamma,
        n_components=n_components,
        eigen_solver="dense",
    ).fit(X_train)

    # RBFKernelPCA stores unit eigenvectors in X_projected_ and scales
    # transformed samples by 1 / eigenvalue. KernelPCA uses 1 / sqrt(eigenvalue),
    # so compare after restoring the common projection scale. Eigenvector signs
    # are arbitrary, hence the absolute comparison.
    actual = pca.transform(X_test) * np.sqrt(pca.e_vals_[:n_components])
    assert_allclose(
        np.abs(actual),
        np.abs(reference.transform(X_test)),
        rtol=1e-7,
        atol=1e-9,
    )
