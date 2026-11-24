# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_moons

from mlxtend.feature_extraction import RBFKernelPCA as KPCA

X1, y1 = make_moons(n_samples=50, random_state=1)
X_ASYMMETRIC = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 3.0], [4.0, 1.0]])


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


@pytest.mark.parametrize("n_components", [None, 1, 2, 3])
def test_transform_reproduces_fitted_projection(n_components):
    pca = KPCA(gamma=0.2, n_components=n_components).fit(X_ASYMMETRIC)

    np.testing.assert_allclose(
        pca.transform(X_ASYMMETRIC), pca.X_projected_, atol=1e-12
    )


def test_transform_centers_new_samples_in_training_feature_space():
    train = X_ASYMMETRIC
    query = np.array([[0.25, 0.25], [5.0, 1.5]])
    pca = KPCA(gamma=0.2, n_components=2).fit(train)

    def kernel(samples):
        squared_distances = ((samples[:, None, :] - train) ** 2).sum(axis=2)
        return np.exp(-0.2 * squared_distances)

    training_kernel = kernel(train)
    cross_kernel = kernel(query)
    centered = (
        cross_kernel
        - cross_kernel.mean(axis=1, keepdims=True)
        - training_kernel.mean(axis=0)
        + training_kernel.mean()
    )
    expected = centered @ (pca.e_vecs_[:, :2] / pca.e_vals_[:2])
    np.testing.assert_allclose(pca.transform(query), expected, atol=1e-12)
    np.testing.assert_allclose(pca.transform(query[:1]), expected[:1], atol=1e-12)


def test_transform_with_previously_fitted_state():
    pca = KPCA(gamma=0.2).fit(X_ASYMMETRIC)
    del pca._kernel_row_means
    np.testing.assert_allclose(
        pca.transform(X_ASYMMETRIC), pca.X_projected_, atol=1e-12
    )


def test_fail_array_fit():
    pca = KPCA(n_components=2)
    with pytest.raises(ValueError):
        pca.fit(X1[1])


def test_fail_array_transform():
    pca = KPCA(n_components=2)
    pca.fit(X1)
    with pytest.raises(ValueError):
        pca.transform(X1[1])
