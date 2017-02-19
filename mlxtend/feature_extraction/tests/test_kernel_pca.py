# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import raises
from mlxtend.feature_extraction import RBFKernelPCA as KPCA
from sklearn.datasets import make_moons

X1, y1 = make_moons(n_samples=50, random_state=1)


def test_default_components():
    pca = KPCA()
    pca.fit(X1)
    assert pca.X_projected_.shape == X1.shape


def test_default_2components():
    pca = KPCA(n_components=2)
    pca.fit(X1)
    assert pca.X_projected_.shape == (X1.shape[0], 2)


@raises(AttributeError)
def test_default_components():
    pca = KPCA(n_components=0)
    pca.fit(X1)


def test_proj():
    pca = KPCA(n_components=2)
    pca.fit(X1[:2])
    exp = np.array([[-0.71, -0.71],
                    [0.71, -0.71]])
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


@raises(ValueError)
def test_fail_array_fit():
    pca = KPCA(n_components=2)
    pca.fit(X1[1])


@raises(ValueError)
def test_fail_array_transform():
    pca = KPCA(n_components=2)
    pca.fit(X1)
    pca.transform(X1[1])
