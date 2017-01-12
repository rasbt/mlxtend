# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_allclose
from mlxtend.utils import assert_raises
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from mlxtend.data import iris_data
from mlxtend.preprocessing import standardize

X, y = iris_data()
X = standardize(X)


def test_default_components():
    pca = PCA()
    res = pca.fit(X).transform(X)
    assert res.shape[1] == 4


def test_default_2components():
    pca = PCA(n_components=2)
    res = pca.fit(X).transform(X)
    assert res.shape[1] == 2


def test_eigen_vs_svd():
    pca = PCA(n_components=2, solver='eigen')
    eigen_res = pca.fit(X).transform(X)
    pca = PCA(n_components=2, solver='svd')
    svd_res = pca.fit(X).transform(X)
    assert_allclose(np.absolute(eigen_res), np.absolute(svd_res), atol=0.0001)


def test_default_components_zero():
    assert_raises(AttributeError,
                  'n_components must be > 1 or None',
                  PCA,
                  0)


def test_evals():
    pca = PCA(n_components=2, solver='eigen')
    pca.fit(X)
    res = pca.fit(X).transform(X)
    assert_almost_equal(pca.e_vals_, [2.93, 0.93, 0.15, 0.02], decimal=2)


def test_fail_array_dimension():
    pca = PCA(n_components=2)
    assert_raises(ValueError,
                  'X must be a 2D array. Try X[:, numpy.newaxis]',
                  pca.fit,
                  X[1])


def test_fail_array_dimension():
    pca = PCA(n_components=2)
    assert_raises(ValueError,
                  'X must be a 2D array. Try X[:, numpy.newaxis]',
                  pca.transform,
                  X[1])
