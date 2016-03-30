# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import raises
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


@raises(AttributeError)
def test_default_components():
    pca = PCA(n_components=0)
    pca.fit(X)
    res = pca.fit(X).transform(X)


def test_evals():
    pca = PCA(n_components=2)
    pca.fit(X)
    res = pca.fit(X).transform(X)
    assert_almost_equal(pca.e_vals_, [2.93, 0.93, 0.15, 0.02], decimal=2)


@raises(ValueError)
def test_fail_array_fit():
    pca = PCA(n_components=2)
    pca.fit(X[1])


@raises(ValueError)
def test_fail_array_transform():
    pca = PCA(n_components=2)
    pca.fit(X)
    exp = pca.transform(X[1])
