# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import raises
from mlxtend.feature_extraction import LinearDiscriminantAnalysis as LDA
from mlxtend.data import iris_data
from mlxtend.preprocessing import standardize

X, y = iris_data()
X = standardize(X)


def test_default_components():
    lda = LDA()
    lda.fit(X, y)
    res = lda.fit(X, y).transform(X)
    assert res.shape[1] == 4


def test_default_2components():
    lda = LDA(n_discriminants=2)
    lda.fit(X, y)
    res = lda.fit(X, y).transform(X)
    assert res.shape[1] == 2


@raises(AttributeError)
def test_default_components_0():
    lda = LDA(n_discriminants=0)
    lda.fit(X, y)


def test_evals_eigen():
    lda = LDA(n_discriminants=2)
    lda.fit(X, y).transform(X)
    np.set_printoptions(suppress=True)
    print('%s' % lda.e_vals_)
    assert_almost_equal(lda.e_vals_, [20.49, 0.14, 0.0, 0.0], decimal=2)


def test_evecs_eigen_vs_svd():

    lda = LDA(n_discriminants=2)
    lda.fit(X, y).transform(X)
    eigen_vecs = lda.e_vecs_
    lda = LDA(n_discriminants=2, solver='svd')
    lda.fit(X, y).transform(X)
    assert_almost_equal(lda.e_vecs_[:, 0],
                        eigen_vecs[:, 0], decimal=2)


@raises(ValueError)
def test_fail_array_fit():
    lda = LDA()
    lda.fit(X[1], y[1])


@raises(ValueError)
def test_fail_array_transform():
    lda = LDA()
    lda.fit(X, y)
    lda.transform(X[1])


def test_loadings():

    expect = np.abs(np.array([[0.7, 0., 0., 0.],
                              [0.7, 0.1, 0., 0.],
                              [3.9, 0.3, 0., 0.],
                              [2.1, 0.2, 0., 0.]]))

    lda = LDA(n_discriminants=2)
    lda.fit(X, y)
    assert_almost_equal(np.abs(lda.loadings_), expect, decimal=1)
