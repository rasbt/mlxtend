# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from mlxtend.data import iris_data
from mlxtend.feature_extraction import LinearDiscriminantAnalysis as LDA
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


def test_default_0components():
    with pytest.raises(AttributeError):
        LDA(n_discriminants=0)


def test_evals():
    lda = LDA(n_discriminants=2)
    lda.fit(X, y).transform(X)
    np.set_printoptions(suppress=True)
    print("%s" % lda.e_vals_)
    assert_almost_equal(lda.e_vals_, [20.90, 0.14, 0.0, 0.0], decimal=2)


def test_fail_array_fit():
    lda = LDA()
    with pytest.raises(ValueError):
        lda.fit(X[1], y[1])


def test_fail_array_transform():
    lda = LDA()
    lda.fit(X, y)
    with pytest.raises(ValueError):
        lda.transform(X[1])
