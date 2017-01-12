# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.preprocessing import CopyTransformer
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import issparse
from mlxtend.utils import assert_raises
import sys
from sklearn.model_selection import GridSearchCV

iris = load_iris()
X, y = iris.data, iris.target


def test_copy():
    copy = CopyTransformer()
    np.testing.assert_array_equal(X, copy.transform(X))


def test_copy_failtype():
    copy = CopyTransformer()

    expect = ("X must be a list or NumPy array or SciPy sparse array."
              " Found <class 'int'>")
    if sys.version_info < (3, 0):
        expect = expect.replace('class', 'type')
    assert_raises(ValueError,
                  expect,
                  copy.transform,
                  1)


def test_pipeline():
    param_grid = [{'logisticregression__C': [1, 0.1, 10]}]
    pipe = make_pipeline(StandardScaler(),
                         CopyTransformer(),
                         LogisticRegression())
    grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=1)
    grid.fit(X, y)


def test_sparse():
    copy = CopyTransformer()
    tfidf = TfidfTransformer()
    X_t = tfidf.fit_transform([[1, 2, 3]])
    assert issparse(X_t)
    X_dense = copy.transform(X_t).toarray()
    expect = np.array([[0.26726124, 0.53452248, 0.80178373]])
    assert np.allclose(X_dense, expect)
