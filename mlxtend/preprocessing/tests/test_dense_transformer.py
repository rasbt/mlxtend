# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from packaging.version import Version
from scipy.sparse import issparse
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxtend.preprocessing import DenseTransformer

iris = load_iris()
X, y = iris.data, iris.target


def test_dense_to_dense():
    todense = DenseTransformer(return_copy=False)
    np.testing.assert_array_equal(X, todense.transform(X))


def test_sparse_to_dense():
    todense = DenseTransformer()
    tfidf = TfidfTransformer()
    X_t = tfidf.fit_transform([[1, 2, 3]])
    assert issparse(X_t)
    X_dense = todense.transform(X_t)
    expect = np.array([[0.26726124, 0.53452248, 0.80178373]])
    assert np.allclose(X_dense, expect)


def test_pipeline():
    rf = RandomForestClassifier(n_estimators=10)
    param_grid = [{"randomforestclassifier__n_estimators": [1, 5, 10]}]
    pipe = make_pipeline(StandardScaler(), DenseTransformer(), rf)
    if Version(sklearn_version) < Version("0.24.1"):
        grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=1, iid=False)
    else:
        grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=1)
    grid.fit(X, y)
