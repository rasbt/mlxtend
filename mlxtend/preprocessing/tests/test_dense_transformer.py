# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.preprocessing import DenseTransformer
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import issparse
from sklearn.model_selection import GridSearchCV

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
    rf = RandomForestClassifier()
    param_grid = [{'randomforestclassifier__n_estimators': [1, 5, 10]}]
    pipe = make_pipeline(StandardScaler(), DenseTransformer(), rf)
    grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=1)
    grid.fit(X, y)
