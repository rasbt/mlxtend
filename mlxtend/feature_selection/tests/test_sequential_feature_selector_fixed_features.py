# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.utils import assert_raises

iris = load_iris()
X_iris = iris.data
y_iris = iris.target


def test_run_forward():
    knn = KNeighborsClassifier()
    sfs = SFS(
        estimator=knn,
        k_features=3,
        forward=True,
        floating=False,
        fixed_features=(0, 1),
        verbose=0,
    )
    sfs.fit(X_iris, y_iris)
    for k in sfs.subsets_:
        assert 0 in sfs.subsets_[k]["feature_idx"]
        assert 1 in sfs.subsets_[k]["feature_idx"]


def test_run_floating_forward():
    knn = KNeighborsClassifier()
    sfs = SFS(
        estimator=knn,
        k_features=3,
        forward=True,
        floating=True,
        fixed_features=(0, 1),
        verbose=0,
    )
    sfs.fit(X_iris, y_iris)
    for k in sfs.subsets_:
        assert 0 in sfs.subsets_[k]["feature_idx"]
        assert 1 in sfs.subsets_[k]["feature_idx"]


def test_run_backward():
    knn = KNeighborsClassifier()
    sfs = SFS(
        estimator=knn,
        k_features=3,
        forward=False,
        floating=False,
        fixed_features=(0, 1),
        verbose=0,
    )
    sfs.fit(X_iris, y_iris)
    for k in sfs.subsets_:
        assert 0 in sfs.subsets_[k]["feature_idx"]
        assert 1 in sfs.subsets_[k]["feature_idx"]


def test_run_floating_backward():
    knn = KNeighborsClassifier()
    sfs = SFS(
        estimator=knn,
        k_features=3,
        forward=False,
        floating=True,
        fixed_features=(0, 1),
        verbose=0,
    )
    sfs.fit(X_iris, y_iris)
    for k in sfs.subsets_:
        assert 0 in sfs.subsets_[k]["feature_idx"]
        assert 1 in sfs.subsets_[k]["feature_idx"]


def test_run_forward_with_range():
    knn = KNeighborsClassifier()
    sfs = SFS(
        estimator=knn,
        k_features=(3, 4),
        forward=True,
        floating=False,
        fixed_features=(0, 1),
        verbose=0,
    )
    sfs.fit(X_iris, y_iris)
    for k in sfs.subsets_:
        assert 0 in sfs.subsets_[k]["feature_idx"]
        assert 1 in sfs.subsets_[k]["feature_idx"]


def test_run_floating_forward_with_range():
    knn = KNeighborsClassifier()
    sfs = SFS(
        estimator=knn,
        k_features=(3, 4),
        forward=True,
        floating=True,
        fixed_features=(0, 1),
        verbose=0,
    )
    sfs.fit(X_iris, y_iris)
    for k in sfs.subsets_:
        assert 0 in sfs.subsets_[k]["feature_idx"]
        assert 1 in sfs.subsets_[k]["feature_idx"]


def test_run_backward_with_range():
    knn = KNeighborsClassifier()
    sfs = SFS(
        estimator=knn,
        k_features=(3, 4),
        forward=False,
        floating=False,
        fixed_features=(0, 1),
        verbose=0,
    )
    sfs.fit(X_iris, y_iris)
    for k in sfs.subsets_:
        assert 0 in sfs.subsets_[k]["feature_idx"]
        assert 1 in sfs.subsets_[k]["feature_idx"]


def test_run_floating_backward_with_range():
    knn = KNeighborsClassifier()
    sfs = SFS(
        estimator=knn,
        k_features=(3, 4),
        forward=False,
        floating=True,
        fixed_features=(0, 1),
        verbose=0,
    )
    sfs.fit(X_iris, y_iris)
    for k in sfs.subsets_:
        assert 0 in sfs.subsets_[k]["feature_idx"]
        assert 1 in sfs.subsets_[k]["feature_idx"]


def test_pandas():
    X_df = pd.DataFrame(
        X_iris, columns=["sepal length", "sepal width", "petal width", "petal width"]
    )
    knn = KNeighborsClassifier()
    sfs = SFS(
        estimator=knn,
        k_features=3,
        forward=True,
        floating=False,
        fixed_features=("sepal length", "sepal width"),
        verbose=0,
    )
    sfs.fit(X_df, y_iris)
    print(sfs.subsets_)
    for k in sfs.subsets_:
        assert 0 in sfs.subsets_[k]["feature_idx"]
        assert 1 in sfs.subsets_[k]["feature_idx"]


def test_wrong_feature_number():
    knn = KNeighborsClassifier()
    expect = "k_features must be between len(fixed_features) and X.shape[1]."
    sfs = SFS(knn, k_features=1, fixed_features=(1, 3))
    assert_raises(AttributeError, expect, sfs.fit, X=X_iris, y=y_iris)


def test_wrong_feature_range():
    knn = KNeighborsClassifier()
    expect = (
        "k_features tuple min value must be between len(fixed_features) and X.shape[1]."
    )
    sfs = SFS(knn, k_features=(1, 3), fixed_features=(1, 3))
    assert_raises(AttributeError, expect, sfs.fit, X=X_iris, y=y_iris)
