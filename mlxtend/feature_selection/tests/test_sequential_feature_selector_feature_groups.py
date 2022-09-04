# Sebastian Raschka 2014-2022
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause
import numpy as np
import pandas as pd
from numpy import nan
from numpy.testing import assert_almost_equal
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_boston, load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from mlxtend.classifier import SoftmaxRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.utils import assert_raises


def nan_roc_auc_score(y_true, y_score, average="macro", sample_weight=None):
    if len(np.unique(y_true)) != 2:
        return np.nan
    else:
        return roc_auc_score(
            y_true, y_score, average=average, sample_weight=sample_weight
        )


def dict_compare_utility(d_actual, d_desired, decimal=3):
    assert d_actual.keys() == d_desired.keys(), "%s != %s" % (d_actual, d_desired)
    for i in d_actual:
        err_msg = "d_actual[%s]['feature_idx']" " != d_desired[%s]['feature_idx']" % (
            i,
            i,
        )
        assert d_actual[i]["feature_idx"] == d_desired[i]["feature_idx"], err_msg
        assert_almost_equal(
            actual=d_actual[i]["avg_score"],
            desired=d_desired[i]["avg_score"],
            decimal=decimal,
            err_msg=(
                "d_actual[%s]['avg_score']" " != d_desired[%s]['avg_score']" % (i, i)
            ),
        )
        assert_almost_equal(
            actual=d_actual[i]["cv_scores"],
            desired=d_desired[i]["cv_scores"],
            decimal=decimal,
            err_msg=(
                "d_actual[%s]['cv_scores']" " != d_desired[%s]['cv_scores']" % (i, i)
            ),
        )


def test_run_default():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    sfs = SFS(estimator=knn, verbose=0)
    sfs.fit(X, y)
    assert sfs.k_feature_idx_ == (3,)


def test_fit_params():
    iris = load_iris()
    X = iris.data
    y = iris.target
    sample_weight = np.ones(X.shape[0])
    forest = RandomForestClassifier(n_estimators=100, random_state=123)
    sfs = SFS(estimator=forest, verbose=0)
    sfs.fit(X, y, sample_weight=sample_weight)
    assert sfs.k_feature_idx_ == (3,)


def test_knn_wo_cv_feature_groups_default():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs1 = SFS(
        knn,
        k_features=3,
        forward=True,
        floating=False,
        cv=0,
        verbose=0,
        feature_groups=[[0], [1], [2], [3]],
    )
    sfs1 = sfs1.fit(X, y)
    expect = {
        1: {
            "avg_score": 0.95999999999999996,
            "cv_scores": np.array([0.96]),
            "feature_idx": (3,),
        },
        2: {
            "avg_score": 0.97333333333333338,
            "cv_scores": np.array([0.97333333]),
            "feature_idx": (2, 3),
        },
        3: {
            "avg_score": 0.97333333333333338,
            "cv_scores": np.array([0.97333333]),
            "feature_idx": (1, 2, 3),
        },
    }
    dict_compare_utility(d_actual=sfs1.subsets_, d_desired=expect, decimal=2)


def test_regression_sbfs():
    boston = load_boston()
    X, y = boston.data, boston.target
    lr = LinearRegression()
    sfs_r = SFS(
        lr,
        k_features=1,  # this is number of groups if `feature_groups` is not None
        forward=False,
        floating=True,
        scoring="neg_mean_squared_error",
        cv=10,
        verbose=0,
        feature_groups=[[7, 10, 12], [0], [1], [2], [3], [4], [5], [6], [8], [9], [11]],
    )
    sfs_r = sfs_r.fit(X, y)
    assert sfs_r.k_feature_idx_ == (7, 10, 12), sfs_r.k_feature_idx_


def test_transform_not_fitted():
    iris = load_iris()
    X = iris.data
    knn = KNeighborsClassifier(n_neighbors=4)

    sfs1 = SFS(
        knn,
        k_features=2,
        forward=True,
        floating=False,
        cv=0,
        clone_estimator=False,
        verbose=0,
        n_jobs=1,
    )

    expect = "SequentialFeatureSelector has not been fitted, yet."

    assert_raises(AttributeError, expect, sfs1.transform, X)


def test_keyboard_interrupt():
    iris = load_iris()
    X = iris.data
    y = iris.target

    knn = KNeighborsClassifier(n_neighbors=4)
    sfs1 = SFS(
        knn,
        k_features=3,
        forward=True,
        floating=False,
        cv=3,
        clone_estimator=False,
        verbose=5,
        n_jobs=1,
        feature_groups=[[0, 1], [2, 3]],
    )

    sfs1._TESTING_INTERRUPT_MODE = True
    out = sfs1.fit(X, y)

    assert len(out.subsets_.keys()) > 0
    assert sfs1.interrupted_


def test_max_feature_subset_best():
    boston = load_boston()
    X, y = boston.data, boston.target
    lr = LinearRegression()

    sfs = SFS(
        lr,
        k_features="best",
        forward=True,
        floating=False,
        cv=10,
        feature_groups=[
            [0],
            [2, 4],
            [1, 3, 5],
            [6],
            [7, 8, 9, 10],
            [11],
            [12],
        ],
    )

    sfs = sfs.fit(X, y)
    assert sfs.k_feature_idx_ == (1, 3, 5, 7, 8, 9, 10, 11, 12)


def test_max_feature_subset_parsimonious():
    boston = load_boston()
    X, y = boston.data, boston.target
    lr = LinearRegression()

    sfs = SFS(
        lr,
        k_features="parsimonious",
        forward=True,
        floating=False,
        cv=10,
        feature_groups=[
            [0],
            [1, 3],
            [2, 4],
            [5, 10, 11, 12],
            [6],
            [7],
            [8, 9],
        ],
    )

    sfs = sfs.fit(X, y)
    assert sfs.k_feature_idx_ == (5, 10, 11, 12)