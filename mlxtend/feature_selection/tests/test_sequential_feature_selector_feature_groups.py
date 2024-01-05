# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.data import boston_housing_data
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.utils import assert_raises


def dict_compare_utility(d_actual, d_desired, decimal=2):
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
    X, y = boston_housing_data()
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
        k_features=2,
        forward=True,
        floating=False,
        cv=3,
        clone_estimator=False,
        verbose=5,
        n_jobs=1,
        feature_groups=[[0, 1], [2], [3]],
    )

    sfs1._TESTING_INTERRUPT_MODE = True
    out = sfs1.fit(X, y)

    assert len(out.subsets_.keys()) > 0
    assert sfs1.interrupted_


def test_max_feature_subset_best():
    X, y = boston_housing_data()
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
    X, y = boston_housing_data()
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


def test_knn_wo_cv_with_fixed_features_and_feature_groups_case1():
    # features (0, 1) gives different score?
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs = SFS(
        knn,
        k_features=(1, 2),
        scoring="accuracy",
        cv=0,
        fixed_features=[0, 1],
        feature_groups=[[0, 1], [2], [3]],
    )
    sfs.fit(X, y)
    # expect is based on what provided in `test_knn_wo_cv`
    expect = {
        1: {
            "feature_idx": (0, 1),
            "feature_names": ("0", "1"),
            "avg_score": 0.8333333333333334,
            "cv_scores": np.array([0.8333333333333334]),
        },
        2: {
            "feature_idx": (0, 1, 3),
            "feature_names": ("0", "1", "3"),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
    }
    dict_compare_utility(d_actual=expect, d_desired=sfs.subsets_)


def test_knn_wo_cv_with_fixed_features_and_feature_groups_case2():
    # similar to case1, but `fixed_features` is now consisting of two groups
    # [0,1] and [3]
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = SFS(
        knn,
        k_features=2,
        scoring="accuracy",
        cv=0,
        fixed_features=[0, 1, 3],
        feature_groups=[[0, 1], [2], [3]],
    )
    efs1 = efs1.fit(X, y)
    # expect is based on what provided in `test_knn_wo_cv`
    expect = {
        2: {
            "feature_idx": (0, 1, 3),
            "feature_names": ("0", "1", "3"),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
    }
    dict_compare_utility(d_actual=expect, d_desired=efs1.subsets_)
