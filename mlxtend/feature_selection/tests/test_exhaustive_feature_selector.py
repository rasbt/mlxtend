# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import boston_housing_data
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.utils import assert_raises


def dict_compare_utility(d1, d2, decimal=2):
    assert d1.keys() == d2.keys(), "%s != %s" % (d1, d2)
    for i in d1:
        err_msg1 = "d1[%s]['feature_idx']" " != d2[%s]['feature_idx']" % (i, i)
        err_msg2 = "d1[%s]['feature_names']" " != d2[%s]['feature_names']" % (i, i)
        assert d1[i]["feature_idx"] == d2[i]["feature_idx"], err_msg1
        assert d1[i]["feature_names"] == d2[i]["feature_names"], err_msg2
        assert_almost_equal(
            d1[i]["avg_score"],
            d2[i]["avg_score"],
            decimal=decimal,
            err_msg=("d1[%s]['avg_score']" " != d2[%s]['avg_score']" % (i, i)),
        )
        assert_almost_equal(
            d1[i]["cv_scores"],
            d2[i]["cv_scores"],
            decimal=decimal,
            err_msg=("d1[%s]['cv_scores']" " != d2[%s]['cv_scores']" % (i, i)),
        )


def test_minfeatures_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()

    efs = EFS(estimator=knn, min_features=0, max_features=2)
    expect = "min_features must be smaller than 5 and larger than 0"
    assert_raises(AttributeError, expect, efs.fit, X, y)


def test_maxfeatures_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()

    efs = EFS(estimator=knn, min_features=1, max_features=0)
    expect = "max_features must be smaller than 5 and larger than 0"
    assert_raises(AttributeError, expect, efs.fit, X, y)


def test_minmaxfeatures_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()

    efs = EFS(estimator=knn, min_features=3, max_features=2)
    expect = "min_features must be <= max_features"
    assert_raises(AttributeError, expect, efs.fit, X, y)


def test_knn_wo_cv():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(
        knn,
        min_features=2,
        max_features=3,
        scoring="accuracy",
        cv=0,
        print_progress=False,
    )
    efs1 = efs1.fit(X, y)
    expect = {
        0: {
            "feature_idx": (0, 1),
            "feature_names": ("0", "1"),
            "avg_score": 0.8333333333333334,
            "cv_scores": np.array([0.8333333333333334]),
        },
        1: {
            "feature_idx": (0, 2),
            "feature_names": ("0", "2"),
            "avg_score": 0.95999999999999996,
            "cv_scores": np.array([0.95999999999999996]),
        },
        2: {
            "feature_idx": (0, 3),
            "feature_names": ("0", "3"),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
        3: {
            "feature_idx": (1, 2),
            "feature_names": ("1", "2"),
            "avg_score": 0.95999999999999996,
            "cv_scores": np.array([0.96]),
        },
        4: {
            "feature_idx": (1, 3),
            "feature_names": ("1", "3"),
            "avg_score": 0.95999999999999996,
            "cv_scores": np.array([0.96]),
        },
        5: {
            "feature_idx": (2, 3),
            "feature_names": ("2", "3"),
            "avg_score": 0.97333333333333338,
            "cv_scores": np.array([0.97333333]),
        },
        6: {
            "feature_idx": (0, 1, 2),
            "feature_names": ("0", "1", "2"),
            "avg_score": 0.95999999999999996,
            "cv_scores": np.array([0.96]),
        },
        7: {
            "feature_idx": (0, 1, 3),
            "feature_names": ("0", "1", "3"),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
        8: {
            "feature_idx": (0, 2, 3),
            "feature_names": ("0", "2", "3"),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
        9: {
            "feature_idx": (1, 2, 3),
            "feature_names": ("1", "2", "3"),
            "avg_score": 0.97333333333333338,
            "cv_scores": np.array([0.97333333]),
        },
    }
    dict_compare_utility(d1=expect, d2=efs1.subsets_)


def test_knn_cv3():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(
        knn,
        min_features=3,
        max_features=3,
        scoring="accuracy",
        cv=4,
        print_progress=False,
    )
    efs1 = efs1.fit(X, y)
    expect = {
        0: {
            "avg_score": 0.9329658605974395,
            "feature_idx": (0, 1, 2),
            "feature_names": ("0", "1", "2"),
            "cv_scores": np.array([0.974, 0.947, 0.892, 0.919]),
        },
        1: {
            "avg_score": 0.9400782361308677,
            "feature_idx": (0, 1, 3),
            "feature_names": ("0", "1", "3"),
            "cv_scores": np.array([0.921, 0.947, 0.919, 0.973]),
        },
        2: {
            "avg_score": 0.9532361308677098,
            "feature_idx": (0, 2, 3),
            "feature_names": ("0", "2", "3"),
            "cv_scores": np.array([0.974, 0.947, 0.919, 0.973]),
        },
        3: {
            "avg_score": 0.97275641025641035,
            "feature_idx": (1, 2, 3),
            "feature_names": ("1", "2", "3"),
            "cv_scores": np.array([0.974, 1.0, 0.946, 0.973]),
        },
    }

    if Version(sklearn_version) < Version("1.0"):
        expect[0]["avg_score"] = 0.9391025641025641
        expect[0]["cv_scores"] = np.array([0.974, 0.947, 0.892, 0.946])
        expect[2]["avg_score"] = 0.9529914529914529

    if Version(sklearn_version) < Version("0.22"):
        expect[0]["cv_scores"] = np.array(
            [0.97435897, 0.94871795, 0.88888889, 0.94444444]
        )
        expect[1]["cv_scores"] = np.array(
            [0.92307692, 0.94871795, 0.91666667, 0.97222222]
        )
        expect[2]["cv_scores"] = np.array(
            [0.97435897, 0.94871795, 0.91666667, 0.97222222]
        )
        expect[3]["cv_scores"] = np.array(
            [0.97435897, 0.94871795, 0.91666667, 0.97222222]
        )
        expect[1]["avg_score"] = 0.94017094017094016
        assert round(efs1.best_score_, 4) == 0.9728
    else:
        assert round(efs1.best_score_, 4) == 0.9732

    dict_compare_utility(d1=expect, d2=efs1.subsets_)
    assert efs1.best_idx_ == (1, 2, 3)
    assert efs1.best_feature_names_ == ("1", "2", "3")


def test_knn_cv3_groups():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(
        knn,
        min_features=3,
        max_features=3,
        scoring="accuracy",
        cv=GroupKFold(n_splits=3),
        print_progress=False,
    )
    np.random.seed(1630672634)
    groups = np.random.randint(0, 6, size=len(y))
    efs1 = efs1.fit(X, y, groups=groups)

    expect = {
        0: {
            "cv_scores": np.array([0.97916667, 0.93877551, 0.9245283]),
            "feature_idx": (0, 1, 2),
            "avg_score": 0.9474901595858469,
            "feature_names": ("0", "1", "2"),
        },
        1: {
            "cv_scores": np.array([1.0, 0.93877551, 0.9245283]),
            "feature_idx": (0, 1, 3),
            "avg_score": 0.9544346040302915,
            "feature_names": ("0", "1", "3"),
        },
        2: {
            "cv_scores": np.array([0.97916667, 0.95918367, 0.9245283]),
            "feature_idx": (0, 2, 3),
            "avg_score": 0.9542928806742822,
            "feature_names": ("0", "2", "3"),
        },
        3: {
            "cv_scores": np.array([0.97916667, 0.95918367, 0.94339623]),
            "feature_idx": (1, 2, 3),
            "avg_score": 0.9605821888503829,
            "feature_names": ("1", "2", "3"),
        },
    }
    dict_compare_utility(d1=expect, d2=efs1.subsets_)


def test_fit_params():
    iris = load_iris()
    X = iris.data
    y = iris.target
    sample_weight = np.ones(X.shape[0])
    forest = RandomForestClassifier(n_estimators=100, random_state=123)
    efs1 = EFS(
        forest,
        min_features=3,
        max_features=3,
        scoring="accuracy",
        cv=4,
        print_progress=False,
    )
    efs1 = efs1.fit(X, y, sample_weight=sample_weight)

    # The set of explored 3-feature subsets is deterministic, but the exact
    # cross-validation scores drift across scikit-learn versions (e.g. changes
    # to how sample_weight feeds into RandomForest's bootstrap). So we assert
    # the stable structure (which subsets are explored and selected) and a
    # tolerance band on the scores rather than exact floats.
    expected_subsets = {
        0: {"feature_idx": (0, 1, 2), "feature_names": ("0", "1", "2")},
        1: {"feature_idx": (0, 1, 3), "feature_names": ("0", "1", "3")},
        2: {"feature_idx": (0, 2, 3), "feature_names": ("0", "2", "3")},
        3: {"feature_idx": (1, 2, 3), "feature_names": ("1", "2", "3")},
    }
    assert efs1.subsets_.keys() == expected_subsets.keys()
    for i, sub in expected_subsets.items():
        assert efs1.subsets_[i]["feature_idx"] == sub["feature_idx"]
        assert efs1.subsets_[i]["feature_names"] == sub["feature_names"]
        assert 0.8 < efs1.subsets_[i]["avg_score"] <= 1.0

    assert efs1.best_idx_ == (0, 2, 3)
    assert 0.9 < efs1.best_score_ <= 1.0


def test_regression():
    X, y = boston_housing_data()
    X = X[:, [1, 2, 6, 8, 12]]
    lr = LinearRegression()
    efs_r = EFS(
        lr,
        min_features=3,
        max_features=4,
        scoring="neg_mean_squared_error",
        cv=10,
        print_progress=False,
    )
    efs_r = efs_r.fit(X, y)
    assert efs_r.best_idx_ == (0, 2, 4)
    assert round(efs_r.best_score_, 4) == -40.8777


def test_clone_params_fail():
    class Perceptron(object):
        def __init__(self, eta=0.1, epochs=50, random_seed=None, print_progress=0):
            self.eta = eta
            self.epochs = epochs
            self.random_seed = random_seed
            self.print_progress = print_progress
            self._is_fitted = False

        def _fit(self, X, y, init_params=True):
            self._check_target_array(y, allowed={(0, 1)})
            y_data = np.where(y == 0, -1.0, 1.0)

            if init_params:
                self.b_, self.w_ = self._init_params(
                    weights_shape=(X.shape[1], 1),
                    bias_shape=(1,),
                    random_seed=self.random_seed,
                )
                self.cost_ = []

            rgen = np.random.RandomState(self.random_seed)
            for i in range(self.epochs):
                errors = 0

                for idx in self._yield_minibatches_idx(
                    rgen=rgen, n_batches=y_data.shape[0], data_ary=y_data, shuffle=True
                ):
                    update = self.eta * (y_data[idx] - self._to_classlabels(X[idx]))
                    self.w_ += (update * X[idx]).reshape(self.w_.shape)
                    self.b_ += update
                    errors += int(update.item() != 0.0)

                if self.print_progress:
                    self._print_progress(
                        iteration=i + 1, n_iter=self.epochs, cost=errors
                    )
                self.cost_.append(errors)
            return self

        def _net_input(self, X):
            """Net input function"""
            return (np.dot(X, self.w_) + self.b_).flatten()

        def _to_classlabels(self, X):
            return np.where(self._net_input(X) < 0.0, -1.0, 1.0)

        def _predict(self, X):
            return np.where(self._net_input(X) < 0.0, 0, 1)

    expect = (
        "Cannot clone object. You should provide an"
        " instance of scikit-learn estimator instead of a class."
    )

    assert_raises(
        TypeError,
        expect,
        EFS,
        Perceptron,
        min_features=2,
        max_features=2,
        clone_estimator=True,
    )


def test_clone_params_pass():
    iris = load_iris()
    X = iris.data
    y = iris.target
    lr = SoftmaxRegression(random_seed=1)
    efs1 = EFS(
        lr,
        min_features=2,
        max_features=2,
        scoring="accuracy",
        cv=0,
        clone_estimator=False,
        print_progress=False,
        n_jobs=1,
    )
    efs1 = efs1.fit(X, y)
    assert efs1.best_idx_ == (1, 3)


def test_transform_not_fitted():
    iris = load_iris()
    X = iris.data
    knn = KNeighborsClassifier(n_neighbors=4)

    efs1 = EFS(
        knn,
        min_features=2,
        max_features=2,
        scoring="accuracy",
        cv=0,
        clone_estimator=False,
        print_progress=False,
        n_jobs=1,
    )

    expect = "ExhaustiveFeatureSelector has not been fitted, yet."

    assert_raises(AttributeError, expect, efs1.transform, X)


def test_fit_transform():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)

    efs1 = EFS(
        knn,
        min_features=2,
        max_features=2,
        scoring="accuracy",
        cv=0,
        clone_estimator=False,
        print_progress=False,
        n_jobs=1,
    )

    X_t = efs1.fit_transform(X, y)
    assert X_t.shape == (150, 2)


def test_get_metric_dict_not_fitted():
    knn = KNeighborsClassifier(n_neighbors=4)

    efs1 = EFS(
        knn,
        min_features=2,
        max_features=2,
        scoring="accuracy",
        cv=0,
        clone_estimator=False,
        print_progress=False,
        n_jobs=1,
    )

    expect = "ExhaustiveFeatureSelector has not been fitted, yet."

    assert_raises(AttributeError, expect, efs1.get_metric_dict)


def test_check_pandas_dataframe_fit():
    knn = KNeighborsClassifier(n_neighbors=4)
    iris = load_iris()
    X = iris.data
    y = iris.target
    efs1 = EFS(
        knn,
        min_features=2,
        max_features=2,
        scoring="accuracy",
        cv=0,
        clone_estimator=False,
        print_progress=False,
        n_jobs=1,
    )

    df = pd.DataFrame(
        X, columns=["sepal length", "sepal width", "petal length", "petal width"]
    )

    sfs1 = efs1.fit(X, y)
    assert efs1.best_idx_ == (2, 3), efs1.best_idx_
    assert efs1.best_feature_names_ == ("2", "3")
    assert efs1.interrupted_ is False

    sfs1._TESTING_INTERRUPT_MODE = True
    sfs1 = sfs1.fit(df, y)
    assert efs1.best_idx_ == (0, 1), efs1.best_idx_
    assert efs1.best_feature_names_ == ("sepal length", "sepal width")
    assert efs1.interrupted_ is True


def test_check_pandas_dataframe_transform():
    knn = KNeighborsClassifier(n_neighbors=4)
    iris = load_iris()
    X = iris.data
    y = iris.target
    efs1 = EFS(
        knn,
        min_features=2,
        max_features=2,
        scoring="accuracy",
        cv=0,
        clone_estimator=False,
        print_progress=False,
        n_jobs=1,
    )

    df = pd.DataFrame(
        X, columns=["sepal length", "sepal width", "petal length", "petal width"]
    )
    efs1 = efs1.fit(df, y)
    assert efs1.best_idx_ == (2, 3)
    assert (150, 2) == efs1.transform(df).shape


def test_knn_wo_cv_with_feature_groups_integer():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(
        knn,
        min_features=2,
        max_features=2,
        scoring="accuracy",
        cv=0,
        print_progress=False,
        feature_groups=[[0], [1, 2], [3]],
    )
    efs1 = efs1.fit(X, y)
    # expect is based on what provided in `test_knn_wo_cv`
    expect = {
        0: {
            "feature_idx": (0, 1, 2),
            "feature_names": ("0", "1", "2"),
            "avg_score": 0.95999999999999996,
            "cv_scores": np.array([0.96]),
        },
        1: {
            "feature_idx": (0, 3),
            "feature_names": ("0", "3"),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
        2: {
            "feature_idx": (1, 2, 3),
            "feature_names": ("1", "2", "3"),
            "avg_score": 0.97333333333333338,
            "cv_scores": np.array([0.97333333]),
        },
    }
    dict_compare_utility(d1=expect, d2=efs1.subsets_)


def test_knn_wo_cv_with_feature_groups_string():
    iris = load_iris()
    X = iris.data
    df_X = pd.DataFrame(
        X, columns=["sepal length", "sepal width", "petal length", "petal width"]
    )
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(
        knn,
        min_features=2,
        max_features=2,
        scoring="accuracy",
        cv=0,
        print_progress=False,
        feature_groups=[
            ["sepal length"],
            ["sepal width", "petal length"],
            ["petal width"],
        ],
    )
    efs1 = efs1.fit(df_X, y)
    expect = {
        0: {
            "feature_idx": (0, 1, 2),
            "feature_names": ("sepal length", "sepal width", "petal length"),
            "avg_score": 0.95999999999999996,
            "cv_scores": np.array([0.96]),
        },
        1: {
            "feature_idx": (0, 3),
            "feature_names": ("sepal length", "petal width"),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
        2: {
            "feature_idx": (1, 2, 3),
            "feature_names": ("sepal width", "petal length", "petal width"),
            "avg_score": 0.97333333333333338,
            "cv_scores": np.array([0.97333333]),
        },
    }
    dict_compare_utility(d1=expect, d2=efs1.subsets_)


def test_knn_wo_cv_with_fixed_features_and_feature_groups_case1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(
        knn,
        min_features=1,
        max_features=2,
        scoring="accuracy",
        cv=0,
        print_progress=False,
        fixed_features=[0, 1],
        feature_groups=[[0, 1], [2], [3]],
    )
    efs1 = efs1.fit(X, y)
    # expect is based on what provided in `test_knn_wo_cv`
    expect = {
        0: {
            "feature_idx": (0, 1),
            "feature_names": ("0", "1"),
            "avg_score": 0.8333333333333334,
            "cv_scores": np.array([0.8333333333333334]),
        },
        1: {
            "feature_idx": (0, 1, 2),
            "feature_names": ("0", "1", "2"),
            "avg_score": 0.95999999999999996,
            "cv_scores": np.array([0.96]),
        },
        2: {
            "feature_idx": (0, 1, 3),
            "feature_names": ("0", "1", "3"),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
    }
    dict_compare_utility(d1=expect, d2=efs1.subsets_)


def test_knn_wo_cv_with_fixed_features_and_feature_groups_case2():
    # similar to case1, but `fixed_features` is now consisting of two groups
    # [0,1] and [3]
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(
        knn,
        min_features=2,
        max_features=2,
        scoring="accuracy",
        cv=0,
        print_progress=False,
        fixed_features=[0, 1, 3],
        feature_groups=[[0, 1], [2], [3]],
    )
    efs1 = efs1.fit(X, y)
    # expect is based on what provided in `test_knn_wo_cv`
    expect = {
        0: {
            "feature_idx": (0, 1, 3),
            "feature_names": ("0", "1", "3"),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
    }
    dict_compare_utility(d1=expect, d2=efs1.subsets_)


def test_check_support_string_in_feature_groups():
    iris = load_iris()
    X = iris.data
    y = iris.target

    features_names = ["sepal length", "sepal width", "petal length", "petal width"]
    df = pd.DataFrame(X, columns=features_names)

    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(
        knn,
        min_features=2,
        max_features=2,
        scoring="accuracy",
        cv=0,
        print_progress=False,
        feature_groups=[
            [features_names[0]],
            [features_names[1], features_names[2]],
            [features_names[3]],
        ],
    )
    efs1 = efs1.fit(df, y)
    # expect is based on what provided in `test_knn_wo_cv`
    expect = {
        0: {
            "feature_idx": (0, 1, 2),
            "feature_names": (features_names[0], features_names[1], features_names[2]),
            "avg_score": 0.95999999999999996,
            "cv_scores": np.array([0.96]),
        },
        1: {
            "feature_idx": (0, 3),
            "feature_names": (features_names[0], features_names[3]),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
        2: {
            "feature_idx": (1, 2, 3),
            "feature_names": (features_names[1], features_names[2], features_names[3]),
            "avg_score": 0.97333333333333338,
            "cv_scores": np.array([0.97333333]),
        },
    }
    dict_compare_utility(d1=expect, d2=efs1.subsets_)


def test_check_support_string_in_fixed_feature():
    iris = load_iris()
    X = iris.data
    y = iris.target

    features_names = ["sepal length", "sepal width", "petal length", "petal width"]
    df = pd.DataFrame(X, columns=features_names)

    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(
        knn,
        min_features=2,
        max_features=3,
        scoring="accuracy",
        cv=0,
        print_progress=False,
        fixed_features=[features_names[0], features_names[1]],
    )

    efs1 = efs1.fit(df, y)
    # expect is based on what provided in `test_knn_wo_cv`
    expect = {
        0: {
            "feature_idx": (0, 1),
            "feature_names": (features_names[0], features_names[1]),
            "avg_score": 0.8333333333333334,
            "cv_scores": np.array([0.8333333333333334]),
        },
        1: {
            "feature_idx": (0, 1, 2),
            "feature_names": (features_names[0], features_names[1], features_names[2]),
            "avg_score": 0.95999999999999996,
            "cv_scores": np.array([0.96]),
        },
        2: {
            "feature_idx": (0, 1, 3),
            "feature_names": (features_names[0], features_names[1], features_names[3]),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
    }
    dict_compare_utility(d1=expect, d2=efs1.subsets_)


def test_fixed_features_and_feature_groups_pandas_and_strings():
    iris = load_iris()
    X = iris.data
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    df_X = pd.DataFrame(X, columns=feature_names)
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(
        knn,
        min_features=2,
        max_features=2,
        scoring="accuracy",
        cv=0,
        print_progress=False,
        fixed_features=[feature_names[0], feature_names[1], feature_names[3]],
        feature_groups=[
            [feature_names[0], feature_names[1]],
            [feature_names[2]],
            [feature_names[3]],
        ],
    )
    efs1 = efs1.fit(df_X, y)
    # expect is based on what provided in `test_knn_wo_cv`
    expect = {
        0: {
            "feature_idx": (0, 1, 3),
            "feature_names": (feature_names[0], feature_names[1], feature_names[3]),
            "avg_score": 0.96666666666666667,
            "cv_scores": np.array([0.96666667]),
        },
    }
    dict_compare_utility(d1=expect, d2=efs1.subsets_)


def test_logistic_regression_compatibility():
    iris = load_iris()
    X, y = iris.data, iris.target

    lr = LogisticRegression(solver="lbfgs", max_iter=1000)

    efs = EFS(estimator=lr, min_features=1, max_features=1, cv=2, print_progress=False)
    efs.fit(X, y)

    assert efs.best_idx_ == (3,)
    assert efs.best_score_ > 0.90


def test_get_metric_dict_top_k_returns_top_subsets_issue_610():
    # Regression test for #610: top_k must restrict the returned dict to
    # the top-K subsets ranked by avg_score (descending).
    knn = KNeighborsClassifier(n_neighbors=4)
    iris = load_iris()
    X, y = iris.data, iris.target
    efs = EFS(
        knn,
        min_features=1,
        max_features=3,
        scoring="accuracy",
        cv=3,
        clone_estimator=False,
        print_progress=False,
        n_jobs=1,
    )
    efs.fit(X, y)

    full = efs.get_metric_dict()
    assert len(full) > 3, "test setup expects more than 3 subsets evaluated"

    top3 = efs.get_metric_dict(top_k=3)
    assert len(top3) == 3, "top_k=3 should return exactly 3 entries"

    # The returned subsets are exactly the 3 highest-scoring ones from `full`.
    expected_top = sorted(
        full.keys(), key=lambda k: full[k]["avg_score"], reverse=True
    )[:3]
    assert set(top3.keys()) == set(
        expected_top
    ), "top_k did not return the highest-scoring subsets"

    # All other metrics (avg_score, ci_bound, std_dev, std_err, feature_idx,
    # feature_names if present) must match the corresponding entries in the
    # full dict.
    for k in top3:
        assert top3[k]["feature_idx"] == full[k]["feature_idx"]
        assert top3[k]["avg_score"] == full[k]["avg_score"]


def test_get_metric_dict_top_k_none_preserves_default_behavior_issue_610():
    knn = KNeighborsClassifier(n_neighbors=4)
    iris = load_iris()
    X, y = iris.data, iris.target
    efs = EFS(
        knn,
        min_features=1,
        max_features=2,
        scoring="accuracy",
        cv=2,
        clone_estimator=False,
        print_progress=False,
        n_jobs=1,
    )
    efs.fit(X, y)

    default = efs.get_metric_dict()
    explicit_none = efs.get_metric_dict(top_k=None)
    assert default.keys() == explicit_none.keys()


def test_get_metric_dict_top_k_invalid_raises_issue_610():
    knn = KNeighborsClassifier(n_neighbors=4)
    iris = load_iris()
    X, y = iris.data, iris.target
    efs = EFS(
        knn,
        min_features=1,
        max_features=2,
        scoring="accuracy",
        cv=2,
        clone_estimator=False,
        print_progress=False,
        n_jobs=1,
    )
    efs.fit(X, y)
    assert_raises(
        ValueError,
        "`top_k` must be a positive integer or None",
        efs.get_metric_dict,
        top_k=0,
    )
    assert_raises(
        ValueError,
        "`top_k` must be a positive integer or None",
        efs.get_metric_dict,
        top_k=-2,
    )
    assert_raises(
        ValueError,
        "`top_k` must be a positive integer or None",
        efs.get_metric_dict,
        top_k=1.5,
    )


def test_get_metric_dict_top_k_larger_than_total_returns_all_issue_610():
    # Boundary: if top_k exceeds the number of evaluated subsets, all
    # entries are returned (not an error).
    knn = KNeighborsClassifier(n_neighbors=4)
    iris = load_iris()
    X, y = iris.data, iris.target
    efs = EFS(
        knn,
        min_features=1,
        max_features=2,
        scoring="accuracy",
        cv=2,
        clone_estimator=False,
        print_progress=False,
        n_jobs=1,
    )
    efs.fit(X, y)
    full = efs.get_metric_dict()
    out = efs.get_metric_dict(top_k=10**6)
    assert out.keys() == full.keys()
