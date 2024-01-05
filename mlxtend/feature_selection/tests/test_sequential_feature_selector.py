# Sebastian Raschka 2014-2024
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
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import boston_housing_data
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.utils import assert_raises


def nan_roc_auc_score(y_true, y_score, average="macro", sample_weight=None):
    if len(np.unique(y_true)) != 2:
        return np.nan
    else:
        return roc_auc_score(
            y_true, y_score, average=average, sample_weight=sample_weight
        )


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


def test_kfeatures_type_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    name = "k_features"
    expect = f"{name} must be between 1 and X.shape[1]."
    sfs = SFS(estimator=knn, verbose=0, k_features=0)
    assert_raises(AttributeError, expect, sfs.fit, X, y)


def test_kfeatures_type_2():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    expect = "k_features must be a positive integer, tuple, or string"
    sfs = SFS(estimator=knn, verbose=0, k_features=set())
    assert_raises(AttributeError, expect, sfs.fit, X, y)


def test_kfeatures_type_3():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    expect = "k_features tuple min value must be between 1 and X.shape[1]."
    sfs = SFS(estimator=knn, verbose=0, k_features=(0, 5))
    assert_raises(AttributeError, expect, sfs.fit, X, y)


def test_kfeatures_type_4():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    expect = "k_features tuple max value must be between 1 and X.shape[1]."
    sfs = SFS(estimator=knn, verbose=0, k_features=(1, 5))
    assert_raises(AttributeError, expect, sfs.fit, X, y)


def test_kfeatures_type_5():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    expect = (
        "The min k_features value must be smaller" " than the max k_features value."
    )
    sfs = SFS(estimator=knn, verbose=0, k_features=(3, 1))
    assert_raises(AttributeError, expect, sfs.fit, X, y)


def test_knn_wo_cv():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs1 = SFS(knn, k_features=3, forward=True, floating=False, cv=0, verbose=0)
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


def test_knn_cv3():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs1 = SFS(knn, k_features=3, forward=True, floating=False, cv=4, verbose=0)
    sfs1 = sfs1.fit(X, y)
    sfs1.subsets_
    expect = {
        1: {
            "avg_score": 0.9599928876244666,
            "cv_scores": np.array([0.974, 0.947, 0.919, 1.0]),
            "feature_idx": (3,),
        },
        2: {
            "avg_score": 0.95993589743589736,
            "cv_scores": np.array([0.974, 0.947, 0.919, 1.0]),
            "feature_idx": (2, 3),
        },
        3: {
            "avg_score": 0.9732,
            "cv_scores": np.array([0.974, 1.0, 0.946, 0.973]),
            "feature_idx": (1, 2, 3),
        },
    }

    if Version(sklearn_version) < Version("1.0"):
        expect[1]["avg_score"] = 0.95299145299145294
        expect[1]["cv_scores"] = (np.array([0.974, 0.947, 0.892, 1.0]),)

    if Version(sklearn_version) < Version("0.22"):
        expect[1]["cv_scores"] = np.array([0.97435897, 0.94871795, 0.88888889, 1.0])
        expect[2]["cv_scores"] = np.array([0.97435897, 0.94871795, 0.91666667, 1.0])
        expect[2]["avg_score"] = 0.97275641025641035
        expect[3]["cv_scores"] = np.array([0.97435897, 1.0, 0.94444444, 0.97222222])

    dict_compare_utility(d_actual=sfs1.subsets_, d_desired=expect, decimal=2)


def test_knn_cv3_groups():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs1 = SFS(
        knn,
        k_features=3,
        forward=True,
        floating=False,
        cv=GroupKFold(n_splits=3),
        verbose=0,
    )
    np.random.seed(1630672634)
    groups = np.random.randint(0, 6, size=len(y))
    sfs1 = sfs1.fit(X, y, groups=groups)
    # print(sfs1.subsets_)
    expect = {
        1: {
            "cv_scores": np.array([0.97916667, 0.93877551, 0.96226415]),
            "feature_idx": (3,),
            "avg_score": 0.9600687759380482,
        },
        2: {
            "cv_scores": np.array([0.95833333, 0.93877551, 0.98113208]),
            "feature_idx": (1, 3),
            "avg_score": 0.9594136396697044,
        },
        3: {
            "cv_scores": np.array([0.97916667, 0.95918367, 0.94339623]),
            "feature_idx": (1, 2, 3),
            "avg_score": 0.9605821888503829,
        },
    }
    dict_compare_utility(d_actual=sfs1.subsets_, d_desired=expect, decimal=3)


def test_knn_rbf_groupkfold():
    nan_roc_auc_scorer = make_scorer(nan_roc_auc_score)
    rng = np.random.RandomState(123)
    iris = load_iris()
    X = iris.data
    # knn = KNeighborsClassifier(n_neighbors=4)
    forest = RandomForestClassifier(n_estimators=100, random_state=123)
    bool_01 = [True if item == 0 else False for item in iris["target"]]
    bool_02 = [True if (item == 1 or item == 2) else False for item in iris["target"]]
    groups = []
    y_new = []
    for ind, _ in enumerate(bool_01):
        if bool_01[ind]:
            groups.append("attribute_A")
            y_new.append(0)
        if bool_02[ind]:
            throw = rng.rand()
            if throw < 0.5:
                groups.append("attribute_B")
            else:
                groups.append("attribute_C")
            throw2 = rng.rand()
            if throw2 < 0.5:
                y_new.append(0)
            else:
                y_new.append(1)
    y_new_bool = [True if item == 1 else False for item in y_new]
    cv_obj = GroupKFold(n_splits=3)
    cv_obj_list = list(cv_obj.split(X, np.array(y_new_bool), groups))
    sfs1 = SFS(
        forest,
        k_features=3,
        forward=True,
        floating=False,
        cv=cv_obj_list,
        scoring=nan_roc_auc_scorer,
        verbose=0,
    )
    sfs1 = sfs1.fit(X, y_new)
    expect = {
        1: {
            "cv_scores": np.array([0.52, nan, 0.72]),
            "avg_score": 0.62,
            "feature_idx": (1,),
        },
        2: {
            "cv_scores": np.array([0.42, nan, 0.65]),
            "avg_score": 0.53,
            "feature_idx": (1, 2),
        },
        3: {
            "cv_scores": np.array([0.47, nan, 0.63]),
            "avg_score": 0.55,
            "feature_idx": (1, 2, 3),
        },
    }

    dict_compare_utility(d_actual=sfs1.subsets_, d_desired=expect, decimal=1)


def test_knn_option_sfs():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs1 = SFS(knn, k_features=3, forward=True, floating=False, cv=4, verbose=0)
    sfs1 = sfs1.fit(X, y)
    assert sfs1.k_feature_idx_ == (1, 2, 3)


def test_knn_option_sffs():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs2 = SFS(knn, k_features=3, forward=True, floating=True, cv=4, verbose=0)
    sfs2 = sfs2.fit(X, y)
    assert sfs2.k_feature_idx_ == (1, 2, 3)


def test_knn_option_sbs():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs3 = SFS(knn, k_features=3, forward=False, floating=False, cv=4, verbose=0)
    sfs3 = sfs3.fit(X, y)
    assert sfs3.k_feature_idx_ == (1, 2, 3)


def test_knn_option_sfbs():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs4 = SFS(knn, k_features=3, forward=False, floating=True, cv=4, verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert sfs4.k_feature_idx_ == (1, 2, 3)


def test_knn_option_sfbs_tuplerange_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=3)
    sfs4 = SFS(knn, k_features=(1, 3), forward=False, floating=True, cv=4, verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert round(sfs4.k_score_, 3) == 0.967, sfs4.k_score_
    assert sfs4.k_feature_idx_ == (0, 2, 3), sfs4.k_feature_idx_


def test_knn_option_sfbs_tuplerange_2():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=3)
    sfs4 = SFS(knn, k_features=(1, 4), forward=False, floating=True, cv=4, verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert round(sfs4.k_score_, 3) == 0.967, sfs4.k_score_
    assert sfs4.k_feature_idx_ == (0, 2, 3), sfs4.k_feature_idx_


def test_knn_option_sffs_tuplerange_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=3)
    sfs4 = SFS(knn, k_features=(1, 3), forward=True, floating=True, cv=4, verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert round(sfs4.k_score_, 3) == 0.967, sfs4.k_score_
    assert sfs4.k_feature_idx_ == (0, 2, 3), sfs4.k_feature_idx_


def test_knn_option_sfs_tuplerange_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=3)
    sfs4 = SFS(knn, k_features=(1, 3), forward=True, floating=False, cv=4, verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert round(sfs4.k_score_, 3) == 0.967, sfs4.k_score_
    assert sfs4.k_feature_idx_ == (0, 2, 3), sfs4.k_feature_idx_


def test_knn_option_sbs_tuplerange_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=3)
    sfs4 = SFS(knn, k_features=(1, 3), forward=False, floating=False, cv=4, verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert round(sfs4.k_score_, 3) == 0.967, sfs4.k_score_
    assert sfs4.k_feature_idx_ == (0, 2, 3), sfs4.k_feature_idx_


def test_knn_scoring_metric():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs5 = SFS(knn, k_features=3, forward=False, floating=True, cv=4, verbose=0)
    sfs5 = sfs5.fit(X, y)

    if Version(sklearn_version) < Version("0.22"):
        assert round(sfs5.k_score_, 4) == 0.9728
    else:
        assert round(sfs5.k_score_, 4) == 0.9732

    sfs6 = SFS(knn, k_features=3, forward=False, floating=True, cv=4, verbose=0)
    sfs6 = sfs6.fit(X, y)
    if Version(sklearn_version) < Version("0.22"):
        assert round(sfs5.k_score_, 4) == 0.9728
    else:
        assert round(sfs5.k_score_, 4) == 0.9732

    sfs7 = SFS(
        knn, k_features=3, forward=False, floating=True, scoring="f1_macro", cv=4
    )
    sfs7 = sfs7.fit(X, y)
    if Version(sklearn_version) < Version("0.22"):
        assert round(sfs5.k_score_, 4) == 0.9727
    else:
        assert round(sfs5.k_score_, 4) == 0.9732


def test_regression():
    X, y = boston_housing_data()
    lr = LinearRegression()
    sfs_r = SFS(
        lr,
        k_features=13,
        forward=True,
        floating=False,
        scoring="neg_mean_squared_error",
        cv=10,
        verbose=0,
    )
    sfs_r = sfs_r.fit(X, y)
    assert len(sfs_r.k_feature_idx_) == 13

    if Version(sklearn_version) < Version("0.20"):
        assert round(sfs_r.k_score_, 4) == -34.7631, round(sfs_r.k_score_, 4)
    else:
        assert round(sfs_r.k_score_, 4) == -34.7053, round(sfs_r.k_score_, 4)


def test_regression_sffs():
    X, y = boston_housing_data()
    lr = LinearRegression()
    sfs_r = SFS(
        lr,
        k_features=11,
        forward=True,
        floating=True,
        scoring="neg_mean_squared_error",
        cv=10,
        verbose=0,
    )
    sfs_r = sfs_r.fit(X, y)
    assert sfs_r.k_feature_idx_ == (0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12)


def test_regression_sbfs():
    X, y = boston_housing_data()
    lr = LinearRegression()
    sfs_r = SFS(
        lr,
        k_features=3,
        forward=False,
        floating=True,
        scoring="neg_mean_squared_error",
        cv=10,
        verbose=0,
    )
    sfs_r = sfs_r.fit(X, y)
    assert sfs_r.k_feature_idx_ == (7, 10, 12), sfs_r.k_feature_idx_


def test_regression_in_range():
    X, y = boston_housing_data()
    lr = LinearRegression()
    sfs_r = SFS(
        lr,
        k_features=(1, 13),
        forward=True,
        floating=False,
        scoring="neg_mean_squared_error",
        cv=10,
        verbose=0,
    )
    sfs_r = sfs_r.fit(X, y)
    assert len(sfs_r.k_feature_idx_) == 9

    if Version(sklearn_version) < Version("0.20"):
        assert round(sfs_r.k_score_, 4) == -31.1537, round(sfs_r.k_score_, 4)
    else:
        assert round(sfs_r.k_score_, 4) == -31.1299, round(sfs_r.k_score_, 4)


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
                    errors += int(update != 0.0)

                if self.print_progress:
                    self.print_progress(
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
        "Cannot clone object. You should provide an "
        "instance of scikit-learn estimator instead of a class."
    )

    assert_raises(
        TypeError,
        expect,
        SFS,
        Perceptron,
        scoring="accuracy",
        k_features=3,
        clone_estimator=True,
    )


def test_clone_params_pass():
    iris = load_iris()
    X = iris.data
    y = iris.target
    lr = SoftmaxRegression(random_seed=1)
    sfs1 = SFS(
        lr,
        k_features=2,
        forward=True,
        floating=False,
        scoring="accuracy",
        cv=0,
        clone_estimator=True,
        verbose=0,
        n_jobs=1,
    )
    sfs1 = sfs1.fit(X, y)
    assert sfs1.k_feature_idx_ == (1, 3)


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


def test_get_metric_dict_not_fitted():
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

    assert_raises(AttributeError, expect, sfs1.get_metric_dict)


def test_cv_generator_raises():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)

    groups = np.arange(len(y)) // 50
    cv_gen = GroupKFold(n_splits=3).split(X, y, groups)

    expect = (
        "Input cv is a generator object, which is not supported. "
        "Instead please input an iterable yielding train, test splits. "
        "This can usually be done by passing a cross-validation "
        "generator to the built-in list function. I.e. "
        "cv=list(<cv-generator>)"
    )

    assert_raises(
        TypeError, expect, SFS, knn, k_features=2, cv=cv_gen, verbose=0, n_jobs=1
    )


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
    )

    sfs1._TESTING_INTERRUPT_MODE = True
    out = sfs1.fit(X, y)

    assert len(out.subsets_.keys()) > 0
    assert sfs1.interrupted_


def test_gridsearch():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=2)

    sfs1 = SFS(estimator=knn, k_features=3, forward=True, floating=False, cv=5)

    pipe = Pipeline([("sfs", sfs1), ("knn", knn)])

    param_grid = [
        {"sfs__k_features": [1, 2, 3, 4], "sfs__estimator__n_neighbors": [1, 2, 3, 4]}
    ]

    if Version(sklearn_version) < Version("0.24.1"):
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            n_jobs=1,
            iid=False,
            cv=5,
            refit=False,
        )
    else:
        gs = GridSearchCV(
            estimator=pipe, param_grid=param_grid, n_jobs=1, cv=5, refit=False
        )

    gs = gs.fit(X, y)

    assert gs.best_params_["sfs__k_features"] == 3


def test_string_scoring_clf():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs1 = SFS(knn, k_features=3, cv=0)
    sfs1 = sfs1.fit(X, y)

    sfs2 = SFS(knn, k_features=3, scoring="accuracy", cv=0)
    sfs2 = sfs2.fit(X, y)

    sfs3 = SFS(knn, k_features=3, scoring=make_scorer(accuracy_score), cv=0)
    sfs3 = sfs2.fit(X, y)

    assert sfs1.k_score_ == sfs2.k_score_
    assert sfs1.k_score_ == sfs3.k_score_


def test_max_feature_subset_size_in_tuple_range():
    X, y = boston_housing_data()
    lr = LinearRegression()

    sfs = SFS(
        lr,
        k_features=(1, 5),
        forward=False,
        floating=True,
        scoring="neg_mean_squared_error",
        cv=10,
    )

    sfs = sfs.fit(X, y)
    assert len(sfs.k_feature_idx_) == 5


def test_max_feature_subset_best():
    X, y = boston_housing_data()
    lr = LinearRegression()

    sfs = SFS(lr, k_features="best", forward=True, floating=False, cv=10)

    sfs = sfs.fit(X, y)
    assert sfs.k_feature_idx_ == (1, 3, 5, 7, 8, 9, 10, 11, 12)


def test_max_feature_subset_parsimonious():
    X, y = boston_housing_data()
    lr = LinearRegression()

    sfs = SFS(lr, k_features="parsimonious", forward=True, floating=False, cv=10)

    sfs = sfs.fit(X, y)
    assert sfs.k_feature_idx_ == (5, 10, 11, 12)


def test_check_pandas_dataframe_fit():
    for floating in [True, False]:
        iris = load_iris()
        X = iris.data
        y = iris.target
        lr = SoftmaxRegression(random_seed=1)
        sfs1 = SFS(
            lr,
            k_features=2,
            forward=True,
            floating=floating,
            scoring="accuracy",
            cv=0,
            verbose=0,
            n_jobs=1,
        )

        df = pd.DataFrame(
            X, columns=["sepal len", "sepal width", "petal len", "petal width"]
        )

        sfs1 = sfs1.fit(X, y)
        assert sfs1.k_feature_idx_ == (1, 3)
        assert sfs1.k_feature_names_ == ("1", "3")
        assert sfs1.subsets_[2]["feature_names"] == ("1", "3")

        sfs1 = sfs1.fit(df, y)
        assert sfs1.subsets_[1]["feature_names"] == ("petal width",)
        assert sfs1.subsets_[2]["feature_names"] == ("sepal width", "petal width")
        assert sfs1.subsets_[1]["feature_idx"] == (3,)
        assert sfs1.subsets_[2]["feature_idx"] == (1, 3)
        assert sfs1.k_feature_idx_ == (1, 3)
        assert sfs1.k_feature_names_ == ("sepal width", "petal width")

        sfs1._TESTING_INTERRUPT_MODE = True
        out = sfs1.fit(df, y)
        assert len(out.subsets_.keys()) > 0
        assert sfs1.interrupted_
        assert sfs1.subsets_[1]["feature_names"] == ("petal width",)
        assert sfs1.k_feature_idx_ == (3,)
        assert sfs1.k_feature_names_ == ("petal width",)


def test_check_pandas_dataframe_fit_backward():
    for floating in [True, False]:
        iris = load_iris()
        X = iris.data
        y = iris.target
        lr = SoftmaxRegression(random_seed=1)
        sfs1 = SFS(
            lr,
            k_features=2,
            forward=False,
            floating=floating,
            scoring="accuracy",
            cv=0,
            verbose=0,
            n_jobs=1,
        )

        df = pd.DataFrame(
            X, columns=["sepal len", "sepal width", "petal len", "petal width"]
        )

        sfs1 = sfs1.fit(X, y)
        assert sfs1.k_feature_idx_ == (1, 2)
        assert sfs1.k_feature_names_ == ("1", "2")
        assert sfs1.subsets_[2]["feature_names"] == ("1", "2")

        sfs1 = sfs1.fit(df, y)
        assert sfs1.subsets_[3]["feature_names"] == (
            "sepal len",
            "sepal width",
            "petal len",
        )
        assert sfs1.subsets_[2]["feature_names"] == ("sepal width", "petal len")
        assert sfs1.subsets_[3]["feature_idx"] == (0, 1, 2)
        assert sfs1.subsets_[2]["feature_idx"] == (1, 2)
        assert sfs1.k_feature_idx_ == (1, 2)
        assert sfs1.k_feature_names_ == ("sepal width", "petal len")

        sfs1._TESTING_INTERRUPT_MODE = True
        out = sfs1.fit(df, y)
        assert len(out.subsets_.keys()) > 0
        assert sfs1.interrupted_
        assert sfs1.subsets_[3]["feature_names"] == (
            "sepal len",
            "sepal width",
            "petal len",
        )
        assert sfs1.k_feature_idx_ == (0, 1, 2)
        assert sfs1.k_feature_names_ == ("sepal len", "sepal width", "petal len")


def test_check_pandas_dataframe_transform():
    iris = load_iris()
    X = iris.data
    y = iris.target
    lr = SoftmaxRegression(random_seed=1)
    sfs1 = SFS(
        lr,
        k_features=2,
        forward=True,
        floating=False,
        scoring="accuracy",
        cv=0,
        verbose=0,
        n_jobs=1,
    )

    df = pd.DataFrame(
        X, columns=["sepal length", "sepal width", "petal length", "petal width"]
    )
    sfs1 = sfs1.fit(df, y)
    assert sfs1.k_feature_idx_ == (1, 3)
    assert (150, 2) == sfs1.transform(df).shape


def test_invalid_estimator():
    expect = "Estimator must have an ._estimator_type for infering `scoring`"
    assert_raises(AttributeError, expect, SFS, PCA())

    class PCA2(PCA):
        def __init__(self):
            super().__init__()
            self._estimator_type = "something"

    expect = "Estimator must be a Classifier or Regressor."
    assert_raises(AttributeError, expect, SFS, PCA2())


def test_invalid_k_features():
    iris = load_iris()
    X = iris.data
    y = iris.target
    lr = SoftmaxRegression(random_seed=1)

    sfs1 = SFS(lr, k_features=(1, 2, 3), scoring="accuracy")
    expect = "k_features tuple must consist of 2 elements, a min and a max value."
    assert_raises(AttributeError, expect, sfs1.fit, X, y)

    sfs1 = SFS(lr, k_features="something", scoring="accuracy")
    expect = 'If a string argument is provided, it must be "best" or "parsimonious"'
    assert_raises(AttributeError, expect, sfs1.fit, X, y)


def test_verbose():
    iris = load_iris()
    X = iris.data
    y = iris.target
    lr = SoftmaxRegression(random_seed=1)

    sfs1 = SFS(lr, k_features=1, scoring="accuracy", verbose=1)
    sfs1.fit(X, y)


def test_check_pandas_dataframe_with_feature_groups():
    iris = load_iris()
    X = iris.data
    y = iris.target
    lr = SoftmaxRegression(random_seed=1)

    df = pd.DataFrame(
        X, columns=["sepal length", "sepal width", "petal length", "petal width"]
    )

    sfs1 = SFS(
        lr,
        k_features=2,
        forward=True,
        floating=False,
        scoring="accuracy",
        feature_groups=[
            ["sepal length", "petal length"],
            ["sepal width"],
            ["petal width"],
        ],
        cv=0,
        verbose=0,
        n_jobs=1,
    )

    sfs1 = sfs1.fit(df, y)
    assert sfs1.k_feature_names_ == (
        "sepal width",
        "petal width",
    ), sfs1.k_feature_names_
    assert (150, 2) == sfs1.transform(df).shape

    # now, test with different `feature_groups`
    sfs1 = SFS(
        lr,
        k_features=2,  # this is num of selected groups to form selected features
        forward=True,
        floating=False,
        scoring="accuracy",
        feature_groups=[
            ["petal length", "petal width"],
            ["sepal length"],
            ["sepal width"],
        ],
        cv=0,
        verbose=0,
        n_jobs=1,
    )

    sfs1 = sfs1.fit(df, y)
    # the selected fetures are sorted according their corresponding indices
    assert sfs1.k_feature_names_ == (
        "sepal width",
        "petal length",
        "petal width",
    ), sfs1.k_feature_names_


def test_check_pandas_dataframe_with_feature_groups_and_fixed_features():
    iris = load_iris()
    X = iris.data
    y = iris.target
    lr = SoftmaxRegression(random_seed=123)

    df = pd.DataFrame(
        X, columns=["sepal length", "sepal width", "petal length", "petal width"]
    )
    sfs1 = SFS(
        lr,
        k_features=2,
        forward=True,
        floating=False,
        scoring="accuracy",
        feature_groups=[
            ["petal length", "petal width"],
            ["sepal length"],
            ["sepal width"],
        ],
        fixed_features=["sepal length", "petal length", "petal width"],
        cv=0,
        verbose=0,
        n_jobs=1,
    )

    sfs1 = sfs1.fit(df, y)
    assert sfs1.k_feature_names_ == (
        "sepal length",
        "petal length",
        "petal width",
    ), sfs1.k_feature_names_


def test_check_feature_groups():
    iris = load_iris()
    X = iris.data
    y = iris.target
    lr = SoftmaxRegression(random_seed=123)
    sfs1 = SFS(
        lr,
        k_features=2,
        forward=True,
        floating=False,
        scoring="accuracy",
        feature_groups=[[2, 3], [0], [1]],
        fixed_features=[0, 2, 3],
        cv=0,
        verbose=0,
        n_jobs=1,
    )

    sfs1 = sfs1.fit(X, y)
    assert sfs1.k_feature_idx_ == (0, 2, 3), sfs1.k_feature_idx_
