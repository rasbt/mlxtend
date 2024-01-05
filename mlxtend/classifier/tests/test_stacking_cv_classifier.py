# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Authors: Sebastian Raschka <sebastianraschka.com>
#          Reiichiro Nakano <github.com/reiinakano>
#
# License: BSD 3 clause

import random

import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn import datasets, exceptions
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from mlxtend.classifier import StackingCVClassifier
from mlxtend.data import iris_data
from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.utils import assert_raises

X_iris, y_iris = iris_data()
X_iris = X_iris[:, 1:3]

breast_cancer = datasets.load_breast_cancer()
X_breast, y_breast = breast_cancer.data[:, 1:3], breast_cancer.target


def test_StackingCVClassifier():
    np.random.seed(123)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], meta_classifier=meta, shuffle=False
    )

    scores = cross_val_score(sclf, X_iris, y_iris, cv=5, scoring="accuracy")
    scores_mean = round(scores.mean(), 2)

    if Version(sklearn_version) < Version("0.20"):
        assert scores_mean == 0.93, scores_mean
    else:
        assert scores_mean == 0.92, scores_mean


def test_use_clones():
    np.random.seed(123)
    X, y = iris_data()

    meta = LogisticRegression(solver="liblinear", multi_class="ovr")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    StackingCVClassifier(
        classifiers=[clf1, clf2], use_clones=True, meta_classifier=meta, shuffle=False
    ).fit(X, y)

    assert_raises(
        exceptions.NotFittedError,
        "This RandomForestClassifier instance is not fitted yet."
        " Call 'fit' with appropriate arguments"
        " before using this estimator.",
        clf1.predict,
        X,
    )

    StackingCVClassifier(
        classifiers=[clf1, clf2],
        use_probas=True,
        use_clones=False,
        meta_classifier=meta,
        shuffle=False,
    ).fit(X, y)

    clf1.predict(X)


def test_sample_weight():
    # with no weight given
    np.random.seed(123)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], meta_classifier=meta, shuffle=False
    )
    prob1 = sclf.fit(X_iris, y_iris).predict_proba(X_iris)

    # with weight = 1
    np.random.seed(123)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], meta_classifier=meta, shuffle=False
    )
    w = np.ones(len(y_iris))
    prob2 = sclf.fit(X_iris, y_iris, sample_weight=w).predict_proba(X_iris)

    # with random weight
    random.seed(87)
    w = np.array([random.random() for _ in range(len(y_iris))])
    np.random.seed(123)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], meta_classifier=meta, shuffle=False
    )
    prob3 = sclf.fit(X_iris, y_iris, sample_weight=w).predict_proba(X_iris)

    diff12 = np.max(np.abs(prob1 - prob2))
    diff23 = np.max(np.abs(prob2 - prob3))
    assert diff12 < 1e-3, "max diff is %.4f" % diff12
    assert diff23 > 1e-3, "max diff is %.4f" % diff23


def test_no_weight_support():
    w = np.array([random.random() for _ in range(len(y_iris))])
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2, clf3], meta_classifier=meta, shuffle=False
    )
    with pytest.raises(TypeError):
        sclf.fit(X_iris, y_iris, sample_weight=w)


def test_no_weight_support_meta():
    w = np.array([random.random() for _ in range(len(y_iris))])
    meta = KNeighborsClassifier()
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], meta_classifier=meta, shuffle=False
    )

    with pytest.raises(TypeError):
        sclf.fit(X_iris, y_iris, sample_weight=w)


def test_no_weight_support_with_no_weight():
    logit = LogisticRegression(multi_class="ovr", solver="liblinear")
    rf = RandomForestClassifier(n_estimators=10)
    gnb = GaussianNB()
    knn = KNeighborsClassifier()
    sclf = StackingCVClassifier(
        classifiers=[logit, rf, gnb], meta_classifier=knn, shuffle=False
    )
    sclf.fit(X_iris, y_iris)

    sclf = StackingCVClassifier(
        classifiers=[logit, knn, gnb], meta_classifier=rf, shuffle=False
    )
    sclf.fit(X_iris, y_iris)


def test_StackingClassifier_proba():
    np.random.seed(12)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], meta_classifier=meta, shuffle=False
    )

    scores = cross_val_score(sclf, X_iris, y_iris, cv=5, scoring="accuracy")
    scores_mean = round(scores.mean(), 2)

    if Version(sklearn_version) < Version("0.20"):
        assert scores_mean == 0.92, scores_mean
    else:
        assert scores_mean == 0.93, scores_mean


def test_gridsearch():
    np.random.seed(123)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], meta_classifier=meta, use_probas=True, shuffle=False
    )

    params = {
        "meta_classifier__C": [1.0, 100.0],
        "randomforestclassifier__n_estimators": [20, 200],
    }

    if Version(sklearn_version) < Version("0.24.1"):
        grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, iid=False)
    else:
        grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    X, y = iris_data()
    grid.fit(X, y)

    mean_scores = [round(s, 2) for s in grid.cv_results_["mean_test_score"]]

    assert mean_scores == [0.96, 0.95, 0.96, 0.95]


def test_gridsearch_enumerate_names():
    np.random.seed(123)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf1, clf2], meta_classifier=meta, shuffle=False
    )

    params = {
        "meta_classifier__C": [1.0, 100.0],
        "randomforestclassifier-1__n_estimators": [5, 10],
        "randomforestclassifier-2__n_estimators": [5, 20],
        "use_probas": [True, False],
    }

    if Version(sklearn_version) < Version("0.24.1"):
        grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, iid=False)
    else:
        grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    X, y = iris_data()
    grid = grid.fit(X, y)


def test_use_probas():
    np.random.seed(123)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], use_probas=True, meta_classifier=meta, shuffle=False
    )

    scores = cross_val_score(sclf, X_iris, y_iris, cv=5, scoring="accuracy")
    scores_mean = round(scores.mean(), 2)
    assert scores_mean == 0.94, scores_mean


def test_use_features_in_secondary():
    np.random.seed(123)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2],
        use_features_in_secondary=True,
        meta_classifier=meta,
        shuffle=False,
    )

    scores = cross_val_score(sclf, X_iris, y_iris, cv=5, scoring="accuracy")
    scores_mean = round(scores.mean(), 2)
    assert scores_mean == 0.93, scores_mean


def test_do_not_stratify():
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], meta_classifier=meta, random_state=42, stratify=False
    )

    scores = cross_val_score(sclf, X_iris, y_iris, cv=5, scoring="accuracy")
    scores_mean = round(scores.mean(), 2)
    assert scores_mean == 0.93, scores.mean()


def test_cross_validation_technique():
    # This is like the `test_do_not_stratify` but instead
    # autogenerating the cross validation strategy it provides
    # a pre-created object
    cv = KFold(n_splits=2, shuffle=True)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10, random_state=42)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], meta_classifier=meta, cv=cv, random_state=42
    )

    scores = cross_val_score(sclf, X_iris, y_iris, cv=5, scoring="accuracy")
    scores_mean = round(scores.mean(), 2)
    assert scores_mean == 0.92, scores.mean()


def test_not_fitted():
    np.random.seed(123)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], use_probas=True, meta_classifier=meta, shuffle=False
    )

    X, y = iris_data()
    assert_raises(
        NotFittedError,
        "This StackingCVClassifier instance is not fitted yet."
        " Call 'fit' with appropriate arguments"
        " before using this method.",
        sclf.predict,
        X,
    )

    assert_raises(
        NotFittedError,
        "This StackingCVClassifier instance is not fitted yet."
        " Call 'fit' with appropriate arguments"
        " before using this method.",
        sclf.predict_proba,
        X,
    )

    assert_raises(
        NotFittedError,
        "This StackingCVClassifier instance is not fitted yet."
        " Call 'fit' with appropriate arguments"
        " before using this method.",
        sclf.predict_meta_features,
        X,
    )


def test_verbose():
    np.random.seed(123)
    meta = LogisticRegression(multi_class="ovr", solver="liblinear")
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2],
        use_probas=True,
        meta_classifier=meta,
        shuffle=False,
        verbose=3,
    )
    sclf.fit(X_iris, y_iris)


def test_get_params():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression(multi_class="ovr", solver="liblinear")
    sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

    got = sorted(list({s.split("__")[0] for s in sclf.get_params().keys()}))

    expect = [
        "classifiers",
        "cv",
        "drop_proba_col",
        "gaussiannb",
        "kneighborsclassifier",
        "meta_classifier",
        "n_jobs",
        "pre_dispatch",
        "random_state",
        "randomforestclassifier",
        "shuffle",
        "store_train_meta_features",
        "stratify",
        "use_clones",
        "use_features_in_secondary",
        "use_probas",
        "verbose",
    ]
    assert got == expect, got


def test_classifier_gridsearch():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(n_estimators=10, random_state=42)
    clf3 = GaussianNB()
    lr = LogisticRegression(multi_class="ovr", solver="liblinear")
    sclf = StackingCVClassifier(classifiers=[clf1], meta_classifier=lr, random_state=42)

    params = {"classifiers": [[clf1], [clf1, clf2, clf3]]}

    if Version(sklearn_version) < Version("0.24.1"):
        grid = GridSearchCV(
            estimator=sclf, param_grid=params, cv=5, iid=False, refit=True
        )
    else:
        grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, refit=True)
    grid.fit(X_iris, y_iris)

    assert len(grid.best_params_["classifiers"]) == 3, len(
        grid.best_params_["classifiers"]
    )


def test_train_meta_features_():
    knn = KNeighborsClassifier()
    lr = LogisticRegression(multi_class="ovr", solver="liblinear")
    gnb = GaussianNB()
    stclf = StackingCVClassifier(
        classifiers=[knn, gnb], meta_classifier=lr, store_train_meta_features=True
    )
    X_train, _, y_train, _ = train_test_split(X_iris, y_iris, test_size=0.3)
    stclf.fit(X_train, y_train)
    train_meta_features = stclf.train_meta_features_
    assert train_meta_features.shape == (X_train.shape[0], 2)


def test_predict_meta_features():
    knn = KNeighborsClassifier()
    lr = LogisticRegression(multi_class="ovr", solver="liblinear")
    gnb = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3)
    #  test default (class labels)
    stclf = StackingCVClassifier(
        classifiers=[knn, gnb], meta_classifier=lr, store_train_meta_features=True
    )
    stclf.fit(X_train, y_train)
    test_meta_features = stclf.predict(X_test)
    assert test_meta_features.shape == (X_test.shape[0],)


def test_meta_feat_reordering():
    knn = KNeighborsClassifier()
    lr = LogisticRegression(multi_class="ovr", solver="liblinear")
    gnb = GaussianNB()
    stclf = StackingCVClassifier(
        classifiers=[knn, gnb],
        meta_classifier=lr,
        shuffle=True,
        random_state=42,
        store_train_meta_features=True,
    )
    X_train, _, y_train, _ = train_test_split(
        X_breast, y_breast, random_state=0, test_size=0.3
    )
    stclf.fit(X_train, y_train)

    if Version(sklearn_version) < Version("0.21"):
        expected_value = 0.86
    elif Version(sklearn_version) < Version("0.22"):
        expected_value = 0.87
    else:
        expected_value = 0.85

    assert (
        round(roc_auc_score(y_train, stclf.train_meta_features_[:, 1]), 2)
        == expected_value
    ), round(roc_auc_score(y_train, stclf.train_meta_features_[:, 1]), 2)


def test_clone():
    knn = KNeighborsClassifier()
    lr = LogisticRegression(multi_class="ovr", solver="liblinear")
    gnb = GaussianNB()
    stclf = StackingCVClassifier(
        classifiers=[knn, gnb], meta_classifier=lr, store_train_meta_features=True
    )
    clone(stclf)


def test_sparse_inputs():
    np.random.seed(123)
    rf = RandomForestClassifier(n_estimators=10)
    lr = LogisticRegression(multi_class="ovr", solver="liblinear")
    stclf = StackingCVClassifier(
        classifiers=[rf, rf], meta_classifier=lr, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_breast, y_breast, test_size=0.3
    )

    # dense
    stclf.fit(X_train, y_train)
    assert round(stclf.score(X_train, y_train), 2) == 0.99

    # sparse
    stclf.fit(sparse.csr_matrix(X_train), y_train)
    assert round(stclf.score(X_train, y_train), 2) == 0.99


def test_sparse_inputs_with_features_in_secondary():
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    lr = LogisticRegression(multi_class="ovr", solver="liblinear")
    stclf = StackingCVClassifier(
        classifiers=[rf, rf],
        meta_classifier=lr,
        random_state=42,
        use_features_in_secondary=True,
    )
    X_train, _, y_train, _ = train_test_split(X_breast, y_breast, test_size=0.3)

    # dense
    stclf.fit(X_train, y_train)

    expected_value = 1.0

    assert round(stclf.score(X_train, y_train), 2) == expected_value, round(
        stclf.score(X_train, y_train), 2
    )

    # sparse
    stclf.fit(sparse.csr_matrix(X_train), y_train)

    if Version(sklearn_version) < Version("0.21"):
        expected_value = 1.0
    if Version(sklearn_version) < Version("0.22"):
        expected_value = 0.99
    else:
        expected_value = 1.00

    assert round(stclf.score(X_train, y_train), 2) == expected_value, round(
        stclf.score(X_train, y_train), 2
    )


def test_StackingClassifier_drop_proba_col():
    np.random.seed(123)
    lr1 = LogisticRegression(solver="liblinear", multi_class="ovr")
    sclf1 = StackingCVClassifier(
        classifiers=[lr1, lr1],
        use_probas=True,
        drop_proba_col=None,
        meta_classifier=lr1,
    )

    sclf1.fit(X_iris, y_iris)
    r1 = sclf1.predict_meta_features(X_iris[:2])
    assert r1.shape == (2, 6)

    sclf2 = StackingCVClassifier(
        classifiers=[lr1, lr1],
        use_probas=True,
        drop_proba_col="last",
        meta_classifier=lr1,
    )

    sclf2.fit(X_iris, y_iris)
    r2 = sclf2.predict_meta_features(X_iris[:2])
    assert r2.shape == (2, 4), r2.shape

    sclf4 = StackingCVClassifier(
        classifiers=[lr1, lr1],
        use_probas=True,
        drop_proba_col="first",
        meta_classifier=lr1,
    )

    sclf4.fit(X_iris, y_iris)
    r4 = sclf4.predict_meta_features(X_iris[:2])
    assert r4.shape == (2, 4), r4.shape

    sclf3 = StackingCVClassifier(
        classifiers=[lr1, lr1],
        use_probas=True,
        drop_proba_col="last",
        meta_classifier=lr1,
    )

    sclf3.fit(X_iris[0:100], y_iris[0:100])  # only 2 classes
    r3 = sclf3.predict_meta_features(X_iris[:2])
    assert r3.shape == (2, 2), r3.shape


def test_works_with_df_if_fold_indexes_missing():
    """This is a regression test to make sure fitting will still work even if
    training data has ids that cannot be indexed using the indexes from the cv
    (e.g. skf)

    Some possibilities:
    + Output of the folds are not neatly consecutive (i.e. [341, 345, 543, ...]
      instead of [0, 1, ... n])
    + Indexes just start from some number greater than the size of the input
      (see test case)

    Training data sometimes has ids that carry other information, and selection
    of rows based on cv should not break.

    This is fixed in the code using `safe_indexing`
    """

    np.random.seed(123)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    lr = LogisticRegression(multi_class="ovr", solver="liblinear")
    stclf = StackingCVClassifier(
        classifiers=[rf, rf],
        meta_classifier=lr,
        random_state=42,
        use_features_in_secondary=True,
    )

    X_modded = pd.DataFrame(X_breast, index=np.arange(X_breast.shape[0]) + 1000)
    y_modded = pd.Series(y_breast, index=np.arange(y_breast.shape[0]) + 1000)

    X_train, X_test, y_train, y_test = train_test_split(
        X_modded, y_modded, test_size=0.3
    )

    # dense
    stclf.fit(X_train, y_train)

    if Version(sklearn_version) < Version("0.22"):
        assert round(stclf.score(X_train, y_train), 2) == 0.99, round(
            stclf.score(X_train, y_train), 2
        )
    else:
        assert round(stclf.score(X_train, y_train), 2) == 0.98, round(
            stclf.score(X_train, y_train), 2
        )


def test_decision_function():
    np.random.seed(123)

    # PassiveAggressiveClassifier has no predict_proba
    meta = PassiveAggressiveClassifier(random_state=42)
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], use_probas=True, meta_classifier=meta
    )

    scores = cross_val_score(sclf, X_breast, y_breast, cv=5, scoring="roc_auc")
    scores_mean = round(scores.mean(), 2)
    assert scores_mean == 0.96, scores_mean

    # another test
    meta = SVC(decision_function_shape="ovo")

    sclf = StackingCVClassifier(
        classifiers=[clf1, clf2], use_probas=True, meta_classifier=meta
    )

    scores = cross_val_score(sclf, X_breast, y_breast, cv=5, scoring="roc_auc")
    scores_mean = round(scores.mean(), 2)

    if Version(sklearn_version) < Version("0.21"):
        assert scores_mean == 0.94, scores_mean
    elif Version(sklearn_version) < Version("0.22"):
        assert scores_mean == 0.96, scores_mean
    else:
        assert scores_mean == 0.90, scores_mean


def test_drop_col_unsupported():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier()

    with pytest.raises(ValueError):
        StackingCVClassifier(
            classifiers=[clf1, clf2, clf3],
            meta_classifier=meta,
            drop_proba_col="invalid value",
        )
