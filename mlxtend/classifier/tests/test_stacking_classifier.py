# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import random
import pytest
from mlxtend.classifier import StackingClassifier
from mlxtend.externals.estimator_checks import NotFittedError
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises
from numpy.testing import assert_almost_equal
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version


X, y = iris_data()
X = X[:, 1:3]


y2 = np.c_[y, y]


def test_StackingClassifier():
    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr',)
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              meta_classifier=meta)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))

    if Version(sklearn_version) < Version("0.20"):
        assert scores_mean == 0.95, scores_mean
    else:
        assert scores_mean == 0.95, scores_mean


def test_sample_weight():
    # Make sure that:
    #    prediction with weight
    # != prediction with no weight
    # == prediction with weight ones
    random.seed(87)
    w = np.array([random.random() for _ in range(len(y))])

    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              meta_classifier=meta)
    prob1 = sclf.fit(X, y, sample_weight=w).predict_proba(X)

    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              meta_classifier=meta)
    prob2 = sclf.fit(X, y, sample_weight=None).predict_proba(X)

    maxdiff = np.max(np.abs(prob1 - prob2))
    assert maxdiff > 1e-3, "max diff is %.4f" % maxdiff

    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              meta_classifier=meta)
    prob3 = sclf.fit(X, y, sample_weight=np.ones(len(y))).predict_proba(X)

    maxdiff = np.max(np.abs(prob2 - prob3))
    assert maxdiff < 1e-3, "max diff is %.4f" % maxdiff


def test_weight_unsupported():
    # Error since KNN does not support sample_weight
    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              meta_classifier=meta)
    random.seed(87)
    w = np.array([random.random() for _ in range(len(y))])

    with pytest.raises(TypeError):
        sclf.fit(X, y, sample_seight=w)


def test_weight_unsupported_no_weight():
    # This is okay since we do not pass sample weight
    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              meta_classifier=meta)
    sclf.fit(X, y)


def test_StackingClassifier_proba_avg_1():

    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr',
                              random_state=1)
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_probas=True,
                              meta_classifier=meta)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.93, scores_mean


def test_StackingClassifier_proba_concat_1():

    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_probas=True,
                              average_probas=False,
                              meta_classifier=meta)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.93, scores_mean


def test_StackingClassifier_avg_vs_concat():
    np.random.seed(123)
    lr1 = LogisticRegression(solver='liblinear',
                             multi_class='ovr')
    sclf1 = StackingClassifier(classifiers=[lr1, lr1],
                               use_probas=True,
                               average_probas=True,
                               meta_classifier=lr1)

    sclf1.fit(X, y)
    r1 = sclf1.predict_meta_features(X[:2])
    assert r1.shape == (2, 3)
    assert_almost_equal(np.sum(r1[0]), 1.0, decimal=6)
    assert_almost_equal(np.sum(r1[1]), 1.0, decimal=6)

    sclf2 = StackingClassifier(classifiers=[lr1, lr1],
                               use_probas=True,
                               average_probas=False,
                               meta_classifier=lr1)

    sclf2.fit(X, y)
    r2 = sclf2.predict_meta_features(X[:2])
    assert r2.shape == (2, 6)
    assert_almost_equal(np.sum(r2[0]), 2.0, decimal=6)
    assert_almost_equal(np.sum(r2[1]), 2.0, decimal=6)
    np.array_equal(r2[0][:3], r2[0][3:])


def test_StackingClassifier_drop_last_proba():
    np.random.seed(123)
    lr1 = LogisticRegression(solver='liblinear',
                             multi_class='ovr')
    sclf1 = StackingClassifier(classifiers=[lr1, lr1],
                               use_probas=True,
                               drop_last_proba=False,
                               meta_classifier=lr1)

    sclf1.fit(X, y)
    r1 = sclf1.predict_meta_features(X[:2])
    assert r1.shape == (2, 6)

    sclf2 = StackingClassifier(classifiers=[lr1, lr1],
                               use_probas=True,
                               drop_last_proba=True,
                               meta_classifier=lr1)

    sclf2.fit(X, y)
    r2 = sclf2.predict_meta_features(X[:2])
    assert r2.shape == (2, 4), r2.shape

    sclf3 = StackingClassifier(classifiers=[lr1, lr1],
                               use_probas=True,
                               drop_last_proba=True,
                               meta_classifier=lr1)

    sclf3.fit(X[0:100], y[0:100])  # only 2 classes
    r3 = sclf3.predict_meta_features(X[:2])
    assert r3.shape == (2, 2), r3.shape


def test_multivariate_class():
    np.random.seed(123)
    meta = KNeighborsClassifier()
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = KNeighborsClassifier()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              meta_classifier=meta)
    y_pred = sclf.fit(X, y2).predict(X)
    ca = .973
    assert round((y_pred == y2).mean(), 3) == ca


def test_gridsearch():
    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              meta_classifier=meta)

    params = {'meta_classifier__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, iid=False)
    X, y = iris_data()
    grid.fit(X, y)

    mean_scores = [round(s, 2) for s
                   in grid.cv_results_['mean_test_score']]

    assert mean_scores == [0.95, 0.97, 0.96, 0.96], mean_scores


def test_gridsearch_enumerate_names():
    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf1, clf2],
                              meta_classifier=meta)

    params = {'meta_classifier__C': [1.0, 100.0],
              'randomforestclassifier-1__n_estimators': [5, 10],
              'randomforestclassifier-2__n_estimators': [5, 20],
              'use_probas': [True, False]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, iid=False)
    X, y = iris_data()
    grid = grid.fit(X, y)


def test_use_probas():
    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_probas=True,
                              meta_classifier=meta)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.93, scores_mean


def test_not_fitted():
    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_probas=True,
                              meta_classifier=meta)

    X, y = iris_data()

    assert_raises(NotFittedError,
                  "This StackingClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict,
                  X)

    assert_raises(NotFittedError,
                  "This StackingClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict_proba,
                  X)

    assert_raises(NotFittedError,
                  "This StackingClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict_meta_features,
                  X)


def test_verbose():
    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_probas=True,
                              meta_classifier=meta,
                              verbose=3)
    X, y = iris_data()
    sclf.fit(X, y)


def test_use_features_in_secondary_predict():
    np.random.seed(123)
    X, y = iris_data()
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_features_in_secondary=True,
                              meta_classifier=meta)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.95, scores_mean


def test_use_features_in_secondary_predict_proba():
    np.random.seed(123)
    X, y = iris_data()
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr',
                              random_state=1)
    clf1 = RandomForestClassifier(n_estimators=10, random_state=1)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_features_in_secondary=True,
                              meta_classifier=meta)

    sclf.fit(X, y)
    idx = [0, 1, 2]
    y_pred = sclf.predict_proba(X[idx])[:, 0]
    expect = np.array([0.916, 0.828, 0.889])
    np.testing.assert_almost_equal(y_pred, expect, 3)


def test_use_features_in_secondary_sparse_input_predict():
    np.random.seed(123)
    X, y = iris_data()
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr',
                              random_state=1)
    clf1 = RandomForestClassifier(n_estimators=10, random_state=1)
    sclf = StackingClassifier(classifiers=[clf1],
                              use_features_in_secondary=True,
                              meta_classifier=meta)

    scores = cross_val_score(sclf,
                             sparse.csr_matrix(X),
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.97, scores_mean


def test_use_features_in_secondary_sparse_input_predict_proba():
    np.random.seed(123)
    meta = LogisticRegression(solver='liblinear',
                              multi_class='ovr')
    clf1 = RandomForestClassifier(n_estimators=10)
    sclf = StackingClassifier(classifiers=[clf1],
                              use_features_in_secondary=True,
                              meta_classifier=meta)

    sclf.fit(sparse.csr_matrix(X), y)
    idx = [0, 1, 2]
    y_pred = sclf.predict_proba(
        sparse.csr_matrix(X[idx])
    )[:, 0]
    expect = np.array([0.910, 0.829, 0.882])
    np.testing.assert_almost_equal(y_pred, expect, 3)


def test_get_params():
    np.random.seed(123)
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()
    lr = LogisticRegression(solver='liblinear',
                            multi_class='ovr')
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              meta_classifier=lr)

    got = sorted(list({s.split('__')[0] for s in sclf.get_params().keys()}))
    expect = ['average_probas',
              'classifiers',
              'drop_last_proba',
              'gaussiannb',
              'kneighborsclassifier',
              'meta_classifier',
              'randomforestclassifier',
              'store_train_meta_features',
              'use_clones',
              'use_features_in_secondary',
              'use_probas',
              'verbose']
    assert got == expect, got


def test_classifier_gridsearch():
    np.random.seed(123)
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()
    lr = LogisticRegression(solver='liblinear',
                            multi_class='ovr')
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              meta_classifier=lr)

    params = {'classifiers': [[clf1, clf1, clf1], [clf2, clf3]]}

    grid = GridSearchCV(estimator=sclf,
                        param_grid=params,
                        cv=5,
                        iid=False,
                        refit=True)
    grid.fit(X, y)

    assert len(grid.best_params_['classifiers']) == 2


def test_train_meta_features_():
    np.random.seed(123)
    knn = KNeighborsClassifier()
    lr = LogisticRegression(solver='liblinear',
                            multi_class='ovr')
    gnb = GaussianNB()
    stclf = StackingClassifier(classifiers=[knn, gnb],
                               meta_classifier=lr,
                               store_train_meta_features=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    stclf.fit(X_train, y_train)
    train_meta_features = stclf.train_meta_features_
    assert train_meta_features.shape == (X_train.shape[0], 2)


def test_predict_meta_features():
    knn = KNeighborsClassifier()
    lr = LogisticRegression(solver='liblinear',
                            multi_class='ovr',
                            random_state=1)
    gnb = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #  test default (class labels)
    stclf = StackingClassifier(classifiers=[knn, gnb],
                               meta_classifier=lr,
                               store_train_meta_features=True)
    stclf.fit(X_train, y_train)
    test_meta_features = stclf.predict(X_test)
    assert test_meta_features.shape == (X_test.shape[0],)


def test_clone():
    np.random.seed(1)
    knn = KNeighborsClassifier()
    lr = LogisticRegression(solver='liblinear',
                            multi_class='ovr')
    gnb = GaussianNB()
    stclf = StackingClassifier(classifiers=[knn, gnb],
                               meta_classifier=lr,
                               store_train_meta_features=True)
    clone(stclf)


def test_decision_function():
    np.random.seed(123)

    # PassiveAggressiveClassifier has no predict_proba
    meta = PassiveAggressiveClassifier(max_iter=1000,
                                       tol=0.001,
                                       random_state=42)
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_probas=True,
                              meta_classifier=meta)

    # binarize target
    y2 = y > 1
    scores = cross_val_score(sclf,
                             X,
                             y2,
                             cv=5,
                             scoring='roc_auc')
    scores_mean = (round(scores.mean(), 2))

    if Version(sklearn_version) < Version("0.21"):
        assert scores_mean == 0.96, scores_mean
    else:
        assert scores_mean == 0.93, scores_mean

    # another test
    meta = SVC(decision_function_shape='ovo')

    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_probas=True,
                              meta_classifier=meta)
    
    scores = cross_val_score(sclf,
                             X,
                             y2,
                             cv=5,
                             scoring='roc_auc')
    scores_mean = (round(scores.mean(), 2))

    if Version(sklearn_version) < Version("0.21"):
        assert scores_mean == 0.95, scores_mean
    else:
        assert scores_mean == 0.95, scores_mean
