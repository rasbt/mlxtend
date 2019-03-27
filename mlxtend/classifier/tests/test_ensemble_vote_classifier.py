# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import random
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from mlxtend.data import iris_data
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from nose.tools import raises

X, y = iris_data()
X = X[:, 1:3]


def test_EnsembleVoteClassifier():

    np.random.seed(123)
    clf1 = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='hard')

    scores = cross_val_score(eclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert(scores_mean == 0.94)


def test_sample_weight():
    # with no weight
    np.random.seed(123)
    clf1 = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='hard')
    prob1 = eclf.fit(X, y).predict_proba(X)

    # with weight = 1
    w = np.ones(len(y))
    np.random.seed(123)
    clf1 = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='hard')
    prob2 = eclf.fit(X, y, sample_weight=w).predict_proba(X)

    # with random weight
    random.seed(87)
    w = np.array([random.random() for _ in range(len(y))])
    np.random.seed(123)
    clf1 = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='hard')
    prob3 = eclf.fit(X, y, sample_weight=w).predict_proba(X)

    diff12 = np.max(np.abs(prob1 - prob2))
    diff23 = np.max(np.abs(prob2 - prob3))
    assert diff12 < 1e-3, "max diff is %.4f" % diff12
    assert diff23 > 1e-3, "max diff is %.4f" % diff23


@raises(TypeError)
def test_no_weight_support():
    random.seed(87)
    w = np.array([random.random() for _ in range(len(y))])
    logi = LogisticRegression(solver='liblinear', multi_class='ovr')
    rf = RandomForestClassifier(n_estimators=10)
    gnb = GaussianNB()
    knn = KNeighborsClassifier()
    eclf = EnsembleVoteClassifier(clfs=[logi, rf, gnb, knn], voting='hard')
    eclf.fit(X, y, sample_weight=w)


def test_no_weight_support_with_no_weight():
    logi = LogisticRegression(solver='liblinear', multi_class='ovr')
    rf = RandomForestClassifier(n_estimators=10)
    gnb = GaussianNB()
    knn = KNeighborsClassifier()
    eclf = EnsembleVoteClassifier(clfs=[logi, rf, gnb, knn], voting='hard')
    eclf.fit(X, y)


def test_1model_labels():
    clf = LogisticRegression(multi_class='multinomial',
                             solver='newton-cg', random_state=123)
    ens_clf_1 = EnsembleVoteClassifier(clfs=[clf], voting='soft', weights=None)
    ens_clf_2 = EnsembleVoteClassifier(clfs=[clf], voting='soft', weights=[1.])

    pred_e1 = ens_clf_1.fit(X, y).predict(X)
    pred_e2 = ens_clf_2.fit(X, y).predict(X)
    pred_e3 = clf.fit(X, y).predict(X)

    np.testing.assert_equal(pred_e1, pred_e2)
    np.testing.assert_equal(pred_e1, pred_e3)


def test_1model_probas():
    clf = LogisticRegression(multi_class='multinomial',
                             solver='newton-cg', random_state=123)
    ens_clf_1 = EnsembleVoteClassifier(clfs=[clf], voting='soft', weights=None)
    ens_clf_2 = EnsembleVoteClassifier(clfs=[clf], voting='soft', weights=[1.])

    pred_e1 = ens_clf_1.fit(X, y).predict_proba(X)
    pred_e2 = ens_clf_2.fit(X, y).predict_proba(X)
    pred_e3 = clf.fit(X, y).predict_proba(X)

    np.testing.assert_almost_equal(pred_e1, pred_e2, decimal=8)
    np.testing.assert_almost_equal(pred_e1, pred_e3, decimal=8)


def test_EnsembleVoteClassifier_weights():

    np.random.seed(123)
    clf1 = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
                                  voting='soft',
                                  weights=[1, 2, 10])

    scores = cross_val_score(eclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert(scores_mean == 0.93)


def test_EnsembleVoteClassifier_gridsearch():

    clf1 = LogisticRegression(solver='liblinear',
                              multi_class='ovr',
                              random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft')

    params = {'logisticregression__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200]}

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5, iid=False)

    X, y = iris_data()
    grid.fit(X, y)

    mean_scores = [round(s, 2) for s
                   in grid.cv_results_['mean_test_score']]

    assert mean_scores == [0.95, 0.96, 0.96, 0.95]


def test_EnsembleVoteClassifier_gridsearch_enumerate_names():

    clf1 = LogisticRegression(solver='liblinear',
                              multi_class='ovr',
                              random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf1, clf2])

    params = {'logisticregression-1__C': [1.0, 100.0],
              'logisticregression-2__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [5, 20],
              'voting': ['hard', 'soft']}

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5, iid=False)

    X, y = iris_data()
    grid = grid.fit(X, y)


def test_get_params():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1, n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3])

    got = sorted(list({s.split('__')[0] for s in eclf.get_params().keys()}))
    expect = ['clfs',
              'gaussiannb',
              'kneighborsclassifier',
              'randomforestclassifier',
              'refit',
              'verbose',
              'voting',
              'weights']
    assert got == expect, got


def test_classifier_gridsearch():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1, n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1])

    params = {'clfs': [[clf1, clf1, clf1], [clf2, clf3]]}

    grid = GridSearchCV(estimator=eclf,
                        param_grid=params,
                        iid=False,
                        cv=5,
                        refit=True)
    grid.fit(X, y)

    assert len(grid.best_params_['clfs']) == 2


def test_string_labels_numpy_array():
    np.random.seed(123)
    clf1 = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='hard')

    y_str = y.copy()
    y_str = y_str.astype(str)
    y_str[:50] = 'a'
    y_str[50:100] = 'b'
    y_str[100:150] = 'c'

    scores = cross_val_score(eclf,
                             X,
                             y_str,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert(scores_mean == 0.94)


def test_string_labels_python_list():
    np.random.seed(123)
    clf1 = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='hard')

    y_str = (['a' for a in range(50)] +
             ['b' for a in range(50)] +
             ['c' for a in range(50)])

    scores = cross_val_score(eclf,
                             X,
                             y_str,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert(scores_mean == 0.94)


def test_string_labels_refit_false():
    np.random.seed(123)
    clf1 = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()

    y_str = y.copy()
    y_str = y_str.astype(str)
    y_str[:50] = 'a'
    y_str[50:100] = 'b'
    y_str[100:150] = 'c'

    clf1.fit(X, y_str)
    clf2.fit(X, y_str)
    clf3.fit(X, y_str)

    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
                                  voting='hard',
                                  refit=False)

    eclf.fit(X, y_str)
    assert round(eclf.score(X, y_str), 2) == 0.97

    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
                                  voting='soft',
                                  refit=False)

    eclf.fit(X, y_str)
    assert round(eclf.score(X, y_str), 2) == 0.97


def test_clone():

    clf1 = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
                                  voting='hard',
                                  refit=False)
    clone(eclf)
