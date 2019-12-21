# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import random
import pytest
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from mlxtend.regressor import EnsembleVotingRegressor
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from mlxtend.data import iris_data
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

X, y = iris_data()
X = X[:, 1:3]


class EnsembleVoteRegressor(object):
    pass


def test_EnsembleVoteRegressor():

    np.random.seed(123)
    clf1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    clf2 = RandomForestRegressor(random_state=1, n_estimators=10)
    clf3 = LinearRegression()
    eclf = EnsembleVoteRegressor(clfs=[clf1, clf2, clf3], voting='hard')

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
    clf1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    clf2 = RandomForestRegressor(random_state=1, n_estimators=10)
    clf3 = LinearRegression()
    eclf = EnsembleVoteRegressor(clfs=[clf1, clf2, clf3], voting='hard')
    prob1 = eclf.fit(X, y).predict_proba(X)

    # with weight = 1
    w = np.ones(len(y))
    np.random.seed(123)
    clf1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    clf2 = RandomForestRegressor(random_state=1, n_estimators=10)
    clf3 = LinearRegression()
    eclf = EnsembleVoteRegressor(clfs=[clf1, clf2, clf3], voting='hard')
    prob2 = eclf.fit(X, y, sample_weight=w).predict_proba(X)

    # with random weight
    random.seed(87)
    w = np.array([random.random() for _ in range(len(y))])
    np.random.seed(123)
    clf1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    clf2 = RandomForestRegressor(random_state=1, n_estimators=10)
    clf3 = LinearRegression()
    eclf = EnsembleVoteRegressor(clfs=[clf1, clf2, clf3], voting='hard')
    prob3 = eclf.fit(X, y, sample_weight=w).predict_proba(X)

    diff12 = np.max(np.abs(prob1 - prob2))
    diff23 = np.max(np.abs(prob2 - prob3))
    assert diff12 < 1e-3, "max diff is %.4f" % diff12
    assert diff23 > 1e-3, "max diff is %.4f" % diff23


def test_no_weight_support():
    random.seed(87)
    w = np.array([random.random() for _ in range(len(y))])
    gbr = GradientBoostingRegressor(random_state=1, n_estimators=10)
    rf = RandomForestRegressor(random_state=1, n_estimators=10)
    lr = LinearRegression()
    eclf = EnsembleVoteRegressor(clfs=[gbr, rf, lr], voting='hard')
    with pytest.raises(TypeError):
        eclf.fit(X, y, sample_weight=w)


def test_no_weight_support_with_no_weight():
    gbr = GradientBoostingRegressor(random_state=1, n_estimators=10)
    rf = RandomForestRegressor(random_state=1, n_estimators=10)
    lr = LinearRegression()
    eclf = EnsembleVoteRegressor(clfs=[gbr, rf, lr], voting='hard')
    eclf.fit(X, y)


def test_1model_labels():
    clf = GradientBoostingRegressor(random_state=123, n_estimators=10)
    ens_clf_1 = EnsembleVoteRegressor(clfs=[clf], voting='soft', weights=None)
    ens_clf_2 = EnsembleVoteRegressor(clfs=[clf], voting='soft', weights=[1.])

    pred_e1 = ens_clf_1.fit(X, y).predict(X)
    pred_e2 = ens_clf_2.fit(X, y).predict(X)
    pred_e3 = clf.fit(X, y).predict(X)

    np.testing.assert_equal(pred_e1, pred_e2)
    np.testing.assert_equal(pred_e1, pred_e3)


def test_1model_probas():
    clf = GradientBoostingRegressor(random_state=123, n_estimators=10)
    ens_clf_1 = EnsembleVoteRegressor(clfs=[clf], voting='soft', weights=None)
    ens_clf_2 = EnsembleVoteRegressor(clfs=[clf], voting='soft', weights=[1.])

    pred_e1 = ens_clf_1.fit(X, y).predict_proba(X)
    pred_e2 = ens_clf_2.fit(X, y).predict_proba(X)
    pred_e3 = clf.fit(X, y).predict_proba(X)

    np.testing.assert_almost_equal(pred_e1, pred_e2, decimal=8)
    np.testing.assert_almost_equal(pred_e1, pred_e3, decimal=8)


def test_EnsembleVoteRegressor_weights():

    np.random.seed(123)
    clf1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    clf2 = RandomForestRegressor(random_state=1, n_estimators=10)
    clf3 = LinearRegression()
    eclf = EnsembleVoteRegressor(clfs=[clf1, clf2, clf3],
                                  voting='soft',
                                  weights=[1, 2, 10])

    scores = cross_val_score(eclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert(scores_mean == 0.93)


def test_EnsembleVoteRegressor_gridsearch():

    clf1 = GradientBoostingRegressor(random_state=1)
    clf2 = RandomForestRegressor(random_state=1)
    clf3 = LinearRegression()
    eclf = EnsembleVoteRegressor(clfs=[clf1, clf2, clf3], voting='soft')

    params = {'GradientBoostingRegressor__n_estimators': [20, 200],
              'RandomForestRegressor__n_estimators': [20, 200]}

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5, iid=False)

    X, y = iris_data()
    grid.fit(X, y)

    mean_scores = [round(s, 2) for s
                   in grid.cv_results_['mean_test_score']]

    assert mean_scores == [0.95, 0.96, 0.96, 0.95]


def test_EnsembleVoteRegressor_gridsearch_enumerate_names():

    clf1 = GradientBoostingRegressor(random_state=1)
    clf2 = EnsembleVoteRegressor(random_state=1)
    eclf = EnsembleVoteRegressor(clfs=[clf1, clf1, clf2])

    params = {'GradientBoostingRegressor-1__n_estimators': [20, 200],
              'GradientBoostingRegressor-2__n_estimators': [20, 200],
              'RandomForestRegressor__n_estimators': [20, 200],
              'voting': ['hard', 'soft']}

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5, iid=False)

    X, y = iris_data()
    grid = grid.fit(X, y)


def test_get_params():
    clf1 = KNeighborsRegressor(n_neighbors=1)
    clf2 = RandomForestRegressor(random_state=1, n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteRegressor(clfs=[clf1, clf2, clf3])

    got = sorted(list({s.split('__')[0] for s in eclf.get_params().keys()}))
    expect = ['clfs',
              'gaussiannb',
              'kneighborsregressor',
              'randomforestregressor',
              'refit',
              'verbose',
              'voting',
              'weights']
    assert got == expect, got


def test_classifier_gridsearch():
    clf1 = KNeighborsRegressor(n_neighbors=1)
    clf2 = RandomForestRegressor(random_state=1, n_estimators=10)
    clf3 = GaussianNB()
    eclf = EnsembleVoteRegressor(clfs=[clf1])

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
    clf1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    clf2 = RandomForestRegressor(random_state=1, n_estimators=10)
    clf3 = LinearRegression()
    eclf = EnsembleVoteRegressor(clfs=[clf1, clf2, clf3], voting='hard')

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
    clf1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    clf2 = RandomForestRegressor(random_state=1, n_estimators=10)
    clf3 = LinearRegression()

    y_str = y.copy()
    y_str = y_str.astype(str)
    y_str[:50] = 'a'
    y_str[50:100] = 'b'
    y_str[100:150] = 'c'

    clf1.fit(X, y_str)
    clf2.fit(X, y_str)
    clf3.fit(X, y_str)

    eclf = EnsembleVoteRegressor(clfs=[clf1, clf2, clf3],
                                  voting='hard',
                                  refit=False)

    eclf.fit(X, y_str)
    assert round(eclf.score(X, y_str), 2) == 0.97

    eclf = EnsembleVoteRegressor(clfs=[clf1, clf2, clf3],
                                  voting='soft',
                                  refit=False)

    eclf.fit(X, y_str)
    assert round(eclf.score(X, y_str), 2) == 0.97


def test_clone():

    clf1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    clf2 = RandomForestRegressor(random_state=1, n_estimators=10)
    clf3 = LinearRegression()
    eclf = EnsembleVoteRegressor(clfs=[clf1, clf2, clf3],
                                  voting='hard',
                                  refit=False)
    clone(eclf)
