# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import NotFittedError
import numpy as np
from sklearn import datasets
from mlxtend.utils import assert_raises
from nose.tools import assert_almost_equal
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
y2 = np.c_[y, y]


def test_StackingClassifier():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              meta_classifier=meta)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.95


def test_StackingClassifier_proba_avg_1():

    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
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
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
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
    lr1 = LogisticRegression()
    sclf1 = StackingClassifier(classifiers=[lr1, lr1],
                               use_probas=True,
                               average_probas=True,
                               meta_classifier=lr1)

    sclf1.fit(X, y)
    r1 = sclf1.predict_meta_features(X[:2])
    assert r1.shape == (2, 3)
    assert_almost_equal(np.sum(r1[0]), 1.0, places=6)
    assert_almost_equal(np.sum(r1[1]), 1.0, places=6)

    sclf2 = StackingClassifier(classifiers=[lr1, lr1],
                               use_probas=True,
                               average_probas=False,
                               meta_classifier=lr1)

    sclf2.fit(X, y)
    r2 = sclf2.predict_meta_features(X[:2])
    assert r2.shape == (2, 6)
    assert_almost_equal(np.sum(r2[0]), 2.0, places=6)
    assert_almost_equal(np.sum(r2[1]), 2.0, places=6)
    np.array_equal(r2[0][:3], r2[0][3:])


def test_multivariate_class():
    np.random.seed(123)
    meta = KNeighborsClassifier()
    clf1 = RandomForestClassifier()
    clf2 = KNeighborsClassifier()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              meta_classifier=meta)
    y_pred = sclf.fit(X, y2).predict(X)
    ca = .973
    assert round((y_pred == y2).mean(), 3) == ca


def test_gridsearch():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              meta_classifier=meta)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)

    mean_scores = [round(s, 2) for s
                   in grid.cv_results_['mean_test_score']]

    assert mean_scores == [0.95, 0.97, 0.96, 0.96]


def test_gridsearch_enumerate_names():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf1, clf2],
                              meta_classifier=meta)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier-1__n_estimators': [5, 10],
              'randomforestclassifier-2__n_estimators': [5, 20],
              'use_probas': [True, False]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    grid = grid.fit(iris.data, iris.target)


def test_use_probas():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
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
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_probas=True,
                              meta_classifier=meta)

    assert_raises(NotFittedError,
                  "This StackingClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict,
                  iris.data)

    assert_raises(NotFittedError,
                  "This StackingClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict_proba,
                  iris.data)


def test_verbose():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_probas=True,
                              meta_classifier=meta,
                              verbose=3)
    sclf.fit(iris.data, iris.target)


def test_use_features_in_secondary_predict():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
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
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingClassifier(classifiers=[clf1, clf2],
                              use_features_in_secondary=True,
                              meta_classifier=meta)

    sclf.fit(X, y)
    idx = [0, 1, 2]
    y_pred = sclf.predict_proba(X[idx])[:, 0]
    expect = np.array([0.911, 0.829, 0.885])
    np.testing.assert_almost_equal(y_pred, expect, 3)


def test_get_params():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              meta_classifier=lr)

    got = sorted(list({s.split('__')[0] for s in sclf.get_params().keys()}))
    expect = ['average_probas',
              'classifiers',
              'gaussiannb',
              'kneighborsclassifier',
              'meta-logisticregression',
              'meta_classifier',
              'randomforestclassifier',
              'store_train_meta_features',
              'use_features_in_secondary',
              'use_probas',
              'verbose']
    assert got == expect, got


def test_classifier_gridsearch():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              meta_classifier=lr)

    params = {'classifiers': [[clf1, clf1, clf1], [clf2, clf3]]}

    grid = GridSearchCV(estimator=sclf,
                        param_grid=params,
                        cv=5,
                        refit=True)
    grid.fit(X, y)

    assert len(grid.best_params_['classifiers']) == 2


def test_train_meta_features_():
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    gnb = GaussianNB()
    stclf = StackingClassifier(classifiers=[knn, gnb],
                               meta_classifier=lr,
                               store_train_meta_features=True)
    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3)
    stclf.fit(X_train, y_train)
    train_meta_features = stclf.train_meta_features_
    assert train_meta_features.shape == (X_train.shape[0], 2)


def test_predict_meta_features():
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    gnb = GaussianNB()
    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3)

    #  test default (class labels)
    stclf = StackingClassifier(classifiers=[knn, gnb],
                               meta_classifier=lr,
                               store_train_meta_features=True)
    stclf.fit(X_train, y_train)
    test_meta_features = stclf.predict(X_test)
    assert test_meta_features.shape == (X_test.shape[0],)
