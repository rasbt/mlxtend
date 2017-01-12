# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Authors: Sebastian Raschka <sebastianraschka.com>
#          Reiichiro Nakano <github.com/reiinakano>
#
# License: BSD 3 clause

from mlxtend.classifier import StackingCVClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import datasets
from mlxtend.utils import assert_raises
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target


def test_StackingClassifier():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.93


def test_StackingClassifier_proba():

    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.93


def test_gridsearch():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                use_probas=True,
                                shuffle=False)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)

    mean_scores = [round(s, 2) for s
                   in grid.cv_results_['mean_test_score']]

    assert mean_scores == [0.96, 0.95, 0.96, 0.95]


def test_gridsearch_enumerate_names():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier-1__n_estimators': [5, 10],
              'randomforestclassifier-2__n_estimators': [5, 20]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    grid = grid.fit(iris.data, iris.target)


def test_use_probas():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.94, scores_mean


def test_use_features_in_secondary():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_features_in_secondary=True,
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.93, scores_mean


def test_do_not_stratify():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                stratify=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.94


def test_not_fitted():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta, shuffle=False)

    assert_raises(NotFittedError,
                  "This StackingCVClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict,
                  iris.data)

    assert_raises(NotFittedError,
                  "This StackingCVClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict_proba,
                  iris.data)


def test_verbose():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta,
                                shuffle=False,
                                verbose=3)
    sclf.fit(iris.data, iris.target)
