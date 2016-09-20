# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Authors: Sebastian Raschka <sebastianraschka.com>
#          Reiichiro Nakano <github.com/reiinakano>
#
# License: BSD 3 clause

from mlxtend.classifier import StackingCVClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import NotFittedError
import numpy as np
from sklearn import datasets
from mlxtend.utils import assert_raises


iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target


def test_StackingClassifier():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta)

    scores = cross_validation.cross_val_score(sclf,
                                              X,
                                              y,
                                              cv=5,
                                              scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.94


def test_StackingClassifier_proba():

    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta)

    scores = cross_validation.cross_val_score(sclf,
                                              X,
                                              y,
                                              cv=5,
                                              scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.94


def test_gridsearch():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)

    mean_scores = []
    for params, mean_score, scores in grid.grid_scores_:
        mean_scores.append(round(mean_score, 2))
    assert mean_scores == [0.96, 0.95, 0.96, 0.95]


def test_gridsearch_enumerate_names():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf1, clf2],
                                meta_classifier=meta)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier-1__n_estimators': [5, 10],
              'randomforestclassifier-2__n_estimators': [5, 20]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    grid = grid.fit(iris.data, iris.target)


def test_do_not_use_probas():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=False,
                                meta_classifier=meta)

    scores = cross_validation.cross_val_score(sclf,
                                              X,
                                              y,
                                              cv=5,
                                              scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.93, scores_mean


def test_use_features_in_secondary():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_features_in_secondary=True,
                                meta_classifier=meta)

    scores = cross_validation.cross_val_score(sclf,
                                              X,
                                              y,
                                              cv=5,
                                              scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.95, scores_mean


def test_not_fitted():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta)

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
                                verbose=3)
    sclf.fit(iris.data, iris.target)
