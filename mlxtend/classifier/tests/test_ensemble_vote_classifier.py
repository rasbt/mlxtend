# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.classifier import EnsembleVoteClassifier


from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.MajorityVote import RandomForestClassifier
import numpy as np
from sklearn import datasets


iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target


def test_EnsembleVoteClassifier():

    np.random.seed(123)
    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier()
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='hard')

    scores = cross_validation.cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert(scores_mean == 0.94)


def test_EnsembleVoteClassifier_weights():

    np.random.seed(123)
    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier()
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[1,2,10])

    scores = cross_validation.cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert(scores_mean == 0.93)


def test_EnsembleVoteClassifier_gridsearch():

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft')

    params = {'logisticregression__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200],}

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)

    mean_scores = []
    for params, mean_score, scores in grid.grid_scores_:
        mean_scores.append(round(mean_score, 2))
    assert(mean_scores == [0.95, 0.96, 0.96, 0.95])


def test_EnsembleVoteClassifier_gridsearch_enumerate_names():

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf1, clf2], voting='soft')

    params = {'logisticregression-1__C': [1.0, 100.0],
              'logisticregression-2__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [5, 20],}

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    gs = grid.fit(iris.data, iris.target)
