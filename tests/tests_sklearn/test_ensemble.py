import numpy as np
from mlxtend.sklearn import EnsembleClassifier

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import datasets


iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target


def test_EnsembleClassifier():

    np.random.seed(123)    
    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier()
    clf3 = GaussianNB()
    eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard')

    scores = cross_validation.cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert(scores_mean == 0.94)


def test_EnsembleClassifier_weights():

    np.random.seed(123)    
    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier()
    clf3 = GaussianNB()
    eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[1,2,10])

    scores = cross_validation.cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert(scores_mean == 0.93)   

	

