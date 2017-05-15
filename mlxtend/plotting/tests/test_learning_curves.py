# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.plotting import plot_learning_curves
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split


def test_training_size():

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = (train_test_split(X, y,
                                        train_size=0.6, random_state=2))

    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    training_errors, test_errors = (plot_learning_curves(X_train, y_train,
                                    X_test, y_test, clf, suppress_plot=True))

    desired1 = [0.22, 0.22, 0.22, 0.31, 0.31, 0.3, 0.33, 0.32, 0.33, 0.32]
    desired2 = [0.45, 0.45, 0.35, 0.35, 0.45, 0.43, 0.35, 0.35, 0.35, 0.35]

    np.testing.assert_almost_equal(training_errors, desired1, decimal=2)
    np.testing.assert_almost_equal(test_errors, desired2, decimal=2)


def test_log_loss():

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = (train_test_split(X, y,
                                        train_size=0.6, random_state=2))

    clf = LogisticRegression(random_state=1)
    training_errors, test_errors = (plot_learning_curves(X_train, y_train,
                                    X_test, y_test, clf, scoring = "log_loss",
                                    suppress_plot=True))

    desired1 = [0.61, 0.49, 0.41, 0.39, 0.38, 0.37, 0.39, 0.37, 0.36, 0.34]
    desired2 = [0.56, 0.51, 0.45, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34]

    np.testing.assert_almost_equal(training_errors, desired1, decimal=2)
    np.testing.assert_almost_equal(test_errors, desired2, decimal=2)


def test_scikit_metrics():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.6,
                                                        random_state=2)

    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    training_errors, test_errors = plot_learning_curves(X_train,
                                                        y_train,
                                                        X_test,
                                                        y_test,
                                                        clf,
                                                        suppress_plot=True,
                                                        scoring='accuracy')

    desired1 = [0.78, 0.78, 0.78, 0.69, 0.69, 0.7, 0.67, 0.68, 0.67, 0.68]
    desired2 = [0.55, 0.55, 0.65, 0.65, 0.55, 0.57, 0.65, 0.65, 0.65, 0.65]

    np.testing.assert_almost_equal(training_errors, desired1, decimal=2)
    np.testing.assert_almost_equal(test_errors, desired2, decimal=2)
