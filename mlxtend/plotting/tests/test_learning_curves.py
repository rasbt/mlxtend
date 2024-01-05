# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.plotting import plot_learning_curves


def test_training_size():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=2
    )

    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    training_errors, test_errors = plot_learning_curves(
        X_train, y_train, X_test, y_test, clf, suppress_plot=True
    )

    desired1 = [0.22, 0.22, 0.22, 0.31, 0.31, 0.3, 0.33, 0.32, 0.33, 0.32]
    desired2 = [0.45, 0.45, 0.35, 0.35, 0.45, 0.43, 0.35, 0.35, 0.35, 0.35]

    np.testing.assert_almost_equal(training_errors, desired1, decimal=2)
    np.testing.assert_almost_equal(test_errors, desired2, decimal=2)


def test_scikit_metrics():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=2
    )

    clf = DecisionTreeClassifier(max_depth=1, random_state=1)
    training_acc, test_acc = plot_learning_curves(
        X_train, y_train, X_test, y_test, clf, scoring="accuracy", suppress_plot=True
    )

    desired1 = np.array([0.22, 0.22, 0.22, 0.31, 0.31, 0.3, 0.33, 0.32, 0.33, 0.32])
    desired2 = np.array([0.45, 0.45, 0.35, 0.35, 0.45, 0.43, 0.35, 0.35, 0.35, 0.35])
    np.testing.assert_almost_equal(training_acc, 1 - desired1, decimal=2)
    np.testing.assert_almost_equal(test_acc, 1 - desired2, decimal=2)
