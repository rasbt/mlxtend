# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from mlxtend.classifier import LogisticRegression as MlxtendLogisticRegression
from mlxtend.classifier import SoftmaxRegression
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


@pytest.fixture(
    params=[
        [0, 1],
        [-1, 1],
        [1, 2],
        ["no", "yes"],
        [0, 1, 2],
        ["a", "b", "c"],
    ],
    ids=[
        "binary",
        "negative",
        "one-based",
        "strings",
        "multiclass",
        "multiclass-strings",
    ],
)
def classification_data(request):
    y_train = np.tile(request.param, 20)
    y_test = np.tile(request.param, 3)
    return (
        np.zeros((len(y_train), 1)),
        y_train,
        np.zeros((len(y_test), 1)),
        y_test,
    )


def test_log_loss_uses_probabilities(classification_data):
    X_train, y_train, X_test, y_test = classification_data
    errors = plot_learning_curves(
        X_train,
        y_train,
        X_test,
        y_test,
        DummyClassifier(strategy="prior"),
        scoring="log_loss",
        suppress_plot=True,
    )

    assert np.shape(errors) == (2, 10)
    np.testing.assert_allclose(errors, np.log(np.unique(y_train).size))


def test_log_loss_matches_probability_loss_at_each_training_size():
    X_train = np.tile([[-2.0], [-1.0], [1.0], [2.0]], (10, 1))
    y_train = np.tile([0, 1, 0, 1], 10)
    X_test = np.array([[-3.0], [-0.5], [0.5], [3.0]])
    y_test = np.array([0, 0, 1, 1])
    clf = LogisticRegression(C=0.5, solver="liblinear", random_state=0)

    training_errors, test_errors = plot_learning_curves(
        X_train,
        y_train,
        X_test,
        y_test,
        clf,
        scoring="log_loss",
        suppress_plot=True,
    )

    expected_train = []
    expected_test = []
    for size in range(4, 41, 4):
        model = clone(clf).fit(X_train[:size], y_train[:size])
        expected_train.append(
            log_loss(y_train[:size], model.predict_proba(X_train[:size]))
        )
        expected_test.append(log_loss(y_test, model.predict_proba(X_test)))

    np.testing.assert_allclose(training_errors, expected_train)
    np.testing.assert_allclose(test_errors, expected_test)


@pytest.mark.parametrize(
    "model_class, n_classes",
    [(MlxtendLogisticRegression, 2), (SoftmaxRegression, 3)],
)
def test_log_loss_with_mlxtend_classifier(model_class, n_classes):
    classes = np.arange(n_classes)
    y_train = np.tile(classes, 20)
    y_test = np.tile(classes, 3)
    X_train = y_train.astype(float).reshape(-1, 1)
    X_test = y_test.astype(float).reshape(-1, 1) + 0.25
    clf = model_class(epochs=5, random_seed=0)

    training_errors, test_errors = plot_learning_curves(
        X_train,
        y_train,
        X_test,
        y_test,
        clf,
        scoring="log_loss",
        suppress_plot=True,
    )

    expected_train = []
    expected_test = []
    for size in range(2 * n_classes, len(y_train) + 1, 2 * n_classes):
        model = model_class(epochs=5, random_seed=0).fit(X_train[:size], y_train[:size])
        expected_train.append(
            log_loss(y_train[:size], model.predict_proba(X_train[:size]))
        )
        expected_test.append(log_loss(y_test, model.predict_proba(X_test)))

    np.testing.assert_allclose(training_errors, expected_train)
    np.testing.assert_allclose(test_errors, expected_test)


def test_log_loss_requires_predict_proba():
    X_train = np.tile([[-1.0], [1.0]], (10, 1))
    y_train = np.tile([0, 1], 10)

    with pytest.raises(AttributeError, match="predict_proba"):
        plot_learning_curves(
            X_train,
            y_train,
            X_train,
            y_train,
            LinearSVC(random_state=0),
            scoring="log_loss",
            suppress_plot=True,
        )


@pytest.mark.parametrize("scoring", ["accuracy", "misclassification error"])
def test_classification_score_controls(classification_data, scoring):
    X_train, y_train, X_test, y_test = classification_data
    errors = plot_learning_curves(
        X_train,
        y_train,
        X_test,
        y_test,
        DummyClassifier(strategy="prior"),
        scoring=scoring,
        suppress_plot=True,
    )

    expected = 1 / np.unique(y_train).size
    if scoring == "misclassification error":
        expected = 1 - expected
    assert np.shape(errors) == (2, 10)
    np.testing.assert_allclose(errors, expected)


@pytest.mark.parametrize(
    "scoring, expected_train, expected_test",
    [
        ("mean_squared_error", 5, 8 / 3),
        ("mean_absolute_error", 2, 4 / 3),
        ("median_absolute_error", 2, 2),
        ("r2", 0, 0),
    ],
)
def test_regression_score_controls(scoring, expected_train, expected_test):
    y_train = np.tile([-3.0, -1.0, 1.0, 3.0], 10)
    y_test = np.tile([-2.0, 0.0, 2.0], 3)
    training_errors, test_errors = plot_learning_curves(
        np.zeros((len(y_train), 1)),
        y_train,
        np.zeros((len(y_test), 1)),
        y_test,
        DummyRegressor(),
        scoring=scoring,
        suppress_plot=True,
    )

    np.testing.assert_allclose(training_errors, np.full(10, expected_train))
    np.testing.assert_allclose(test_errors, np.full(10, expected_test))


def test_log_loss_plot():
    y = np.tile([0, 1], 10)
    X = np.zeros((len(y), 1))
    fig, ax = plt.subplots()
    try:
        errors = plot_learning_curves(
            X, y, X, y, DummyClassifier(strategy="prior"), scoring="log_loss"
        )
        np.testing.assert_allclose(errors, np.log(2))
        assert ax.get_ylabel() == "Performance (log_loss)"
        assert [line.get_label() for line in ax.lines] == [
            "training set",
            "test set",
        ]
        for line in ax.lines:
            np.testing.assert_array_equal(line.get_xdata(), np.arange(10, 101, 10))
            np.testing.assert_allclose(line.get_ydata(), np.log(2))
    finally:
        plt.close(fig)


def test_unknown_scoring():
    with pytest.raises(AttributeError, match="scoring must be in"):
        plot_learning_curves(
            np.zeros((20, 1)),
            np.tile([0, 1], 10),
            np.zeros((2, 1)),
            np.array([0, 1]),
            DummyClassifier(),
            scoring="unknown",
            suppress_plot=True,
        )
