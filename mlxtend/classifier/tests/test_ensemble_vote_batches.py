import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxtend.classifier import EnsembleVoteClassifier


class ColumnClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, column=0):
        self.column = column

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.classes_[np.asarray(X[:, self.column], dtype=int)]


def _rowwise_oracle(model, X):
    predictions = model.transform(X)
    modes = [np.argmax(np.bincount(row, weights=model.weights)) for row in predictions]
    return model.le_.inverse_transform(modes)


@pytest.mark.parametrize(
    "weights", [None, [1, 1, 1, 1, 1], [0, 2, 0.25, 1, 0.5], [0] * 5]
)
@pytest.mark.parametrize(
    "labels", [[0, 1, 2, 3], [-5, 2, 19, 24], ["a", "b", "d", "z"]]
)
@pytest.mark.parametrize("prefit", [False, True])
def test_batched_votes_match_rowwise_oracle_across_block_boundaries(
    weights, labels, prefit
):
    rng = np.random.RandomState(11)
    X_fit = rng.randint(0, 4, size=(4, 5))
    labels = np.asarray(labels)
    classifiers = [ColumnClassifier(i) for i in range(5)]
    if prefit:
        for classifier in classifiers:
            classifier.fit(X_fit, labels)
    model = EnsembleVoteClassifier(
        classifiers, voting="hard", weights=weights, fit_base_estimators=not prefit
    ).fit(X_fit, labels)
    X = rng.randint(0, 4, size=(9001, 5))
    X[0] = [3, 2, 1, 0, 0]
    X[-1] = [2, 1, 3, 1, 2]
    original = X.copy()

    np.testing.assert_array_equal(model.predict(X), _rowwise_oracle(model, X))
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("weights", [[1, -2, -3], [np.nan, 1, 2], [np.inf, 1, 2]])
def test_unusual_weights_retain_rowwise_semantics(weights):
    X = np.array([[0, 1, 2], [1, 1, 0], [2, 2, 2]])
    model = EnsembleVoteClassifier(
        [ColumnClassifier(i) for i in range(3)], weights=weights
    ).fit(X, np.arange(3))
    with np.errstate(invalid="ignore"):
        np.testing.assert_array_equal(model.predict(X), _rowwise_oracle(model, X))


def test_ties_still_select_the_smallest_encoded_label():
    X = np.array([[2, 1], [1, 2], [2, 0], [0, 2]])
    model = EnsembleVoteClassifier([ColumnClassifier(0), ColumnClassifier(1)]).fit(
        np.zeros((3, 2)), np.array(["a", "b", "c"])
    )
    np.testing.assert_array_equal(model.predict(X), ["b", "b", "a", "a"])


def test_real_fitted_classifiers_keep_weighted_predictions():
    X, y = load_iris(return_X_y=True)
    model = EnsembleVoteClassifier(
        [
            LogisticRegression(max_iter=500),
            DecisionTreeClassifier(max_depth=1, random_state=2),
            DecisionTreeClassifier(max_depth=3, random_state=2),
        ],
        weights=[1, 0.25, 2],
    ).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), _rowwise_oracle(model, X))


def test_empty_prediction_batch_keeps_the_existing_error():
    model = EnsembleVoteClassifier([ColumnClassifier()]).fit(
        np.zeros((2, 1)), np.arange(2)
    )
    with pytest.raises(ValueError):
        model.predict(np.empty((0, 1), dtype=int))
