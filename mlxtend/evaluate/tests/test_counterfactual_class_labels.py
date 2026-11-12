# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.linear_model import LogisticRegression

from mlxtend.evaluate import create_counterfactual


def _problem():
    X = np.array([[-4.0], [-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [4.0]])
    return X, np.array([0, 0, 0, 0, 1, 1, 1, 1])


def _counterfactual(model, X, target, column):
    return create_counterfactual(
        x_reference=np.array([3.0 if column == 0 else -3.0]),
        y_desired=target,
        model=model,
        X_dataset=X,
        y_desired_proba=0.9,
        lammbda=100,
        random_seed=17,
    )


@pytest.mark.parametrize("labels", [[0, 1], [10, 20], [-1, 1], ["cold", "warm"]])
@pytest.mark.parametrize("column", [0, 1])
def test_probability_counterfactual_is_invariant_to_class_relabeling(labels, column):
    X, y = _problem()
    labels = np.asarray(labels)
    reference_model = LogisticRegression().fit(X, y)
    relabeled_model = LogisticRegression().fit(X, labels[y])
    expected = _counterfactual(reference_model, X, column, column)

    actual = _counterfactual(relabeled_model, X, labels[column], column)

    assert_allclose(actual, expected, rtol=0, atol=1e-8)
    assert relabeled_model.predict(actual.reshape(1, -1))[0] == labels[column]


def test_probability_counterfactual_rejects_an_unknown_class_label():
    X, y = _problem()
    model = LogisticRegression().fit(X, y)

    with pytest.raises(ValueError, match="y_desired"):
        _counterfactual(model, X, 7, 1)


class ProbabilitiesOnly:
    def __init__(self, estimator):
        self.estimator = estimator

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def test_probability_counterfactual_retains_positional_fallback_without_classes():
    X, y = _problem()
    model = LogisticRegression().fit(X, y)
    expected = _counterfactual(model, X, 1, 1)

    actual = _counterfactual(ProbabilitiesOnly(model), X, 1, 1)

    assert_allclose(actual, expected, rtol=0, atol=1e-8)
