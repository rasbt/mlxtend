import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlxtend.evaluate import bootstrap_point632_score


@pytest.mark.parametrize("seed", [0, 7, 123])
@pytest.mark.parametrize("clone_estimator", [False, True])
def test_oob_binary_probabilities_match_manual_bootstrap(seed, clone_estimator):
    X, y = make_classification(
        n_samples=80, n_features=4, n_redundant=0, random_state=23
    )
    estimator = LogisticRegression(solver="liblinear", random_state=5)
    rng = np.random.RandomState(seed)
    expected = []
    for _ in range(5):
        train = rng.choice(len(y), size=len(y), replace=True)
        test = np.setdiff1d(np.arange(len(y)), train)
        model = clone(estimator).fit(X[train], y[train])
        expected.append(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))

    actual = bootstrap_point632_score(
        estimator,
        X,
        y,
        n_splits=5,
        method="oob",
        scoring_func=roc_auc_score,
        predict_proba=True,
        random_seed=seed,
        clone_estimator=clone_estimator,
    )

    np.testing.assert_allclose(actual, expected)
    assert hasattr(estimator, "classes_") is not clone_estimator


@pytest.mark.parametrize("method", ["oob", ".632"])
def test_bootstrap_probability_shapes(method):
    X, y = make_classification(
        n_samples=90,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=23,
    )
    prediction_shapes = []

    def score(targets, probabilities):
        prediction_shapes.append(probabilities.shape)
        assert probabilities.shape == (len(targets), 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        return np.mean(probabilities[np.arange(len(targets)), targets])

    result = bootstrap_point632_score(
        LogisticRegression(),
        X,
        y,
        n_splits=3,
        method=method,
        scoring_func=score,
        predict_proba=True,
        random_seed=7,
    )

    assert len(prediction_shapes) == (3 if method == "oob" else 6)
    assert result.shape == (3,)
    assert np.isfinite(result).all()
