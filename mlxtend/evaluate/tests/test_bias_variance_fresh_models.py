import pickle
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxtend.evaluate import bias_variance_decomp


def _data(classification=False):
    X = np.linspace(-2, 2, 20).reshape(-1, 1)
    X_test = np.linspace(-1.9, 1.9, 11).reshape(-1, 1)
    if classification:
        y = (np.sin(4 * X[:, 0]) > 0).astype(int)
        y_test = (np.sin(4 * X_test[:, 0]) > 0).astype(int)
    else:
        y = np.sin(3 * X[:, 0]) + X[:, 0] ** 2
        y_test = np.sin(3 * X_test[:, 0]) + X_test[:, 0] ** 2
    return X, y, X_test, y_test


def _estimator(kind, warm_start, pipeline):
    classes = {
        "forest": RandomForestRegressor,
        "boosting": GradientBoostingRegressor,
        "classifier": RandomForestClassifier,
    }
    estimator = classes[kind](
        n_estimators=3, max_depth=2, warm_start=warm_start, random_state=17
    )
    return make_pipeline(StandardScaler(), estimator) if pipeline else estimator


def _independent_decomposition(kind, warm_start, pipeline, data):
    X, y, X_test, y_test = data
    rng = np.random.RandomState(42)
    predictions = []
    for _ in range(10):
        indices = rng.choice(len(X), size=len(X), replace=True)
        fitted = _estimator(kind, warm_start, pipeline).fit(X[indices], y[indices])
        predictions.append(fitted.predict(X_test))
    predictions = np.asarray(predictions)
    if kind == "classifier":
        mean_prediction = (np.sum(predictions, axis=0) > len(predictions) / 2).astype(
            int
        )
        return (
            np.mean(predictions != y_test),
            np.mean(mean_prediction != y_test),
            np.mean(predictions != mean_prediction),
        )
    mean_prediction = np.mean(predictions, axis=0)
    return (
        np.mean((predictions - y_test) ** 2),
        np.mean((mean_prediction - y_test) ** 2),
        np.mean((predictions - mean_prediction) ** 2),
    )


@pytest.mark.parametrize("kind", ["forest", "boosting", "classifier"])
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("prefitted", [False, True])
def test_warm_start_replicates_match_independent_fits(kind, pipeline, prefitted):
    data = _data(classification=kind == "classifier")
    X, y, X_test, y_test = data
    estimator = _estimator(kind, warm_start=True, pipeline=pipeline)
    if prefitted:
        estimator.fit(X, 1 - y if kind == "classifier" else y + 4.0)
    before = pickle.dumps(estimator)
    expected = _independent_decomposition(kind, True, pipeline, data)
    assert expected[2] > 0.001

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = bias_variance_decomp(
            estimator,
            X,
            y,
            X_test,
            y_test,
            loss="0-1_loss" if kind == "classifier" else "mse",
            num_rounds=10,
            random_seed=42,
        )

    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert pickle.dumps(estimator) == before
    assert not any("Warm-start fitting" in str(warning.message) for warning in seen)


@pytest.mark.parametrize("kind", ["forest", "boosting", "classifier"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_cold_start_statistics_keep_their_existing_values(kind, pipeline):
    data = _data(classification=kind == "classifier")
    expected = _independent_decomposition(kind, False, pipeline, data)
    result = bias_variance_decomp(
        _estimator(kind, warm_start=False, pipeline=pipeline),
        *data,
        loss="0-1_loss" if kind == "classifier" else "mse",
        num_rounds=10,
        random_seed=42,
    )
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_fit_parameters_reach_each_fresh_estimator():
    data = _data()
    X, y, X_test, y_test = data
    estimator = _estimator("forest", warm_start=True, pipeline=True)
    result = bias_variance_decomp(
        estimator,
        X,
        y,
        X_test,
        y_test,
        loss="mse",
        num_rounds=10,
        random_seed=42,
        randomforestregressor__sample_weight=np.ones(len(X)),
    )
    expected = _independent_decomposition("forest", True, True, data)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
