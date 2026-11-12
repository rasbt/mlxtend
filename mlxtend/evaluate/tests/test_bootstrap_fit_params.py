# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from mlxtend.evaluate import bias_variance_decomp, bootstrap_point632_score


def _data():
    X = np.arange(12).reshape(-1, 1)
    y = np.array([0, 4, 1, 9, 2, 7, 3, 15, 8, 5, 20, 6], dtype=float)
    weights = np.array([1, 9, 2, 3, 1, 5, 8, 4, 20, 2, 3, 6], dtype=float)
    return X, y, weights


def _weight_container(weights, kind):
    if kind == "list":
        return weights.tolist()
    if kind == "series":
        return pd.Series(weights, index=np.arange(len(weights))[::-1] + 100)
    return weights.copy()


def _estimator(pipeline):
    estimator = DummyRegressor(strategy="mean")
    if pipeline:
        return make_pipeline(estimator), "dummyregressor__sample_weight"
    return estimator, "sample_weight"


def _bootstrap_means(y, weights, rounds, seed):
    rng = np.random.RandomState(seed)
    rows = np.arange(len(y))
    for _ in range(rounds):
        indices = rng.choice(rows, size=len(rows), replace=True)
        mean = np.dot(y[indices], weights[indices]) / weights[indices].sum()
        yield indices, mean


@pytest.mark.parametrize("kind", ["array", "list", "series"])
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("method", ["oob", ".632"])
def test_bootstrap_scores_resample_fit_weights(kind, pipeline, method):
    X, y, weights = _data()
    supplied_weights = _weight_container(weights, kind)
    estimator, weight_key = _estimator(pipeline)
    rounds, seed = 9, 17

    scores = bootstrap_point632_score(
        estimator,
        X,
        y,
        n_splits=rounds,
        method=method,
        random_seed=seed,
        **{weight_key: supplied_weights},
    )

    expected = []
    for indices, mean in _bootstrap_means(y, weights, rounds, seed):
        test = np.setdiff1d(np.arange(len(y)), indices)
        test_error = np.mean((y[test] - mean) ** 2)
        if method == "oob":
            expected.append(test_error)
        else:
            train_error = np.mean((y - mean) ** 2)
            expected.append(0.632 * test_error + 0.368 * train_error)

    assert_allclose(scores, expected, rtol=1e-12)
    assert_array_equal(supplied_weights, weights)
    original = estimator[-1] if pipeline else estimator
    assert not hasattr(original, "constant_")


@pytest.mark.parametrize("kind", ["array", "list", "series"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_bias_variance_resamples_fit_weights(kind, pipeline):
    X, y, weights = _data()
    supplied_weights = _weight_container(weights, kind)
    X_test = np.array([[13], [14], [15]])
    y_test = np.array([2.0, 11.0, 8.0])
    estimator, weight_key = _estimator(pipeline)
    rounds, seed = 9, 17

    result = bias_variance_decomp(
        estimator,
        X,
        y,
        X_test,
        y_test,
        loss="mse",
        num_rounds=rounds,
        random_seed=seed,
        **{weight_key: supplied_weights},
    )

    means = np.array([mean for _, mean in _bootstrap_means(y, weights, rounds, seed)])
    expected_loss = np.mean((means[:, None] - y_test) ** 2)
    expected_bias = np.mean((means.mean() - y_test) ** 2)
    expected_var = np.var(means)
    assert_allclose(result, [expected_loss, expected_bias, expected_var], rtol=1e-12)
    assert_allclose(result[0], result[1] + result[2], rtol=1e-12)
    assert_array_equal(supplied_weights, weights)


class FitParameterRegressor(LinearRegression):
    def fit(self, X, y, sample_weight=None, epochs=None, callbacks=None):
        self.fit_parameters_ = (sample_weight, epochs, callbacks)
        return super().fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("sample_weight", [None, 2.0])
@pytest.mark.parametrize("evaluation", ["bootstrap", "bias_variance"])
def test_bootstrap_preserves_non_sample_fit_parameters(sample_weight, evaluation):
    X, y, _ = _data()
    callbacks = [object(), object()]
    estimator = FitParameterRegressor()
    fit_params = dict(sample_weight=sample_weight, epochs=5, callbacks=callbacks)
    if evaluation == "bootstrap":
        bootstrap_point632_score(
            estimator,
            X,
            y,
            method="oob",
            n_splits=3,
            random_seed=17,
            clone_estimator=False,
            **fit_params,
        )
    else:
        bias_variance_decomp(
            estimator,
            X,
            y,
            X[:3],
            y[:3],
            loss="mse",
            num_rounds=3,
            random_seed=17,
            **fit_params,
        )

    passed_weight, epochs, passed_callbacks = estimator.fit_parameters_
    assert passed_weight is sample_weight
    assert epochs == 5
    assert passed_callbacks is callbacks
