# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# License: BSD 3 clause

import gc
import tracemalloc
from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from mlxtend.evaluate import bias_variance_decomp


class _MeanRegressor:
    def __init__(self, shape="vector"):
        self.shape = shape

    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        if self.shape == "scalar":
            return self.mean_
        values = np.full(len(X), self.mean_)
        return values[None, :] if self.shape == "row" else values


class _SequenceRegressor:
    def __init__(self, values):
        self.values = iter(values)

    def fit(self, X, y):
        self.prediction_ = next(self.values)
        return self

    def predict(self, X):
        return np.full(len(X), self.prediction_)


def _materialized_statistics(predictions, target):
    predictions = np.asarray(predictions, dtype=float)
    mean = predictions.mean(axis=0)
    return (
        np.mean((predictions - target) ** 2),
        np.mean((mean - target) ** 2),
        np.mean((predictions - mean) ** 2),
    )


@pytest.mark.parametrize("rounds", [1, 5, 17])
@pytest.mark.parametrize(
    "model",
    [
        LinearRegression(),
        DecisionTreeRegressor(max_depth=3, random_state=11),
        RandomForestRegressor(n_estimators=3, max_depth=2, random_state=17),
    ],
)
def test_streaming_statistics_match_independently_materialized_predictions(
    model, rounds
):
    X = np.linspace(-2, 2, 40)[:, None]
    y = np.sin(3 * X[:, 0]) + X[:, 0] ** 2
    X_test = np.linspace(-1.9, 1.9, 11)[:, None]
    y_test = np.sin(3 * X_test[:, 0]) + X_test[:, 0] ** 2
    rng = np.random.RandomState(7)
    predictions = []
    for _ in range(rounds):
        indices = rng.choice(len(X), len(X), replace=True)
        fitted = clone(model).fit(X[indices], y[indices])
        predictions.append(fitted.predict(X_test))
    actual = bias_variance_decomp(
        clone(model), X, y, X_test, y_test, loss="mse", num_rounds=rounds, random_seed=7
    )
    assert_allclose(actual, _materialized_statistics(predictions, y_test), rtol=1e-12)
    assert_allclose(actual[0], actual[1] + actual[2], rtol=1e-12)


@pytest.mark.parametrize("shape", ["scalar", "vector", "row"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prediction_assignment_shapes_and_float_promotion(shape, dtype):
    X = np.arange(20, dtype=dtype)[:, None]
    y = np.sin(X[:, 0]).astype(dtype)
    X_test = X[:7]
    y_test = y[:7]
    rng = np.random.RandomState(7)
    predictions = []
    for _ in range(13):
        indices = rng.choice(len(X), len(X), replace=True)
        predictions.append(np.full(len(y_test), np.mean(y[indices]), dtype=float))
    actual = bias_variance_decomp(
        _MeanRegressor(shape),
        X,
        y,
        X_test,
        y_test,
        loss="mse",
        num_rounds=13,
        random_seed=7,
    )
    assert_allclose(actual, _materialized_statistics(predictions, y_test), rtol=1e-12)


@pytest.mark.parametrize("offset", [1e12, -1e12])
def test_centered_accumulator_preserves_variance_at_large_offsets(offset):
    values = offset + np.array([0.0, 0.25, 0.5, 1.25, 0.75])
    with localcontext() as context:
        context.prec = 60
        decimals = [Decimal.from_float(value) for value in values]
        mean = sum(decimals) / len(decimals)
        expected_variance = float(
            sum((value - mean) ** 2 for value in decimals) / len(values)
        )
    X = np.zeros((8, 1))
    y = np.full(8, offset + 0.5)
    actual = bias_variance_decomp(
        _SequenceRegressor(values),
        X,
        y,
        X,
        y,
        loss="mse",
        num_rounds=len(values),
        random_seed=7,
    )
    assert_allclose(actual[2], expected_variance, rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "values", [[np.inf, 1.0], [np.inf, -np.inf], [np.nan, 1.0], [-np.inf, -1.0]]
)
def test_nonfinite_predictions_keep_materialized_numpy_results(values):
    X = np.zeros((8, 1))
    y = np.ones(8)
    with np.errstate(all="ignore"):
        actual = bias_variance_decomp(
            _SequenceRegressor(values),
            X,
            y,
            X,
            y,
            loss="mse",
            num_rounds=len(values),
            random_seed=7,
        )
        expected = _materialized_statistics(np.tile(values, (len(y), 1)).T, y)
    assert_allclose(actual, expected, equal_nan=True)


def test_zero_rounds_remain_invalid():
    with pytest.raises(ValueError):
        bias_variance_decomp(
            _MeanRegressor(),
            np.ones((8, 1)),
            np.ones(8),
            np.ones((4, 1)),
            np.ones(4),
            loss="mse",
            num_rounds=0,
            random_seed=7,
        )


def test_negative_round_count_remains_invalid():
    with pytest.raises(ValueError):
        bias_variance_decomp(
            _MeanRegressor(),
            np.ones((8, 1)),
            np.ones(8),
            np.ones((4, 1)),
            np.ones(4),
            loss="mse",
            num_rounds=-1,
            random_seed=7,
        )


def test_peak_memory_does_not_scale_with_bootstrap_round_count():
    X = np.arange(64, dtype=float)[:, None]
    y = np.sin(X[:, 0])
    X_test = np.zeros((20000, 1))
    y_test = np.ones(20000)
    peaks = []
    for rounds in [5, 50]:
        gc.collect()
        tracemalloc.start()
        try:
            bias_variance_decomp(
                _MeanRegressor(),
                X,
                y,
                X_test,
                y_test,
                loss="mse",
                num_rounds=rounds,
                random_seed=7,
            )
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    assert peaks[1] < 3 * peaks[0], peaks
