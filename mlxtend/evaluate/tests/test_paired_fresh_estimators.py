import pickle
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mlxtend.evaluate import (
    combined_ftest_5x2cv,
    paired_ttest_5x2cv,
    paired_ttest_kfold_cv,
    paired_ttest_resampled,
)

CASES = [
    (paired_ttest_resampled, {"num_rounds": 8, "test_size": 0.3}),
    (paired_ttest_kfold_cv, {"cv": 4, "shuffle": True}),
    (paired_ttest_5x2cv, {}),
    (combined_ftest_5x2cv, {}),
]


def _data(classification=False):
    X = np.linspace(-2, 2, 40)[:, None]
    y = np.sin(3 * X[:, 0]) + X[:, 0] ** 2
    if classification:
        y = (np.sin(4 * X[:, 0]) > 0).astype(int)
    return X, y


def _models(warm_start, pipeline, classification=False):
    forest = RandomForestClassifier if classification else RandomForestRegressor
    first = forest(n_estimators=3, max_depth=2, warm_start=warm_start, random_state=17)
    second = (
        DecisionTreeClassifier(max_depth=1, random_state=11)
        if classification
        else DummyRegressor()
    )
    if pipeline:
        first = make_pipeline(StandardScaler(), first)
        second = make_pipeline(StandardScaler(), second)
    return first, second


def _independent_result(function, pipeline, classification=False):
    X, y = _data(classification)
    rng = np.random.RandomState(42)
    indices = np.arange(len(X))
    if function is paired_ttest_kfold_cv:
        splits = list(KFold(4, shuffle=True, random_state=42).split(X))
    else:
        count = 8 if function is paired_ttest_resampled else 5
        test_size = 0.3 if function is paired_ttest_resampled else 0.5
        splits = []
        for _ in range(count):
            train, test = train_test_split(
                indices, test_size=test_size, random_state=rng.randint(32767)
            )
            splits.append((train, test))
            if count == 5:
                splits.append((test, train))
    scorer = get_scorer("accuracy" if classification else "neg_mean_squared_error")
    differences = []
    for train, test in splits:
        first, second = _models(True, pipeline, classification)
        first.fit(X[train], y[train])
        second.fit(X[train], y[train])
        differences.append(
            scorer(first, X[test], y[test]) - scorer(second, X[test], y[test])
        )
    differences = np.asarray(differences)
    if function in (paired_ttest_resampled, paired_ttest_kfold_cv):
        result = stats.ttest_1samp(differences, popmean=0)
        return result.statistic, result.pvalue
    pairs = differences.reshape(5, 2)
    variance_sum = np.sum((pairs - pairs.mean(axis=1, keepdims=True)) ** 2)
    if function is paired_ttest_5x2cv:
        statistic = differences[0] / np.sqrt(variance_sum / 5)
        return statistic, 2 * stats.t.sf(abs(statistic), 5)
    statistic = np.sum(differences**2) / (2 * variance_sum)
    return statistic, stats.f.sf(statistic, 10, 5)


@pytest.mark.parametrize("function,kwargs", CASES)
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("prefitted", [False, True])
def test_warm_start_statistics_match_independent_folds(
    function, kwargs, pipeline, prefitted
):
    X, y = _data()
    first, second = _models(True, pipeline)
    if prefitted:
        first.fit(X, y + 4)
        second.fit(X, y + 4)
    before = pickle.dumps((first, second))
    expected = _independent_result(function, pipeline)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        actual = function(
            first,
            second,
            X,
            y,
            scoring="neg_mean_squared_error",
            random_seed=42,
            **kwargs,
        )
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert pickle.dumps((first, second)) == before
    assert not any("Warm-start fitting" in str(w.message) for w in seen)


@pytest.mark.parametrize("function,kwargs", CASES)
def test_cold_start_statistics_are_preserved(function, kwargs):
    X, y = _data()
    first, second = _models(False, False)
    result = function(
        first,
        second,
        X,
        y,
        scoring="neg_mean_squared_error",
        random_seed=42,
        **kwargs,
    )
    assert_allclose(
        result, _independent_result(function, False), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("function,kwargs", CASES)
def test_classifier_default_scoring_uses_independent_models(function, kwargs):
    X, y = _data(classification=True)
    first, second = _models(True, False, classification=True)
    result = function(first, second, X, y, random_seed=42, **kwargs)
    assert_allclose(
        result,
        _independent_result(function, False, classification=True),
        rtol=1e-12,
        atol=1e-12,
    )


class _RememberingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, offset=0.0):
        self.offset = offset

    def fit(self, X, y):
        self.seen_ = getattr(self, "seen_", set()) | set(X[:, 0])
        self.mean_ = np.mean(y) + self.offset
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


@pytest.mark.parametrize("function,kwargs", CASES)
def test_callable_scorer_never_sees_training_state_from_another_fold(function, kwargs):
    X, y = _data()
    observed = []

    def scorer(estimator, test_X, test_y):
        assert estimator.seen_.isdisjoint(test_X[:, 0])
        observed.append(estimator)
        return -np.mean((estimator.predict(test_X) - test_y) ** 2)

    result = function(
        _RememberingRegressor(),
        _RememberingRegressor(offset=0.7),
        X,
        y,
        scoring=scorer,
        random_seed=42,
        **kwargs,
    )
    assert np.all(np.isfinite(result))
    assert len({id(estimator) for estimator in observed}) == len(observed)
