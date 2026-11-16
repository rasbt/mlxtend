from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import Ridge

from mlxtend.evaluate import (
    combined_ftest_5x2cv,
    paired_ttest_5x2cv,
    paired_ttest_kfold_cv,
    paired_ttest_resampled,
)

PROCEDURES = [
    (paired_ttest_resampled, {"num_rounds": 8}, "resampled"),
    (paired_ttest_kfold_cv, {"cv": 5, "shuffle": True}, "kfold"),
    (paired_ttest_5x2cv, {}, "5x2-t"),
    (combined_ftest_5x2cv, {}, "5x2-f"),
]


def _data():
    rng = np.random.RandomState(7)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(size=60) * 0.4
    return X, y


def _decimal_statistic(scores, kind):
    with localcontext() as context:
        context.prec = 80
        differences = [
            Decimal.from_float(a) - Decimal.from_float(b)
            for a, b in zip(scores[::2], scores[1::2])
        ]
        if kind in ("resampled", "kfold"):
            count = Decimal(len(differences))
            mean = sum(differences) / count
            variance = sum((d - mean) ** 2 for d in differences) / (count - 1)
            return float(mean * count.sqrt() / variance.sqrt())
        variances = []
        for a, b in zip(differences[::2], differences[1::2]):
            mean = (a + b) / 2
            variances.append((a - mean) ** 2 + (b - mean) ** 2)
        if kind == "5x2-t":
            return float(differences[0] / (sum(variances) / 5).sqrt())
        return float(sum(d * d for d in differences) / (2 * sum(variances)))


@pytest.mark.parametrize("procedure,options,kind", PROCEDURES)
@pytest.mark.parametrize(
    "scale", [1e-200, 1e-100, 0.125, 1.0, 8.0, 1e100, 1e200, -1e-200, -3.0, -1e200]
)
def test_statistic_and_pvalue_match_decimal_oracle_across_score_units(
    procedure, options, kind, scale
):
    X, y = _data()
    scores = []

    def scorer(estimator, X_test, y_test):
        score = float(-np.mean((estimator.predict(X_test) - y_test) ** 2))
        scores.append(score)
        return score * scale

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        actual = procedure(
            Ridge(alpha=0.1),
            Ridge(alpha=4),
            X,
            y,
            scoring=scorer,
            random_seed=9,
            **options,
        )
    expected_statistic = _decimal_statistic(scores, kind)
    if kind == "5x2-f":
        expected_pvalue = stats.f.sf(expected_statistic, 10, 5)
    else:
        expected_statistic *= np.sign(scale)
        degrees = {"resampled": 7, "kfold": 4, "5x2-t": 5}[kind]
        expected_pvalue = 2 * stats.t.sf(abs(expected_statistic), degrees)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual, [expected_statistic, expected_pvalue], rtol=2e-12, atol=2e-14
    )
    assert len(scores) == {"resampled": 16, "kfold": 10, "5x2-t": 20, "5x2-f": 20}[kind]


@pytest.mark.parametrize("procedure,options,kind", PROCEDURES)
@pytest.mark.parametrize("scale", [1.0, 1e200, 1e-200])
@pytest.mark.parametrize("equal_scores", [False, True])
def test_zero_variance_and_identical_scores_keep_expected_degeneracy(
    procedure, options, kind, scale, equal_scores
):
    X, y = _data()

    def scorer(estimator, X_test, y_test):
        return scale * (1.0 if equal_scores or estimator.alpha == 4 else 2.0)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        statistic, pvalue = procedure(
            Ridge(alpha=0.1),
            Ridge(alpha=4),
            X,
            y,
            scoring=scorer,
            random_seed=9,
            **options,
        )

    if equal_scores:
        assert np.isnan(statistic) and np.isnan(pvalue)
    else:
        assert statistic == np.inf
        assert pvalue == 0.0


@pytest.mark.parametrize("procedure,options,kind", PROCEDURES)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_scores_remain_undefined(procedure, options, kind, value):
    X, y = _data()

    def scorer(estimator, X_test, y_test):
        return value if estimator.alpha == 0.1 else 1.0

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        statistic, pvalue = procedure(
            Ridge(alpha=0.1),
            Ridge(alpha=4),
            X,
            y,
            scoring=scorer,
            random_seed=9,
            **options,
        )
    assert np.isnan(statistic) and np.isnan(pvalue)


@pytest.mark.parametrize("num_rounds", [0, 1])
def test_insufficient_resampling_rounds_remain_undefined(num_rounds):
    X, y = _data()
    with np.errstate(invalid="ignore", divide="ignore"):
        statistic, pvalue = paired_ttest_resampled(
            Ridge(alpha=0.1), Ridge(alpha=4), X, y, num_rounds=num_rounds, random_seed=9
        )
    assert np.isnan(statistic) and np.isnan(pvalue)
