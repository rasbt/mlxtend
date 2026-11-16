# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# License: BSD 3 clause

from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mlxtend.evaluate import bootstrap


def _decimal_standard_error(values, ddof):
    with localcontext() as context:
        context.prec = 100
        values = [Decimal.from_float(float(value)) for value in values]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values)
        return float((variance / (len(values) - ddof)).sqrt())


def _recorded_bootstrap(x, func, ddof=1, num_rounds=50):
    values = []

    def recorded(sample):
        value = func(sample)
        values.append(value)
        return value

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = bootstrap(x, recorded, num_rounds=num_rounds, ddof=ddof, seed=7)
    return result, values[1:]


@pytest.mark.parametrize("scale", [1.0, 1e200, 1e-200, -1e200, -1e-200, 1e-310])
@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_finite_standard_error_matches_high_precision_oracle(scale, ddof):
    x = np.array([-5.0, -1.0, 0.0, 2.0, 7.0]) * scale
    result, replicates = _recorded_bootstrap(x, np.mean, ddof=ddof)
    expected = _decimal_standard_error(replicates, ddof)
    assert np.isfinite(result[1])
    assert result[1] > 0
    assert_allclose(result[1], expected, rtol=3e-14, atol=np.nextafter(0.0, 1.0))


@pytest.mark.parametrize("constant", [0.0, 1e308, -1e308, 1e-308])
@pytest.mark.parametrize("ddof", [0, 1])
def test_constant_statistics_have_zero_standard_error(constant, ddof):
    result, _ = _recorded_bootstrap(np.ones(4), lambda sample: constant, ddof=ddof)
    assert result == (constant, 0.0, (constant, constant))


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_representable_standard_error_near_float_limit(sign):
    x = sign * np.array([-1.0, -0.75, 0.5, 0.75]) * np.finfo(float).max
    result, replicates = _recorded_bootstrap(x, lambda sample: sample[0])
    expected = _decimal_standard_error(replicates, 1)
    assert np.isfinite(expected)
    assert_allclose(result[1], expected, rtol=3e-14, atol=0)


@pytest.mark.parametrize("offset", [1e200, -1e200])
def test_small_variations_about_large_offset_remain_resolved(offset):
    x = offset + np.array([0.0, 2.0, 5.0, 9.0]) * abs(np.spacing(offset))
    result, replicates = _recorded_bootstrap(x, lambda sample: sample[0])
    expected = _decimal_standard_error(replicates, 1)
    assert_allclose(result[1], expected, rtol=3e-14, atol=0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_statistics_keep_nan_standard_error(value):
    result, _ = _recorded_bootstrap(np.ones(4), lambda sample: value)
    assert np.isnan(result[1])
    assert_allclose(result[0], value, equal_nan=True)
    assert_allclose(result[2], [value, value], equal_nan=True)


@pytest.mark.parametrize("ddof", [1, 2, 3])
def test_undefined_standard_error_keeps_numpy_behavior(ddof):
    with pytest.warns(RuntimeWarning, match="Degrees of freedom"):
        result, _ = _recorded_bootstrap(np.ones(4), np.mean, ddof=ddof, num_rounds=1)
    assert np.isnan(result[1])


def test_resampling_original_statistic_and_percentile_convention_are_unchanged():
    x = np.array([1.0, 2.0, 3.0, 5.0]) * 1e200
    before = x.copy()
    result, replicates = _recorded_bootstrap(x, np.mean)
    rng = np.random.RandomState(7)
    expected = [np.mean(x[rng.choice(len(x), len(x), replace=True)]) for _ in range(50)]
    assert_array_equal(replicates, expected)
    assert_array_equal(x, before)
    assert result[0] == np.mean(x)
    ordered = sorted(expected)
    assert result[2] == (ordered[0], ordered[48])
