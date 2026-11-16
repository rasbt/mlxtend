# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from mlxtend.preprocessing import minmax_scaling


def _decimal_reference(values, lower, upper):
    with localcontext() as context:
        context.prec = 400
        source = [Decimal.from_float(float(value)) for value in values]
        low, high = min(source), max(source)
        target_low = Decimal.from_float(float(lower))
        target_high = Decimal.from_float(float(upper))
        if low == high:
            return np.full(len(values), lower, dtype=float)
        return np.array(
            [
                float(
                    target_low
                    + (value - low) / (high - low) * (target_high - target_low)
                )
                for value in source
            ]
        )


@pytest.mark.parametrize(
    "values",
    [
        [-np.finfo(float).max, 0.0, np.finfo(float).max],
        [-1e308, -5e307, 0.0, 5e307, 1e308],
        [-np.finfo(float).max, -np.finfo(float).max / 2, 2.0],
        [2.0, np.finfo(float).max / 2, np.finfo(float).max],
        [np.nextafter(1e308, 0), 1e308, np.nextafter(1e308, np.inf)],
        [-1e308, -1e308, -1e308],
        [-4.0, -0.5, 2.0],
    ],
)
@pytest.mark.parametrize(
    "bounds",
    [
        (0.0, 1.0),
        (-2.0, 3.0),
        (-1e308, 1e308),
        (np.finfo(float).max, -np.finfo(float).max),
        (-np.finfo(float).max, np.finfo(float).max / 4),
    ],
)
def test_finite_ranges_match_decimal_reference(values, bounds):
    values = np.array(values).reshape(-1, 1)
    original = values.copy()
    values.flags.writeable = False
    expected = _decimal_reference(values[:, 0], *bounds)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = minmax_scaling(
            values, columns=[0], min_val=bounds[0], max_val=bounds[1]
        )
    assert np.isfinite(result).all()
    assert_allclose(
        result[:, 0],
        expected,
        rtol=2e-14,
        atol=4 * np.finfo(float).eps * max(abs(bounds[0]), abs(bounds[1])),
    )
    assert result[0, 0] == bounds[0]
    if np.ptp(values[:, 0] / 2) > 0:
        assert result[-1, 0] == bounds[1]
    assert_array_equal(values, original)


def test_mixed_dataframe_ranges_preserve_selection_and_input():
    frame = pd.DataFrame(
        {
            "constant": [1e308] * 5,
            "unused": [10, 20, 30, 40, 50],
            "ordinary": [-4.0, -2.0, 0.0, 2.0, 4.0],
            "wide": [-1e308, -5e307, 0.0, 5e307, 1e308],
        },
        index=[4, 8, 15, 16, 23],
    )
    original = frame.copy(deep=True)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = minmax_scaling(frame, columns=["wide", "constant", "ordinary"])
    expected = pd.DataFrame(
        {
            "wide": [0.0, 0.25, 0.5, 0.75, 1.0],
            "constant": [0.0] * 5,
            "ordinary": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
        index=frame.index,
    )
    assert_frame_equal(result, expected)
    assert_frame_equal(frame, original)


def test_one_dimensional_wide_input():
    values = np.array([-1e308, 0.0, 1e308])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = minmax_scaling(values, columns=[0])
    assert_array_equal(result, [[0.0], [0.5], [1.0]])


def test_ordinary_range_arithmetic_is_unchanged():
    values = np.array([[-4.0, 5.0], [-0.5, 8.0], [2.0, 12.0]])
    expected = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0))
    assert_array_equal(minmax_scaling(values, [0, 1]), expected)
    assert_array_equal(
        minmax_scaling(values, [1, 0], min_val=-3, max_val=7),
        expected[:, [1, 0]] * 10 - 3,
    )
