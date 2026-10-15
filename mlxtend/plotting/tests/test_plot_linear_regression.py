import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge

from mlxtend.plotting import plot_linear_regression


@pytest.mark.parametrize("input_kind", ["array", "list", "column"])
@pytest.mark.parametrize("limits", ["auto", (-2.0, 6.0)])
@pytest.mark.parametrize("estimator", [LinearRegression, Ridge])
def test_regression_line_uses_single_feature_endpoint_samples(
    input_kind, limits, estimator
):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2 * x + 1
    expected = estimator().fit(x[:, np.newaxis], y)
    model = estimator()
    data = x.tolist() if input_kind == "list" else x
    if input_kind == "column":
        data = data[:, np.newaxis]
    original = np.asarray(data).copy()
    fig, ax = plt.subplots()
    try:
        result = plot_linear_regression(data, y, model=model, xlim=limits)
        expected_limits = [0.8, 4.8] if limits == "auto" else limits
        line = ax.lines[-1]
        assert_allclose(line.get_xdata(), expected_limits)
        assert_allclose(
            line.get_ydata(), expected.predict(np.asarray(expected_limits)[:, None])
        )
        assert_allclose(result, [expected.intercept_, expected.coef_[0], 1.0])
        assert_allclose(np.asarray(data), original)
        assert ax.get_legend() is not None
        assert_allclose(ax.collections[0].get_offsets(), np.column_stack([x, y]))
    finally:
        plt.close(fig)


def test_default_model_with_user_guide_data():
    x = np.array(
        [
            4,
            8,
            13,
            26,
            31,
            10,
            8,
            30,
            18,
            12,
            20,
            5,
            28,
            18,
            6,
            31,
            12,
            12,
            27,
            11,
            6,
            14,
            25,
            7,
            13,
            4,
            15,
            21,
            15,
        ]
    )
    y = np.array(
        [
            14,
            24,
            22,
            59,
            66,
            25,
            18,
            60,
            39,
            32,
            53,
            18,
            55,
            41,
            28,
            61,
            35,
            36,
            52,
            23,
            19,
            25,
            73,
            16,
            32,
            14,
            31,
            43,
            34,
        ]
    )
    fig, ax = plt.subplots()
    try:
        intercept, slope, correlation = plot_linear_regression(x, y)
        expected_slope, expected_intercept = np.polyfit(x, y, 1)
        assert_allclose([intercept, slope], [expected_intercept, expected_slope])
        assert_allclose(correlation, np.corrcoef(x, y)[0, 1])
        line = ax.lines[-1]
        assert_allclose(
            line.get_ydata(), slope * np.asarray(line.get_xdata()) + intercept
        )
    finally:
        plt.close(fig)


def test_correlation_and_plot_options_are_preserved():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x**2
    fig, ax = plt.subplots()
    try:
        result = plot_linear_regression(
            x,
            y,
            model=LinearRegression(),
            corr_func=spearmanr,
            scattercolor="green",
            fit_style="r-",
            legend=False,
            xlim=(0.0, 5.0),
        )
        expected_slope, expected_intercept = np.polyfit(x, y, 1)
        assert_allclose(result, [expected_intercept, expected_slope, 1.0])
        assert ax.get_legend() is None
        assert ax.lines[-1].get_color() == "r"
        assert ax.lines[-1].get_linestyle() == "-"
        assert_allclose(
            ax.lines[-1].get_ydata(),
            [expected_intercept, 5 * expected_slope + expected_intercept],
        )
    finally:
        plt.close(fig)
