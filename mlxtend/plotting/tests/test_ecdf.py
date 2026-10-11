# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlxtend.data import iris_data
from mlxtend.plotting import ecdf


def test_threshold():
    X, y = iris_data()
    ax, threshold, count = ecdf(x=X[:, 0], x_label="sepal length (cm)", percentile=0.8)
    assert threshold == 6.5
    assert count == 120


@pytest.mark.parametrize("label", ["Cumulative probability", "", "F(x)"])
@pytest.mark.parametrize("provide_axes", [False, True])
def test_y_label(label, provide_axes):
    fig, ax = plt.subplots()
    try:
        kwargs = {"ax": ax} if provide_axes else {}
        result, threshold, count = ecdf(
            [1, 4, 2, 3], y_label=label, x_label="Score", percentile=0.75, **kwargs
        )
        assert result is ax
        assert ax.get_ylabel() == label
        assert ax.get_xlabel() == "Score"
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3, 4])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.25, 0.5, 0.75, 1])
        assert threshold == 3
        assert count == 3
    finally:
        plt.close(fig)


@pytest.mark.parametrize("provide_axes", [False, True])
def test_default_y_label(provide_axes):
    fig, ax = plt.subplots()
    try:
        kwargs = {"ax": ax} if provide_axes else {}
        result, threshold, count = ecdf([3, 1, 2], **kwargs)
        assert result is ax
        assert ax.get_ylabel() == "ECDF"
        assert threshold is None
        assert count is None
    finally:
        plt.close(fig)
