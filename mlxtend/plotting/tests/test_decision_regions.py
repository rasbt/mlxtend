# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions
from mlxtend.utils import assert_raises

plt.switch_backend("agg")
X, y = iris_data()
sr = SoftmaxRegression(epochs=15, eta=0.01, random_seed=1)


def test_pass():
    sr.fit(X[:, :2], y)
    plot_decision_regions(X=X[:, :2], y=y, clf=sr)


def test_ylist():
    sr.fit(X[:, :2], y)
    assert_raises(
        ValueError,
        "y must be a NumPy array. Found {}".format(type([])),
        plot_decision_regions,
        X[:, :2],
        list(y),
        sr,
    )


def test_filler_feature_values_fail():
    sr.fit(X, y)
    assert_raises(
        ValueError,
        "Filler values must be provided when " "X has more than 2 training features.",
        plot_decision_regions,
        X,
        y,
        sr,
    )


def test_feature_index_fail():
    sr.fit(X, y)
    assert_raises(
        ValueError,
        "Unable to unpack feature_index. Make sure feature_index "
        "only has two dimensions.",
        plot_decision_regions,
        X,
        y,
        sr,
        feature_index=(0, 1, 2),
        filler_feature_values={2: 0.5},
    )


def test_X_dim_fail():
    sr.fit(X[:, :2], y)
    assert_raises(
        ValueError,
        "X must be a 2D array",
        plot_decision_regions,
        X[:, :2].flatten(),
        y,
        sr,
    )


def test_X_highlight_fail_if_1d():
    sr.fit(X[:, :2], y)
    assert_raises(
        ValueError,
        "X_highlight must be a 2D array",
        plot_decision_regions,
        X[:, :2],
        y,
        sr,
        X_highlight=y,
    )


def test_y_int_ary():
    sr.fit(X[:, :2], y)
    assert_raises(
        ValueError,
        "y must be an integer array. Found float64. "
        "Try passing the array as y.astype(np.int_)",
        plot_decision_regions,
        X[:, :2],
        y.astype(np.float_),
        sr,
    )


def test_y_ary_dim():
    sr.fit(X[:, :2], y)
    assert_raises(
        ValueError,
        "y must be a 1D array",
        plot_decision_regions,
        X[:, :2],
        y[:, np.newaxis],
        sr,
    )


def test_scatter_kwargs_type():
    kwargs = "not a dictionary"
    sr.fit(X[:, :2], y)
    message = "d must be of type dict or None, but got " "{} instead".format(
        type(kwargs)
    )
    assert_raises(
        TypeError,
        message,
        plot_decision_regions,
        X[:, :2],
        y,
        sr,
        scatter_kwargs=kwargs,
    )


def test_contourf_kwargs_type():
    kwargs = "not a dictionary"
    sr.fit(X[:, :2], y)
    message = "d must be of type dict or None, but got " "{} instead".format(
        type(kwargs)
    )
    assert_raises(
        TypeError,
        message,
        plot_decision_regions,
        X[:, :2],
        y,
        sr,
        contourf_kwargs=kwargs,
    )


def test_scatter_highlight_kwargs_type():
    kwargs = "not a dictionary"
    sr.fit(X[:, :2], y)
    message = "d must be of type dict or None, but got " "{} instead".format(
        type(kwargs)
    )
    assert_raises(
        TypeError,
        message,
        plot_decision_regions,
        X[:, :2],
        y,
        sr,
        X_highlight=X[:, :2],
        scatter_highlight_kwargs=kwargs,
    )
