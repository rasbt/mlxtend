# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.contour import QuadContourSet
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

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


@pytest.mark.parametrize(
    "class_labels", [(0, 1), (-2, -1), (-3, -2), (-1, 0), (99, 100), (0, 3)]
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_contour_levels_use_class_indices(class_labels, n_jobs):
    if n_jobs > mp.cpu_count():
        pytest.skip("The parallel plotting check needs at least two CPU cores.")
    features = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.repeat(class_labels, 2)
    classifier = DecisionTreeClassifier(random_state=0).fit(features, labels)
    fig, ax = plt.subplots(figsize=(2, 2), dpi=40)
    try:
        plot_decision_regions(
            features, labels, classifier, ax=ax, n_jobs=n_jobs, colors="red,blue"
        )
        regions = next(c for c in ax.collections if isinstance(c, QuadContourSet))
        np.testing.assert_array_equal(regions.levels, [-0.5, 0.5, 1.5])
        np.testing.assert_allclose(
            regions.get_facecolors()[:, :3], [[1, 0, 0], [0, 0, 1]]
        )
        assert ax.get_legend_handles_labels()[1] == [str(c) for c in class_labels]
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "class_labels, predicted_index",
    [((-3,), 0), ((-3, -2, -1), 0), ((-3, -2, -1), 1), ((0, 1, 2), 0), ((0, 1, 2), 1)],
)
def test_contour_levels_include_classes_missing_from_predictions(
    class_labels, predicted_index
):
    features = np.arange(2 * len(class_labels), dtype=float).reshape(-1, 1)
    labels = np.repeat(class_labels, 2)
    classifier = DummyClassifier(
        strategy="constant", constant=class_labels[predicted_index]
    ).fit(features, labels)
    fig, ax = plt.subplots(figsize=(2, 2), dpi=40)
    try:
        plot_decision_regions(features, labels, classifier, ax=ax)
        regions = next(c for c in ax.collections if isinstance(c, QuadContourSet))
        np.testing.assert_array_equal(
            regions.levels, np.arange(len(class_labels) + 1) - 0.5
        )
        scatter_colors = [
            c.get_facecolors()[0, :3]
            for c in ax.collections
            if isinstance(c, PathCollection)
        ]
        np.testing.assert_allclose(regions.get_facecolors()[:, :3], scatter_colors)
        assert ax.get_legend_handles_labels()[1] == [str(c) for c in class_labels]
        fig.canvas.draw()
    finally:
        plt.close(fig)


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
        y.astype(np.float64),
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
