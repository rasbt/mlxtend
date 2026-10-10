# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mlxtend.plotting import plot_reliability_diagram  # noqa: E402

Y = [0, 1, 1, 1]
P = [0.1, 0.9, 0.8, 0.3]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _curve(ax):
    """Return the (x, y) data of the first non-diagonal line."""
    line = [ln for ln in ax.lines if ln.get_label() != "Perfectly calibrated"][0]
    return np.asarray(line.get_xdata()), np.asarray(line.get_ydata())


def test_curve_matches_hand_computed_bins():
    fig, ax = plot_reliability_diagram(Y, P, n_bins=2)
    x, y = _curve(ax)
    np.testing.assert_allclose(x, [0.2, 0.85])
    np.testing.assert_allclose(y, [0.5, 1.0])


def test_returns_fig_and_axis():
    fig, ax = plot_reliability_diagram(Y, P, n_bins=2)
    assert isinstance(fig, plt.Figure)
    assert ax in fig.axes


def test_ece_and_label_in_legend():
    _, ax = plot_reliability_diagram(Y, P, n_bins=2, label="Model A")
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Model A (ECE = 0.225)" in texts
    assert "Perfectly calibrated" in texts


def test_ece_can_be_hidden():
    _, ax = plot_reliability_diagram(Y, P, n_bins=2, label="Model A", show_ece=False)
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Model A" in texts


def test_histogram_panel():
    fig, _ = plot_reliability_diagram(Y, P, n_bins=2, show_histogram=True)
    assert len(fig.axes) == 2
    heights = [patch.get_height() for patch in fig.axes[1].patches]
    assert heights == [2, 2]  # two predictions in each bin


def test_no_histogram_panel():
    fig, _ = plot_reliability_diagram(Y, P, show_histogram=False)
    assert len(fig.axes) == 1


def test_empty_bins_are_skipped():
    # all predictions fall into the last of 10 bins
    _, ax = plot_reliability_diagram([1, 0, 1], [0.95, 0.92, 0.99])
    x, y = _curve(ax)
    assert len(x) == 1 and not np.isnan(y).any()


def test_overlay_on_existing_axis_draws_diagonal_once():
    fig, ax = plt.subplots()
    _, ax1 = plot_reliability_diagram(Y, P, n_bins=2, label="A", ax=ax)
    _, ax2 = plot_reliability_diagram(
        Y, [0.2, 0.7, 0.6, 0.5], n_bins=2, label="B", ax=ax
    )
    assert ax1 is ax and ax2 is ax
    labels = [ln.get_label() for ln in ax.lines]
    assert labels.count("Perfectly calibrated") == 1
    assert len(ax.lines) == 3
    assert len(fig.axes) == 1  # histogram is skipped for a user axis


def test_axes_limits():
    _, ax = plot_reliability_diagram(Y, P)
    assert ax.get_xlim() == (0.0, 1.0)
    assert ax.get_ylim() == (0.0, 1.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="same length"):
        plot_reliability_diagram([0, 1], [0.5])
    with pytest.raises(ValueError, match="range"):
        plot_reliability_diagram([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError, match="n_bins"):
        plot_reliability_diagram([0, 1], [0.5, 0.5], n_bins=0)


def test_xlabel_only_on_bottom_panel_when_overlaying():
    fig, ax = plot_reliability_diagram(Y, P, n_bins=2, label="A")
    plot_reliability_diagram(Y, P, n_bins=2, label="B", ax=ax)
    assert ax.get_xlabel() == ""
    assert fig.axes[1].get_xlabel() == "Mean predicted probability"


def test_xlabel_on_single_axis():
    _, ax = plot_reliability_diagram(Y, P, show_histogram=False)
    assert ax.get_xlabel() == "Mean predicted probability"
