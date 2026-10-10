# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Supriya Selvan Ganesan
#
# A function for plotting a reliability diagram (calibration curve).
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from mlxtend.evaluate.calibration import (
    _bin_statistics,
    _check_binary_probs,
    expected_calibration_error,
)

_DIAGONAL_LABEL = "Perfectly calibrated"


def plot_reliability_diagram(
    y_target,
    y_prob,
    n_bins=10,
    label=None,
    show_ece=True,
    show_histogram=True,
    figsize=None,
    ax=None,
):
    """Plot a reliability diagram (calibration curve) for a binary classifier.

    Predictions are grouped into equal-width probability bins. For each
    non-empty bin, the mean predicted probability (x-axis) is plotted
    against the observed fraction of positives (y-axis). A perfectly
    calibrated classifier lies on the diagonal; points below the diagonal
    indicate over-confidence and points above it indicate under-confidence.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True binary class labels (0 or 1).
    y_prob : array-like, shape=[n_samples]
        Predicted probabilities of the positive class, in the range [0, 1].
    n_bins : int (default: 10)
        Number of equal-width probability bins.
    label : str or None (default: None)
        Legend label of the curve, e.g., the model name.
    show_ece : bool (default: True)
        Appends the expected calibration error to the legend label.
    show_histogram : bool (default: True)
        Adds a small histogram of the predicted probabilities below the
        diagram. Ignored if an existing axis is passed via `ax`.
    figsize : tuple (default: None)
        Height and width of the figure. Ignored if `ax` is passed.
    ax : matplotlib axis (default: None)
        Existing axis to draw on. Calling the function several times with
        the same axis overlays several models (the diagonal is drawn only
        once). If None, a new figure is created.

    Returns
    -----------
    fig, ax : matplotlib.pyplot subplot objects
        Figure and the main (diagram) axis.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/plotting/plot_reliability_diagram/

    """
    y_target, y_prob = _check_binary_probs(y_target, y_prob)
    counts, mean_prob, frac_pos = _bin_statistics(y_target, y_prob, n_bins)

    ax_hist = None
    if ax is None:
        if show_histogram:
            fig, (ax, ax_hist) = plt.subplots(
                2,
                1,
                figsize=figsize,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if not any(line.get_label() == _DIAGONAL_LABEL for line in ax.lines):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label=_DIAGONAL_LABEL)

    curve_label = label
    if show_ece:
        ece = expected_calibration_error(y_target, y_prob, n_bins=n_bins)
        ece_text = "ECE = %.3f" % ece
        curve_label = ece_text if label is None else "%s (%s)" % (label, ece_text)

    nonempty = counts > 0
    ax.plot(mean_prob[nonempty], frac_pos[nonempty], marker="o", label=curve_label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of positives")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    if ax_hist is not None:
        ax_hist.bar(
            (np.arange(n_bins) + 0.5) / n_bins,
            counts,
            width=1.0 / n_bins,
            edgecolor="white",
        )
        ax_hist.set_ylabel("Count")
        ax_hist.set_xlabel("Mean predicted probability")
        ax_hist.set_xlim(0, 1)
    elif len(fig.axes) == 1:
        # no histogram panel below that already carries the x-axis label
        ax.set_xlabel("Mean predicted probability")

    return fig, ax
