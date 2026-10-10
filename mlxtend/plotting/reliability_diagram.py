# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# A function for plotting a reliability diagram (calibration curve).
# Author: Supriya Selvan Ganesan
#
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
    """Plot a reliability diagram (also called a calibration curve).

    The predictions are sorted into equal-width probability bins. For every
    bin that has data, the average predicted probability (x axis) is plotted
    against the fraction of samples that are really positive (y axis). A
    perfectly calibrated model follows the diagonal. Points below the
    diagonal mean the model is overconfident, and points above it mean the
    model is underconfident.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True labels, either 0 or 1.
    y_prob : array-like, shape=[n_samples]
        Predicted probability of the positive class, between 0 and 1.
    n_bins : int (default: 10)
        Number of equal-width probability bins.
    label : str or None (default: None)
        Name of the curve in the legend, for example the model name.
    show_ece : bool (default: True)
        Adds the expected calibration error to the legend entry.
    show_histogram : bool (default: True)
        Adds a small histogram under the diagram that shows how many
        predictions fall into each bin. It is skipped if you pass your own
        `ax`.
    figsize : tuple (default: None)
        Width and height of the figure. Not used if `ax` is given.
    ax : matplotlib axis (default: None)
        An existing axis to draw on. Call the function again with the same
        axis to put several models in one plot (the diagonal is only drawn
        once). If None, a new figure is created.

    Returns
    -----------
    fig, ax : matplotlib.pyplot subplot objects
        The figure and the main axis, the one with the curve.

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
            # curve on top, a smaller histogram below it
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

    # draw the diagonal only once, in case this is a second model on the same ax
    if not any(line.get_label() == _DIAGONAL_LABEL for line in ax.lines):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label=_DIAGONAL_LABEL)

    # put the ECE in the legend text so it is visible without extra code
    curve_label = label
    if show_ece:
        ece = expected_calibration_error(y_target, y_prob, n_bins=n_bins)
        ece_text = "ECE = %.3f" % ece
        curve_label = ece_text if label is None else "%s (%s)" % (label, ece_text)

    # empty bins have NaN values, leave them out so the line is not broken
    nonempty = counts > 0
    ax.plot(mean_prob[nonempty], frac_pos[nonempty], marker="o", label=curve_label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of positives")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    if ax_hist is not None:
        # one bar per bin, centered on the bin
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
