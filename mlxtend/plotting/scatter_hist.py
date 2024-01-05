# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scatter_hist(x, y, xlabel=None, ylabel=None, figsize=(5, 5)):
    """
    Scatter plot and individual feature histograms along axes.

    Parameters
    ----------
    x : 1D array-like or Pandas Series
        X-axis values.

    y : 1D array-like or Pandas Series
        Y-axis values.

    xlabel : str (default: `None`)
        Label for the X-axis values. If `x` is a pandas Series,
        and `xlabel` is `None`, the label is inferred automatically.

    ylabel : str (default: `None`)
        Label for the X-axis values. If `y` is a pandas Series,
        and `ylabel` is `None`, the label is inferred automatically.

    figsize : tuple (default: `(5, 5)`)
        Matplotlib figure size.

    Returns
    ---------
    plot : Matplotlib Figure object

    """
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.001
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    if hasattr(x, "values"):
        x_values = x.values
        if xlabel is None:
            xlabel = x.name
    else:
        x_values = x

    if hasattr(y, "values"):
        y_values = y.values
        if ylabel is None:
            ylabel = y.name
    else:
        y_values = y

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect_scatter)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.axis("off")
    ax_histy.axis("off")
    ax_histx.hist(x_values, edgecolor="white", bins="auto")
    ax_histy.hist(y_values, edgecolor="white", orientation="horizontal", bins="auto")
    plot = ax.scatter(x_values, y_values)
    return plot
