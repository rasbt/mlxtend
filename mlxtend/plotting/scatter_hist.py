# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def scatter_hist(x, y, data):
    """
    Scatter plot and individual feature histograms along axes.
    Parameters
    ----------
    x : str or int
        DataFrame column name of the x-axis values or
        integer for the numpy ndarray column index.
    y : str or int
        DataFrame column name of the y-axis values or
        integer for the numpy ndarray column index
    data : Pandas DataFrame object or NumPy ndarray.
    Returns
    ---------
    plot : matplotlib pyplot figure object
    """
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.001
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    if isinstance(data, pd.DataFrame):
        for i in (x, y):
            assert (isinstance(i, str))
        frame = True
        xlabel = x
        ylabel = y
        x = data[x]
        y = data[y]

    elif isinstance(data, np.ndarray):
        for i in (x, y):
            assert (isinstance(i, int))
        frame = False
        x = data[:, x]
        y = data[:, y]

    else:
        raise ValueError('df must be pandas.DataFrame or numpy.ndarray object')

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes(rect_scatter)
    if frame:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.axis("off")
    ax_histy.axis("off")
    ax_histx.hist(x, edgecolor='white', bins='auto')
    ax_histy.hist(y, edgecolor='white', orientation='horizontal', bins='auto')
    plot = ax.scatter(x, y)
    return plot
