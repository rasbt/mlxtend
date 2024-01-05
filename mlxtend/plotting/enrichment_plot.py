# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# A function for plotting enrichment plots.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def enrichment_plot(
    df,
    colors="bgrkcy",
    markers=" ",
    linestyles="-",
    alpha=0.5,
    lw=2,
    where="post",
    grid=True,
    count_label="Count",
    xlim="auto",
    ylim="auto",
    invert_axes=False,
    legend_loc="best",
    ax=None,
):
    """Plot stacked barplots

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame where columns represent the different categories.
    colors: str (default: 'bgrcky')
        The colors of the bars.
    markers : str (default: ' ')
        Matplotlib markerstyles, e.g,
        'sov' for square,circle, and triangle markers.
    linestyles : str (default: '-')
        Matplotlib linestyles, e.g.,
        '-,--' to cycle normal and dashed lines. Note
        that the different linestyles need to be separated by commas.
    alpha : float (default: 0.5)
        Transparency level from 0.0 to 1.0.
    lw : int or float (default: 2)
        Linewidth parameter.
    where : {'post', 'pre', 'mid'} (default: 'post')
        Starting location of the steps.
    grid : bool (default: `True`)
        Plots a grid if True.
    count_label : str (default: 'Count')
        Label for the "Count"-axis.
    xlim : 'auto' or array-like [min, max] (default: 'auto')
        Min and maximum position of the x-axis range.
    ylim : 'auto' or array-like [min, max] (default: 'auto')
        Min and maximum position of the y-axis range.
    invert_axes : bool (default: False)
        Plots count on the x-axis if True.
    legend_loc : str (default: 'best')
        Location of the plot legend
        {best, upper left, upper right, lower left, lower right}
        No legend if legend_loc=False
    ax : matplotlib axis, optional (default: None)
        Use this axis for plotting or make a new one otherwise

    Returns
    ----------
    ax : matplotlib axis

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/plotting/enrichment_plot/

    """
    if isinstance(df, pd.Series):
        df_temp = pd.DataFrame(df)
    else:
        df_temp = df

    if ax is None:
        ax = plt.gca()

    color_gen = cycle(colors)
    marker_gen = cycle(markers)
    linestyle_gen = cycle(linestyles.split(","))
    r = range(1, len(df_temp.index) + 1)
    labels = df_temp.columns

    x_data = df_temp
    y_data = r

    for lab in labels:
        x, y = sorted(x_data[lab]), y_data
        if invert_axes:
            x, y = y, x

        ax.step(
            x,
            y,
            where=where,
            label=lab,
            color=next(color_gen),
            alpha=alpha,
            lw=lw,
            marker=next(marker_gen),
            linestyle=next(linestyle_gen),
        )

    if invert_axes:
        ax.set_ylim, ax.set_xlim = ax.set_xlim, ax.set_ylim

    if ylim == "auto":
        ax.set_ylim([np.min(y_data) - 1, np.max(y_data) + 1])
    else:
        ax.set_ylim(ylim)

    if xlim == "auto":
        df_min, df_max = np.min(x_data.min()), np.max(x_data.max())
        ax.set_xlim([df_min - 1, df_max + 1])

    else:
        ax.set_xlim(xlim)

    if legend_loc:
        plt.legend(loc=legend_loc, scatterpoints=1)

    if grid:
        plt.grid()

    if count_label:
        if invert_axes:
            plt.xlabel(count_label)
        else:
            plt.ylabel(count_label)

    return ax
