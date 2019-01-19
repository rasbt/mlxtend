# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# Matplotlib wrapper for generating stacked barplots.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle


def stacked_barplot(df, bar_width='auto', colors='bgrcky',
                    labels='index', rotation=90, legend_loc='best'):
    """
    Function to plot stacked barplots

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame where the index denotes the
        x-axis labels, and the columns contain the different
        measurements for each row.
    bar_width: 'auto' or float (default: 'auto')
        Parameter to set the widths of the bars. if
        'auto', the width is automatically determined by
        the number of columns in the dataset.
    colors: str (default: 'bgrcky')
        The colors of the bars.
    labels: 'index' or iterable (default: 'index')
        If 'index', the DataFrame index will be used as
        x-tick labels.
    rotation: int (default: 90)
        Parameter to rotate the x-axis labels.
    legend_loc : str (default: 'best')
        Location of the plot legend
        {best, upper left, upper right, lower left, lower right}
        No legend if legend_loc=False

    Returns
    ---------
    fig : matplotlib.pyplot figure object

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/stacked_barplot/

    """
    # Setting the positions and width for the bars
    pos = np.array(range(len(df.index)))

    if bar_width == 'auto':
        width = 1 / (len(df.columns[1:]) * 2)
    else:
        width = bar_width

    if labels == 'index':
        labels = df.index

    color_gen = cycle(colors)

    label_pos = [pos]

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(12, 6))

    plt.bar(pos,
            df.iloc[:, 0],
            width,
            alpha=0.8,
            color=next(color_gen),
            label=df.columns[0])

    for i, c in enumerate(df.columns[1:]):

        bar_pos = [p + width*(i+1) for p in pos]
        label_pos.append(bar_pos)
        plt.bar(bar_pos,
                df.iloc[:, i+1],
                width,
                alpha=0.5,
                color=next(color_gen),
                label=c)

    # Setting axis labels and ticks

    label_pos = np.asarray(label_pos).mean(axis=0) + width*0.5

    ax.set_xticks(label_pos)
    ax.set_xticklabels(labels, rotation=rotation, horizontalalignment='center')

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos) + width*7)

    if legend_loc:
        plt.legend(loc=legend_loc, scatterpoints=1)
    return fig
