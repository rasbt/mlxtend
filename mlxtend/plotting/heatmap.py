# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# A function for plotting decision regions of classifiers.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np


def heatmap(matrix,
            hide_spines=False,
            hide_ticks=False,
            figsize=None,
            cmap=None,
            colorbar=True,
            row_names=None,
            column_names=None,
            column_name_rotation=45,
            cell_values=True,
            cell_fmt='.2f',
            cell_font_size=None):
    """Plot a heatmap via matplotlib.

    Parameters
    -----------
    conf_mat : array-like, shape = [n_rows, n_columns]
        And arbitrary 2D array.

    hide_spines : bool (default: False)
        Hides axis spines if True.

    hide_ticks : bool (default: False)
        Hides axis ticks if True

    figsize : tuple (default: (2.5, 2.5))
        Height and width of the figure

    cmap : matplotlib colormap (default: `None`)
        Uses matplotlib.pyplot.cm.viridis if `None`

    colorbar : bool (default: True)
        Shows a colorbar if True

    row_names : array-like, shape = [n_rows] (default: None)
        List of row names to be used as y-axis tick labels.

    column_names : array-like, shape = [n_columns] (default: None)
        List of column names to be used as x-axis tick labels.

    column_name_rotation : int (default: 45)
        Number of degrees for rotating column x-tick labels.

    cell_values : bool (default: True)
        Plots cell values if True.

    cell_fmt : string (default: '.2f')
        Format specification for cell values (if `cell_values=True`)

    cell_font_size : int (default: None)
        Font size for cell values (if `cell_values=True`)

    Returns
    -----------
    fig, ax : matplotlib.pyplot subplot objects
        Figure and axis elements of the subplot.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/heatmap/

    """

    if row_names is not None and len(row_names) != matrix.shape[0]:
        raise AssertionError(f'len(row_names) (got {len(row_names)})'
                             ' should be equal to number of'
                             ' rows in the input '
                             f' array (expect {matrix.shape[0]}).')

    if column_names is not None and len(column_names) != matrix.shape[1]:
        raise AssertionError(f'len(column_names)'
                             ' (got {len(column_names)})'
                             ' should be equal to number of'
                             ' columns in the'
                             f' input array (expect {matrix.shape[1]}).')

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)

    if cmap is None:
        cmap = plt.cm.viridis

    if figsize is None:
        figsize = (len(matrix)*1.25, len(matrix)*1.25)

    matshow = ax.matshow(matrix, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    normed_matrix = matrix.astype('float') / matrix.max()

    if cell_values:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                cell_text = format(matrix[i, j], cell_fmt)

                ax.text(x=j,
                        y=i,
                        size=cell_font_size,
                        s=cell_text,
                        va='center',
                        ha='center',
                        color="white" if normed_matrix[i, j] < 0.5
                              else "black")

    if row_names is not None:
        tick_marks = np.arange(len(row_names))
        plt.yticks(tick_marks, row_names)

    if column_names is not None:
        tick_marks = np.arange(len(column_names))
        plt.xticks(tick_marks, column_names, rotation=column_name_rotation)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    return fig, ax
