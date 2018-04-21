# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Implementation of the mulitnomial logistic regression algorithm for
# classification.
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from matplotlib.pyplot import subplots
from matplotlib.table import Table
import numpy as np


def checkerboard_plot(ary,
                      cell_colors=('white', 'black'),
                      font_colors=('black', 'white'),
                      fmt='%.1f',
                      figsize=None,
                      row_labels=None,
                      col_labels=None,
                      fontsize=None):
    """
    Plot a checkerboard table / heatmap via matplotlib.

    Parameters
    -----------
    ary : array-like, shape = [n, m]
        A 2D Nnumpy array.
    cell_colors : tuple or list (default: ('white', 'black'))
        Tuple or list containing the two colors of the
        checkerboard pattern.
    font_colors : tuple or list (default: ('black', 'white'))
        Font colors corresponding to the cell colors.
    figsize : tuple (default: (2.5, 2.5))
        Height and width of the figure
    fmt : str (default: '%.1f')
        Python string formatter for cell values.
        The default '%.1f' results in floats with 1 digit after
        the decimal point. Use '%d' to show numbers as integers.
    row_labels : list (default: None)
        List of the row labels. Uses the array row
        indices 0 to n by default.
    col_labels : list (default: None)
        List of the column labels. Uses the array column
        indices 0 to m by default.
    fontsize : int (default: None)
        Specifies the font size of the checkerboard table.
        Uses matplotlib's default if None.

    Returns
    -----------
    fig : matplotlib Figure object.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/checkerboard_plot/

    """

    fig, ax = subplots(figsize=figsize)
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    n_rows, n_cols = ary.shape

    if row_labels is None:
        row_labels = np.arange(n_rows)
    if col_labels is None:
        col_labels = np.arange(n_cols)

    width, height = 1.0 / n_cols, 1.0 / n_rows

    for (row_idx, col_idx), cell_val in np.ndenumerate(ary):

        idx = (col_idx + row_idx) % 2
        tb.add_cell(row_idx, col_idx, width, height,
                    text=fmt % cell_val,
                    loc='center',
                    facecolor=cell_colors[idx])

    for row_idx, label in enumerate(row_labels):
        tb.add_cell(row_idx, -1,
                    width, height,
                    text=label, loc='right',
                    edgecolor='none', facecolor='none')

    for col_idx, label in enumerate(col_labels):
        tb.add_cell(-1, col_idx,
                    width, height / 2.,
                    text=label, loc='center',
                    edgecolor='none', facecolor='none')

    for (row_idx, col_idx), cell_val in np.ndenumerate(ary):
        idx = (col_idx + row_idx) % 2
        tb._cells[(row_idx, col_idx)]._text.set_color(font_colors[idx])

    ax.add_table(tb)
    tb.set_fontsize(fontsize)

    return fig
