# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# A function for plotting a confusion matrix.
# License: BSD 3 clause

import matplotlib.pyplot as plt


def plot_confusion_matrix(conf_mat,
                          hide_spines=False, hide_ticks=False,
                          figsize=(2.5, 2.5),
                          cmap=None, alpha=0.3):
    """Plot a confusion matrix via matplotlib.

    Parameters
    -----------
    conf_mat : array-like, shape = [n_classes, n_classes]
        Confusion matrix from evaluate.confusion matrix.
    hide_spines : bool (default: False)
        Hides axis spines if True.
    hide_ticks : bool (default: False)
        Hides axis ticks if True
    figsize : tuple (default: (2.5, 2.5))
        Height and width of the figure
    cmap : matplotlib colormap (default: `None`)
        Uses matplotlib.pyplot.cm.Blues if `None`

    Returns
    -----------
    fig, ax : matplotlib.pyplot subplot objects
        Figure and axis elements of the subplot.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/

    """
    fig, ax = plt.subplots(figsize=figsize)
    if cmap is None:
        cmap = plt.cm.Blues
    ax.matshow(conf_mat, cmap=cmap, alpha=alpha)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(x=j,
                    y=i,
                    s=conf_mat[i, j],
                    va='center',
                    ha='center')

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

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    return fig, ax
