# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Matplotlib wrapper for generating stacked barplots.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np


def scatterplotmatrix(X, fig_axes=None, names=None,
                      figsize=(8, 8), alpha=1.0, **kwargs):
    """
    Lower triangular of a scatterplot matrix

    Parameters
    -----------
    X : array-like, shape={num_examples, num_features}
      Design matrix containing data instances (examples)
      with multiple exploratory variables (features).

    fix_axes : tuple (default: None)
      A `(fig, axes)` tuple, where fig is an figure object
      and axes is an axes object created via matplotlib,
      for example, by calling the pyplot `subplot` function
      `fig, axes = plt.subplots(...)`

    names : list (default: None)
      A list of string names, which should have the same number
      of elements as there are features (columns) in `X`.

    figsize : tuple (default: (8, 8))
      Height and width of the subplot grid. Ignored if
      fig_axes is not `None`.

    alpha : float (default: 1.0)
      Transparency for both the scatter plots and the
      histograms along the diagonal.

    **kwargs : kwargs
      Keyword arguments for the scatterplots.

    Returns
    --------
    fix_axes : tuple
      A `(fig, axes)` tuple, where fig is an figure object
      and axes is an axes object created via matplotlib,
      for example, by calling the pyplot `subplot` function
      `fig, axes = plt.subplots(...)`

    Examples
    ----------
    For more usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/scatterplotmatrix/

    """

    num_examples, num_features = X.shape

    if fig_axes is None:
        fig, axes = plt.subplots(nrows=num_features,
                                 ncols=num_features,
                                 figsize=figsize)
    else:
        fig, axes = fig_axes

    if names is None:
        names = ['X%d' % (i+1) for i in range(num_features)]

    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        axes[j, i].scatter(X[:, j], X[:, i], alpha=alpha, **kwargs)
        axes[j, i].set_xlabel(names[j])
        axes[j, i].set_ylabel(names[i])
        axes[i, j].set_axis_off()

    for i in range(num_features):
        axes[i, i].hist(X[:, i], alpha=alpha)
        axes[i, i].set_ylabel('Count')
        axes[i, i].set_xlabel(names[i])

    return fig, axes
