# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# A function for plotting decision regions of classifiers.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np


def ecdf(x, y_label='ECDF', x_label=None, ax=None, percentile=None,
         ecdf_color=None, ecdf_marker='o', percentile_color='black',
         percentile_linestyle='--'):
    """Plots an Empirical Cumulative Distribution Function

    Parameters
    ----------
    x : array or list, shape=[n_samples,]
        Array-like object containing the feature values
    y_label : str (default='ECDF')
        Text label for the y-axis
    x_label : str (default=None)
        Text label for the x-axis
    ax : matplotlib.axes.Axes (default: None)
        An existing matplotlib Axes. Creates
        one if ax=None
    percentile : float (default=None)
        Float between 0 and 1 for plotting a percentile
        threshold line
    ecdf_color : matplotlib color (default=None)
        Color for the ECDF plot; uses matplotlib defaults
        if None
    ecdf_marker : matplotlib marker (default='o')
        Marker style for the ECDF plot
    percentile_color : matplotlib color (default='black')
        Color for the percentile threshold if percentile is not None
    percentile_linestyle : matplotlib linestyle (default='--')
        Line style for the percentile threshold if percentile is not None

    Returns
    ---------
    ax : matplotlib.axes.Axes object
    percentile_threshold : float
        Feature threshold at the percentile or None if `percentile=None`
    percentile_count : Number of if percentile is not None
        Number of samples that have a feature less or equal than
        the feature threshold at a percentile threshold
        or None if `percentile=None`

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/ecdf/

    """
    if ax is None:
        ax = plt.gca()

    x = np.sort(x)
    y = np.arange(1, x.shape[0] + 1) / float(x.shape[0])

    ax.plot(x, y,
            marker=ecdf_marker,
            linestyle='',
            color=ecdf_color)
    ax.set_ylabel('ECDF')
    if x_label is not None:
        ax.set_xlabel(x_label)
    if percentile:
        targets = x[y <= percentile]
        percentile_threshold = targets.max()
        percentile_count = targets.shape[0]
        ax.axvline(percentile_threshold,
                   color=percentile_color,
                   linestyle=percentile_linestyle)

    else:
        percentile_threshold = None
        percentile_count = None

    return ax, percentile_threshold, percentile_count
