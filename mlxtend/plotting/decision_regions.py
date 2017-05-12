# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# A function for plotting decision regions of classifiers.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections


def plot_decision_regions(X, y, clf,
                          feature_index=None,
                          filler_feature_dict=None,
                          ax=None,
                          X_highlight=None,
                          res=0.02, legend=1,
                          hide_spines=True,
                          markers='s^oxv<>',
                          colors='red,blue,limegreen,gray,cyan'):
    """Plot decision regions of a classifier.

    Please note that this functions assumes that class labels are
    labeled consecutively, e.g,. 0, 1, 2, 3, 4, and 5. If you have class
    labels with integer labels > 4, you may want to provide additional colors
    and/or markers as `colors` and `markers` arguments.
    See http://matplotlib.org/examples/color/named_colors.html for more
    information.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Feature Matrix.
    y : array-like, shape = [n_samples]
        True class labels.
    clf : Classifier object.
        Must have a .predict method.
    feature_index : array-like (default: (0,) for 1D, (0, 1) otherwise)
        Feature indices to use for plotting. The first index in
        feature_index will be on the x-axis, the second index will be
        on the y-axis.
    filler_feature_dict : dict (default: None)
        Only needed for number features > 2. Dictionary of feature
        index-value pairs for the features not being plotted.
    ax : matplotlib.axes.Axes (default: None)
        An existing matplotlib Axes. Creates
        one if ax=None.
    X_highlight : array-like, shape = [n_samples, n_features] (default: None)
        An array with data points that are used to highlight samples in `X`.
    res : float (default: 0.02)
        Grid width. Lower values increase the resolution but
        slow down the plotting.
    hide_spines : bool (default: True)
        Hide axis spines if True.
    legend : int (default: 1)
        Integer to specify the legend location.
        No legend if legend is 0.
    markers : list
        Scatterplot markers.
    colors : str (default 'red,blue,limegreen,gray,cyan')
        Comma separated list of colors.

    Returns
    ---------
    ax : matplotlib.axes.Axes object

    """

    if not isinstance(X, np.ndarray):
        raise ValueError('X must be a 2D NumPy array')
    if not isinstance(y, np.ndarray):
        raise ValueError('y must be a 1D NumPy array')
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError('y must have be an integer array. '
                         'Try passing the array as y.astype(np.integer)')

    if ax is None:
        ax = plt.gca()

    plot_testdata = True
    if not isinstance(X_highlight, np.ndarray):
        if X_highlight is not None:
            raise ValueError('X_highlight must be a NumPy array or None')
        else:
            plot_testdata = False

    if len(X.shape) != 2:
        raise ValueError('X must be a 2D array')
    elif isinstance(X_highlight, np.ndarray) and len(X_highlight.shape) < 2:
        raise ValueError('X_highlight must be a 2D array')
    elif len(y.shape) > 1:
        raise ValueError('y must be a 1D array')
    else:
        dim = X.shape[1]

    # Extra input validations for higher number of training features
    if dim > 2:
        if filler_feature_dict is None:
            raise ValueError('Filler values must be provided when X has more than 2 features. ')
        if feature_index is not None:
            try:
                x_index, y_index = feature_index
            except ValueError:
                raise ValueError('Unable to unpack feature_index. '
                                 'Make sure feature_index has two dimensions.')

    marker_gen = cycle(list(markers))

    n_classes = np.unique(y).shape[0]
    colors = colors.split(',')
    colors_gen = cycle(colors)
    colors = [next(colors_gen) for c in range(n_classes)]

    if dim == 1:
        y_min, y_max = -1, 1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    elif dim == 2:
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    else:
        if feature_index is None:
            feature_index = (0, 1)
        x_index, y_index = feature_index
        y_min, y_max = X[:, y_index].min() - 1, X[:, y_index].max() + 1
        x_min, x_max = X[:, x_index].min() - 1, X[:, x_index].max() + 1


    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    if dim == 1:
        Z = clf.predict(np.array([xx.ravel()]).T)
    elif dim == 2:
        Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    else:
        # Need to create feature array with filled in values
        X = np.array([xx.ravel(), yy.ravel()]).T
        X_predict = np.zeros((X.shape[0], dim))
        X_predict[:, x_index] = X[:, 0]
        X_predict[:, y_index] = X[:, 1]
        for feature_index in filler_feature_dict:
            X_predict[:, feature_index] = filler_feature_dict[feature_index]
        Z = clf.predict(X_predict)

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z,
                alpha=0.3,
                colors=colors,
                levels=np.arange(Z.max() + 2) - 0.5)

    ax.axis(xmin=xx.min(), xmax=xx.max(), y_min=yy.min(), y_max=yy.max())

    if dim <= 2:
        for idx, c in enumerate(np.unique(y)):
            if dim == 2:
                y_data = X[y == c, 1]
            else:
                y_data = [0 for i in X[y == c]]

            ax.scatter(x=X[y == c, 0],
                       y=y_data,
                       alpha=0.8,
                       c=colors[idx],
                       marker=next(marker_gen),
                       edgecolor='black',
                       label=c)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if dim == 1:
        ax.axes.get_yaxis().set_ticks([])

    if legend and dim <= 2:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, framealpha=0.3, scatterpoints=1, loc=legend)

    if plot_testdata and dim <= 2:
        if dim == 2:
            ax.scatter(X_highlight[:, 0],
                       X_highlight[:, 1],
                       c='',
                       edgecolor='black',
                       alpha=1.0,
                       linewidths=1,
                       marker='o',
                       s=80)
        else:
            ax.scatter(X_highlight,
                       [0 for i in X_highlight],
                       c='',
                       edgecolor='black',
                       alpha=1.0,
                       linewidths=1,
                       marker='o',
                       s=80)

    return ax


def plot_decision_region_slices(xkey, ykey, data, training_features, target_feature, clf,
        filler_feature_dict=None, xres=0.1, yres=0.1, xlim=None, ylim=None,
        colors='C0,C1,C2,C3,C4', ax=None):
    '''Function to plot 2D decision region of a scikit-learn classifier

    Parameters
    ----------
    xkey : str
        Key for feature on x-axis.
    ykey : str
        Key for feature on y-axis.
    data : pandas.DataFrame
        DataFrame containing the training dataset. Must contain columns with training features and target used in training the classifier clf.
    training_features : list
        List of the training features used to train clf.
    target_feature : str
        Target feature column name.
    clf : fitted scikit-learn classifier or pipeline
        The fitted scikit-learn classifier for which you would like to vizulaize the decision regions
    filler_feature_dict : dict, optional
        Dictionary containing key-value pairs for the training features other than those given by xkey and ykey. Required if number of training features is larger than two.
    xres : float, optional
        The grid spacing used along the x-axis when evaluating the decision region (default is 0.1).
    yres : float, optional
        The grid spacing used along the y-axis when evaluating the decision region (default is 0.1).
    xlim : tuple, int, optional
        If specified, will be used to set the x-axis limit.
    ylim : tuple, int, optional
        If specified, will be used to set the y-axis limit.
    colors: str, optional
        Comma separated list of colors. (default is 'C0,C1,C2,C3,C4')
    ax : matplotlib.axes
        If specified, will plot decision region on ax. Otherwise will create an ax instance.

    Returns
    -------
    matplotlib.axes
        Matplotlib axes with the classifier decision region added.

    '''
    # Validate input types
    if not isinstance(data, pd.DataFrame):
        raise ValueError('data must be a pandas DataFrame')
    if not all([key in data.columns for key in [xkey, ykey]]):
        raise ValueError('Both xkey and ykey must be in data.columns')
    if not isinstance(filler_feature_dict, dict):
        raise ValueError('filler_feature_dict must be a dictionary')

    n_features = len(training_features)
    # Check to see that all the specified featues are consistant
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    # If less than 3 training features, use plot_decision_regions
    if n_features <= 2:
        plot_decision_regions(data[training_features].values,
                              data[target_feature].values, clf,
                              ax=ax,
                              X_highlight=None,
                              res=xres, legend=1,
                              markers='s^oxv<>',
                              colors=colors)
    else:
        if not filler_feature_dict:
            raise ValueError('filler_feature_dict must be provided if using more than 2 training features')
        if not compare(training_features, list(filler_feature_dict.keys()) + [xkey, ykey]):
            raise ValueError('The xkey, ykey, and filler feature keys are not the same as data.columns')

    # Extract the minimum and maximum values of the x-y decision region features
    x_min = data[xkey].min()
    x_max = data[xkey].max()
    y_min = data[ykey].min()
    y_max = data[ykey].max()
    # Construct x-y meshgrid for the specified features
    x_array = np.arange(x_min, x_max, xres)
    y_array = np.arange(y_min, y_max, yres)
    xx1, xx2 = np.meshgrid(x_array, y_array)
    # X should have a row for each x-y point in the meshgrid (will be used in pipeline.predict later)
    X = np.array([xx1.ravel(), xx2.ravel()]).T
    # Now we need to include the filler values for the other training features
    # Construct a DataFrame from X
    df_temp = pd.DataFrame({xkey: X[:, 0], ykey: X[:, 1]}, columns=[xkey, ykey])
    # Add a new column for each other the other non-plotted training features
    for key in filler_feature_dict:
        df_temp[key] = filler_feature_dict[key]
    # Reorder the columns of df_temp to match those used in training clf
    df_temp = df_temp[training_features]
    X_predict = df_temp.values

    Z = clf.predict(X_predict)
    Z = Z.reshape(xx1.shape)

    if ax is None:
        ax = plt.gca()

    n_classes = np.unique(data[target_feature]).shape[0]
    colors = colors.split(',')
    colors_gen = cycle(colors)
    colors = [next(colors_gen) for c in range(n_classes)]
    ax.contourf(xx1, xx2, Z, alpha=0.3, colors=colors, levels=np.arange(Z.max() + 2) - 0.5)

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    return ax
