## plot_decision_regions

*plot_decision_regions(X, y, clf, feature_index=None, filler_feature_values=None, filler_feature_ranges=None, ax=None, X_highlight=None, res=None, legend=1, hide_spines=True, markers='s^oxv<>', colors='#1f77b4,#ff7f0e,#3ca02c,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf', scatter_kwargs=None, contourf_kwargs=None, scatter_highlight_kwargs=None)*

Plot decision regions of a classifier.

Please note that this functions assumes that class labels are
labeled consecutively, e.g,. 0, 1, 2, 3, 4, and 5. If you have class
labels with integer labels > 4, you may want to provide additional colors
and/or markers as `colors` and `markers` arguments.
See http://matplotlib.org/examples/color/named_colors.html for more
information.

**Parameters**

- `X` : array-like, shape = [n_samples, n_features]

    Feature Matrix.

- `y` : array-like, shape = [n_samples]

    True class labels.

- `clf` : Classifier object.

    Must have a .predict method.

- `feature_index` : array-like (default: (0,) for 1D, (0, 1) otherwise)

    Feature indices to use for plotting. The first index in
    `feature_index` will be on the x-axis, the second index will be
    on the y-axis.

- `filler_feature_values` : dict (default: None)

    Only needed for number features > 2. Dictionary of feature
    index-value pairs for the features not being plotted.

- `filler_feature_ranges` : dict (default: None)

    Only needed for number features > 2. Dictionary of feature
    index-value pairs for the features not being plotted. Will use the
    ranges provided to select training samples for plotting.

- `ax` : matplotlib.axes.Axes (default: None)

    An existing matplotlib Axes. Creates
    one if ax=None.

- `X_highlight` : array-like, shape = [n_samples, n_features] (default: None)

    An array with data points that are used to highlight samples in `X`.

- `res` : float or array-like, shape = (2,) (default: None)

    This parameter was used to define the grid width,
    but it has been deprecated in favor of
    determining the number of points given the figure DPI and size
    automatically for optimal results and computational efficiency.
    To increase the resolution, it's is recommended to use to provide
    a `dpi argument via matplotlib, e.g., `plt.figure(dpi=600)`.

- `hide_spines` : bool (default: True)

    Hide axis spines if True.

- `legend` : int (default: 1)

    Integer to specify the legend location.
    No legend if legend is 0.

- `markers` : str (default: 's^oxv<>')

    Scatterplot markers.

- `colors` : str (default: 'red,blue,limegreen,gray,cyan')

    Comma separated list of colors.

- `scatter_kwargs` : dict (default: None)

    Keyword arguments for underlying matplotlib scatter function.

- `contourf_kwargs` : dict (default: None)

    Keyword arguments for underlying matplotlib contourf function.

- `scatter_highlight_kwargs` : dict (default: None)

    Keyword arguments for underlying matplotlib scatter function.

**Returns**

- `ax` : matplotlib.axes.Axes object


**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/](http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/)

