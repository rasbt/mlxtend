mlxtend version: 0.15.0dev 
## category_scatter

*category_scatter(x, y, label_col, data, markers='sxo^v', colors=('blue', 'green', 'red', 'purple', 'gray', 'cyan'), alpha=0.7, markersize=20.0, legend_loc='best')*

Scatter plot to plot categories in different colors/markerstyles.

**Parameters**

- `x` : str or int

    DataFrame column name of the x-axis values or
    integer for the numpy ndarray column index.

- `y` : str

    DataFrame column name of the y-axis values or
    integer for the numpy ndarray column index

- `data` : Pandas DataFrame object or NumPy ndarray.


- `markers` : str

    Markers that are cycled through the label category.

- `colors` : tuple

    Colors that are cycled through the label category.

- `alpha` : float (default: 0.7)

    Parameter to control the transparency.

- `markersize` : float (default` : 20.0)

    Parameter to control the marker size.

- `legend_loc` : str (default: 'best')

    Location of the plot legend
    {best, upper left, upper right, lower left, lower right}
    No legend if legend_loc=False

**Returns**

- `fig` : matplotlig.pyplot figure object


**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/plotting/category_scatter/](http://rasbt.github.io/mlxtend/user_guide/plotting/category_scatter/)




## checkerboard_plot

*checkerboard_plot(ary, cell_colors=('white', 'black'), font_colors=('black', 'white'), fmt='%.1f', figsize=None, row_labels=None, col_labels=None, fontsize=None)*

Plot a checkerboard table / heatmap via matplotlib.

**Parameters**

- `ary` : array-like, shape = [n, m]

    A 2D Nnumpy array.

- `cell_colors` : tuple or list (default: ('white', 'black'))

    Tuple or list containing the two colors of the
    checkerboard pattern.

- `font_colors` : tuple or list (default: ('black', 'white'))

    Font colors corresponding to the cell colors.

- `figsize` : tuple (default: (2.5, 2.5))

    Height and width of the figure

- `fmt` : str (default: '%.1f')

    Python string formatter for cell values.
    The default '%.1f' results in floats with 1 digit after
    the decimal point. Use '%d' to show numbers as integers.

- `row_labels` : list (default: None)

    List of the row labels. Uses the array row
    indices 0 to n by default.

- `col_labels` : list (default: None)

    List of the column labels. Uses the array column
    indices 0 to m by default.

- `fontsize` : int (default: None)

    Specifies the font size of the checkerboard table.
    Uses matplotlib's default if None.

**Returns**

- `fig` : matplotlib Figure object.


**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/plotting/checkerboard_plot/](http://rasbt.github.io/mlxtend/user_guide/plotting/checkerboard_plot/)




## ecdf

*ecdf(x, y_label='ECDF', x_label=None, ax=None, percentile=None, ecdf_color=None, ecdf_marker='o', percentile_color='black', percentile_linestyle='--')*

Plots an Empirical Cumulative Distribution Function

**Parameters**

- `x` : array or list, shape=[n_samples,]

    Array-like object containing the feature values

- `y_label` : str (default='ECDF')

    Text label for the y-axis

- `x_label` : str (default=None)

    Text label for the x-axis

- `ax` : matplotlib.axes.Axes (default: None)

    An existing matplotlib Axes. Creates
    one if ax=None

- `percentile` : float (default=None)

    Float between 0 and 1 for plotting a percentile
    threshold line

- `ecdf_color` : matplotlib color (default=None)

    Color for the ECDF plot; uses matplotlib defaults
    if None

- `ecdf_marker` : matplotlib marker (default='o')

    Marker style for the ECDF plot

- `percentile_color` : matplotlib color (default='black')

    Color for the percentile threshold if percentile is not None

- `percentile_linestyle` : matplotlib linestyle (default='--')

    Line style for the percentile threshold if percentile is not None

**Returns**

- `ax` : matplotlib.axes.Axes object


- `percentile_threshold` : float

    Feature threshold at the percentile or None if `percentile=None`

- `percentile_count` : Number of if percentile is not None

    Number of samples that have a feature less or equal than
    the feature threshold at a percentile threshold
    or None if `percentile=None`

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/plotting/ecdf/](http://rasbt.github.io/mlxtend/user_guide/plotting/ecdf/)




## enrichment_plot

*enrichment_plot(df, colors='bgrkcy', markers=' ', linestyles='-', alpha=0.5, lw=2, where='post', grid=True, count_label='Count', xlim='auto', ylim='auto', invert_axes=False, legend_loc='best', ax=None)*

Plot stacked barplots

**Parameters**

- `df` : pandas.DataFrame

    A pandas DataFrame where columns represent the different categories.
    colors: str (default: 'bgrcky')
    The colors of the bars.

- `markers` : str (default: ' ')

    Matplotlib markerstyles, e.g,
    'sov' for square,circle, and triangle markers.

- `linestyles` : str (default: '-')

    Matplotlib linestyles, e.g.,
    '-,--' to cycle normal and dashed lines. Note
    that the different linestyles need to be separated by commas.

- `alpha` : float (default: 0.5)

    Transparency level from 0.0 to 1.0.

- `lw` : int or float (default: 2)

    Linewidth parameter.

- `where` : {'post', 'pre', 'mid'} (default: 'post')

    Starting location of the steps.

- `grid` : bool (default: `True`)

    Plots a grid if True.

- `count_label` : str (default: 'Count')

    Label for the "Count"-axis.

- `xlim` : 'auto' or array-like [min, max] (default: 'auto')

    Min and maximum position of the x-axis range.

- `ylim` : 'auto' or array-like [min, max] (default: 'auto')

    Min and maximum position of the y-axis range.

- `invert_axes` : bool (default: False)

    Plots count on the x-axis if True.

- `legend_loc` : str (default: 'best')

    Location of the plot legend
    {best, upper left, upper right, lower left, lower right}
    No legend if legend_loc=False

- `ax` : matplotlib axis, optional (default: None)

    Use this axis for plotting or make a new one otherwise

**Returns**

- `ax` : matplotlib axis


**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/plotting/enrichment_plot/](http://rasbt.github.io/mlxtend/user_guide/plotting/enrichment_plot/)




## plot_confusion_matrix

*plot_confusion_matrix(conf_mat, hide_spines=False, hide_ticks=False, figsize=None, cmap=None, colorbar=False, show_absolute=True, show_normed=False)*

Plot a confusion matrix via matplotlib.
**Parameters**

- `conf_mat` : array-like, shape = [n_classes, n_classes]

    Confusion matrix from evaluate.confusion matrix.

- `hide_spines` : bool (default: False)

    Hides axis spines if True.

- `hide_ticks` : bool (default: False)

    Hides axis ticks if True

- `figsize` : tuple (default: (2.5, 2.5))

    Height and width of the figure

- `cmap` : matplotlib colormap (default: `None`)

    Uses matplotlib.pyplot.cm.Blues if `None`

- `colorbar` : bool (default: False)

    Shows a colorbar if True

- `show_absolute` : bool (default: True)

    Shows absolute confusion matrix coefficients if True.
    At least one of  `show_absolute` or `show_normed`
    must be True.

- `show_normed` : bool (default: False)

    Shows normed confusion matrix coefficients if True.
    The normed confusion matrix coefficients give the
    proportion of training examples per class that are
    assigned the correct label.
    At least one of  `show_absolute` or `show_normed`
    must be True.
**Returns**

- `fig, ax` : matplotlib.pyplot subplot objects

    Figure and axis elements of the subplot.
**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/](http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/)




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




## plot_learning_curves

*plot_learning_curves(X_train, y_train, X_test, y_test, clf, train_marker='o', test_marker='^', scoring='misclassification error', suppress_plot=False, print_model=True, style='fivethirtyeight', legend_loc='best')*

Plots learning curves of a classifier.

**Parameters**

- `X_train` : array-like, shape = [n_samples, n_features]

    Feature matrix of the training dataset.

- `y_train` : array-like, shape = [n_samples]

    True class labels of the training dataset.

- `X_test` : array-like, shape = [n_samples, n_features]

    Feature matrix of the test dataset.

- `y_test` : array-like, shape = [n_samples]

    True class labels of the test dataset.

- `clf` : Classifier object. Must have a .predict .fit method.


- `train_marker` : str (default: 'o')

    Marker for the training set line plot.

- `test_marker` : str (default: '^')

    Marker for the test set line plot.

- `scoring` : str (default: 'misclassification error')

    If not 'misclassification error', accepts the following metrics
    (from scikit-learn):
    {'accuracy', 'average_precision', 'f1_micro', 'f1_macro',
    'f1_weighted', 'f1_samples', 'log_loss',
    'precision', 'recall', 'roc_auc',
    'adjusted_rand_score', 'mean_absolute_error', 'mean_squared_error',
    'median_absolute_error', 'r2'}

- `suppress_plot=False` : bool (default: False)

    Suppress matplotlib plots if True. Recommended
    for testing purposes.

- `print_model` : bool (default: True)

    Print model parameters in plot title if True.

- `style` : str (default: 'fivethirtyeight')

    Matplotlib style

- `legend_loc` : str (default: 'best')

    Where to place the plot legend:
    {'best', 'upper left', 'upper right', 'lower left', 'lower right'}

**Returns**

- `errors` : (training_error, test_error): tuple of lists


**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/plotting/plot_learning_curves/](http://rasbt.github.io/mlxtend/user_guide/plotting/plot_learning_curves/)




## plot_linear_regression

*plot_linear_regression(X, y, model=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False), corr_func='pearsonr', scattercolor='blue', fit_style='k--', legend=True, xlim='auto')*

Plot a linear regression line fit.

**Parameters**

- `X` : numpy array, shape = [n_samples,]

    Samples.

- `y` : numpy array, shape (n_samples,)

    Target values
    model: object (default: sklearn.linear_model.LinearRegression)
    Estimator object for regression. Must implement
    a .fit() and .predict() method.
    corr_func: str or function (default: 'pearsonr')
    Uses `pearsonr` from scipy.stats if corr_func='pearsonr'.
    to compute the regression slope. If not 'pearsonr', the `corr_func`,
    the `corr_func` parameter expects a function of the form
    func(<x-array>, <y-array>) as inputs, which is expected to return
    a tuple `(<correlation_coefficient>, <some_unused_value>)`.
    scattercolor: string (default: blue)
    Color of scatter plot points.
    fit_style: string (default: k--)
    Style for the line fit.
    legend: bool (default: True)
    Plots legend with corr_coeff coef.,
    fit coef., and intercept values.
    xlim: array-like (x_min, x_max) or 'auto' (default: 'auto')
    X-axis limits for the linear line fit.

**Returns**

- `regression_fit` : tuple

    intercept, slope, corr_coeff (float, float, float)

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/plotting/plot_linear_regression/](http://rasbt.github.io/mlxtend/user_guide/plotting/plot_linear_regression/)




## plot_sequential_feature_selection

*plot_sequential_feature_selection(metric_dict, kind='std_dev', color='blue', bcolor='steelblue', marker='o', alpha=0.2, ylabel='Performance', confidence_interval=0.95)*

Plot feature selection results.

**Parameters**

- `metric_dict` : mlxtend.SequentialFeatureSelector.get_metric_dict() object


- `kind` : str (default: "std_dev")

    The kind of error bar or confidence interval in
    {'std_dev', 'std_err', 'ci', None}.

- `color` : str (default: "blue")

    Color of the lineplot (accepts any matplotlib color name)

- `bcolor` : str (default: "steelblue").

    Color of the error bars / confidence intervals
    (accepts any matplotlib color name).

- `marker` : str (default: "o")

    Marker of the line plot
    (accepts any matplotlib marker name).

- `alpha` : float in [0, 1] (default: 0.2)

    Transparency of the error bars / confidence intervals.

- `ylabel` : str (default: "Performance")

    Y-axis label.

- `confidence_interval` : float (default: 0.95)

    Confidence level if `kind='ci'`.

**Returns**

- `fig` : matplotlib.pyplot.figure() object


**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/plotting/plot_sequential_feature_selection/](http://rasbt.github.io/mlxtend/user_guide/plotting/plot_sequential_feature_selection/)




## remove_borders

*remove_borders(axes, left=False, bottom=False, right=True, top=True)*

Remove chart junk from matplotlib plots.

**Parameters**

- `axes` : iterable

    An iterable containing plt.gca()
    or plt.subplot() objects, e.g. [plt.gca()].

- `left` : bool (default: `False`)

    Hide left axis spine if True.

- `bottom` : bool (default: `False`)

    Hide bottom axis spine if True.

- `right` : bool (default: `True`)

    Hide right axis spine if True.

- `top` : bool (default: `True`)

    Hide top axis spine if True.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/plotting/remove_chartjunk/](http://rasbt.github.io/mlxtend/user_guide/plotting/remove_chartjunk/)




## scatterplotmatrix

*scatterplotmatrix(X, fig_axes=None, names=None, figsize=(8, 8), alpha=1.0, **kwargs)*

Lower triangular of a scatterplot matrix

**Parameters**

- `X` : array-like, shape={num_examples, num_features}

    Design matrix containing data instances (examples)
    with multiple exploratory variables (features).


- `fix_axes` : tuple (default: None)

    A `(fig, axes)` tuple, where fig is an figure object
    and axes is an axes object created via matplotlib,
    for example, by calling the pyplot `subplot` function
    `fig, axes = plt.subplots(...)`


- `names` : list (default: None)

    A list of string names, which should have the same number
    of elements as there are features (columns) in `X`.


- `figsize` : tuple (default: (8, 8))

    Height and width of the subplot grid. Ignored if
    fig_axes is not `None`.


- `alpha` : float (default: 1.0)

    Transparency for both the scatter plots and the
    histograms along the diagonal.


- `**kwargs` : kwargs

    Keyword arguments for the scatterplots.

**Returns**

- `fix_axes` : tuple

    A `(fig, axes)` tuple, where fig is an figure object
    and axes is an axes object created via matplotlib,
    for example, by calling the pyplot `subplot` function
    `fig, axes = plt.subplots(...)`




## stacked_barplot

*stacked_barplot(df, bar_width='auto', colors='bgrcky', labels='index', rotation=90, legend_loc='best')*

Function to plot stacked barplots

**Parameters**

- `df` : pandas.DataFrame

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

- `legend_loc` : str (default: 'best')

    Location of the plot legend
    {best, upper left, upper right, lower left, lower right}
    No legend if legend_loc=False

**Returns**

- `fig` : matplotlib.pyplot figure object


**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/plotting/stacked_barplot/](http://rasbt.github.io/mlxtend/user_guide/plotting/stacked_barplot/)




