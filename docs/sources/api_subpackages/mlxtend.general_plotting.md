mlxtend version: 0.3.0
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


## enrichment_plot

*enrichment_plot(df, colors='bgrkcy', markers=' ', linestyles='-', alpha=0.5, lw=2, legend=True, where='post', grid=True, count_label='Count', xlim='auto', ylim='auto', invert_axes=False, legend_loc='best', ax=None)*

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

- `legend` : bool (default: True)

    Plots legend if True.

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


