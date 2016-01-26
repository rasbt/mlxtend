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
