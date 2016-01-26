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
