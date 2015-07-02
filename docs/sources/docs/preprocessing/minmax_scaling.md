mlxtend  
Sebastian Raschka, last updated: 06/02/2015


<hr>

# Minmax Scaling

> from mlxtend.preprocessing import minmax_scaling

A function that applies minmax scaling to pandas DataFrame or NumPy array columns.

<hr>

## Example

	from mlxtend.preprocessing import minmax_scaling

![](./img/scaling_minmax_scaling.png)

## Default Parameters

<pre>def minmax_scaling(array, columns, min_val=0, max_val=1):
    """
    Min max scaling for pandas DataFrames

    Parameters
    ----------
    array : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].

    columns : array-like, shape = [n_columns]
      Array-like with column names, e.g., ['col1', 'col2', ...]
      or column indices [0, 2, 4, ...]

    min_val : `int` or `float`, optional (default=`0`)
      minimum value after rescaling.

    min_val : `int` or `float`, optional (default=`1`)
      maximum value after rescaling.

    Returns
    ----------

    df_new: pandas DataFrame object.
      Copy of the array or DataFrame with rescaled columns.

    """</pre>