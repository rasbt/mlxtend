mlxtend  
Sebastian Raschka, 05/14/2015


<hr>

# Standardizing

A function to standardize columns in pandas DataFrames so that they have properties of a standard normal distribution (mean=0, standard deviation=1).

<hr>

## Example


	from mlxtend.pandas import standardizing

![](./img/pandas_scaling_standardizing.png)
    

## Default Parameters

<pre>def standardizing(df, columns):
    """
    Standardizing columns in pandas DataFrames.

    Parameters
    ----------
    df : pandas DataFrame object.

    columns : array-like, shape = [n_columns]
      Array-like with pandas DataFrame column names, e.g., ['col1', 'col2', ...]

    Returns
    ----------

    df_new: pandas DataFrame object.
      Copy of the DataFrame with standardized columns.

    """</pre>