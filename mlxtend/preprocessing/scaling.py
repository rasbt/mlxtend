"""
Created: 01/20/2015
Author: Sebastian Raschka
"""
import pandas as pd
import numpy as np

def minmax_scaling(array, columns, min_val=0, max_val=1):
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

    """
    if not (all([isinstance(c, str) for c in columns]) == True
            or all([isinstance(c, int) for c in columns]) == True):
        raise AttributeError('Columns must be of equal type str or int')

    ary_new = array.astype(float)
    if len(ary_new.shape) == 1:
        ary_new = ary_new[:, np.newaxis]

    if isinstance(ary_new, pd.DataFrame):
        ary_newt = ary_new.loc
    elif isinstance(ary_new, np.ndarray):
        ary_newt = ary_new
    else:
        raise AttributeError('Input array must be a pandas DataFrame or NumPy array')

    ary_newt[:, columns] = (ary_newt[:, columns] - ary_newt[:, columns].min(axis=0)) / \
                     (ary_newt[:, columns].max(axis=0) - ary_newt[:, columns].min(axis=0))

    if not min_val == 0 and not max_val == 1:
            ary_newt[:, columns] = ary_newt[:, columns] * (max_val - min_val) + min_val

    return ary_newt[:, columns]

def standardizing(array, columns, ddof=0):
    """
    Standardizing columns in pandas DataFrames.

    Parameters
    ----------
    array : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].

    columns : array-like, shape = [n_columns]
      Array-like with column names, e.g., ['col1', 'col2', ...]
      or column indices [0, 2, 4, ...]

    ddof : int (default: 0)
      Delta Degrees of Freedom. The divisor used in calculations
      is N - ddof, where N represents the number of elements.

    Returns
    ----------
    df_new: pandas DataFrame object.
      Copy of the array or DataFrame with standardized columns.

    """
    if not (all([isinstance(c, str) for c in columns]) == True
            or all([isinstance(c, int) for c in columns]) == True):
        raise AttributeError('Columns must be of equal type str or int')

    ary_new = array.astype(float)
    if len(ary_new.shape) == 1:
        ary_new = ary_new[:, np.newaxis]

    if isinstance(ary_new, pd.DataFrame):
        ary_newt = ary_new.loc
    elif isinstance(ary_new, np.ndarray):
        ary_newt = ary_new
    else:
        raise AttributeError('Input array must be a pandas DataFrame or NumPy array')

    ary_newt[:, columns] = (ary_newt[:, columns] - ary_newt[:, columns].mean(axis=0)) /\
                       ary_newt[:, columns].std(axis=0, ddof=ddof)

    return ary_newt[:, columns]