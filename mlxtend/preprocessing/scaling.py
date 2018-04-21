# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Classes for column-based scaling of datasets
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import pandas as pd
import numpy as np


def minmax_scaling(array, columns, min_val=0, max_val=1):
    """Min max scaling of pandas' DataFrames.

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
    df_new : pandas DataFrame object.
        Copy of the array or DataFrame with rescaled columns.

    Examples
    ----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/preprocessing/minmax_scaling/

    """
    ary_new = array.astype(float)
    if len(ary_new.shape) == 1:
        ary_new = ary_new[:, np.newaxis]

    if isinstance(ary_new, pd.DataFrame):
        ary_newt = ary_new.loc
    elif isinstance(ary_new, np.ndarray):
        ary_newt = ary_new
    else:
        raise AttributeError('Input array must be a pandas'
                             'DataFrame or NumPy array')

    numerator = ary_newt[:, columns] - ary_newt[:, columns].min(axis=0)
    denominator = (ary_newt[:, columns].max(axis=0) -
                   ary_newt[:, columns].min(axis=0))
    ary_newt[:, columns] = numerator / denominator

    if not min_val == 0 and not max_val == 1:
        ary_newt[:, columns] = (ary_newt[:, columns] *
                                (max_val - min_val) + min_val)

    return ary_newt[:, columns]


def standardize(array, columns=None, ddof=0, return_params=False, params=None):
    """Standardize columns in pandas DataFrames.

    Parameters
    ----------
    array : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].
    columns : array-like, shape = [n_columns] (default: None)
        Array-like with column names, e.g., ['col1', 'col2', ...]
        or column indices [0, 2, 4, ...]
        If None, standardizes all columns.
    ddof : int (default: 0)
        Delta Degrees of Freedom. The divisor used in calculations
        is N - ddof, where N represents the number of elements.
    return_params : dict (default: False)
        If set to True, a dictionary is returned in addition to the
        standardized array. The parameter dictionary contains the
        column means ('avgs') and standard deviations ('stds') of
        the individual columns.
    params : dict (default: None)
        A dictionary with column means and standard deviations as
        returned by the `standardize` function if `return_params`
        was set to True. If a `params` dictionary is provided, the
        `standardize` function will use these instead of computing
        them from the current array.

    Notes
    ----------
    If all values in a given column are the same, these values are all
    set to `0.0`. The standard deviation in the `parameters` dictionary
    is consequently set to `1.0` to avoid dividing by zero.

    Returns
    ----------
    df_new : pandas DataFrame object.
        Copy of the array or DataFrame with standardized columns.

    Examples
    ----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/preprocessing/standardize/

    """
    ary_new = array.astype(float)
    dim = ary_new.shape
    if len(dim) == 1:
        ary_new = ary_new[:, np.newaxis]

    if isinstance(ary_new, pd.DataFrame):
        ary_newt = ary_new.loc
        if columns is None:
            columns = ary_new.columns
    elif isinstance(ary_new, np.ndarray):
        ary_newt = ary_new
        if columns is None:
            columns = list(range(ary_new.shape[1]))

    else:
        raise AttributeError('Input array must be a pandas '
                             'DataFrame or NumPy array')

    if params is not None:
        parameters = params
    else:
        parameters = {'avgs': ary_newt[:, columns].mean(axis=0),
                      'stds': ary_newt[:, columns].std(axis=0, ddof=ddof)}
    are_constant = np.all(ary_newt[:, columns] == ary_newt[0, columns], axis=0)

    for c, b in zip(columns, are_constant):
        if b:
            ary_newt[:, c] = np.zeros(dim[0])
            parameters['stds'][c] = 1.0

    ary_newt[:, columns] = ((ary_newt[:, columns] - parameters['avgs']) /
                            parameters['stds'])

    if return_params:
        return ary_newt[:, columns], parameters
    else:
        return ary_newt[:, columns]
