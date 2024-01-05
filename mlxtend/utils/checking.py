# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# A counter class for printing the progress of an iterator.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


def check_Xy(X, y, y_int=True):
    # check types
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a NumPy array. Found %s" % type(X))
    if not isinstance(y, np.ndarray):
        raise ValueError("y must be a NumPy array. Found %s" % type(y))

    if "int" not in str(y.dtype):
        raise ValueError(
            "y must be an integer array. Found %s. "
            "Try passing the array as y.astype(np.int_)" % y.dtype
        )

    if not ("float" in str(X.dtype) or "int" in str(X.dtype)):
        raise ValueError("X must be an integer or float array. Found %s." % X.dtype)

    # check dim
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array. Found %s" % str(X.shape))
    if len(y.shape) > 1:
        raise ValueError("y must be a 1D array. Found %s" % str(y.shape))

    # check other
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            "y and X must contain the same number of samples. "
            "Got y: %d, X: %d" % (y.shape[0], X.shape[0])
        )


def format_kwarg_dictionaries(
    default_kwargs=None, user_kwargs=None, protected_keys=None
):
    """Function to combine default and user specified kwargs dictionaries

    Parameters
    ----------
    default_kwargs : dict, optional
        Default kwargs (default is None).
    user_kwargs : dict, optional
        User specified kwargs (default is None).
    protected_keys : array_like, optional
        Sequence of keys to be removed from the returned dictionary
        (default is None).

    Returns
    -------
    formatted_kwargs : dict
        Formatted kwargs dictionary.
    """
    formatted_kwargs = {}
    for d in [default_kwargs, user_kwargs]:
        if not isinstance(d, (dict, type(None))):
            raise TypeError(
                "d must be of type dict or None, but " "got {} instead".format(type(d))
            )
        if d is not None:
            formatted_kwargs.update(d)
    if protected_keys is not None:
        for key in protected_keys:
            formatted_kwargs.pop(key, None)

    return formatted_kwargs
