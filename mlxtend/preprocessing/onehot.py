# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from .transactionencoder import TransactionEncoder
import warnings


def one_hot(y, num_labels='auto', dtype='float'):
    """One-hot encoding of class labels

    Parameters
    ----------
    y : array-like, shape = [n_classlabels]
        Python list or numpy array consisting of class labels.
    num_labels : int or 'auto'
        Number of unique labels in the class label array. Infers the number
        of unique labels from the input array if set to 'auto'.
    dtype : str
        NumPy array type (float, float32, float64) of the output array.

    Returns
    ----------
    ary : numpy.ndarray, shape = [n_classlabels]
        One-hot encoded array, where each sample is represented as
        a row vector in the returned array.

    Examples
    ----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/preprocessing/one_hot/

    """
    if not (num_labels == 'auto' or isinstance(num_labels, int)):
        raise AttributeError('num_labels must be an integer or "auto"')
    if isinstance(y, list):
        yt = np.asarray(y)
    else:
        yt = y
    if not len(yt.shape) == 1:
        raise AttributeError('y array must be 1-dimensional')
    if num_labels == 'auto':
        # uniq = np.unique(yt).shape[0]
        uniq = np.max(yt + 1)
    else:
        uniq = num_labels
    if uniq == 1:
        ary = np.array([[0.]], dtype=dtype)

    else:
        ary = np.zeros((len(y), uniq))
        for i, val in enumerate(y):
            ary[i, val] = 1

    return ary.astype(dtype)


class OnehotTransactions(TransactionEncoder):
    def __init__(self, *args, **kwargs):
        warnings.simplefilter('default')
        msg = ("OnehotTransactions has been deprecated and will be removed"
               " in future. Please use TransactionEncoder instead.")
        warnings.warn(msg, DeprecationWarning)

        super().__init__(*args, **kwargs)
