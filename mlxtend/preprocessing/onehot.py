# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


def one_hot(y, num_labels="auto", dtype="float"):
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
    ary : numpy.ndarray, shape = [n_classlabels, num_labels]
        One-hot encoded array, where each sample is represented as
        a row vector in the returned array.

    Examples
    ----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/preprocessing/one_hot/

    """
    if not (num_labels == "auto" or isinstance(num_labels, int)):
        raise AttributeError('num_labels must be an integer or "auto"')
    yt = np.asarray(y)
    if yt.ndim != 1:
        raise AttributeError("y array must be 1-dimensional")
    if yt.size and (yt.dtype.kind not in "iu" or np.min(yt) < 0):
        raise ValueError("class labels must be non-negative integers")
    if num_labels == "auto":
        num_labels = int(np.max(yt)) + 1

    ary = np.zeros((len(yt), num_labels), dtype=dtype)
    # Keep the historical zero encoding for a single class.
    if yt.size:
        if np.max(yt) >= num_labels:
            raise ValueError("class label exceeds the requested number of labels")
        if num_labels != 1:
            ary[np.arange(len(yt)), yt] = 1
    return ary
