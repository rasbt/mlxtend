# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


def mcnemar_table(y_target, y_model1, y_model2):
    """
    Compute a 2x2 contigency table for McNemar's test.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels as 1D NumPy array.
    y_model1 : array-like, shape=[n_samples]
        Predicted class labels from model as 1D NumPy array.
    y_model2 : array-like, shape=[n_samples]
        Predicted class labels from model 2 as 1D NumPy array.

    Returns
    ----------
    tb : array-like, shape=[2, 2]
       - Cell [0, 0]: number of samples that both models predicted correctly
       - Cell [1, 1]: number of samples that both models predicted incorrectly
       - Cell [1, 0]: # of samples that model 1 got right and model 2 got wrong
       - Cell [1, 0]: # of samples that model 2 got right and model 1 got wrong

    """

    for ary in (y_target, y_model1, y_model2):
        if len(ary.shape) != 1:
            raise ValueError('One or more input arrays are not 1-dimensional.')

    if y_target.shape[0] != y_model1.shape[0]:
        raise ValueError('y_target and y_model1 contain a different number'
                         ' of elements.')

    if y_target.shape[0] != y_model2.shape[0]:
        raise ValueError('y_target and y_model2 contain a different number'
                         ' of elements.')

    m1_vs_true = (y_target == y_model1).astype(int)
    m2_vs_true = (y_target == y_model2).astype(int)

    plus_true = m1_vs_true + m2_vs_true
    minus_true = m1_vs_true - m2_vs_true

    tb = np.zeros((2, 2), dtype=int)

    tb[0, 0] = np.sum(plus_true == 2)
    tb[1, 1] = np.sum(plus_true == 0)
    tb[1, 0] = np.sum(minus_true == 1)
    tb[0, 1] = np.sum(minus_true == -1)

    return tb
