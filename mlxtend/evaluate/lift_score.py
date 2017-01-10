# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# A function for generating a confusion matrix.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


def lift_score(y_target, y_predicted, binary=True, positive_label=0):
    """Compute a lift score.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels.
    y_predicted : array-like, shape=[n_samples]
        Predicted class labels.
    binary : bool (default: True)
        Maps a multi-class problem onto a
        binary, where
        the positive class is 1 and
        all other classes are 0.
    positive_label : int (default: 0)
        Class label of the positive class.

    Returns
    ----------
    mat : array-like, shape=[n_classes, n_classes]

    """
    if not isinstance(y_target, np.ndarray):
        targ_tmp = np.asarray(y_target)
    else:
        targ_tmp = y_target
    if not isinstance(y_predicted, np.ndarray):
        pred_tmp = np.asarray(y_predicted)
    else:
        pred_tmp = y_predicted

    pred_tmp = pred_tmp.T
    targ_tmp = targ_tmp.T

    if len(pred_tmp) != len(targ_tmp):
        raise AttributeError('`y_target` and `y_predicted`'
                             ' don\'t have the same number of elements.')
    if binary:
        targ_tmp = np.where(targ_tmp != positive_label, 1, 0)
        pred_tmp = np.where(pred_tmp != positive_label, 1, 0)
    return (support(targ_tmp, pred_tmp)/(support(targ_tmp)*support(pred_tmp)))


def support(y_target, y_predicted=None):
    if y_predicted is None:
        if y_target.ndim == 1:
            return (y_target == 1).sum() / float(y_target.shape[0])
        return (y_target == 1).all(axis=1).sum() / float(y_target.shape[0])
    else:
        all_prod = np.column_stack([y_target, y_predicted])
        return (all_prod == 1).all(axis=1).sum() / float(all_prod.shape[0])
