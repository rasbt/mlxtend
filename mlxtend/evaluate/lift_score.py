# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# A function for generating a confusion matrix.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


def lift_score(y_target, y_predicted, binary=True, positive_label=1):
    """Lift measures the degree to which the predictions of a
    classification model are better than randomly-generated predictions.

    The in terms of True Positives (TP), True Negatives (TN),
    False Positives (FP), and False Negatives (FN), the lift score is
    computed as:
    [ TP/(TP+FN) ] / [ (TP+FP) / (TP+TN+FP+FN) ]


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
    score : float
        Lift score in the range [0, $\infty$]

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score/

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
                             'don\'t have the same number of elements.')
    if binary:
        targ_tmp = np.where(targ_tmp != positive_label, 0, 1)
        pred_tmp = np.where(pred_tmp != positive_label, 0, 1)

    binary_check_targ_tmp = np.extract(targ_tmp > 1, targ_tmp)
    binary_check_pred_tmp = np.extract(pred_tmp > 1, pred_tmp)

    if len(binary_check_targ_tmp) or len(binary_check_pred_tmp):
        raise AttributeError('`y_target` and `y_predicted`'
                             ' have different elements from 0 and 1.')

    return (support(targ_tmp, pred_tmp) /
            (support(targ_tmp) * support(pred_tmp)))


def support(y_target, y_predicted=None):
    """Support is the fraction of the true value
        in predictions and target values.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels.
    y_predicted : array-like, shape=[n_samples]
        Predicted class labels.

    Returns
    ----------
    score : float
        Support score in the range [0, 1]

    """

    if y_predicted is None:
        if y_target.ndim == 1:
            return (y_target == 1).sum() / float(y_target.shape[0])
        return (y_target == 1).all(axis=1).sum() / float(y_target.shape[0])
    else:
        all_prod = np.column_stack([y_target, y_predicted])
        return (all_prod == 1).all(axis=1).sum() / float(all_prod.shape[0])
