# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# A function for generating per class accuracy
#
# License: BSD 3 clause
import numpy as np


def _compute_metric(target_temp, predicted_temp, normalize=True):
    if normalize:
        return (target_temp == predicted_temp).mean()
    else:
        return (target_temp == predicted_temp).sum()


def accuracy_score(y_target, y_predicted, method="standard",
                   pos_label=1, normalize=True):
    """General accuracy function for supervised learning.
    Parameters
    ------------
    y_target : array-like, shape=[n_values]
        True class labels or target values.
    y_predicted : array-like, shape=[n_values]
        Predicted class labels or target values.
    method : str, 'standard' by default.
        The chosen method for accuracy computation.
        If set to 'standard', computes overall accuracy.
        If set to 'binary', computes accuracy for class pos_label.
        If set to 'average', computes average per class accuracy.
    pos_label : str or int, 1 by default.
        The class whose accuracy score is to be reported.
        Used only when `method` is set to 'binary'
    normalize : bool, True by default.
        If True, returns fraction of correctly classified samples.
        If False, returns number of correctly classified samples.

    Returns
    ------------
    score: float

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/accuracy_score/

    """

    target_temp = np.asarray(y_target)
    predicted_temp = np.asarray(y_predicted)
    unique_labels = np.unique(target_temp)

    if len(y_target) != len(y_predicted):
        raise AttributeError('`y_target` and `y_predicted`'
                             ' don\'t have the same number of elements.')
    if method == "standard":
        return _compute_metric(target_temp, predicted_temp, normalize)

    elif method == "binary":
        if pos_label not in unique_labels:
            raise AttributeError("Chosen value of pos_label doesn't exist")

        target_temp = np.where(target_temp == pos_label, 1, 0)
        predicted_temp = np.where(predicted_temp == pos_label, 1, 0)
        return _compute_metric(target_temp, predicted_temp, normalize)

    elif method == "average":
        return sum([_compute_metric(np.where(target_temp != l, 1, 0),
                    np.where(predicted_temp != l, 1, 0))
                    for l in unique_labels]) / float(unique_labels.shape[0])

    else:
        raise ValueError('`method` must be "standard", "average"'
                         'or "binary". Got "%s".' % method)
