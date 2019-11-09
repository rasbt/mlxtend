# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# A function for generating per class accuracy
#
# License: BSD 3 clause
import numpy as np


def per_class_accuracy(y_target, y_predicted, pos_label=None, binary=False):
    """Compute per_class_accuracy for supervised learning.
    Parameters
    ------------
    y_target : array-like, shape=[n_values]
        True class labels or target values.
    y_predicted : array-like, shape=[n_values]
        Predicted class labels or target values.
    pos_label : str or int, None by default.
        The class whose accuracy score is to be reported.
        'binary' has to be set to True.
    binary : bool
        Projects a multi-label problem to a binary problem.
        If set to True, it uses `pos_label` to binarize.

    Returns
    ------------
    score: float
    """

    target_temp = np.asarray(y_target)
    predicted_temp = np.asarray(y_predicted)

    if len(y_target) != len(y_predicted):
        raise AttributeError('`y_target` and `y_predicted`'
                             ' don\'t have the same number of elements.')
    if binary and pos_label is not None:
        if pos_label not in np.unique(target_temp):
            raise AttributeError("Chosen value of pos_label doesn't exist")

        target_temp = np.where(target_temp == pos_label, 1, 0)
        predicted_temp = np.where(predicted_temp == pos_label, 1, 0)

    return (target_temp == predicted_temp).sum() / float(target_temp.shape[0])
