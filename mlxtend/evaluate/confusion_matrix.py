# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# A function for generating a confusion matrix.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from itertools import product
import numpy as np


def confusion_matrix(y_target, y_predicted, binary=False, positive_label=1):
    """Compute a confusion matrix/contingency table.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels.
    y_predicted : array-like, shape=[n_samples]
        Predicted class labels.
    binary : bool (default: False)
        Maps a multi-class problem onto a
        binary confusion matrix, where
        the positive class is 1 and
        all other classes are 0.
    positive_label : int (default: 1)
        Class label of the positive class.

    Returns
    ----------
    mat : array-like, shape=[n_classes, n_classes]

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix/

    """
    if not isinstance(y_target, np.ndarray):
        targ_tmp = np.asarray(y_target)
    else:
        targ_tmp = y_target
    if not isinstance(y_predicted, np.ndarray):
        pred_tmp = np.asarray(y_predicted)
    else:
        pred_tmp = y_predicted

    if len(y_target) != len(y_predicted):
        raise AttributeError('`y_target` and `y_predicted`'
                             ' don\'t have the same number of elements.')

    if binary:
        targ_tmp = np.where(targ_tmp != positive_label, 0, 1)
        pred_tmp = np.where(pred_tmp != positive_label, 0, 1)

    class_labels = np.unique(np.concatenate((targ_tmp, pred_tmp)))
    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])
    n_labels = class_labels.shape[0]
    lst = []
    z = list(zip(targ_tmp, pred_tmp))
    for combi in product(class_labels, repeat=2):
        lst.append(z.count(combi))
    mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)
    return mat
