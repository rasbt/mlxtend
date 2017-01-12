# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# A function for scoring predictions.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.evaluate.confusion_matrix import confusion_matrix


def _accuracy(true, pred):
    return (true == pred).sum() / float(true.shape[0])


def _error(true, pred):
    return 1.0 - _accuracy(true, pred)


def _macro(true, pred, func, unique_labels):
    scores = []
    for l in unique_labels:
        scores.append(func(np.where(true != l, 1, 0),
                           np.where(pred != l, 1, 0)))
    return float(sum(scores)) / len(scores)


def scoring(y_target, y_predicted, metric='error',
            positive_label=1, unique_labels='auto'):
    """Compute a scoring metric for supervised learning.

    Parameters
    ------------
    y_target : array-like, shape=[n_values]
        True class labels or target values.
    y_predicted : array-like, shape=[n_values]
        Predicted class labels or target values.
    metric : str (default: 'error')
        Performance metric:
        'accuracy': (TP + TN)/(FP + FN + TP + TN) = 1-ERR\n
        'per-class accuracy': Average per-class accuracy\n
        'per-class error':  Average per-class error\n
        'error': (TP + TN)/(FP+ FN + TP + TN) = 1-ACC\n
        'false_positive_rate': FP/N = FP/(FP + TN)\n
        'true_positive_rate': TP/P = TP/(FN + TP)\n
        'true_negative_rate': TN/N = TN/(FP + TN)\n
        'precision': TP/(TP + FP)\n
        'recall': equal to 'true_positive_rate'\n
        'sensitivity': equal to 'true_positive_rate' or 'recall'\n
        'specificity': equal to 'true_negative_rate'\n
        'f1': 2 * (PRE * REC)/(PRE + REC)\n
        'matthews_corr_coef':  (TP*TN - FP*FN)
           / (sqrt{(TP + FP)( TP + FN )( TN + FP )( TN + FN )})\n
        Where:
        [TP: True positives, TN = True negatives,\n
         TN: True negatives, FN = False negatives]\n
    positive_label : int (default: 1)
        Label of the positive class for binary classification
        metrics.
    unique_labels : str or array-like (default: 'auto')
        If 'auto', deduces the unique class labels from
        y_target

    Returns
    ------------
    score : float

    """
    implemented = {'error',
                   'accuracy',
                   'per-class accuracy',
                   'per-class error',
                   'false_positive_rate',
                   'true_positive_rate',
                   'true_negative_rate',
                   'precision',
                   'recall',
                   'sensitivity',
                   'specificity',
                   'matthews_corr_coef',
                   'f1'}

    if metric not in implemented:
        raise AttributeError('`metric` not in %s' % implemented)

    if len(y_target) != len(y_predicted):
        raise AttributeError('`y_target` and `y_predicted`'
                             ' don\'t have the same number of elements.')

    if unique_labels == 'auto':
        unique_labels = np.unique(y_target)

    if not isinstance(y_target, np.ndarray):
        targ_tmp = np.asarray(y_target)
    else:
        targ_tmp = y_target
    if not isinstance(y_predicted, np.ndarray):
        pred_tmp = np.asarray(y_predicted)
    else:
        pred_tmp = y_predicted

    # multi-class metrics
    if metric == 'accuracy':
        res = _accuracy(targ_tmp, pred_tmp)
    elif metric == 'error':
        res = _error(targ_tmp, pred_tmp)
    elif metric == 'per-class accuracy':
        res = _macro(targ_tmp,
                     pred_tmp,
                     func=_accuracy,
                     unique_labels=unique_labels)
    elif metric == 'per-class error':
        res = _macro(targ_tmp,
                     pred_tmp,
                     func=_error,
                     unique_labels=unique_labels)

    # binary classification metrics
    else:
        if len(unique_labels) > 2 or len(np.unique(pred_tmp)) > 2:
            raise AttributeError('Metrics precision, '
                                 'recall, and f1 only support binary'
                                 ' class labels')

        # `binary=True` makes sure
        # that positive label is 1 and negative label is 0
        cm = confusion_matrix(y_target=targ_tmp,
                              y_predicted=pred_tmp,
                              binary=True,
                              positive_label=positive_label)
        tp = cm[-1, -1]
        fp = cm[0, -1]
        tn = cm[0, 0]
        fn = cm[-1, 0]

        if metric == 'false_positive_rate':
            res = float(fp) / (fp + tn)
        elif metric in {'true_positive_rate', 'recall', 'sensitivity'}:
            res = float(tp) / (fn + tp)
        elif metric in {'true_negative_rate', 'specificity'}:
            res = float(tn) / (fp + tn)
        elif metric == 'precision':
            res = float(tp) / (tp + fp)
        elif metric == 'f1':
            pre = float(tp) / (tp + fp)
            rec = float(tp) / (fn + tp)
            res = 2.0 * (pre * rec) / (pre + rec)
        elif metric == 'matthews_corr_coef':
            res = float(tp * tn - fp * fn)
            res = res / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return res
