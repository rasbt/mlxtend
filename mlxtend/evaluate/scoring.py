# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# A function for scoring predictions.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from .confusion_matrix import confusion_matrix


def scoring(y_target, y_predicted, metric='error', positive_label=1):
    """Compute a scoring metric for supervised learning.

    Parameters
    ------------
    y_target : array-like, shape=[n_values]
        True class labels or target values.
    y_predicted : array-like, shape=[n_values]
        Predicted class labels or target values.
    metric : str (default: 'error')
        Performance metric.
        [TP = True positives, TN = True negatives,
         TN = True negatives, FN = False negatives]
        'accuracy': (TP + TN)/(FP + FN + TP + TN) = 1-ERR
        'error': (TP + TN)/(FP+ FN + TP + TN) = 1-ACC
        'false_positive_rate': FP/N = FP/(FP + TN)
        'true_positive_rate': TP/P = TP/(FN + TP)
        'true_negative_rate': TN/N = TN/(FP + TN)
        'precision': TP/(TP + FP)
        'recall': equal to 'true_positive_rate'
        'sensitivity': equal to 'true_positive_rate' or 'recall'
        'specificity': equal to 'true_negative_rate'
        'f1': 2 * (PRE * REC)/(PRE + REC)
        'matthews_corr_coef':  (TP*TN - FP*FN)
            /(sqrt{(TP + FP)( TP + FN )( TN + FP )( TN + FN )})
    positive_label : int (default: 1)
        Label of the positive class for binary classification
        metrics.

    Returns
    ------------
    score : float

    """
    implemented = {'error',
                   'accuracy',
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

    if not isinstance(y_target, np.ndarray):
        targ_tmp = np.asarray(y_target)
    else:
        targ_tmp = y_target
    if not isinstance(y_predicted, np.ndarray):
        pred_tmp = np.asarray(y_predicted)
    else:
        pred_tmp = y_predicted

    # multi-class metrics
    if metric in {'accuracy', 'error'}:
        res = float((targ_tmp == pred_tmp).sum()) / pred_tmp.shape[0]
        if metric == 'error':
            res = 1.0 - res

    # binary classification metrics
    else:
        if len(np.unique(targ_tmp)) > 2 or len(np.unique(pred_tmp)) > 2:
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
            res = 2.0 * (pre * rec)/(pre + rec)
        elif metric == 'matthews_corr_coef':
            res = float(tp*tn - fp*fn)
            res = res / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))

    return res
