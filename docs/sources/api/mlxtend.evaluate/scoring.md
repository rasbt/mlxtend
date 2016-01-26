## scoring



*scoring(y_target, y_predicted, metric='error', positive_label=1)*

Compute a scoring metric for supervised learning.

**Parameters**


- `y_target` : array-like, shape=[n_values]

    True class labels or target values.

- `y_predicted` : array-like, shape=[n_values]

    Predicted class labels or target values.

- `metric` : str (default: 'error')

    Performance metric.
    [TP: True positives, TN = True negatives,

    TN: True negatives, FN = False negatives]

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
    / (sqrt{(TP + FP)( TP + FN )( TN + FP )( TN + FN )})


- `positive_label` : int (default: 1)

    Label of the positive class for binary classification
    metrics.

**Returns**


- `score` : float
