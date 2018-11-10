## lift_score

*lift_score(y_target, y_predicted, binary=True, positive_label=1)*

Lift measures the degree to which the predictions of a
classification model are better than randomly-generated predictions.

The in terms of True Positives (TP), True Negatives (TN),
False Positives (FP), and False Negatives (FN), the lift score is
computed as:
[ TP / (TP+FP) ] / [ (TP+FN) / (TP+TN+FP+FN) ]


**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels.

- `y_predicted` : array-like, shape=[n_samples]

    Predicted class labels.

- `binary` : bool (default: True)

    Maps a multi-class problem onto a
    binary, where
    the positive class is 1 and
    all other classes are 0.

- `positive_label` : int (default: 0)

    Class label of the positive class.

**Returns**

- `score` : float

    Lift score in the range [0, $\infty$]

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score/](http://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score/)

