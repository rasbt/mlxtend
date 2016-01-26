## confusion_matrix



*confusion_matrix(y_target, y_predicted, binary=False, positive_label=1)*

Compute a confusion matrix/contingency table.

**Parameters**


- `y_target` : array-like, shape=[n_samples]

    True class labels.

- `y_predicted` : array-like, shape=[n_samples]

    Predicted class labels.

- `binary` : bool (default: True)

    Maps a multi-class problem onto a
    binary confusion matrix, where
    the positive class is 1 and
    all other classes are 0.

- `positive_label` : int (default: 1)

    Class label of the positive class.