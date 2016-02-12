mlxtend version: 0.3.1dev
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

## plot_confusion_matrix

*plot_confusion_matrix(conf_mat, hide_spines=False, hide_ticks=False, figsize=(2.5, 2.5), cmap=None, alpha=0.3)*

Plot a confusion matrix via matplotlib.

**Parameters**

- `conf_mat` : array-like, shape = [n_classes, n_classes]

    Confusion matrix from evaluate.confusion matrix.

- `hide_spines` : bool (default: False)

    Hides axis spines if True.

- `hide_ticks` : bool (default: False)

    Hides axis ticks if True

- `figsize` : tuple (default: (2.5, 2.5))

    Height and width of the figure

- `cmap` : matplotlib colormap (default: `None`)

    Uses matplotlib.pyplot.cm.Blues if `None`

**Returns**

- `fig, ax` : matplotlib.pyplot subplot objects

    Figure and axis elements of the subplot.

## plot_decision_regions

*plot_decision_regions(X, y, clf, X_highlight=None, res=0.02, legend=1, hide_spines=True, markers='s^oxv<>', colors=['red', 'blue', 'limegreen', 'gray', 'cyan'])*

Plot decision regions of a classifier.

**Parameters**

- `X` : array-like, shape = [n_samples, n_features]

    Feature Matrix.

- `y` : array-like, shape = [n_samples]

    True class labels.

- `clf` : Classifier object.

    Must have a .predict method.

- `X_highlight` : array-like, shape = [n_samples, n_features] (default: None)

    An array with data points that are used to highlight samples in `X`.

- `res` : float (default: 0.02)

    Grid width. Lower values increase the resolution but
    slow down the plotting.

- `hide_spines` : bool (default: True)

    Hide axis spines if True.

- `legend` : int (default: 1)

    Integer to specify the legend location.
    No legend if legend is 0.

- `markers` : list

    Scatterplot markers.

- `colors` : list

    Colors.

**Returns**

- `fig` : matplotlib.pyplot.figure object


## plot_learning_curves

*plot_learning_curves(X_train, y_train, X_test, y_test, clf, train_marker='o', test_marker='^', scoring='misclassification error', suppress_plot=False, print_model=True, style='fivethirtyeight', legend_loc='best')*

Plots learning curves of a classifier.

**Parameters**

- `X_train` : array-like, shape = [n_samples, n_features]

    Feature matrix of the training dataset.

- `y_train` : array-like, shape = [n_samples]

    True class labels of the training dataset.

- `X_test` : array-like, shape = [n_samples, n_features]

    Feature matrix of the test dataset.

- `y_test` : array-like, shape = [n_samples]

    True class labels of the test dataset.

- `clf` : Classifier object. Must have a .predict .fit method.


- `train_marker` : str (default: 'o')

    Marker for the training set line plot.

- `test_marker` : str (default: '^')

    Marker for the test set line plot.

- `scoring` : str (default: 'misclassification error')

    If not 'misclassification error', accepts the following metrics
    (from scikit-learn):
    {'accuracy', 'average_precision', 'f1_micro', 'f1_macro',
    'f1_weighted', 'f1_samples', 'log_loss',
    'precision', 'recall', 'roc_auc',
    'adjusted_rand_score', 'mean_absolute_error', 'mean_squared_error',
    'median_absolute_error', 'r2'}

- `suppress_plot=False` : bool (default: False)

    Suppress matplotlib plots if True. Recommended
    for testing purposes.

- `print_model` : bool (default: True)

    Print model parameters in plot title if True.

- `style` : str (default: 'fivethirtyeight')

    Matplotlib style

- `legend_loc` : str (default: 'best')

    Where to place the plot legend:
    {'best', 'upper left', 'upper right', 'lower left', 'lower right'}

**Returns**

- `errors` : (training_error, test_error): tuple of lists


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


