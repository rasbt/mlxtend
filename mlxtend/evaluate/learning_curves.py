# Function for plotting learning curves.
# mlxtend Machine Learning Library Extensions

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(X_train, y_train, X_test, y_test, clf, kind='training_size',
            marker='o', scoring='misclassification error', suppress_plot=False, print_model=True):
    """
    Plots learning curves of a classifier.

    Parameters
    ----------
    X_train : array-like, shape = [n_samples, n_features]
      Feature matrix of the training dataset.

    y_train : array-like, shape = [n_samples]
      True class labels of the training dataset.

    X_test : array-like, shape = [n_samples, n_features]
      Feature matrix of the test dataset.

    y_test : array-like, shape = [n_samples]
      True class labels of the test dataset.

    clf : Classifier object. Must have a .predict .fit method.

    kind : str (default: 'training_size')
      'training_size' or 'n_features'
      Plots missclassifications vs. training size or number of features.

    marker : str (default: 'o')
      Marker for the line plot.

    scoring : str (default: 'misclassification error')
      If not 'accuracy', accepts the following metrics (from scikit-learn):
      {'accuracy', 'average_precision', 'f1_micro', 'f1_macro',
      'f1_weighted', 'f1_samples', 'log_loss', 'precision', 'recall', 'roc_auc',
      'adjusted_rand_score', 'mean_absolute_error', 'mean_squared_error',
      'median_absolute_error', 'r2'}

    suppress_plot=False : bool (default: False)
      Suppress matplotlib plots if True. Recommended
      for testing purposes.

    print_model : bool (default: True)
      Print model parameters in plot title if True.

    Returns
    ---------
    (training_error, test_error): tuple of lists
    """

    if scoring != 'misclassification error':
        from sklearn import metrics

        scoring_func = {
            'accuracy': metrics.accuracy_score,
            'average_precision': metrics.average_precision_score,
            'f1': metrics.f1_score,
            'f1_micro': metrics.f1_score,
            'f1_macro': metrics.f1_score,
            'f1_weighted': metrics.f1_score,
            'f1_samples': metrics.f1_score,
            'log_loss': metrics.log_loss,
            'precision': metrics.precision_score,
            'recall': metrics.recall_score,
            'roc_auc': metrics.roc_auc_score,
            'adjusted_rand_score': metrics.adjusted_rand_score,
            'mean_absolute_error': metrics.mean_absolute_error,
            'mean_squared_error': metrics.mean_squared_error,
            'median_absolute_error': metrics.median_absolute_error,
            'r2': metrics.r2_score}

        if scoring not in scoring_func.keys():
            raise AttributeError('scoring must be in', scoring_func.keys())


    else:
        def misclf_err(y_predict, y):
            return  (y_predict != y).sum() / float(len(y))

        scoring_func = {
            'misclassification error': misclf_err}

    training_errors = []
    test_errors = []

    if kind not in ('training_size', 'n_features'):
        raise AttributeError('kind must be training_size or n_features')

    if kind == 'training_size':
        rng = [int(i) for i in np.linspace(0, X_train.shape[0], 11)][1:][::-1]
        for r in rng:
            model = clf.fit(X_train[:r], y_train[:r])

            y_train_predict = clf.predict(X_train[:r])
            y_test_predict = clf.predict(X_test)

            train_misclf = scoring_func[scoring](y_train[:r], y_train_predict)
            training_errors.append(train_misclf)

            test_misclf = scoring_func[scoring](y_test, y_test_predict)
            test_errors.append(test_misclf)

        if not suppress_plot:
            plt.plot(np.arange(10,101,10), training_errors, label='training error', marker=marker)
            plt.plot(np.arange(10,101,10), test_errors, label='test error', marker=marker)
            plt.xlabel('training set size in percent')

    elif kind == 'n_features':
        rng = np.arange(1, X_train.shape[1]+1)
        for r in rng:
            model = clf.fit(X_train[:, 0:r], y_train)

            y_train_predict = clf.predict(X_train[:, 0:r])
            train_misclf = scoring_func[scoring](y_train, y_train_predict)
            training_errors.append(train_misclf)

            y_test_predict = clf.predict(X_test[:, 0:r])
            test_misclf = scoring_func[scoring](y_test, y_test_predict)
            test_errors.append(test_misclf)

        if not suppress_plot:
            plt.plot(rng, training_errors, label='training error', marker=marker)
            plt.plot(rng, test_errors, label='test error', marker=marker)
            plt.xlabel('number of features')

    if not suppress_plot:
        plt.ylabel('Performance ({})'.format(scoring))
        if print_model:
            plt.title('Learning Curves\n\n{}\n'.format(model))
        plt.legend(loc='best')
        if kind == 'training_size':
            plt.xlim([0, 110])
        else:
            plt.xlim([0, X_train.shape[1]+1])

    return (training_errors, test_errors)