# Sebastian Raschka 08/13/2014
# mlxtend Machine Learning Library Extensions
# matplotlib utilities for removing chartchunk

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(X_train, y_train, X_test, y_test, clf, kind='training_size', marker='o'):
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

    Returns
    ---------
    (training_error, test_error): tuple of lists
    """
    training_errors = []
    test_errors = []

    if kind not in ('training_size', 'n_features'):
        raise ArgumentError('kind must be training_size or n_features')

    if kind == 'training_size':
        rng = [int(i) for i in np.linspace(0, X_train.shape[0], 11)][1:][::-1]
        for r in rng:
            clf.fit(X_train[:r], y_train[:r])

            y_train_predict = clf.predict(X_train[:r])
            train_misclf = (y_train_predict != y_train[:r]).sum()
            training_errors.append(train_misclf / X_train[:r].shape[0])

            y_test_predict = clf.predict(X_test)
            test_misclf = (y_test_predict != y_test).sum()
            test_errors.append(test_misclf / X_test.shape[0])

        plt.plot(np.arange(10,101,10), training_errors, label='training error', marker=marker)
        plt.plot(np.arange(10,101,10), test_errors, label='test error', marker=marker)
        plt.xlabel('training set size in percent')

    elif kind == 'n_features':
        rng = np.arange(1, X_train.shape[1]+1)
        for r in rng:
            clf.fit(X_train[:, 0:r], y_train)
            y_train_predict = clf.predict(X_train[:, 0:r])
            train_misclf = (y_train_predict != y_train).sum()
            training_errors.append(train_misclf / X_train.shape[0])

            y_test_predict = clf.predict(X_test[:, 0:r])
            test_misclf = (y_test_predict != y_test).sum()
            test_errors.append(test_misclf / X_test.shape[0])
        plt.plot(rng, training_errors, label='training error', marker=marker)
        plt.plot(rng, test_errors, label='test error', marker=marker)
        plt.xlabel('number of features')

    plt.ylabel('missclassification error')
    plt.title('Learning Curves')
    plt.ylim([0,1])
    plt.legend(loc=1)
    return (training_errors, test_errors)