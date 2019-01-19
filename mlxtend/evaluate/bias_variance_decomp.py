# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# Nonparametric Permutation Test
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause
import numpy as np


def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(sample_indices,
                                   size=sample_indices.shape[0],
                                   replace=True)
    return X[bootstrap_indices], y[bootstrap_indices]


def bias_variance_decomp(estimator, X_train, y_train, X_test, y_test,
                         loss='0-1_loss', num_rounds=200, random_seed=None):
    """
    estimator : object
        A classifier or regressor object or class implementing a `fit`
        `predict` method similar to the scikit-learn API.

    X_train : array-like, shape=(num_examples, num_features)
        A training dataset for drawing the bootstrap samples to carry
        out the bias-variance decomposition.

    y_train : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_train` examples.

    X_test : array-like, shape=(num_examples, num_features)
        The test dataset for computing the average loss, bias,
        and variance.

    y_test : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_test` examples.

    loss : str (default='0-1_loss')
        Loss function for performing the bias-variance decomposition.
        Currently allowed values are '0-1_loss' and 'mse'.

    num_rounds : int (default=200)
        Number of bootstrap rounds for performing the bias-variance
        decomposition.

    random_seed : int (default=None)
        Random seed for the bootstrap sampling used for the
        bias-variance decomposition.

    Returns
    ----------
    avg_expected_loss, avg_bias, avg_var : returns the average expected
        average bias, and average bias (all floats), where the average
        is computed over the data points in the test set.

    """
    supported = ['0-1_loss', 'mse']
    if loss not in supported:
        raise NotImplementedError('loss must be one of the following: %s' %
                                  supported)

    rng = np.random.RandomState(random_seed)

    all_pred = np.zeros((num_rounds, y_test.shape[0]), dtype=np.int)

    for i in range(num_rounds):
        X_boot, y_boot = _draw_bootstrap_sample(rng, X_train, y_train)
        pred = estimator.fit(X_boot, y_boot).predict(X_test)
        all_pred[i] = pred

    if loss == '0-1_loss':
        main_predictions = np.apply_along_axis(lambda x:
                                               np.argmax(np.bincount(x)),
                                               axis=0,
                                               arr=all_pred)

        avg_expected_loss = (main_predictions != y_test).sum()/y_test.size

        avg_expected_loss = np.apply_along_axis(lambda x:
                                                (x != y_test).mean(),
                                                axis=1,
                                                arr=all_pred).mean()

        avg_bias = np.sum(main_predictions != y_test) / y_test.size
        avg_var = np.sum(main_predictions != all_pred) / all_pred.size

        var = np.zeros(pred.shape)

        for pred in all_pred:
            var += (pred != main_predictions).astype(np.int)
        var /= num_rounds

        avg_var = var.sum()/y_test.shape[0]

    else:
        avg_expected_loss = np.apply_along_axis(
            lambda x:
            ((x - y_test)**2).mean(),
            axis=1,
            arr=all_pred).mean()

        main_predictions = np.mean(all_pred, axis=0)

        avg_bias = np.sum((main_predictions - y_test)**2) / y_test.size
        avg_var = np.sum((main_predictions - all_pred)**2) / all_pred.size

    return avg_expected_loss, avg_bias, avg_var
