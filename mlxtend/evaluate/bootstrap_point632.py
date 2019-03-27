# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# Bootstrap functions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from .bootstrap_outofbag import BootstrapOutOfBag
from sklearn.base import clone


def _check_arrays(X, y=None):
    if isinstance(X, list):
        raise ValueError('X must be a numpy array')
    if not len(X.shape) == 2:
        raise ValueError('X must be a 2D array. Try X[:, numpy.newaxis]')
    try:
        if y is None:
            return
    except(AttributeError):
        if not len(y.shape) == 1:
            raise ValueError('y must be a 1D array.')

    if not len(y) == X.shape[0]:
        raise ValueError('X and y must contain the'
                         'same number of samples')


def no_information_rate(targets, predictions, loss_fn):
    combinations = np.array(np.meshgrid(targets, predictions)).reshape(-1, 2)
    return loss_fn(combinations[:, 0], combinations[:, 1])


def accuracy(targets, predictions):
    return np.mean(np.array(targets) == np.array(predictions))


def mse(targets, predictions):
    return np.mean((np.array(targets) - np.array(predictions))**2)


def bootstrap_point632_score(estimator, X, y, n_splits=200,
                             method='.632', scoring_func=None,
                             random_seed=None,
                             clone_estimator=True):
    """
    Implementation of the .632 [1] and .632+ [2] bootstrap
    for supervised learning

    References:

    - [1] Efron, Bradley. 1983. "Estimating the Error Rate
      of a Prediction Rule: Improvement on Cross-Validation."
      Journal of the American Statistical Association
      78 (382): 316. doi:10.2307/2288636.
    - [2] Efron, Bradley, and Robert Tibshirani. 1997.
      "Improvements on Cross-Validation: The .632+ Bootstrap Method."
      Journal of the American Statistical Association
      92 (438): 548. doi:10.2307/2965703.

    Parameters
    ----------
    estimator : object
        An estimator for classification or regression that
        follows the scikit-learn API and implements "fit" and "predict"
        methods.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    n_splits : int (default=200)
        Number of bootstrap iterations.
        Must be larger than 1.

    method : str (default='.632')
        The bootstrap method, which can be either
        - 1) '.632' bootstrap (default)
        - 2) '.632+' bootstrap
        - 3) 'oob' (regular out-of-bag, no weighting)
        for comparison studies.

    scoring_func : callable,
        Score function (or loss function) with signature
        ``scoring_func(y, y_pred, **kwargs)``.
        If none, uses classification accuracy if the
        estimator is a classifier and mean squared error
        if the estimator is a regressor.

    random_seed : int (default=None)
        If int, random_seed is the seed used by
        the random number generator.

    clone_estimator : bool (default=True)
        Clones the estimator if true, otherwise fits
        the original.

    Returns
    -------
    scores : array of float, shape=(len(list(n_splits)),)
        Array of scores of the estimator for each bootstrap
        replicate.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from mlxtend.evaluate import bootstrap_point632_score
    >>> iris = datasets.load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> lr = linear_model.LogisticRegression()
    >>> scores = bootstrap_point632_score(lr, X, y)
    >>> acc = np.mean(scores)
    >>> print('Accuracy:', acc)
    0.953023146884
    >>> lower = np.percentile(scores, 2.5)
    >>> upper = np.percentile(scores, 97.5)
    >>> print('95%% Confidence interval: [%.2f, %.2f]' % (lower, upper))
    95% Confidence interval: [0.90, 0.98]

    For more usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/bootstrap_point632_score/

    """
    if not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError('Number of splits must be'
                         ' greater than 1. Got %s.' % n_splits)

    allowed_methods = ('.632', '.632+', 'oob')
    if not isinstance(method, str) or method not in allowed_methods:
        raise ValueError('The `method` must '
                         'be in %s. Got %s.' % (allowed_methods, method))

    _check_arrays(X, y)

    if clone_estimator:
        cloned_est = clone(estimator)
    else:
        cloned_est = estimator

    if scoring_func is None:
        if cloned_est._estimator_type == 'classifier':
            scoring_func = accuracy
        elif cloned_est._estimator_type == 'regressor':
            scoring_func = mse
        else:
            raise AttributeError('Estimator type undefined.'
                                 'Please provide a scoring_func argument.')

    oob = BootstrapOutOfBag(n_splits=n_splits, random_seed=random_seed)
    scores = np.empty(dtype=np.float, shape=(n_splits,))
    cnt = 0
    for train, test in oob.split(X):
        cloned_est.fit(X[train], y[train])

        test_acc = scoring_func(y[test], cloned_est.predict(X[test]))

        if method == 'oob':
            acc = test_acc

        else:
            train_acc = scoring_func(y[train], cloned_est.predict(X[train]))
            if method == '.632+':
                gamma = no_information_rate(y,
                                            cloned_est.predict(X),
                                            scoring_func)
                R = (-(test_acc - train_acc)) / (gamma - (1 - test_acc))
                weight = 0.632 / (1-0.368 * R)

            else:
                weight = 0.632

            acc = weight*test_acc + (1. - weight)*train_acc

        scores[cnt] = acc
        cnt += 1
    return scores
