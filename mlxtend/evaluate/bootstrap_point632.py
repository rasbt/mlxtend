# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Bootstrap functions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from .bootstrap_outofbag import BootstrapOutOfBag
from sklearn.metrics.scorer import check_scoring
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
        raise ValueError('X and y must contain the same number of samples')


def bootstrap_point632_score(estimator, X, y, n_splits=200,
                             method='.632', scoring=None, random_seed=None):
    """
    Implementation of the 0.632 bootstrap for supervised learning

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
        The bootstrap method, which can be either the
        regular '.632' bootstrap (default) or the '.632+'
        bootstrap (not implemented, yet).

    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {'accuracy', 'f1', 'precision', 'recall', 'roc_auc', etc.}
        for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2', etc.} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.

    random_seed : int (default=None)
        If int, random_seed is the seed used by
        the random number generator.

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

    allowed_methods = ('.632', '.632+')
    if not isinstance(method, str) or method not in allowed_methods:
        raise ValueError('The `method` must '
                         'be in %s. Got %s.' % (allowed_methods, method))

    if method == '.632+':
        raise NotImplementedError('The .632+ method is not implemented, yet.')

    _check_arrays(X, y)

    estimator = clone(estimator)
    scorer = check_scoring(estimator, scoring=scoring)

    oob = BootstrapOutOfBag(n_splits=n_splits, random_seed=random_seed)
    scores = np.empty(dtype=np.float, shape=(n_splits,))
    cnt = 0
    for train, test in oob.split(X):
        estimator.fit(X[train], y[train])
        train_acc = scorer(estimator, X[train], y[train])
        test_acc = scorer(estimator, X[test], y[test])
        acc = 0.368*train_acc + 0.632*test_acc
        scores[cnt] = acc
        cnt += 1
    return scores
