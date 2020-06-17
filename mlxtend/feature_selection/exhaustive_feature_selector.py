# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Algorithm for exhaustive feature selection.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import numpy as np
import scipy as sp
import scipy.stats
import sys
import operator as op
from copy import deepcopy
from itertools import combinations
from itertools import chain
from functools import reduce
from sklearn.metrics import get_scorer
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from ..externals.name_estimators import _name_estimators
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed


def _calc_score(selector, X, y, indices, groups=None, **fit_params):
    if selector.cv:
        scores = cross_val_score(selector.est_,
                                 X[:, indices], y,
                                 groups=groups,
                                 cv=selector.cv,
                                 scoring=selector.scorer,
                                 n_jobs=1,
                                 pre_dispatch=selector.pre_dispatch,
                                 fit_params=fit_params)
    else:
        selector.est_.fit(X[:, indices], y, **fit_params)
        scores = np.array([selector.scorer(selector.est_, X[:, indices], y)])
    return indices, scores


def _get_featurenames(subsets_dict, feature_idx, custom_feature_names, X):
    feature_names = None
    if feature_idx is not None:
        if custom_feature_names is not None:
            feature_names = tuple((custom_feature_names[i]
                                   for i in feature_idx))
        elif hasattr(X, 'loc'):
            feature_names = tuple((X.columns[i] for i in feature_idx))
        else:
            feature_names = tuple(str(i) for i in feature_idx)

    subsets_dict_ = deepcopy(subsets_dict)
    for key in subsets_dict_:
        if custom_feature_names is not None:
            new_tuple = tuple((custom_feature_names[i]
                               for i in subsets_dict[key]['feature_idx']))
        elif hasattr(X, 'loc'):
            new_tuple = tuple((X.columns[i]
                               for i in subsets_dict[key]['feature_idx']))
        else:
            new_tuple = tuple(str(i) for i in subsets_dict[key]['feature_idx'])
        subsets_dict_[key]['feature_names'] = new_tuple

    return subsets_dict_, feature_names


class ExhaustiveFeatureSelector(BaseEstimator, MetaEstimatorMixin):

    """Exhaustive Feature Selection for Classification and Regression.
       (new in v0.4.3)

    Parameters
    ----------
    estimator : scikit-learn classifier or regressor
    min_features : int (default: 1)
        Minumum number of features to select
    max_features : int (default: 1)
        Maximum number of features to select
    print_progress : bool (default: True)
        Prints progress as the number of epochs
        to stderr.
    scoring : str, (default='accuracy')
        Scoring metric in {accuracy, f1, precision, recall, roc_auc}
        for classifiers,
        {'mean_absolute_error', 'mean_squared_error',
        'median_absolute_error', 'r2'} for regressors,
        or a callable object or function with
        signature ``scorer(estimator, X, y)``.
    cv : int (default: 5)
        Scikit-learn cross-validation generator or `int`.
        If estimator is a classifier (or y consists of integer class labels),
        stratified k-fold is performed, and regular k-fold cross-validation
        otherwise.
        No cross-validation if cv is None, False, or 0.
    n_jobs : int (default: 1)
        The number of CPUs to use for evaluating different feature subsets
        in parallel. -1 means 'all CPUs'.
    pre_dispatch : int, or string (default: '2*n_jobs')
        Controls the number of jobs that get dispatched
        during parallel execution if `n_jobs > 1` or `n_jobs=-1`.
        Reducing this number can be useful to avoid an explosion of
        memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
        None, in which case all the jobs are immediately created and spawned.
            Use this for lightweight and fast-running jobs,
            to avoid delays due to on-demand spawning of the jobs
        An int, giving the exact number of total jobs that are spawned
        A string, giving an expression as a function
            of n_jobs, as in `2*n_jobs`
    clone_estimator : bool (default: True)
        Clones estimator if True; works with the original estimator instance
        if False. Set to False if the estimator doesn't
        implement scikit-learn's set_params and get_params methods.
        In addition, it is required to set cv=0, and n_jobs=1.

    Attributes
    ----------
    best_idx_ : array-like, shape = [n_predictions]
        Feature Indices of the selected feature subsets.
    best_feature_names_ : array-like, shape = [n_predictions]
        Feature names of the selected feature subsets. If pandas
        DataFrames are used in the `fit` method, the feature
        names correspond to the column names. Otherwise, the
        feature names are string representation of the feature
        array indices. New in v 0.13.0.
    best_score_ : float
        Cross validation average score of the selected subset.
    subsets_ : dict
        A dictionary of selected feature subsets during the
        exhaustive selection, where the dictionary keys are
        the lengths k of these feature subsets. The dictionary
        values are dictionaries themselves with the following
        keys: 'feature_idx' (tuple of indices of the feature subset)
              'feature_names' (tuple of feature names of the feat. subset)
              'cv_scores' (list individual cross-validation scores)
              'avg_score' (average cross-validation score)
        Note that if pandas
        DataFrames are used in the `fit` method, the 'feature_names'
        correspond to the column names. Otherwise, the
        feature names are string representation of the feature
        array indices. The 'feature_names' is new in v 0.13.0.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/

    """
    def __init__(self, estimator, min_features=1, max_features=1,
                 print_progress=True, scoring='accuracy',
                 cv=5, n_jobs=1,
                 pre_dispatch='2*n_jobs',
                 clone_estimator=True):
        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features
        self.pre_dispatch = pre_dispatch
        self.scoring = scoring
        self.scorer = get_scorer(scoring)
        self.cv = cv
        self.print_progress = print_progress
        self.n_jobs = n_jobs
        self.named_est = {key: value for key, value in
                          _name_estimators([self.estimator])}
        self.clone_estimator = clone_estimator
        if self.clone_estimator:
            self.est_ = clone(self.estimator)
        else:
            self.est_ = self.estimator
        self.fitted = False
        self.interrupted_ = False

        # don't mess with this unless testing
        self._TESTING_INTERRUPT_MODE = False

    def fit(self, X, y, custom_feature_names=None, groups=None, **fit_params):
        """Perform feature selection and learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        y : array-like, shape = [n_samples]
            Target values.
        custom_feature_names : None or tuple (default: tuple)
            Custom feature names for `self.k_feature_names` and
            `self.subsets_[i]['feature_names']`.
            (new in v 0.13.0)
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.
        fit_params : dict of string -> object, optional
            Parameters to pass to to the fit method of classifier.

        Returns
        -------
        self : object

        """

        # reset from a potential previous fit run
        self.subsets_ = {}
        self.fitted = False
        self.interrupted_ = False
        self.best_idx_ = None
        self.best_feature_names_ = None
        self.best_score_ = None

        if hasattr(X, 'loc'):
            X_ = X.values
        else:
            X_ = X

        if (custom_feature_names is not None
                and len(custom_feature_names) != X.shape[1]):
            raise ValueError('If custom_feature_names is not None, '
                             'the number of elements in custom_feature_names '
                             'must equal the number of columns in X.')

        if (not isinstance(self.max_features, int) or
                (self.max_features > X.shape[1] or self.max_features < 1)):
            raise AttributeError('max_features must be'
                                 ' smaller than %d and larger than 0' %
                                 (X.shape[1] + 1))

        if (not isinstance(self.min_features, int) or
                (self.min_features > X.shape[1] or self.min_features < 1)):
            raise AttributeError('min_features must be'
                                 ' smaller than %d and larger than 0'
                                 % (X.shape[1] + 1))

        if self.max_features < self.min_features:
            raise AttributeError('min_features must be <= max_features')

        candidates = chain.from_iterable(
            combinations(range(X_.shape[1]), r=i) for i in
            range(self.min_features, self.max_features + 1)
        )

        def ncr(n, r):
            """Return the number of combinations of length r from n items.

            Parameters
            ----------
            n : {integer}
            Total number of items
            r : {integer}
            Number of items to select from n

            Returns
            -------
            Number of combinations, integer

            """

            r = min(r, n-r)
            if r == 0:
                return 1
            numer = reduce(op.mul, range(n, n-r, -1))
            denom = reduce(op.mul, range(1, r+1))
            return numer//denom

        all_comb = np.sum([ncr(n=X_.shape[1], r=i)
                           for i in range(self.min_features,
                                          self.max_features + 1)])

        n_jobs = min(self.n_jobs, all_comb)
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=self.pre_dispatch)
        work = enumerate(parallel(delayed(_calc_score)
                                  (self, X_, y, c, groups=groups, **fit_params)
                                  for c in candidates))

        try:
            for iteration, (c, cv_scores) in work:

                self.subsets_[iteration] = {'feature_idx': c,
                                            'cv_scores': cv_scores,
                                            'avg_score': np.mean(cv_scores)}

                if self.print_progress:
                    sys.stderr.write('\rFeatures: %d/%d' % (
                        iteration + 1, all_comb))
                    sys.stderr.flush()

                if self._TESTING_INTERRUPT_MODE:
                    self.subsets_, self.best_feature_names_ = \
                        _get_featurenames(self.subsets_,
                                          self.best_idx_,
                                          custom_feature_names,
                                          X)
                    raise KeyboardInterrupt

        except KeyboardInterrupt as e:
            self.interrupted_ = True
            sys.stderr.write('\nSTOPPING EARLY DUE TO KEYBOARD INTERRUPT...')

        max_score = float('-inf')
        for c in self.subsets_:
            if self.subsets_[c]['avg_score'] > max_score:
                max_score = self.subsets_[c]['avg_score']
                best_subset = c
        score = max_score
        idx = self.subsets_[best_subset]['feature_idx']

        self.best_idx_ = idx
        self.best_score_ = score
        self.fitted = True
        self.subsets_, self.best_feature_names_ = \
            _get_featurenames(self.subsets_,
                              self.best_idx_,
                              custom_feature_names,
                              X)
        return self

    def transform(self, X):
        """Return the best selected features from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.

        Returns
        -------
        Feature subset of X, shape={n_samples, k_features}

        """
        self._check_fitted()
        if hasattr(X, 'loc'):
            X_ = X.values
        else:
            X_ = X
        return X_[:, self.best_idx_]

    def fit_transform(self, X, y, groups=None, **fit_params):
        """Fit to training data and return the best selected features from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        y : array-like, shape = [n_samples]
            Target values.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.
        fit_params : dict of string -> object, optional
            Parameters to pass to to the fit method of classifier.

        Returns
        -------
        Feature subset of X, shape={n_samples, k_features}

        """
        self.fit(X, y, groups=groups, **fit_params)
        return self.transform(X)

    def get_metric_dict(self, confidence_interval=0.95):
        """Return metric dictionary

        Parameters
        ----------
        confidence_interval : float (default: 0.95)
            A positive float between 0.0 and 1.0 to compute the confidence
            interval bounds of the CV score averages.

        Returns
        ----------
        Dictionary with items where each dictionary value is a list
        with the number of iterations (number of feature subsets) as
        its length. The dictionary keys corresponding to these lists
        are as follows:
            'feature_idx': tuple of the indices of the feature subset
            'cv_scores': list with individual CV scores
            'avg_score': of CV average scores
            'std_dev': standard deviation of the CV score average
            'std_err': standard error of the CV score average
            'ci_bound': confidence interval bound of the CV score average

        """
        self._check_fitted()
        fdict = deepcopy(self.subsets_)
        for k in fdict:
            std_dev = np.std(self.subsets_[k]['cv_scores'])
            bound, std_err = self._calc_confidence(
                self.subsets_[k]['cv_scores'],
                confidence=confidence_interval)
            fdict[k]['ci_bound'] = bound
            fdict[k]['std_dev'] = std_dev
            fdict[k]['std_err'] = std_err
        return fdict

    def _calc_confidence(self, ary, confidence=0.95):
        std_err = scipy.stats.sem(ary)
        bound = std_err * sp.stats.t._ppf((1 + confidence) / 2.0, len(ary))
        return bound, std_err

    def _check_fitted(self):
        if not self.fitted:
            raise AttributeError('ExhaustiveFeatureSelector has not been'
                                 ' fitted, yet.')
