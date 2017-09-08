# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Algorithm for sequential feature selection.
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
from sklearn.externals.joblib import Parallel, delayed


def _calc_score(selector, X, y, indices):
    if selector.cv:
        scores = cross_val_score(selector.est_,
                                 X[:, indices], y,
                                 cv=selector.cv,
                                 scoring=selector.scorer,
                                 n_jobs=1,
                                 pre_dispatch=selector.pre_dispatch)
    else:
        selector.est_.fit(X[:, indices], y)
        scores = np.array([selector.scorer(selector.est_, X[:, indices], y)])
    return indices, scores


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
    best_score_ : float
        Cross validation average score of the selected subset.
    subsets_ : dict
        A dictionary of selected feature subsets during the
        sequential selection, where the dictionary keys are
        the lengths k of these feature subsets. The dictionary
        values are dictionaries themselves with the following
        keys: 'feature_idx' (tuple of indices of the feature subset)
              'cv_scores' (list individual cross-validation scores)
              'avg_score' (average cross-validation score)

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

    def fit(self, X, y):
        """Perform feature selection and learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """

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

        candidates = chain(*((combinations(range(X.shape[1]), r=i))
                          for i in range(self.min_features,
                                         self.max_features + 1)))

        self.subsets_ = {}
        
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
        
        all_comb = np.sum([ncr(n=X.shape[1], r=i)
                           for i in range(self.min_features,
                                          self.max_features + 1)])
        
        n_jobs = min(self.n_jobs, all_comb)
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=self.pre_dispatch)
        work = enumerate(parallel(delayed(_calc_score)(self, X, y, c)
                                  for c in candidates))

        for iteration, (c, cv_scores) in work:

            self.subsets_[iteration] = {'feature_idx': c,
                                        'cv_scores': cv_scores,
                                        'avg_score': np.mean(cv_scores)}

            if self.print_progress:
                sys.stderr.write('\rFeatures: %d/%d' % (
                    iteration + 1, all_comb))
                sys.stderr.flush()

        max_score = float('-inf')
        for c in self.subsets_:
            if self.subsets_[c]['avg_score'] > max_score:
                max_score = self.subsets_[c]['avg_score']
                best_subset = c
        score = max_score
        idx = self.subsets_[best_subset]['feature_idx']

        self.best_idx_ = idx
        self.best_score_ = score
        self.subsets_plus_ = dict()
        self.fitted = True
        return self

    def transform(self, X):
        """Return the best selected features from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        Feature subset of X, shape={n_samples, k_features}

        """
        self._check_fitted()
        return X[:, self.best_idx_]

    def fit_transform(self, X, y):
        """Fit to training data and return the best selected features from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        Feature subset of X, shape={n_samples, k_features}

        """
        self.fit(X, y)
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
