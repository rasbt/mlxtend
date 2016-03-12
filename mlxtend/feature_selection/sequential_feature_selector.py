# Sebastian Raschka 2014-2016
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
from copy import deepcopy
from itertools import combinations
from collections import deque
from sklearn.metrics import get_scorer
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.cross_validation import cross_val_score
from ..externals.name_estimators import _name_estimators


class SequentialFeatureSelector(BaseEstimator, MetaEstimatorMixin):

    """Sequential Feature Selection for Classification and Regression.

    Parameters
    ----------
    estimator : scikit-learn classifier or regressor
    k_features : int
        Number of features to select,
        where k_features < the full feature set.
    forward : bool (default: True)
        Forward selection if True,
        backward selection otherwise
    floating : bool (default: False)
        Adds a conditional exclusion/inclusion if True.
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
    skip_if_stuck: bool (default: True)
        Set to True to skip conditional
        exlusion/inclusion if floating=True and
        algorithm gets stuck in cycles.
    n_jobs : int (default: 1)
        The number of CPUs to use for cross validation. -1 means 'all CPUs'.
    pre_dispatch : int, or string
        Controls the number of jobs that get dispatched
        during parallel execution in cross_val_score.
        Reducing this number can be useful to avoid an explosion of
        memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
        None, in which case all the jobs are immediately created and spawned.
            Use this for lightweight and fast-running jobs,
            to avoid delays due to on-demand spawning of the jobs
        An int, giving the exact number of total jobs that are spawned
        A string, giving an expression as a function
            of n_jobs, as in `2*n_jobs`

    Attributes
    ----------
    k_feature_idx_ : array-like, shape = [n_predictions]
        Feature Indices of the selected feature subsets.
    k_score_ : float
        Cross validation average score of the selected subset.
    subsets_ : dict
        A dictionary of selected feature subsets during the
        sequential selection, where the dictionary keys are
        the lenghts k of these feature subsets. The dictionary
        values are dictionaries themselves with the following
        keys: 'feature_idx' (tuple of indices of the feature subset)
              'cv_scores' (list individual cross-validation scores)
              'avg_score' (average cross-validation score)

    """
    def __init__(self, estimator, k_features,
                 forward=True, floating=False,
                 print_progress=True, scoring='accuracy',
                 cv=5, skip_if_stuck=True, n_jobs=1,
                 pre_dispatch='2*n_jobs'):
        self.estimator = estimator
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
        self.pre_dispatch = pre_dispatch
        self.scoring = scoring
        self.scorer = get_scorer(scoring)
        self.skip_if_stuck = skip_if_stuck
        self.cv = cv
        self.print_progress = print_progress
        self.n_jobs = n_jobs
        self.named_est = {key: value for key, value in
                          _name_estimators([self.estimator])}

    def fit(self, X, y):
        self.est_ = clone(self.estimator)
        if X.shape[1] < self.k_features:
            raise AttributeError('Features in X < k_features')
        if self.skip_if_stuck:
            sdq = deque(maxlen=4)
        else:
            sdq = deque(maxlen=0)

        self.subsets_ = {}
        orig_set = set(range(X.shape[1]))
        if self.forward:
            k_idx = ()
            k = 0
        else:
            k_idx = tuple(range(X.shape[1]))
            k = len(k_idx)
            k_score = self._calc_score(X, y, k_idx)
            self.subsets_[k] = {'feature_idx': k_idx,
                                'cv_scores': k_score,
                                'avg_score': k_score.mean()}

        while k != self.k_features:
            prev_subset = set(k_idx)
            if self.forward:
                k_idx, k_score, cv_scores = \
                    self._inclusion(orig_set=orig_set,
                                    subset=prev_subset,
                                    X=X, y=y)
            else:
                k_idx, k_score, cv_scores = \
                    self._exclusion(feature_set=prev_subset, X=X, y=y)

            if self.floating and not self._is_stuck(sdq):
                (new_feature,) = set(k_idx) ^ prev_subset
                if self.forward:
                    k_idx_c, k_score_c, cv_scores_c = \
                        self._exclusion(feature_set=k_idx,
                                        fixed_feature=new_feature,
                                        X=X, y=y)
                else:
                    k_idx_c, k_score_c, cv_scores_c = \
                        self._inclusion(orig_set=orig_set - {new_feature},
                                        subset=set(k_idx),
                                        X=X, y=y)

                if k_score_c and k_score_c > k_score:
                    k_idx, k_score, cv_scores = \
                        k_idx_c, k_score_c, cv_scores_c

            k = len(k_idx)
            # floating can lead to multiple same-sized subsets
            if k not in self.subsets_ or (self.subsets_[k]['avg_score'] >
                                          k_score):
                self.subsets_[k] = {'feature_idx': k_idx,
                                    'cv_scores': cv_scores,
                                    'avg_score': k_score}
            sdq.append(k_idx)

            if self.print_progress:
                sys.stderr.write('\rFeatures: %d/%d' % (
                    len(k_idx), self.k_features))
                sys.stderr.flush()

        self.k_feature_idx_ = k_idx
        self.k_score_ = k_score
        self.subsets_plus_ = dict()
        return self

    def _is_stuck(self, sdq):
        stuck = False
        if len(sdq) == 4 and (sdq[0] == sdq[2] or sdq[1] == sdq[3]):
            stuck = True
        return stuck

    def _calc_score(self, X, y, indices):
        if self.cv:
            scores = cross_val_score(self.est_,
                                     X[:, indices], y,
                                     cv=self.cv,
                                     scoring=self.scorer,
                                     n_jobs=self.n_jobs,
                                     pre_dispatch=self.pre_dispatch)
        else:
            self.est_.fit(X[:, indices], y)
            scores = np.array([self.scorer(self.est_, X[:, indices], y)])
        return scores

    def _inclusion(self, orig_set, subset, X, y):
        all_avg_scores = []
        all_cv_scores = []
        all_subsets = []
        res = (None, None, None)
        remaining = orig_set - subset
        if remaining:
            for feature in remaining:
                new_subset = tuple(subset | {feature})
                cv_scores = self._calc_score(X, y, new_subset)
                all_avg_scores.append(cv_scores.mean())
                all_cv_scores.append(cv_scores)
                all_subsets.append(new_subset)
            best = np.argmax(all_avg_scores)
            res = (all_subsets[best],
                   all_avg_scores[best],
                   all_cv_scores[best])
        return res

    def _exclusion(self, feature_set, X, y, fixed_feature=None):
        n = len(feature_set)
        res = (None, None, None)
        if n > 1:
            all_avg_scores = []
            all_cv_scores = []
            all_subsets = []
            for p in combinations(feature_set, r=n - 1):
                if fixed_feature and fixed_feature not in set(p):
                    continue
                cv_scores = self._calc_score(X, y, p)
                all_avg_scores.append(cv_scores.mean())
                all_cv_scores.append(cv_scores)
                all_subsets.append(p)
            best = np.argmax(all_avg_scores)
            res = (all_subsets[best],
                   all_avg_scores[best],
                   all_cv_scores[best])
        return res

    def transform(self, X):
        return X[:, self.k_feature_idx_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_metric_dict(self, confidence_interval=0.95):
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
