# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Algorithm for sequential feature selection.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import datetime
import types
import numpy as np
import scipy as sp
import scipy.stats
import sys
from copy import deepcopy
from itertools import combinations
from sklearn.metrics import get_scorer
from sklearn.base import clone
from sklearn.base import MetaEstimatorMixin
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed


def _calc_score(selector, X, y, indices, groups=None, **fit_params):
    if selector.cv:
        scores = cross_val_score(selector.est_,
                                 X, y,
                                 groups=groups,
                                 cv=selector.cv,
                                 scoring=selector.scorer,
                                 n_jobs=1,
                                 pre_dispatch=selector.pre_dispatch,
                                 fit_params=fit_params)
    else:
        selector.est_.fit(X, y, **fit_params)
        scores = np.array([selector.scorer(selector.est_, X, y)])
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


class SequentialFeatureSelector(_BaseXComposition, MetaEstimatorMixin):

    """Sequential Feature Selection for Classification and Regression.

    Parameters
    ----------
    estimator : scikit-learn classifier or regressor
    k_features : int or tuple or str (default: 1)
        Number of features to select,
        where k_features < the full feature set.
        New in 0.4.2: A tuple containing a min and max value can be provided,
            and the SFS will consider return any feature combination between
            min and max that scored highest in cross-validation. For example,
            the tuple (1, 4) will return any combination from
            1 up to 4 features instead of a fixed number of features k.
        New in 0.8.0: A string argument "best" or "parsimonious".
            If "best" is provided, the feature selector will return the
            feature subset with the best cross-validation performance.
            If "parsimonious" is provided as an argument, the smallest
            feature subset that is within one standard error of the
            cross-validation performance will be selected.
    forward : bool (default: True)
        Forward selection if True,
        backward selection otherwise
    floating : bool (default: False)
        Adds a conditional exclusion/inclusion if True.
    verbose : int (default: 0), level of verbosity to use in logging.
        If 0, no output,
        if 1 number of features in current set, if 2 detailed logging i
        ncluding timestamp and cv scores at step.
    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.
    cv : int (default: 5)
        Integer or iterable yielding train, test splits. If cv is an integer
        and `estimator` is a classifier (or y consists of integer class
        labels) stratified k-fold. Otherwise regular k-fold cross-validation
        is performed. No cross-validation if cv is None, False, or 0.
    n_jobs : int (default: 1)
        The number of CPUs to use for evaluating different feature subsets
        in parallel. -1 means 'all CPUs'.
    early_stop_rounds : int (default 0)
        Enable early stopping criterion when > 0, this value determines the
        number of iterations after which, if no performance boost has been
        seen, execution is stopped.
        Used only when `k_features == 'best'` or `k_features == 'parsimonious'`
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
    fixed_features : tuple (default: None)
        If not `None`, the feature indices provided as a tuple will be
        regarded as fixed by the feature selector. For example, if
        `fixed_features=(1, 3, 7)`, the 2nd, 4th, and 8th feature are
        guaranteed to be present in the solution. Note that if
        `fixed_features` is not `None`, make sure that the number of
        features to be selected is greater than `len(fixed_features)`.
        In other words, ensure that `k_features > len(fixed_features)`.
        New in mlxtend v. 0.18.0.

    Attributes
    ----------
    k_feature_idx_ : array-like, shape = [n_predictions]
        Feature Indices of the selected feature subsets.
    k_feature_names_ : array-like, shape = [n_predictions]
        Feature names of the selected feature subsets. If pandas
        DataFrames are used in the `fit` method, the feature
        names correspond to the column names. Otherwise, the
        feature names are string representation of the feature
        array indices. New in v 0.13.0.
    k_score_ : float
        Cross validation average score of the selected subset.
    subsets_ : dict
        A dictionary of selected feature subsets during the
        sequential selection, where the dictionary keys are
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
    http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

    """
    def __init__(self, estimator, k_features=1,
                 forward=True, floating=False,
                 verbose=0, scoring=None,
                 cv=5, n_jobs=1,
                 early_stop_rounds=0,
                 pre_dispatch='2*n_jobs',
                 clone_estimator=True,
                 fixed_features=None):

        self.estimator = estimator
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
        self.pre_dispatch = pre_dispatch
        # Want to raise meaningful error message if a
        # cross-validation generator is inputted
        if isinstance(cv, types.GeneratorType):
            err_msg = ('Input cv is a generator object, which is not '
                       'supported. Instead please input an iterable yielding '
                       'train, test splits. This can usually be done by '
                       'passing a cross-validation generator to the '
                       'built-in list function. I.e. cv=list(<cv-generator>)')
            raise TypeError(err_msg)
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.clone_estimator = clone_estimator

        if not isinstance(early_stop_rounds, int) or early_stop_rounds < 0:
            raise ValueError('Number of early stopping round should be '
                             'an integer value greater than or equal to 0.'
                             'Got %d' % early_stop_rounds)

        self.early_stop_rounds = early_stop_rounds

        if fixed_features is not None:
            if isinstance(self.k_features, int) and \
                    self.k_features <= len(fixed_features):
                raise ValueError('Number of features to be selected must'
                                 ' be larger than the number of'
                                 ' features specified via `fixed_features`.'
                                 ' Got `k_features=%d` and'
                                 ' `fixed_features=%d`' %
                                 (k_features, len(fixed_features)))

            elif isinstance(self.k_features, tuple) and \
                    self.k_features[0] <= len(fixed_features):
                raise ValueError('The minimum number of features to'
                                 ' be selected must'
                                 ' be larger than the number of'
                                 ' features specified via `fixed_features`.'
                                 ' Got `k_features=%s` and '
                                 '`len(fixed_features)=%d`' %
                                 (k_features, len(fixed_features)))

        self.fixed_features = fixed_features

        if self.clone_estimator:
            self.est_ = clone(self.estimator)
        else:
            self.est_ = self.estimator
        self.scoring = scoring

        if scoring is None:
            if self.est_._estimator_type == 'classifier':
                scoring = 'accuracy'
            elif self.est_._estimator_type == 'regressor':
                scoring = 'r2'
            else:
                raise AttributeError('Estimator must '
                                     'be a Classifier or Regressor.')
        if isinstance(scoring, str):
            self.scorer = get_scorer(scoring)
        else:
            self.scorer = scoring

        self.fitted = False
        self.subsets_ = {}
        self.interrupted_ = False

        # don't mess with this unless testing
        self._TESTING_INTERRUPT_MODE = False

    @property
    def named_estimators(self):
        """
        Returns
        -------
        List of named estimator tuples, like [('svc', SVC(...))]
        """
        return _name_estimators([self.estimator])

    def get_params(self, deep=True):
        #
        # Return estimator parameter names for GridSearch support.
        #
        return self._get_params('named_estimators', deep=deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params('estimator', 'named_estimators', **params)
        return self

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
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for y.
        custom_feature_names : None or tuple (default: tuple)
            Custom feature names for `self.k_feature_names` and
            `self.subsets_[i]['feature_names']`.
            (new in v 0.13.0)
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.
        fit_params : various, optional
            Additional parameters that are being passed to the estimator.
            For example, `sample_weights=weights`.

        Returns
        -------
        self : object

        """

        # reset from a potential previous fit run
        self.subsets_ = {}
        self.fitted = False
        self.interrupted_ = False
        self.k_feature_idx_ = None
        self.k_feature_names_ = None
        self.k_score_ = None

        self.fixed_features_ = self.fixed_features
        self.fixed_features_set_ = set()

        if hasattr(X, 'loc'):
            X_ = X.values
            if self.fixed_features is not None:
                self.fixed_features_ = tuple(X.columns.get_loc(c)
                                             if isinstance(c, str) else c
                                             for c in self.fixed_features
                                             )
        else:
            X_ = X

        if self.fixed_features is not None:
            self.fixed_features_set_ = set(self.fixed_features_)

        if (custom_feature_names is not None
                and len(custom_feature_names) != X.shape[1]):
            raise ValueError('If custom_feature_names is not None, '
                             'the number of elements in custom_feature_names '
                             'must equal the number of columns in X.')

        if not isinstance(self.k_features, int) and\
                not isinstance(self.k_features, tuple)\
                and not isinstance(self.k_features, str):
            raise AttributeError('k_features must be a positive integer'
                                 ', tuple, or string')

        if (isinstance(self.k_features, int) and (
                self.k_features < 1 or self.k_features > X_.shape[1])):
            raise AttributeError('k_features must be a positive integer'
                                 ' between 1 and X.shape[1], got %s'
                                 % (self.k_features, ))

        if isinstance(self.k_features, tuple):
            if len(self.k_features) != 2:
                raise AttributeError('k_features tuple must consist of 2'
                                     ' elements a min and a max value.')

            if self.k_features[0] not in range(1, X_.shape[1] + 1):
                raise AttributeError('k_features tuple min value must be in'
                                     ' range(1, X.shape[1]+1).')

            if self.k_features[1] not in range(1, X_.shape[1] + 1):
                raise AttributeError('k_features tuple max value must be in'
                                     ' range(1, X.shape[1]+1).')

            if self.k_features[0] > self.k_features[1]:
                raise AttributeError('The min k_features value must be smaller'
                                     ' than the max k_features value.')

        if isinstance(self.k_features, tuple) or\
                isinstance(self.k_features, str):

            select_in_range = True

            if isinstance(self.k_features, str):
                if self.k_features not in {'best', 'parsimonious'}:
                    raise AttributeError('If a string argument is provided, '
                                         'it must be "best" or "parsimonious"')
                else:
                    min_k = 1
                    max_k = X_.shape[1]
            else:
                min_k = self.k_features[0]
                max_k = self.k_features[1]

        else:
            select_in_range = False
            k_to_select = self.k_features

        orig_set = set(range(X_.shape[1]))
        n_features = X_.shape[1]

        if self.forward and self.fixed_features is not None:
            orig_set = set(range(X_.shape[1])) - self.fixed_features_set_
            n_features = len(orig_set)

        if self.forward:
            if select_in_range:
                k_to_select = max_k

            if self.fixed_features is not None:
                k_idx = self.fixed_features_
                k = len(k_idx)
                k_idx, k_score = _calc_score(self, X_[:, k_idx], y, k_idx,
                                             groups=groups, **fit_params)
                self.subsets_[k] = {
                    'feature_idx': k_idx,
                    'cv_scores': k_score,
                    'avg_score': np.nanmean(k_score)
                }

            else:
                k_idx = ()
                k = 0
        else:
            if select_in_range:
                k_to_select = min_k
            k_idx = tuple(orig_set)
            k = len(k_idx)
            k_idx, k_score = _calc_score(self, X_[:, k_idx], y, k_idx,
                                         groups=groups, **fit_params)
            self.subsets_[k] = {
                'feature_idx': k_idx,
                'cv_scores': k_score,
                'avg_score': np.nanmean(k_score)
            }
        best_subset = None
        k_score = 0
        best_score = -np.inf
        early_stop_count = self.early_stop_rounds

        try:
            while k != k_to_select:
                prev_subset = set(k_idx)

                if self.forward:
                    k_idx, k_score, cv_scores = self._inclusion(
                        orig_set=orig_set,
                        subset=prev_subset,
                        X=X_,
                        y=y,
                        groups=groups,
                        **fit_params
                    )
                else:
                    k_idx, k_score, cv_scores = self._exclusion(
                        feature_set=prev_subset,
                        X=X_,
                        y=y,
                        groups=groups,
                        fixed_feature=self.fixed_features_set_,
                        **fit_params
                    )

                if self.floating:

                    if self.forward:
                        continuation_cond_1 = len(k_idx)
                    else:
                        continuation_cond_1 = n_features - len(k_idx)

                    continuation_cond_2 = True
                    ran_step_1 = True
                    new_feature = None

                    while continuation_cond_1 >= 2 and continuation_cond_2:
                        k_score_c = None

                        if ran_step_1:
                            (new_feature,) = set(k_idx) ^ prev_subset

                        if self.forward:

                            fixed_features_ok = True
                            if self.fixed_features is not None and \
                                    len(self.fixed_features) - len(k_idx) <= 1:
                                fixed_features_ok = False
                            if fixed_features_ok:
                                k_idx_c, k_score_c, cv_scores_c = \
                                    self._exclusion(
                                        feature_set=k_idx,
                                        fixed_feature=(
                                            {new_feature} |
                                            self.fixed_features_set_),
                                        X=X_,
                                        y=y,
                                        groups=groups,
                                        **fit_params
                                    )

                        else:
                            k_idx_c, k_score_c, cv_scores_c = self._inclusion(
                                orig_set=orig_set - {new_feature},
                                subset=set(k_idx),
                                X=X_,
                                y=y,
                                groups=groups,
                                **fit_params
                            )

                        if k_score_c is not None and k_score_c > k_score:

                            if len(k_idx_c) in self.subsets_:
                                cached_score = self.subsets_[len(
                                    k_idx_c)]['avg_score']
                            else:
                                cached_score = None

                            if cached_score is None or \
                                    k_score_c > cached_score:
                                prev_subset = set(k_idx)
                                k_idx, k_score, cv_scores = \
                                    k_idx_c, k_score_c, cv_scores_c
                                continuation_cond_1 = len(k_idx)
                                ran_step_1 = False

                            else:
                                continuation_cond_2 = False

                        else:
                            continuation_cond_2 = False

                k = len(k_idx)
                # floating can lead to multiple same-sized subsets
                if k not in self.subsets_ or (k_score >
                                              self.subsets_[k]['avg_score']):

                    k_idx = tuple(sorted(k_idx))
                    self.subsets_[k] = {
                        'feature_idx': k_idx,
                        'cv_scores': cv_scores,
                        'avg_score': k_score
                    }

                if self.verbose == 1:
                    sys.stderr.write('\rFeatures: %d/%s' % (
                        len(k_idx),
                        k_to_select
                    ))
                    sys.stderr.flush()
                elif self.verbose > 1:
                    sys.stderr.write('\n[%s] Features: %d/%s -- score: %s' % (
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        len(k_idx),
                        k_to_select,
                        k_score
                    ))

                if self._TESTING_INTERRUPT_MODE:
                    self.subsets_, self.k_feature_names_ = \
                        _get_featurenames(self.subsets_,
                                          self.k_feature_idx_,
                                          custom_feature_names,
                                          X)
                    raise KeyboardInterrupt

                # early stop
                if self.early_stop_rounds \
                        and k != k_to_select \
                        and self.k_features in {'best', 'parsimonious'}:
                    if k_score <= best_score:
                        early_stop_count -= 1
                        if early_stop_count == 0:
                            print('Performances not improved for %d rounds. '
                                  'Stopping now!' % self.early_stop_rounds)
                            break
                    else:
                        early_stop_count = self.early_stop_rounds
                        best_score = k_score

        except KeyboardInterrupt:
            self.interrupted_ = True
            sys.stderr.write('\nSTOPPING EARLY DUE TO KEYBOARD INTERRUPT...')

        if select_in_range:
            max_score = float('-inf')

            max_score = float('-inf')
            for k in self.subsets_:
                if k < min_k or k > max_k:
                    continue
                if self.subsets_[k]['avg_score'] > max_score:
                    max_score = self.subsets_[k]['avg_score']
                    best_subset = k
            k_score = max_score
            k_idx = self.subsets_[best_subset]['feature_idx']

            if self.k_features == 'parsimonious':
                for k in self.subsets_:
                    if k >= best_subset:
                        continue
                    if self.subsets_[k]['avg_score'] >= (
                            max_score - np.std(self.subsets_[k]['cv_scores']) /
                            self.subsets_[k]['cv_scores'].shape[0]):
                        max_score = self.subsets_[k]['avg_score']
                        best_subset = k
                k_score = max_score
                k_idx = self.subsets_[best_subset]['feature_idx']

        self.k_feature_idx_ = k_idx
        self.k_score_ = k_score
        self.fitted = True
        self.subsets_, self.k_feature_names_ = \
            _get_featurenames(self.subsets_,
                              self.k_feature_idx_,
                              custom_feature_names,
                              X)
        return self

    def _inclusion(self, orig_set, subset, X, y, ignore_feature=None,
                   groups=None, **fit_params):
        all_avg_scores = []
        all_cv_scores = []
        all_subsets = []
        res = (None, None, None)
        remaining = orig_set - subset
        if remaining:
            features = len(remaining)
            n_jobs = min(self.n_jobs, features)
            parallel = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                                pre_dispatch=self.pre_dispatch)
            work = parallel(delayed(_calc_score)
                            (self, X[:, tuple(subset | {feature})], y,
                             tuple(subset | {feature}),
                             groups=groups, **fit_params)
                            for feature in remaining
                            if feature != ignore_feature)

            for new_subset, cv_scores in work:
                all_avg_scores.append(np.nanmean(cv_scores))
                all_cv_scores.append(cv_scores)
                all_subsets.append(new_subset)

            best = np.argmax(all_avg_scores)
            res = (all_subsets[best],
                   all_avg_scores[best],
                   all_cv_scores[best])
        return res

    def _exclusion(self, feature_set, X, y, fixed_feature=None,
                   groups=None, **fit_params):
        n = len(feature_set)
        res = (None, None, None)
        if n > 1:
            all_avg_scores = []
            all_cv_scores = []
            all_subsets = []
            features = n
            n_jobs = min(self.n_jobs, features)
            parallel = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                                pre_dispatch=self.pre_dispatch)
            work = parallel(delayed(_calc_score)(self, X[:, p], y, p,
                                                 groups=groups, **fit_params)
                            for p in combinations(feature_set, r=n - 1)
                            if not fixed_feature or
                            fixed_feature.issubset(set(p)))

            for p, cv_scores in work:

                all_avg_scores.append(np.nanmean(cv_scores))
                all_cv_scores.append(cv_scores)
                all_subsets.append(p)

            best = np.argmax(all_avg_scores)
            res = (all_subsets[best],
                   all_avg_scores[best],
                   all_cv_scores[best])
        return res

    def transform(self, X):
        """Reduce X to its most important features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.

        Returns
        -------
        Reduced feature subset of X, shape={n_samples, k_features}

        """
        self._check_fitted()
        if hasattr(X, 'loc'):
            X_ = X.values
        else:
            X_ = X
        return X_[:, self.k_feature_idx_]

    def fit_transform(self, X, y, groups=None, **fit_params):
        """Fit to training data then reduce X to its most important features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        y : array-like, shape = [n_samples]
            Target values.
            New in v 0.13.0: a pandas Series are now also accepted as
            argument for y.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.
        fit_params : various, optional
            Additional parameters that are being passed to the estimator.
            For example, `sample_weights=weights`.

        Returns
        -------
        Reduced feature subset of X, shape={n_samples, k_features}

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
            raise AttributeError('SequentialFeatureSelector has not been'
                                 ' fitted, yet.')
