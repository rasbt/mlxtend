# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Algorithm for generic feature selection.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause
#
# Modified for generic search
# Jonathan Taylor 2021

import types
import sys
from copy import deepcopy

import numpy as np
import scipy as sp

from sklearn.metrics import get_scorer
from sklearn.base import (clone, MetaEstimatorMixin)
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed

from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition

class FeatureSelector(_BaseXComposition, MetaEstimatorMixin):

    """Feature Selection for Classification and Regression.

    Parameters
    ----------
    estimator: scikit-learn classifier or regressor
    strategy: Strategy
        Description of search strategy: a named tuple
        with fields `initial_state`, 
        `candidate_states`, `build_submodel`, 
        `check_finished` and `postprocess`.

    verbose: int (default: 0), level of verbosity to use in logging.
        If 0, no output,
        if 1 number of features in current set, if 2 detailed logging 
        including timestamp and cv scores at step.
    scoring: str, callable, or None (default: None)
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
    cv: int (default: 5)
        Integer or iterable yielding train, test splits. If cv is an integer
        and `estimator` is a classifier (or y consists of integer class
        labels) stratified k-fold. Otherwise regular k-fold cross-validation
        is performed. No cross-validation if cv is None, False, or 0.
    n_jobs: int (default: 1)
        The number of CPUs to use for evaluating different feature subsets
        in parallel. -1 means 'all CPUs'.
    pre_dispatch: int, or string (default: '2*n_jobs')
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
    clone_estimator: bool (default: True)
        Clones estimator if True; works with the original estimator instance
        if False. Set to False if the estimator doesn't
        implement scikit-learn's set_params and get_params methods.
        In addition, it is required to set cv=0, and n_jobs=1.

    Attributes
    ----------
    results_: dict
        A dictionary of selected feature subsets during the
        selection, where the dictionary keys are
        the states of these feature selector. The dictionary
        values are dictionaries themselves with the following
        keys: 'scores' (list individual cross-validation scores)
              'avg_score' (average cross-validation score)

    Notes
    -----

    See `Strategy` for explanation of the fields.

    Examples
    -----------
    For usage examples, please see
    TBD

    """
    def __init__(self,
                 estimator,
                 strategy,
                 verbose=0,
                 scoring=None,
                 cv=5,
                 n_jobs=1,
                 pre_dispatch='2*n_jobs',
                 clone_estimator=True,
                 fixed_features=None):

        self.estimator = estimator
        self.strategy = strategy
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
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        y: array-like, shape = [n_samples]
            Target values.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for y.
        custom_feature_names: None or tuple (default: tuple)
            Custom feature names for `self.k_feature_names` and
            `self.subsets_[i]['feature_names']`.
            (new in v 0.13.0)
        groups: array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.
        fit_params: various, optional
            Additional parameters that are being passed to the estimator.
            For example, `sample_weights=weights`.

        Returns
        -------
        self: object

        """

        # reset from a potential previous fit run
        self.interrupted_ = False
        self.finished_ = False

        results_ = {}

        # unpack the strategy
        
        (initial_state,
         candidate_states,
         build_submodel,
         check_finished,
         postprocess) = self.strategy

        # fit initial model

        _state, _scores = _calc_score(self.estimator,
                                      self.scorer,
                                      build_submodel,
                                      X,
                                      y,
                                      initial_state,
                                      groups=groups,
                                      cv=self.cv,
                                      pre_dispatch=self.pre_dispatch,
                                      **fit_params)

        # keep a running track of the best state

        self.path_ = [deepcopy(_state)]
        self.best_state_ = _state
        self.best_score_ = np.nanmean(_scores)

        self.update_results_check(results_,
                                  {_state: {'scores': _scores,
                                            'avg_score': np.nanmean(_scores)}},
                                  check_finished)
                
        try:
            while not self.finished_:

                batch_results = self._batch(_state,
                                            candidate_states(_state),
                                            build_submodel,
                                            X,
                                            y,
                                            groups=groups,
                                            **fit_params)

                _state, _score, self.finished_ = self.update_results_check(results_,
                                                                           batch_results,
                                                                           check_finished)
                                           
                self.path_.append(deepcopy(_state))

                if self._TESTING_INTERRUPT_MODE:
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            self.interrupted_ = True
            sys.stderr.write('\nSTOPPING EARLY DUE TO KEYBOARD INTERRUPT...')

        self.selected_state_, self.results_ = postprocess(results_)
        self.fitted = True
        return self

    def transform(self, X):
        """Reduce X to its most important features.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.

        Returns
        -------
        Reduced feature subset of X, shape={n_samples, k_features}

        """
        self._check_fitted()
        build_submodel = self.strategy.build_submodel
        return build_submodel(X, self.selected_state_)

    def fit_transform(self,
                      X,
                      y,
                      groups=None,
                      **fit_params):
        """Fit to training data then reduce X to its most important features.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        y: array-like, shape = [n_samples]
            Target values.
            New in v 0.13.0: a pandas Series are now also accepted as
            argument for y.
        groups: array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.
        fit_params: various, optional
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
        confidence_interval: float (default: 0.95)
            A positive float between 0.0 and 1.0 to compute the confidence
            interval bounds of the CV score averages.

        Returns
        ----------
        Dictionary with items where each dictionary value is a list
        with the number of iterations (number of feature subsets) as
        its length. The dictionary keys corresponding to these lists
        are as follows:
            'state': tuple of the indices of the feature subset
            'scores': list with individual CV scores
            'avg_score': of CV average scores
            'std_dev': standard deviation of the CV score average
            'std_err': standard error of the CV score average
            'ci_bound': confidence interval bound of the CV score average

        """
        self._check_fitted()
        fdict = deepcopy(self.results_)

        def _calc_confidence(ary, confidence=0.95):
            std_err = sp.stats.sem(ary)
            bound = std_err * sp.stats.t._ppf((1 + confidence) / 2.0, len(ary))
            return bound, std_err

        for k in fdict:
            std_dev = np.std(self.results_[k]['scores'])
            bound, std_err = self._calc_confidence(
                self.results_[k]['scores'],
                confidence=confidence_interval)
            fdict[k]['ci_bound'] = bound
            fdict[k]['std_dev'] = std_dev
            fdict[k]['std_err'] = std_err
        return fdict

    # private methods

    def _batch(self,
               state,
               candidates,
               build_submodel,
               X,
               y,
               groups=None,
               **fit_params):

        results = {}

        if candidates is not None:

            parallel = Parallel(n_jobs=self.n_jobs,
                                verbose=self.verbose,
                                pre_dispatch=self.pre_dispatch)

            work = parallel(delayed(_calc_score)
                            (self.estimator,
                             self.scorer,
                             build_submodel,
                             X,
                             y,
                             state,
                             groups=groups,
                             cv=self.cv,
                             pre_dispatch=self.pre_dispatch,
                             **fit_params)
                            for state in candidates)

            for state, scores in work:
                results[state] = {'scores': scores,
                                  'avg_score': np.nanmean(scores)}

        return results

    def _check_fitted(self):
        if not self.fitted:
            raise AttributeError('{} has not been fitted yet.'.format(self.__class__))

    def update_results_check(self,
                             results,
                             batch_results,
                             check_finished):
        """
        Update `results_` with current batch
        and return a boolean about whether 
        we should continue or not.

        Parameters
        ----------

        results: dict
            Dictionary of all results.
            Keys are state with values
            dictionaries having keys
            `scores`, `avg_scores`.

        batch_results: dict
            Dictionary of results from a batch fit.
            Keys are tate with values
            dictionaries having keys
            `scores`, `avg_scores`.

        check_finished: callable
            Callable taking three arguments 
            `(results, best_state, batch_results)` which determines if
            the state generator should step. Often will just check
            if there is a better score than that at current best state
            but can use entire set of results if desired.

        Returns
        -------

        best_state: object
            State that had the best `avg_score`

        fitted: bool
            If batch_results is empty, fitting
            has terminated so return True.
            Otherwise False.

        """

        finished = batch_results == {}

        if not finished:
            results.update(batch_results)

            (cur_state,
             cur_score,
             finished) = check_finished(results,
                                        self.best_state_,
                                        batch_results)
            if cur_score > self.best_score_:
                self.best_state_ = cur_state
                self.best_score_ = cur_score
            return cur_state, cur_score, finished
        else:
            return None, None, True




# private functions


def _calc_score(estimator,
                scorer,
                build_submodel,
                X,
                y,
                state,
                groups=None,
                cv=None,
                pre_dispatch='2*n_jobs',
                **fit_params):
    
    X_state = build_submodel(X, state)
    
    if cv:
        scores = cross_val_score(estimator,
                                 X_state,
                                 y,
                                 groups=groups,
                                 cv=cv,
                                 scoring=scorer,
                                 n_jobs=1,
                                 pre_dispatch=pre_dispatch,
                                 fit_params=fit_params)
    else:
        estimator.fit(X_state,
                      y,
                          **fit_params)
        scores = np.array([scorer(estimator,
                                  X_state,
                                  y)])
    return state, scores

