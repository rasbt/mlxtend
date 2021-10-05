# Jonathan Taylor 2021
# mlxtend Machine Learning Library Extensions
#
# Objects describing search strategy
# Author: Jonathan Taylor <jonathan.taylor@stanford.edu>
# 

from typing import NamedTuple, Any, Callable
from itertools import chain, combinations
from functools import partial

import numpy as np
from sklearn.utils import check_random_state

from .columns import (_get_column_info,
                     Column,
                     _categorical_from_df,
                     _check_categories)

class Strategy(NamedTuple):

    """
    initial_state: object
        Initial state of feature selector.
    state_generator: callable
        Callable taking single argument `state` and returning
        candidates for next batch of scores to be calculated.
    build_submodel: callable
        Callable taking two arguments `(X, state)` that returns
        model matrix represented by `state`.
    check_finished: callable
        Callable taking three arguments 
        `(results, best_state, batch_results)` which determines if
        the state generator should step. Often will just check
        if there is a better score than that at current best state
        but can use entire set of results if desired.
    """

    initial_state: Any
    candidate_states: Callable
    build_submodel: Callable
    check_finished: Callable
    postprocess: Callable

      
class MinMaxCandidates(object):

    def __init__(self,
                 X,
                 min_features=1,
                 max_features=1,
                 fixed_features=None,
                 custom_feature_names=None,
                 categorical_features=None):
        """
        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        min_features: int (default: 1)
            Minumum number of features to select
        max_features: int (default: 1)
            Maximum number of features to select
        fixed_features: column identifiers, default=None
            Subset of features to keep. Stored as `self.columns[fixed_features]`
            where `self.columns` will correspond to columns if X is a `pd.DataFrame`
            or an array of integers if X is an `np.ndarray`
        custom_feature_names: None or tuple (default: tuple)
                Custom feature names for `self.k_feature_names` and
                `self.subsets_[i]['feature_names']`.
                (new in v 0.13.0)
        categorical_features: array-like of {bool, int} of shape (n_features) 
                or shape (n_categorical_features,), default=None.
            Indicates the categorical features.

            - None: no feature will be considered categorical.
            - boolean array-like: boolean mask indicating categorical features.
            - integer array-like: integer indices indicating categorical
              features.

            For each categorical feature, there must be at most `max_bins` unique
            categories, and each categorical value must be in [0, max_bins -1].

        """

        if hasattr(X, 'loc'):
            X_ = X.values
            is_categorical, is_ordinal = _categorical_from_df(X)
            self.columns = X.columns
        else:
            X_ = X
            is_categorical = _check_categories(categorical_features,
                                               X_)[0]
            if is_categorical is None:
                is_categorical = np.zeros(X_.shape[1], np.bool)
            is_ordinal = np.zeros_like(is_categorical)
            self.columns = np.arange(X.shape[1])

        nfeatures = X_.shape[0]

        if (not isinstance(max_features, int) or
                (max_features > nfeatures or max_features < 1)):
            raise AttributeError('max_features must be'
                                 ' smaller than %d and larger than 0' %
                                 (nfeatures + 1))

        if (not isinstance(min_features, int) or
                (min_features > nfeatures or min_features < 1)):
            raise AttributeError('min_features must be'
                                 ' smaller than %d and larger than 0'
                                 % (nfeatures + 1))

        if max_features < min_features:
            raise AttributeError('min_features must be <= max_features')

        self.min_features, self.max_features = min_features, max_features

        # make a mapping from the column info to columns in
        # implied design matrix

        self.column_info_ = _get_column_info(X,
                                             self.columns,
                                             is_categorical,
                                             is_ordinal)
        self.column_map_ = {}
        idx = 0
        for col in self.columns:
            l = self.column_info_[col].columns
            self.column_map_[col] = range(idx, idx +
                                          len(l))
            idx += len(l)
        if (custom_feature_names is not None
                and len(custom_feature_names) != nfeatures):
            raise ValueError('If custom_feature_names is not None, '
                             'the number of elements in custom_feature_names '
                             'must equal %d the number of columns in X.' % idx)
        if custom_feature_names is not None:
            # recompute the Column info using custom_feature_names
            for i, col in enumerate(self.columns):
                cur_col = self.column_info_[col]
                new_name = custom_feature_names[i]
                old_name = cur_col.name
                self.column_info_[col] = Column(col,
                                                new_name,
                                                col.is_categorical,
                                                col.is_ordinal,
                                                tuple([n.replace(old_name,
                                                                 new_name) for n in col.columns]),
                                                col.encoder)

        self._have_already_run = False
        if fixed_features is not None:
            self.fixed_features = set([self.column_info_[f].idx for f in fixed_features])
        else:
            self.fixed_features = set([])
            
    def candidate_states(self, state):
        """
        Produce candidates for fitting.

        Parameters
        ----------

        state: ignored

        Returns
        -------
        candidates: iterator
            A generator of (indices, label) where indices
            are columns of X and label is a name for the 
            given model. The iterator cycles through
            all combinations of columns of nfeature total
            of size ranging between min_features and max_features.
            If appropriate, restricts combinations to include
            a set of fixed features.
            Models are labeled with a tuple of the feature names.
            The names of the columns default to strings of integers
            from range(nfeatures).

        """

        if not self._have_already_run:
            self._have_already_run = True # maybe could be done with a StopIteration on candidates?
            def chain_(i):
                return (c for c in combinations(self.columns, r=i)
                        if self.fixed_features.issubset(c))
            
            candidates = chain.from_iterable(chain_(i) for i in
                                             range(self.min_features,
                                                   self.max_features+1))
            return candidates
        
    def check_finished(self,
                       results,
                       best_state,
                       batch_results):
        """
        Check if we should continue or not. 
        For exhaustive search we stop because
        all models are fit in a single batch.
        """
        return best_state, results[best_state]['avg_score'], True


class StepCandidates(MinMaxCandidates):

    def __init__(self,
                 X,
                 direction='forward',
                 min_features=1,
                 max_features=1,
                 fixed_features=None,
                 custom_feature_names=None,
                 categorical_features=None):
        """
        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        direction: str
            One of ['forward', 'backward', 'both']
        min_features: int (default: 1)
            Minumum number of features to select
        max_features: int (default: 1)
            Maximum number of features to select
        fixed_features: column identifiers, default=None
            Subset of features to keep. Stored as `self.columns[fixed_features]`
            where `self.columns` will correspond to columns if X is a `pd.DataFrame`
            or an array of integers if X is an `np.ndarray`
        custom_feature_names: None or tuple (default: tuple)
                Custom feature names for `self.k_feature_names` and
                `self.subsets_[i]['feature_names']`.
                (new in v 0.13.0)
        categorical_features: array-like of {bool, int} of shape (n_features) 
                or shape (n_categorical_features,), default=None.
            Indicates the categorical features.

            - None: no feature will be considered categorical.
            - boolean array-like: boolean mask indicating categorical features.
            - integer array-like: integer indices indicating categorical
              features.

            For each categorical feature, there must be at most `max_bins` unique
            categories, and each categorical value must be in [0, max_bins -1].

        """

        self.direction = direction
        MinMaxCandidates.__init__(self,
                                  X,
                                  min_features,
                                  max_features,
                                  fixed_features,
                                  custom_feature_names,
                                  categorical_features)
            
    def candidate_states(self, state):
        """
        Produce candidates for fitting.
        For stepwise search this depends on the direction.

        If 'forward', all columns not in the current state
        are added (maintaining an upper limit on the number of columns 
        at `self.max_features`).

        If 'backward', all columns not in the current state
        are dropped (maintaining a lower limit on the number of columns 
        at `self.min_features`).

        All candidates include `self.fixed_features` if any.
        
        Parameters
        ----------

        state: ignored

        Returns
        -------
        candidates: iterator
            A generator of (indices, label) where indices
            are columns of X and label is a name for the 
            given model. The iterator cycles through
            all combinations of columns of nfeature total
            of size ranging between min_features and max_features.
            If appropriate, restricts combinations to include
            a set of fixed features.
            Models are labeled with a tuple of the feature names.
            The names of the columns default to strings of integers
            from range(nfeatures).

        """

        state = set(state)
        if len(state) < self.max_features: # union
            forward = (tuple(sorted(state | set([c]))) for c in self.columns if (c not in state and
                                                                self.fixed_features.issubset(state | set([c]))))
        else:
            forward = None

        if len(state) > self.min_features: # symmetric difference
            backward = (tuple(sorted(state ^ set([c]))) for c in self.columns if (c in state and
                                                                   self.fixed_features.issubset(state ^ set([c]))))
        else:
            backward = None

        if self.direction == 'forward':
            return forward
        elif self.direction == 'backward':
            return backward
        else:
            return chain.from_iterable([forward, backward])
        
    def check_finished(self,
                       results,
                       best_state,
                       batch_results):
        """
        Check if we should continue or not. 

        For stepwise search we stop if we cannot improve
        over our current best score.

        """
        batch_best_score = -np.inf
        batch_best_state = None

        for state in batch_results:
            if batch_results[state]['avg_score'] > batch_best_score:
                batch_best_score = batch_results[state]['avg_score']
                batch_best_state = state
                
        finished = batch_best_score <= results[best_state]['avg_score']
        return batch_best_state, batch_best_score, finished

    def validate(self,
                 current_state,
                 proposed_state):
        """
        Determine if `proposed_state`
        is a valid transition from `current_state`.

        Can be used in subclasses to enforce hierarchy or
        other properties of the model.

        Parameters
        ----------

        state: object
            Current state of the selection procedure.

        proposed_state: object
            Candidate state of the selection procedure.

        Returns
        -------

        valid: bool
            Is this a valid transition?
        """
    
        return True

def min_max(X,
            min_features=1,
            max_features=1,
            fixed_features=None,
            custom_feature_names=None,
            categorical_features=None,
            parsimonious=True):
    """
    Parameters
    ----------
    X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
        New in v 0.13.0: pandas DataFrames are now also accepted as
        argument for X.
    min_features: int (default: 1)
        Minumum number of features to select
    max_features: int (default: 1)
        Maximum number of features to select
    fixed_features: column identifiers, default=None
        Subset of features to keep. Stored as `self.columns[fixed_features]`
        where `self.columns` will correspond to columns if X is a `pd.DataFrame`
        or an array of integers if X is an `np.ndarray`
    custom_feature_names: None or tuple (default: tuple)
            Custom feature names for `self.k_feature_names` and
            `self.subsets_[i]['feature_names']`.
            (new in v 0.13.0)
    categorical_features: array-like of {bool, int} of shape (n_features) 
            or shape (n_categorical_features,), default=None.
        Indicates the categorical features.

        - None: no feature will be considered categorical.
        - boolean array-like: boolean mask indicating categorical features.
        - integer array-like: integer indices indicating categorical
          features.

        For each categorical feature, there must be at most `max_bins` unique
        categories, and each categorical value must be in [0, max_bins -1].

    parsimonious: bool
        If True, use the 1sd rule: among the shortest models
        within one standard deviation of the best score
        pick the one with the best average score. 

    Returns
    -------

    initial_state: tuple
        (column_names, feature_idx)

    state_generator: callable
        Object that proposes candidates
        based on current state. Takes a single 
        argument `state`

    build_submodel: callable
        Candidate generator that enumerate
        all valid subsets of columns.

    check_finished: callable
        Check whether to stop. Takes two arguments:
        `best_result` a dict with keys ['scores', 'avg_score'];
        and `state`.

    """

    min_max = MinMaxCandidates(X,
                               min_features,
                               max_features,
                               fixed_features,
                               custom_feature_names,
                               categorical_features)
    
    # if any categorical features or an intercept
    # is included then we must
    # create a new design matrix

    build_submodel = partial(_build_submodel, min_max.column_info_)

    if min_max.fixed_features:
        initial_features = sorted(min_max.fixed_features)
    else:
        initial_features = range(min_max.min_features)
    initial_state = tuple(initial_features)

    if not parsimonious:
        _postprocess = _postprocess_best
    else:
        _postprocess = _postprocess_best_1sd

    return Strategy(initial_state,
                    min_max.candidate_states,
                    build_submodel,
                    min_max.check_finished,
                    _postprocess)

def step(X,
         direction='forward',
         min_features=1,
         max_features=1,
         random_state=0,
         fixed_features=None,
         initial_features=None,
         custom_feature_names=None,
         categorical_features=None,
         parsimonious=True):
    """
    Parameters
    ----------
    X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
        New in v 0.13.0: pandas DataFrames are now also accepted as
        argument for X.
    direction: str
        One of ['forward', 'backward', 'both']
    min_features: int (default: 1)
        Minumum number of features to select
    max_features: int (default: 1)
        Maximum number of features to select
    fixed_features: column identifiers, default=None
        Subset of features to keep. Stored as `self.columns[fixed_features]`
        where `self.columns` will correspond to columns if X is a `pd.DataFrame`
        or an array of integers if X is an `np.ndarray`
    initial_features: column identifiers, default=None
        Subset of features to be used to initialize when direction
        is `both`. If None defaults to behavior of `forward`.
        where `self.columns` will correspond to columns if X is a `pd.DataFrame`
        or an array of integers if X is an `np.ndarray`
    custom_feature_names: None or tuple (default: tuple)
            Custom feature names for `self.k_feature_names` and
            `self.subsets_[i]['feature_names']`.
            (new in v 0.13.0)
    categorical_features: array-like of {bool, int} of shape (n_features) 
            or shape (n_categorical_features,), default=None.
        Indicates the categorical features.

        - None: no feature will be considered categorical.
        - boolean array-like: boolean mask indicating categorical features.
        - integer array-like: integer indices indicating categorical
          features.

        For each categorical feature, there must be at most `max_bins` unique
        categories, and each categorical value must be in [0, max_bins -1].

    parsimonious: bool
        If True, use the 1sd rule: among the shortest models
        within one standard deviation of the best score
        pick the one with the best average score. 

    Returns
    -------

    initial_state: tuple
        (column_names, feature_idx)

    state_generator: callable
        Object that proposes candidates
        based on current state. Takes a single 
        argument `state`

    build_submodel: callable
        Candidate generator that enumerate
        all valid subsets of columns.

    check_finished: callable
        Check whether to stop. Takes two arguments:
        `best_result` a dict with keys ['scores', 'avg_score'];
        and `state`.

    """

    step = StepCandidates(X,
                          direction,
                          min_features,
                          max_features,
                          fixed_features,
                          custom_feature_names,
                          categorical_features)
    
    # if any categorical features or an intercept
    # is included then we must
    # create a new design matrix

    build_submodel = partial(_build_submodel, step.column_info_)

    # pick an initial state

    if direction in ['forward', 'both']:
        if step.fixed_features:
            forward_features = sorted(step.fixed_features)
        else:
            forward_features = range(step.min_features)
        if direction == 'forward':
            initial_features = forward_features
        else:
            if initial_features is None:
                initial_features = forward_features
    elif direction == 'backward':
        if initial_features is None:
            random_state = check_random_state(random_state)
            initial_features = sorted(random_state.choice([col for col in step.column_info_],
                                                          step.max_features,
                                                          replace=False))
    initial_state = tuple(initial_features)

    if len(initial_features) > step.max_features:
        raise ValueError('initial_features should be of length <= %d' % step.max_features)
    if len(initial_features) < step.min_features:
        raise ValueError('initial_features should be of length >= %d' % step.min_features)
    if not step.fixed_features.issubset(initial_features):
        raise ValueError('initial_features should contain %s' % str(step.fixed_features))

    if not parsimonious:
        _postprocess = _postprocess_best
    else:
        _postprocess = _postprocess_best_1sd

    return Strategy(initial_state,
                    step.candidate_states,
                    build_submodel,
                    step.check_finished,
                    _postprocess)


# private functions


def _build_submodel(column_info, X, cols):
    return np.column_stack([column_info[col].get_columns(X, fit=True)[0] for col in cols])

    
def _postprocess_best(results):
    """
    Find the best state from `results`
    based on `avg_score`.

    Return best state and results
    """

    best_state = None
    best_score = -np.inf

    for state, result in results.items():
        if result['avg_score'] > best_score:
            best_state = state
            best_score = result['avg_score']

    return best_state, results

def _postprocess_best_1sd(results):
    """
    Find the best state from `results`
    based on `avg_score`.

    Find models satisfying the 1sd rule
    and choose the state with best score
    among the smallest such states.

    Return best state and results

    Models are compared by length of state
    """

    best_state = None
    best_score = -np.inf

    for state, result in results.items():
        if result['avg_score'] > best_score:
            best_state = state
            best_score = result['avg_score']

    states_1sd = []

    for state, result in results.items():
        if len(state) >= len(best_state):
            continue
        scores = result['scores']
        _limit = (result['avg_score'] +
                  np.nanstd(scores) / np.sqrt(scores.shape[0]))
        if _limit >= best_score:
            states_1sd.append(state)

    shortest_1sd = np.inf

    for state in states_1sd:
        if len(state) < shortest_1sd:
            shortest_1sd = len(state)
            
    best_state_1sd = None
    best_score_1sd = -np.inf

    for state in states_1sd:
        if ((len(state) == shortest_1sd)
            and (results[state]['avg_score'] <=
                 best_score_1sd)):
            best_state_1sd = state
            best_score_1sd = result['avg_score'][state]
            
    if best_state_1sd:
        return best_state_1sd, results
    else:
        return best_state, results
