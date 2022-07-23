# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Algorithm for exhaustive feature selection.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import operator as op
import sys
from copy import deepcopy
from functools import reduce
from itertools import chain, combinations

import numpy as np
import scipy as sp
import scipy.stats
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score

from ..externals.name_estimators import _name_estimators


def _merge_lists(nested_list, high_level_indices=None):
    """
    merge elements of lists (of a nested_list) into one single list

    Parameters
    ----------
    nested_list: List
        a  list whose elements must be list as well.

    high_level_indices: list or tuple, default None
        a list or tuple that contains integers that are between 0 (inclusive) and
        the length of `nested_lst` (exclusive). If None, the merge of all
        lists nested in `nested_list` will be returned.

    Returns
    -------
    out: tuple
        a tuple, with elements sorted in ascending order, that is the merge of inner
        lists whose indices are provided in `high_level_indices`

    Example:
    nested_list = [[1],[2, 3],[4]]
    high_level_indices = [1, 2]
    >>> _merge_lists(nested_list, high_level_indices)
    [2, 3, 4] # merging [2, 3] and [4]
    """
    if high_level_indices is None:
        high_level_indices = list(range(len(nested_list)))

    out = []
    for idx in high_level_indices:
        out.extend(nested_list[idx])

    return tuple(sorted(out))


def _calc_score(selector, X, y, indices, groups=None, **fit_params):
    if selector.cv:
        scores = cross_val_score(
            selector.est_,
            X[:, indices],
            y,
            groups=groups,
            cv=selector.cv,
            scoring=selector.scorer,
            n_jobs=1,
            pre_dispatch=selector.pre_dispatch,
            fit_params=fit_params,
        )
    else:
        selector.est_.fit(X[:, indices], y, **fit_params)
        scores = np.array([selector.scorer(selector.est_, X[:, indices], y)])
    return indices, scores


def _get_featurenames(subsets_dict, feature_idx, custom_feature_names, X):
    feature_names = None
    if feature_idx is not None:
        if custom_feature_names is not None:
            feature_names = tuple((custom_feature_names[i] for i in feature_idx))
        elif hasattr(X, "loc"):
            feature_names = tuple((X.columns[i] for i in feature_idx))
        else:
            feature_names = tuple(str(i) for i in feature_idx)

    subsets_dict_ = deepcopy(subsets_dict)
    for key in subsets_dict_:
        if custom_feature_names is not None:
            new_tuple = tuple(
                (custom_feature_names[i] for i in subsets_dict[key]["feature_idx"])
            )
        elif hasattr(X, "loc"):
            new_tuple = tuple((X.columns[i] for i in subsets_dict[key]["feature_idx"]))
        else:
            new_tuple = tuple(str(i) for i in subsets_dict[key]["feature_idx"])
        subsets_dict_[key]["feature_names"] = new_tuple

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
        Maximum number of features to select. If parameter `feature_groups` is not
        None, the number of features is equal to the number of feature groups, i.e.
        `len(feature_groups)`. For  example, if `feature_groups = [[0], [1], [2, 3],
        [4]]`, then the `max_features` value cannot exceed 4.
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

    fixed_features : tuple (default: None)
        If not `None`, the feature indices provided as a tuple will be
        regarded as fixed by the feature selector. For example, if
        `fixed_features=(1, 3, 7)`, the 2nd, 4th, and 8th feature are
        guaranteed to be present in the solution. Note that if
        `fixed_features` is not `None`, make sure that the number of
        features to be selected is greater than `len(fixed_features)`.
        In other words, ensure that `k_features > len(fixed_features)`.

    feature_groups : list or None (default: None)
        Optional argument for treating certain features as a group.
        For example `[[1], [2], [3, 4, 5]]`, which can be useful for
        interpretability, for example, if features 3, 4, 5 are one-hot
        encoded features.  (for  more details, please read the notes at the
        bottom of this docstring).

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

    Notes
    -----
    (1) If parameter `feature_groups` is not None, the
    number of features is equal to the number of feature groups, i.e.
    `len(feature_groups)`. For  example, if `feature_groups = [[0], [1], [2, 3],
    [4]]`, then the `max_features` value cannot exceed 4.

    (2) Although two or more individual features may be considered as one group
    throughout the feature-selection process, it does not mean the individual
    features of that group have the same impact on the outcome. For instance, in
    linear regression, the coefficient of the feature 2 and 3 can be different
    even if they are considered as one group in feature_groups.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/

    """

    def __init__(
        self,
        estimator,
        min_features=1,
        max_features=1,
        print_progress=True,
        scoring="accuracy",
        cv=5,
        n_jobs=1,
        pre_dispatch="2*n_jobs",
        clone_estimator=True,
        fixed_features=None,
        feature_groups=None,
    ):
        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features
        self.pre_dispatch = pre_dispatch
        self.scoring = scoring
        self.scorer = get_scorer(scoring)
        self.cv = cv
        self.print_progress = print_progress
        self.n_jobs = n_jobs
        self.named_est = {
            key: value for key, value in _name_estimators([self.estimator])
        }
        self.clone_estimator = clone_estimator
        if self.clone_estimator:
            self.est_ = clone(self.estimator)
        else:
            self.est_ = self.estimator

        self.fixed_features = fixed_features
        self.feature_groups = feature_groups

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
        self.feature_names = None
        self.best_idx_ = None
        self.best_feature_names_ = None
        self.best_score_ = None

        if hasattr(X, "loc"):
            X_ = X.values
            self.feature_names = list(X.columns)
        else:
            X_ = X

        self.feature_names_to_idx_mapper = None
        if self.feature_names is not None:
            self.feature_names_to_idx_mapper = {
                name: idx for idx, name in enumerate(self.feature_names)
            }

        if (
            custom_feature_names is not None
            and len(custom_feature_names) != X_.shape[1]
        ):
            raise ValueError(
                "If custom_feature_names is not None, "
                "the number of elements in custom_feature_names "
                "must equal the number of columns in X."
            )

        # preprocessing on fixed_featuress
        if self.fixed_features is None:
            self.fixed_features = tuple()

        if len(self.fixed_features) > 0 and isinstance(self.fixed_features[0], str):
            # ASSUME all values provided in fixed_feature are string values
            if self.feature_names_to_idx_mapper is None:
                raise ValueError(
                    "The input X does not contain name of features provived in"
                    " `fixed_features`. Try passing input X as pandas DataFrames."
                )

            self.fixed_features = tuple(
                self.feature_names_to_idx_mapper[name] for name in self.fixed_features
            )

        if not set(self.fixed_features).issubset(set(range(X_.shape[1]))):
            raise ValueError(
                "`fixed_features` contains at least one feature that is not in the"
                " input data `X`."
            )

        # preprocessing on feature_groups
        if self.feature_groups is None:
            self.feature_groups = [[i] for i in range(X_.shape[1])]

        for fg in self.feature_groups:
            if len(fg) == 0:
                raise ValueError(
                    "Each list in the nested lists `features_group`" "cannot be empty"
                )

        if isinstance(self.feature_groups[0][0], str):
            # ASSUME all values provided in feature_groups are string values
            if self.feature_names_to_idx_mapper is None:
                raise ValueError(
                    "The input X does not contain name of features provived in"
                    " `feature_groups`. Try passing input X as pandas DataFrames"
                    " in which the name of features match the ones provided in"
                    " `feature_groups`"
                )

            lst = []
            for item in self.feature_groups:
                tmp = [self.feature_names_to_idx_mapper[name] for name in item]
                lst.append(tmp)

            self.feature_groups[:] = lst

        if sorted(_merge_lists(self.feature_groups)) != sorted(
            list(range(X_.shape[1]))
        ):
            raise ValueError(
                "`feature_group` must contain all features within `range(X.shape[1])`"
                " and there should be no common feature betweeen any two distinct"
                " group of features provided in `feature_group`"
            )

        # label-encoding fixed_features according to the groups in `feature_groups`
        # and replace each individual feature in `fixed_features` with their correspondig
        # group id
        features_encoded_by_groupID = np.full(X_.shape[1], -1, dtype=np.int64)
        for id, group in enumerate(self.feature_groups):
            for idx in group:
                features_encoded_by_groupID[idx] = id

        lst = [features_encoded_by_groupID[idx] for idx in self.fixed_features]
        self.fixed_features_group_set = set(lst)

        n_fixed_features_expected = sum(
            len(self.feature_groups[id]) for id in self.fixed_features_group_set
        )
        if n_fixed_features_expected != len(self.fixed_features):
            raise ValueError(
                "For each feature specified in the `fixed feature`, its group-mates"
                "must be specified as `fix_features` as well when `feature_groups`"
                "is provided."
            )

        n_features_ub = len(self.feature_groups)
        n_features_lb = max(1, len(self.fixed_features_group_set))
        # check `self.max_features`
        if not isinstance(self.max_features, int) or (
            self.max_features > n_features_ub or self.max_features < n_features_lb
        ):
            raise AttributeError(
                f"max_features must be smaller than {n_features_ub + 1}"
                f" and larger than {n_features_lb - 1}"
            )

        # check `self.min_features`
        if not isinstance(self.min_features, int) or (
            self.min_features > n_features_ub or self.min_features < n_features_lb
        ):
            raise AttributeError(
                f"min_features must be smaller than {n_features_ub + 1}"
                f" and larger than {n_features_lb - 1}"
            )

        if self.max_features < self.min_features:
            raise AttributeError("min_features must be <= max_features")

        non_fixed_groups = set(range(len(self.feature_groups))) - set(
            self.fixed_feature_groups
        )
        non_fixed_groups = sorted(list(non_fixed_groups))

        # candidates in the following lines are the non-fixed-features candidates
        # (the fixed features will be added later to each combination)
        min_num_candidates = self.min_features - len(self.fixed_features_group_set)
        max_num_candidates = self.max_features - len(self.fixed_features_group_set)
        candidates = chain.from_iterable(
            combinations(non_fixed_groups, r=i)
            for i in range(min_num_candidates, max_num_candidates + 1)
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

            r = min(r, n - r)
            if r == 0:
                return 1
            numer = reduce(op.mul, range(n, n - r, -1))
            denom = reduce(op.mul, range(1, r + 1))
            return numer // denom

        all_comb = np.sum(
            [
                ncr(n=len(non_fixed_groups), r=i)
                for i in range(min_num_candidates, max_num_candidates + 1)
            ]
        )

        n_jobs = min(self.n_jobs, all_comb)
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=self.pre_dispatch)
        work = enumerate(
            parallel(
                delayed(_calc_score)(
                    self,
                    X_,
                    y,
                    _merge_lists(
                        self.feature_groups,
                        list(set(c).union(self.fixed_features_group_set)),
                    ),
                    groups=groups,
                    **fit_params,
                )
                for c in candidates
            )
        )

        try:
            for iteration, (c, cv_scores) in work:
                self.subsets_[iteration] = {
                    "feature_idx": c,
                    "cv_scores": cv_scores,
                    "avg_score": np.mean(cv_scores),
                }

                if self.print_progress:
                    sys.stderr.write("\rFeatures: %d/%d" % (iteration + 1, all_comb))
                    sys.stderr.flush()

                if self._TESTING_INTERRUPT_MODE:
                    self.subsets_, self.best_feature_names_ = _get_featurenames(
                        self.subsets_, self.best_idx_, custom_feature_names, X
                    )
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            self.interrupted_ = True
            sys.stderr.write("\nSTOPPING EARLY DUE TO KEYBOARD INTERRUPT...")

        max_score = float("-inf")
        for c in self.subsets_:
            if self.subsets_[c]["avg_score"] > max_score:
                max_score = self.subsets_[c]["avg_score"]
                best_subset = c
        score = max_score
        idx = self.subsets_[best_subset]["feature_idx"]

        self.best_idx_ = idx
        self.best_score_ = score
        self.fitted = True
        self.subsets_, self.best_feature_names_ = _get_featurenames(
            self.subsets_, self.best_idx_, custom_feature_names, X
        )
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
        if hasattr(X, "loc"):
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
            std_dev = np.std(self.subsets_[k]["cv_scores"])
            bound, std_err = self._calc_confidence(
                self.subsets_[k]["cv_scores"], confidence=confidence_interval
            )
            fdict[k]["ci_bound"] = bound
            fdict[k]["std_dev"] = std_dev
            fdict[k]["std_err"] = std_err
        return fdict

    def _calc_confidence(self, ary, confidence=0.95):
        std_err = scipy.stats.sem(ary)
        bound = std_err * sp.stats.t._ppf((1 + confidence) / 2.0, len(ary))
        return bound, std_err

    def _check_fitted(self):
        if not self.fitted:
            raise AttributeError(
                "ExhaustiveFeatureSelector has not been" " fitted, yet."
            )
