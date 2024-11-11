# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Algorithm for exhaustive feature selection.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import operator as op
import sys
import types
from copy import deepcopy
from functools import reduce
from itertools import chain, combinations

import numpy as np
import scipy as sp
import scipy.stats
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.metrics import get_scorer

from ..externals.name_estimators import _name_estimators
from .utilities import _calc_score, _get_featurenames, _merge_lists, _preprocess


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
        This means, the features within a group are always selected together,
        never split.
        For example, `feature_groups=[[1], [2], [3, 4, 5]]`
        specifies 3 feature groups.In this case,
        possible feature selection results with `k_features=2`
        are `[[1], [2]`, `[[1], [3, 4, 5]]`, or `[[2], [3, 4, 5]]`.
        Feature groups can be useful for
        interpretability, for example, if features 3, 4, 5 are one-hot
        encoded features.  (For  more details, please read the notes at the
        bottom of this docstring).  New in mlxtend v. 0.21.0.

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
        array indices. The 'feature_names' is new in v. 0.13.0.

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

    (3) If both fixed_features and feature_groups are specified, ensure that each
    feature group contains the fixed_features selection. E.g., for a 3-feature set
    fixed_features=[0, 1] and feature_groups=[[0, 1], [2]] is valid;
    fixed_features=[0, 1] and feature_groups=[[0], [1, 2]] is not valid.

    (4) In case of KeyboardInterrupt, the dictionary subsets may not be completed.
    If user is still interested in getting the best score, they can use method
    `finalize_fit`.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/

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
        # Want to raise meaningful error message if a
        # cross-validation generator is inputted
        if isinstance(cv, types.GeneratorType):
            err_msg = (
                "Input cv is a generator object, which is not "
                "supported. Instead please input an iterable yielding "
                "train, test splits. This can usually be done by "
                "passing a cross-validation generator to the "
                "built-in list function. I.e. cv=list(<cv-generator>)"
            )
            raise TypeError(err_msg)

        self.cv = cv
        self.n_jobs = n_jobs
        self.print_progress = print_progress

        self.clone_estimator = clone_estimator
        if self.clone_estimator:
            self.est_ = clone(self.estimator)
        else:
            self.est_ = self.estimator

        self.scoring = scoring
        if self.scoring is None:
            if not hasattr(self.est_, "_estimator_type"):
                raise AttributeError(
                    "Estimator must have an ._estimator_type for infering `scoring`"
                )

            if self.est_._estimator_type == "classifier":
                self.scoring = "accuracy"
            elif self.est_._estimator_type == "regressor":
                self.scoring = "r2"
            else:
                raise AttributeError("Estimator must be a Classifier or Regressor.")

        if isinstance(self.scoring, str):
            self.scorer = get_scorer(self.scoring)
        else:
            self.scorer = self.scoring

        self.named_est = {
            key: value for key, value in _name_estimators([self.estimator])
        }

        self.fixed_features = fixed_features
        self.feature_groups = feature_groups

        self.fitted = False
        self.interrupted_ = False

        # don't mess with this unless testing
        self._TESTING_INTERRUPT_MODE = False

    def fit(self, X, y, groups=None, **fit_params):
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

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.

        fit_params : dict of string -> object, optional
            Parameters to pass to to the fit method of classifier.

        Returns
        -------
        self : object

        """

        self.subsets_ = {}
        self.fitted = False
        self.interrupted_ = False
        self.best_idx_ = None
        self.best_feature_names_ = None
        self.best_score_ = None

        X_, self.feature_names = _preprocess(X)
        self.n_features = X_.shape[1]

        self.feature_names_to_idx_mapper = None
        if self.feature_names is not None:
            self.feature_names_to_idx_mapper = {
                name: idx for idx, name in enumerate(self.feature_names)
            }

        self.fixed_features_ = self.fixed_features
        if self.fixed_features_ is None:
            self.fixed_features_ = tuple()

        fixed_feature_types = {type(i) for i in self.fixed_features_}
        if len(fixed_feature_types) > 1:
            raise ValueError(
                f"fixed_features values must have the same type. Found {fixed_feature_types}."
            )

        if len(self.fixed_features_) > 0 and isinstance(self.fixed_features_[0], str):
            if self.feature_names_to_idx_mapper is None:
                raise ValueError(
                    "The input X does not contain name of features provived in"
                    " `fixed_features`. Try passing input X as pandas DataFrames."
                )

            self.fixed_features_ = tuple(
                self.feature_names_to_idx_mapper[name] for name in self.fixed_features_
            )

        if not set(self.fixed_features_).issubset(set(range(self.n_features))):
            raise ValueError(
                "`fixed_features` contains at least one feature that is not in the"
                " input data `X`."
            )

        if self.feature_groups is None:
            self.feature_groups = [[i] for i in range(self.n_features)]

        for fg in self.feature_groups:
            if len(fg) == 0:
                raise ValueError(
                    "Each list in the nested lists `features_group` cannot be empty"
                )

        feature_group_types = {
            type(i) for sublist in self.feature_groups for i in sublist
        }
        if len(feature_group_types) > 1:
            raise ValueError(
                f"feature_group values must have the same type. Found {feature_group_types}."
            )

        if isinstance(self.feature_groups[0][0], str):
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

            self.feature_groups = lst

        if sorted(_merge_lists(self.feature_groups)) != sorted(
            list(range(self.n_features))
        ):
            raise ValueError(
                "`feature_group` must contain all features within `range(X.shape[1])`"
                " and there should be no common feature betweeen any two distinct"
                " group of features provided in `feature_group`"
            )

        # label-encoding fixed_features according to the groups in `feature_groups`
        # and replace each individual feature in `fixed_features` with their correspondig
        # group id
        features_encoded_by_groupID = np.full(self.n_features, -1, dtype=np.int64)
        for id, group in enumerate(self.feature_groups):
            for idx in group:
                features_encoded_by_groupID[idx] = id

        lst = [features_encoded_by_groupID[idx] for idx in self.fixed_features_]
        self.fixed_features_group_set = set(lst)

        n_fixed_features_expected = sum(
            len(self.feature_groups[id]) for id in self.fixed_features_group_set
        )
        if n_fixed_features_expected != len(self.fixed_features_):
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

        non_fixed_groups = (
            set(range(len(self.feature_groups))) - self.fixed_features_group_set
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
                    list(set(c).union(self.fixed_features_group_set)),
                    groups=groups,
                    feature_groups=self.feature_groups,
                    **fit_params,
                )
                for c in candidates
            )
        )

        try:
            for iteration, (indices, cv_scores) in work:
                self.subsets_[iteration] = {
                    "feature_idx": _merge_lists(self.feature_groups, indices),
                    "cv_scores": cv_scores,
                    "avg_score": np.mean(cv_scores),
                }

                if self.print_progress:
                    sys.stderr.write("\rFeatures: %d/%d" % (iteration + 1, all_comb))
                    sys.stderr.flush()

                if self._TESTING_INTERRUPT_MODE:  # this is just for testing
                    self.finalize_fit()
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            self.interrupted_ = True
            sys.stderr.write("\nSTOPPING EARLY DUE TO KEYBOARD INTERRUPT...")

        if self.interrupted_:
            self.fitted = False
        else:
            self.fitted = True  # the completion of sequential selection process.
            self.finalize_fit()

        return self

    def finalize_fit(self):
        if np.__version__ < "2.0":
            ninf = np.NINF
        else:
            ninf = -np.inf
        max_score = ninf
        for c in self.subsets_:
            if self.subsets_[c]["avg_score"] > max_score:
                best_subset = c
                max_score = self.subsets_[c]["avg_score"]

        self.best_idx_ = self.subsets_[best_subset]["feature_idx"]
        self.best_score_ = max_score
        self.subsets_, self.best_feature_names_ = _get_featurenames(
            self.subsets_, self.best_idx_, self.feature_names, self.n_features
        )

        return

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
        X_, _ = _preprocess(X)
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
