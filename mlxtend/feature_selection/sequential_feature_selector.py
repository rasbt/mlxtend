# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Algorithm for sequential feature selection.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import datetime
import sys
import types
from copy import deepcopy
from itertools import combinations

import numpy as np
import scipy as sp
import scipy.stats
from joblib import Parallel, delayed
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.metrics import get_scorer

from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition
from .utilities import _calc_score, _get_featurenames, _merge_lists, _preprocess


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
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.

    cv : int (default: 5)
        Integer or iterable yielding train, test splits. If cv is an integer
        and `estimator` is a classifier (or y consists of integer class
        labels) stratified k-fold. Otherwise regular k-fold cross-validation
        is performed. No cross-validation if cv is None, False, or 0.

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
        New in mlxtend v. 0.18.0.

    feature_groups : list or None (default: None)
        Optional argument for treating certain features as a group.
        This means, the features within a group are always selected together,
        never split.
        For example, `feature_groups=[[1], [2], [3, 4, 5]]`
        specifies 3 feature groups. In this case,
        possible feature selection results with `k_features=2`
        are `[[1], [2]`, `[[1], [3, 4, 5]]`, or `[[2], [3, 4, 5]]`.
        Feature groups can be useful for
        interpretability, for example, if features 3, 4, 5 are one-hot
        encoded features.  (For  more details, please read the notes at the
        bottom of this docstring).  New in mlxtend v. 0.21.0.

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
        the lengths k of these feature subsets. If the parameter
        `feature_groups` is not None, the value of key indicates
        the number of groups that are selected together. The dictionary
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
    https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

    """

    def __init__(
        self,
        estimator,
        k_features=1,
        forward=True,
        floating=False,
        verbose=0,
        scoring=None,
        cv=5,
        n_jobs=1,
        pre_dispatch="2*n_jobs",
        clone_estimator=True,
        fixed_features=None,
        feature_groups=None,
    ):
        self.estimator = estimator
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
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
        self.verbose = verbose

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

        self.fixed_features = fixed_features
        self.feature_groups = feature_groups

        self.fitted = False
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
        return self._get_params("named_estimators", deep=deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params("estimator", "named_estimators", **params)
        return self

    def generate_error_message_k_features(self, name):
        if (
            len(self.fixed_features_) == 0
            and len(self.feature_groups_) == self.n_features
        ):
            err_msg = f"{name} must be between 1 and X.shape[1]."

        elif (
            len(self.fixed_features_) > 0
            and len(self.feature_groups_) == self.n_features
        ):
            err_msg = f"{name} must be between len(fixed_features) and X.shape[1]."

        elif (
            len(self.fixed_features_) == 0
            and len(self.feature_groups_) < self.n_features
        ):
            err_msg = f"{name} must be between 1 and len(feature_groups)."

        else:  # both fixed_features and feature_groups are provided
            err_msg = f"{name} must be between the number of groups that appear in fixed_features and len(feature_groups)."

        return err_msg

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
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for y.
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

        if not set(self.fixed_features_).issubset(set(range(X_.shape[1]))):
            raise ValueError(
                "`fixed_features` contains at least one feature that is not in the"
                " input data `X`."
            )

        self.feature_groups_ = self.feature_groups
        if self.feature_groups_ is None:
            self.feature_groups_ = [[i] for i in range(X_.shape[1])]

        for fg in self.feature_groups_:
            if len(fg) == 0:
                raise ValueError(
                    "Each list in the nested lists `features_group` cannot be empty"
                )

        feature_group_types = {
            type(i) for sublist in self.feature_groups_ for i in sublist
        }
        if len(feature_group_types) > 1:
            raise ValueError(
                f"feature_group values must have the same type. Found {feature_group_types}."
            )

        if isinstance(self.feature_groups_[0][0], str):
            if self.feature_names_to_idx_mapper is None:
                raise ValueError(
                    "The input X does not contain name of features provived in"
                    " `feature_groups`. Try passing input X as pandas DataFrames"
                    " in which the name of features match the ones provided in"
                    " `feature_groups`"
                )

            lst = []
            for item in self.feature_groups_:
                tmp = [self.feature_names_to_idx_mapper[name] for name in item]
                lst.append(tmp)

            self.feature_groups_ = lst

        if sorted(_merge_lists(self.feature_groups_)) != sorted(
            list(range(X_.shape[1]))
        ):
            raise ValueError(
                "`feature_group` must contain all features within `range(X.shape[1])`"
                " and there should be no common feature betweeen any two distinct"
                " group of features provided in `feature_group`"
            )

        features_encoded_by_groupID = np.full(X_.shape[1], -1, dtype=np.int64)
        for group_id, group in enumerate(self.feature_groups_):
            for idx in group:
                features_encoded_by_groupID[idx] = group_id

        lst = [features_encoded_by_groupID[idx] for idx in self.fixed_features_]
        self.fixed_features_group_set = set(lst)

        n_fixed_features_expected = sum(
            len(self.feature_groups_[group_id])
            for group_id in self.fixed_features_group_set
        )
        if n_fixed_features_expected != len(self.fixed_features_):
            raise ValueError(
                "For each feature specified in the `fixed feature`, its group-mates"
                "must be specified as `fix_features` as well when `feature_groups`"
                "is provided."
            )

        self.k_lb = max(1, len(self.fixed_features_group_set))
        self.k_ub = len(self.feature_groups_)

        if (
            not isinstance(self.k_features, int)
            and not isinstance(self.k_features, tuple)
            and not isinstance(self.k_features, str)
        ):
            raise AttributeError(
                "k_features must be a positive integer" ", tuple, or string"
            )

        eligible_k_values_range = range(self.k_lb, self.k_ub + 1)
        if isinstance(self.k_features, int) and (
            self.k_features not in eligible_k_values_range
        ):
            err_msg = self.generate_error_message_k_features("k_features")
            raise AttributeError(err_msg)

        if isinstance(self.k_features, tuple):
            if len(self.k_features) != 2:
                raise AttributeError(
                    "k_features tuple must consist of 2"
                    " elements, a min and a max value."
                )

            if self.k_features[0] > self.k_features[1]:
                raise AttributeError(
                    "The min k_features value must be smaller"
                    " than the max k_features value."
                )

            if self.k_features[0] not in eligible_k_values_range:
                err_msg = self.generate_error_message_k_features(
                    "k_features tuple min value"
                )
                raise AttributeError(err_msg)

                # raise AttributeError(
                #    "k_features tuple min value must be in" " range(1, X.shape[1]+1)."
                # )

            if self.k_features[1] not in eligible_k_values_range:
                err_msg = self.generate_error_message_k_features(
                    "k_features tuple max value"
                )
                raise AttributeError(err_msg)

                # raise AttributeError(
                #    "k_features tuple max value must be in" " range(1, X.shape[1]+1)."
                # )

        self.is_parsimonious = False
        if isinstance(self.k_features, str):
            if self.k_features not in {"best", "parsimonious"}:
                raise AttributeError(
                    "If a string argument is provided, "
                    'it must be "best" or "parsimonious"'
                )
            if self.k_features == "parsimonious":
                self.is_parsimonious = True

        if isinstance(self.k_features, str):
            self.k_features = (self.k_lb, self.k_ub)
        elif isinstance(self.k_features, int):
            # we treat k_features as k group of features
            self.k_features = (self.k_features, self.k_features)

        self.min_k = self.k_features[0]
        self.max_k = self.k_features[1]

        if self.forward:
            k_idx = tuple(sorted(self.fixed_features_group_set))
            k_stop = self.max_k
        else:
            k_idx = tuple(range(self.k_ub))
            k_stop = self.min_k

        k = len(k_idx)
        if k > 0:
            k_idx, k_score = _calc_score(
                self,
                X_,
                y,
                k_idx,
                groups=groups,
                feature_groups=self.feature_groups_,
                **fit_params,
            )
            self.subsets_[k] = {
                "feature_idx": k_idx,
                "cv_scores": k_score,
                "avg_score": np.nanmean(k_score),
            }

        orig_set = set(range(self.k_ub))
        try:
            while k != k_stop:
                prev_subset = set(k_idx)
                if self.forward:
                    search_set = orig_set
                    must_include_set = prev_subset
                else:
                    search_set = prev_subset
                    must_include_set = self.fixed_features_group_set

                k_idx, k_score, cv_scores = self._feature_selector(
                    search_set,
                    must_include_set,
                    X=X_,
                    y=y,
                    is_forward=self.forward,
                    groups=groups,
                    feature_groups=self.feature_groups_,
                    **fit_params,
                )

                k = len(k_idx)
                # floating can lead to multiple same-sized subsets
                if k not in self.subsets_ or (k_score > self.subsets_[k]["avg_score"]):
                    k_idx = tuple(sorted(k_idx))
                    self.subsets_[k] = {
                        "feature_idx": k_idx,
                        "cv_scores": cv_scores,
                        "avg_score": k_score,
                    }

                if self.floating:
                    # floating direction is opposite of self.forward, i.e. in
                    # forward selection, we do floating in backward manner,
                    # and in backward selection, we do floating in forward manner
                    is_float_forward = not self.forward
                    (new_feature_idx,) = set(k_idx) ^ prev_subset
                    for _ in range(X_.shape[1]):
                        if (
                            self.forward
                            and (len(k_idx) - len(self.fixed_features_group_set)) <= 2
                        ):
                            break
                        if not self.forward and (len(orig_set) - len(k_idx) <= 2):
                            break

                        if is_float_forward:
                            # corresponding to self.forward=False
                            search_set = orig_set - {new_feature_idx}
                            must_include_set = set(k_idx)
                        else:
                            # corresponding to self.forward=True
                            search_set = set(k_idx)
                            must_include_set = self.fixed_features_group_set | {
                                new_feature_idx
                            }

                        (
                            k_idx_c,
                            k_score_c,
                            cv_scores_c,
                        ) = self._feature_selector(
                            search_set,
                            must_include_set,
                            X=X_,
                            y=y,
                            is_forward=is_float_forward,
                            groups=groups,
                            feature_groups=self.feature_groups_,
                            **fit_params,
                        )

                        if k_score_c <= k_score:
                            break

                        # In the floating process, we basically revisit our previous
                        # steps. so, len(k_idx_c) definitely exists as a key in
                        # the dictionary `self.subsets_`
                        if k_score_c <= self.subsets_[len(k_idx_c)]["avg_score"]:
                            break
                        else:
                            k_idx, k_score, cv_scores = k_idx_c, k_score_c, cv_scores_c
                            k_idx = tuple(sorted(k_idx))
                            k = len(k_idx)
                            self.subsets_[k] = {
                                "feature_idx": k_idx,
                                "cv_scores": cv_scores,
                                "avg_score": k_score,
                            }

                if self.verbose == 1:
                    sys.stderr.write("\rFeatures: %d/%s" % (len(k_idx), k_stop))
                    sys.stderr.flush()
                elif self.verbose > 1:
                    sys.stderr.write(
                        "\n[%s] Features: %d/%s -- score: %s"
                        % (
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            len(k_idx),
                            k_stop,
                            k_score,
                        )
                    )

                if self._TESTING_INTERRUPT_MODE:  # just to test `KeyboardInterrupt`
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
        for k in self.subsets_:
            if (
                k >= self.min_k
                and k <= self.max_k
                and self.subsets_[k]["avg_score"] > max_score
            ):
                max_score = self.subsets_[k]["avg_score"]
                best_subset = k

        k_score = max_score
        if k_score == ninf:
            # i.e. all keys of self.subsets_ are not in interval `[self.min_k, self.max_k]`
            # this happens if KeyboardInterrupt happens
            keys = list(self.subsets_.keys())
            scores = [self.subsets_[k]["avg_score"] for k in keys]
            arg = np.argmax(scores)

            k_score = scores[arg]
            best_subset = keys[arg]

        k_idx = self.subsets_[best_subset]["feature_idx"]

        if self.is_parsimonious:
            for k in self.subsets_:
                if k >= best_subset:
                    continue
                if self.subsets_[k]["avg_score"] >= (
                    max_score
                    - np.std(self.subsets_[k]["cv_scores"])
                    / self.subsets_[k]["cv_scores"].shape[0]
                ):
                    max_score = self.subsets_[k]["avg_score"]
                    best_subset = k

            k_score = max_score
            k_idx = self.subsets_[best_subset]["feature_idx"]

        for k in self.subsets_:
            self.subsets_[k]["feature_idx"] = _merge_lists(
                self.feature_groups_, self.subsets_[k]["feature_idx"]
            )
        self.k_feature_idx_ = _merge_lists(self.feature_groups_, k_idx)
        self.k_score_ = k_score
        self.subsets_, self.k_feature_names_ = _get_featurenames(
            self.subsets_, self.k_feature_idx_, self.feature_names, self.n_features
        )

        return

    def _feature_selector(
        self,
        search_set,
        must_include_set,
        X,
        y,
        is_forward,
        groups=None,
        feature_groups=None,
        **fit_params,
    ):
        """Perform one round of feature selection. When `is_forward=True`, it is
        a forward selection that searches the `search_set` to find one feature that
        with `must_include_set` results in highest average score. When
        `is_forward=False`, it is a backward selection that searches the `search_set`
        for a feature that its exclusion results in a set of features that includes
        `must_include_set` and has the highest averege score.

        Parameters
        ----------
        self : object
            an instance of class `SequentialFeatureSelector`

        search_set : set
            a set of features through which a feature must be selected to be included
            (when `is_forward=True`) or to be excluded (when `is_forward=False`)

        must_include_set : set
            a set of features that must be present in the selected subset of features

        X : numpy.ndarray
            a 2D numpy array. Each row corresponds to one observation and each
            column corresponds to one feature.

        y : numpy.ndarray
            the target variable

        is_forward : bool
            True if it is forward selection. False if it is backward selection

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.

        feature_groups : list or None (default: None)
            Optional argument for treating certain features as a group.

        fit_params : various, optional
            Additional parameters that are being passed to the estimator.
            For example, `sample_weights=weights`.

        Returns
        -------
        out1 : the selected set of features that has the highest mean of cv scores
        out2 : the mean of cv scores for the selected set of features.
        out3 : all cv scores for the selected set of features
        """
        out = (None, None, None)

        if feature_groups is None:
            feature_groups = [[i] for i in range(X.shape[1])]

        remaining_set = search_set - must_include_set
        remaining = list(remaining_set)
        n = len(remaining)
        if n > 0:
            if is_forward:
                feature_explorer = combinations(remaining, r=1)
            else:
                feature_explorer = combinations(remaining, r=n - 1)

            n_jobs = min(self.n_jobs, n)
            parallel = Parallel(
                n_jobs=n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch
            )
            work = parallel(
                delayed(_calc_score)(
                    self,
                    X,
                    y,
                    tuple(set(p) | must_include_set),
                    groups=groups,
                    feature_groups=feature_groups,
                    **fit_params,
                )
                for p in feature_explorer
            )

            all_avg_scores = []
            all_cv_scores = []
            all_subsets = []
            for new_subset, cv_scores in work:
                all_avg_scores.append(np.nanmean(cv_scores))
                all_cv_scores.append(cv_scores)
                all_subsets.append(new_subset)

            if len(all_avg_scores) > 0:
                best = np.argmax(all_avg_scores)
                out = (all_subsets[best], all_avg_scores[best], all_cv_scores[best])

        return out

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
        X_, _ = _preprocess(X)
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
                "SequentialFeatureSelector has not been" " fitted, yet."
            )
