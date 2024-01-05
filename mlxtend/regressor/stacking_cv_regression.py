# Out-of-fold stacking regressor
#
# For explanation of approach, see:
# https://web.archive.org/web/20170720114355/dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/
#
# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-regressor for out-of-fold stacking regression
# Authors:
#  Eike Dehling <e.e.dehling@gmail.com>
#  Sebastian Raschka <https://sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from scipy import sparse
from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection._split import check_cv
from sklearn.utils import check_X_y

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition


class StackingCVRegressor(_BaseXComposition, RegressorMixin, TransformerMixin):
    """A 'Stacking Cross-Validation' regressor for scikit-learn estimators.

    Parameters
    ----------
    regressors : array-like, shape = [n_regressors]
        A list of regressors.
        Invoking the `fit` method on the `StackingCVRegressor` will fit clones
        of these original regressors that will
        be stored in the class attribute `self.regr_`.

    meta_regressor : object
        The meta-regressor to be fitted on the ensemble of
        regressor

    cv : int, cross-validation generator or iterable, optional (default: 5)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.
        For integer/None inputs, it will use `KFold` cross-validation

    shuffle : bool (default: True)
        If True,  and the `cv` argument is integer, the training data will
        be shuffled at fitting stage prior to cross-validation. If the `cv`
        argument is a specific cross validation technique, this argument is
        omitted.

    random_state : int, RandomState instance or None, optional (default: None)
        Constrols the randomness of the cv splitter. Used when `cv` is
        integer and `shuffle=True`. New in v0.16.0.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process. New in v0.16.0

    refit : bool (default: True)
        Clones the regressors for stacking regression if True (default)
        or else uses the original ones, which will be refitted on the dataset
        upon calling the `fit` method. Setting refit=False is
        recommended if you are working with estimators that are supporting
        the scikit-learn fit/predict API interface but are not compatible
        to scikit-learn's `clone` function.

    use_features_in_secondary : bool (default: False)
        If True, the meta-regressor will be trained both on
        the predictions of the original regressors and the
        original dataset.
        If False, the meta-regressor will be trained only on
        the predictions of the original regressors.

    store_train_meta_features : bool (default: False)
        If True, the meta-features computed from the training data
        used for fitting the
        meta-regressor stored in the `self.train_meta_features_` array,
        which can be
        accessed after calling `fit`.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details. New in v0.16.0.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    multi_output : bool (default: False)
        If True, allow multi-output targets, but forbid nan or inf values.
        If False, `y` will be checked to be a vector. (New in v0.19.0.)

    Attributes
    ----------
    train_meta_features : numpy array, shape = [n_samples, n_regressors]
        meta-features for training data, where n_samples is the
        number of samples
        in training data and len(self.regressors) is the number of regressors.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/

    """

    def __init__(
        self,
        regressors,
        meta_regressor,
        cv=5,
        shuffle=True,
        random_state=None,
        verbose=0,
        refit=True,
        use_features_in_secondary=False,
        store_train_meta_features=False,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        multi_output=False,
    ):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.refit = refit
        self.use_features_in_secondary = use_features_in_secondary
        self.store_train_meta_features = store_train_meta_features
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.multi_output = multi_output

    def fit(self, X, y, groups=None, sample_weight=None):
        """Fit ensemble regressors and the meta-regressor.

        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : numpy array, shape = [n_samples] or [n_samples, n_targets]
            Target values. Multiple targets are supported only if
            self.multi_output is True.

        groups : numpy array/None, shape = [n_samples]
            The group that each sample belongs to. This is used by specific
            folding strategies such as GroupKFold()

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights passed as sample_weights to each regressor
            in the regressors list as well as the meta_regressor.
            Raises error if some regressor does not support
            sample_weight in the fit() method.

        Returns
        -------
        self : object

        """
        if self.refit:
            self.regr_ = [clone(clf) for clf in self.regressors]
            self.meta_regr_ = clone(self.meta_regressor)
        else:
            self.regr_ = self.regressors
            self.meta_regr_ = self.meta_regressor

        X, y = check_X_y(
            X,
            y,
            accept_sparse=["csc", "csr"],
            dtype=None,
            multi_output=self.multi_output,
        )

        kfold = check_cv(self.cv, y)
        if isinstance(self.cv, int):
            # Override shuffle parameter in case of self generated
            # cross-validation strategy
            kfold.shuffle = self.shuffle
            kfold.random_state = self.random_state
        #
        # The meta_features are collection of the prediction data,
        # in shape of [n_samples, len(self.regressors)]. Each column
        # corresponds to the result of `corss_val_predict` using every
        # base regressors.
        # Advantage of this complex approach is that data points we're
        # predicting have not been trained on by the algorithm, so it's
        # less susceptible to overfitting.
        if sample_weight is None:
            fit_params = None
        else:
            fit_params = dict(sample_weight=sample_weight)
        meta_features = np.column_stack(
            [
                cross_val_predict(
                    regr,
                    X,
                    y,
                    groups=groups,
                    cv=kfold,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    fit_params=fit_params,
                    pre_dispatch=self.pre_dispatch,
                )
                for regr in self.regr_
            ]
        )

        # save meta-features for training data
        if self.store_train_meta_features:
            self.train_meta_features_ = meta_features

        # Train meta-model on the out-of-fold predictions
        if not self.use_features_in_secondary:
            pass
        elif sparse.issparse(X):
            meta_features = sparse.hstack((X, meta_features))
        else:
            meta_features = np.hstack((X, meta_features))

        if sample_weight is None:
            self.meta_regr_.fit(meta_features, y)
        else:
            self.meta_regr_.fit(meta_features, y, sample_weight=sample_weight)

        # Retrain base models on all data
        for regr in self.regr_:
            if sample_weight is None:
                regr.fit(X, y)
            else:
                regr.fit(X, y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """Predict target values for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        y_target : array-like, shape = [n_samples] or [n_samples, n_targets]
            Predicted target values.
        """

        #
        # First we make predictions with the base-models then we predict with
        # the meta-model from that info.
        #

        check_is_fitted(self, "regr_")

        meta_features = np.column_stack([regr.predict(X) for regr in self.regr_])

        if not self.use_features_in_secondary:
            return self.meta_regr_.predict(meta_features)
        elif sparse.issparse(X):
            return self.meta_regr_.predict(sparse.hstack((X, meta_features)))
        else:
            return self.meta_regr_.predict(np.hstack((X, meta_features)))

    def predict_meta_features(self, X):
        """Get meta-features of test-data.

        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
            Test vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        meta-features : numpy array, shape = [n_samples, len(self.regressors)]
            meta-features for test data, where n_samples is the number of
            samples in test data and len(self.regressors) is the number
            of regressors. If self.multi_output is True, then the number of
            columns is len(self.regressors) * n_targets.

        """
        check_is_fitted(self, "regr_")
        return np.column_stack([regr.predict(X) for regr in self.regr_])

    @property
    def named_regressors(self):
        """
        Returns
        -------
        List of named estimator tuples, like [('svc', SVC(...))]
        """
        return _name_estimators(self.regressors)

    def get_params(self, deep=True):
        #
        # Return estimator parameter names for GridSearch support.
        #
        return self._get_params("named_regressors", deep=deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params("regressors", "named_regressors", **params)
        return self
