# Stacking regressor

# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-regressor for stacking regression
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sparse
from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.utils import check_X_y

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition


class StackingRegressor(_BaseXComposition, RegressorMixin, TransformerMixin):
    """A Stacking regressor for scikit-learn estimators for regression.

    Parameters
    ----------
    regressors : array-like, shape = [n_regressors]
        A list of regressors.
        Invoking the `fit` method on the `StackingRegressor` will fit clones
        of those original regressors that will
        be stored in the class attribute
        `self.regr_`.

    meta_regressor : object
        The meta-regressor to be fitted on the ensemble of
        regressors

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the regressor being fitted
        - `verbose=2`: Prints info about the parameters of the
                       regressor being fitted
        - `verbose>2`: Changes `verbose` param of the underlying regressor to
           self.verbose - 2

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


    Attributes
    ----------
    regr_ : list, shape=[n_regressors]
        Fitted regressors (clones of the original regressors)

    meta_regr_ : estimator
        Fitted meta-regressor (clone of the original meta-estimator)

    coef_ : array-like, shape = [n_features]
        Model coefficients of the fitted meta-estimator

    intercept_ : float
        Intercept of the fitted meta-estimator

    train_meta_features : numpy array,
        shape = [n_samples, len(self.regressors)]
        meta-features for training data, where n_samples is the
        number of samples
        in training data and len(self.regressors) is the number of regressors.

    refit : bool (default: True)
        Clones the regressors for stacking regression if True (default)
        or else uses the original ones, which will be refitted on the dataset
        upon calling the `fit` method. Setting refit=False is
        recommended if you are working with estimators that are supporting
        the scikit-learn fit/predict API interface but are not compatible
        to scikit-learn's `clone` function.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/

    """

    def __init__(
        self,
        regressors,
        meta_regressor,
        verbose=0,
        use_features_in_secondary=False,
        store_train_meta_features=False,
        refit=True,
        multi_output=False,
    ):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.verbose = verbose
        self.use_features_in_secondary = use_features_in_secondary
        self.store_train_meta_features = store_train_meta_features
        self.refit = refit
        self.multi_output = multi_output

    @property
    def named_regressors(self):
        return _name_estimators(self.regressors)

    def fit(self, X, y, sample_weight=None):
        """Learn weight coefficients from training data for each regressor.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : numpy array, shape = [n_samples] or [n_samples, n_targets]
             Target values. Multiple targets are supported only if
             self.multi_output is True.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights passed as sample_weights to each regressor
            in the regressors list as well as the meta_regressor.
            Raises error if some regressor does not support
            sample_weight in the fit() method.

        Returns
        -------
        self : object

        """
        X, y = check_X_y(
            X,
            y,
            accept_sparse=["csc", "csr"],
            dtype=None,
            multi_output=self.multi_output,
        )

        if self.refit:
            self.regr_ = clone(self.regressors)
            self.meta_regr_ = clone(self.meta_regressor)
        else:
            self.regr_ = self.regressors
            self.meta_regr_ = self.meta_regressor

        if self.verbose > 0:
            print("Fitting %d regressors..." % (len(self.regressors)))

        for regr in self.regr_:
            if self.verbose > 0:
                i = self.regr_.index(regr) + 1
                print(
                    "Fitting regressor%d: %s (%d/%d)"
                    % (i, _name_estimators((regr,))[0][0], i, len(self.regr_))
                )

            if self.verbose > 2:
                if hasattr(regr, "verbose"):
                    regr.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(_name_estimators((regr,))[0][1])

            if sample_weight is None:
                regr.fit(X, y)
            else:
                regr.fit(X, y, sample_weight=sample_weight)

        meta_features = self.predict_meta_features(X)

        if not self.use_features_in_secondary:
            # meta model uses the prediction outcomes only
            pass
        elif sparse.issparse(X):
            meta_features = sparse.hstack((X, meta_features))
        else:
            meta_features = np.hstack((X, meta_features))

        if sample_weight is None:
            self.meta_regr_.fit(meta_features, y)
        else:
            self.meta_regr_.fit(meta_features, y, sample_weight=sample_weight)

        # save meta-features for training data
        if self.store_train_meta_features:
            self.train_meta_features_ = meta_features
        return self

    @property
    def coef_(self):
        return self.meta_regr_.coef_

    @property
    def intercept_(self):
        return self.meta_regr_.intercept_

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support."""
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
            columns is len(self.regressors) * n_targets

        """
        check_is_fitted(self, "regr_")
        return np.column_stack([r.predict(X) for r in self.regr_])

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
        check_is_fitted(self, "regr_")
        meta_features = self.predict_meta_features(X)

        if not self.use_features_in_secondary:
            return self.meta_regr_.predict(meta_features)
        elif sparse.issparse(X):
            return self.meta_regr_.predict(sparse.hstack((X, meta_features)))
        else:
            return self.meta_regr_.predict(np.hstack((X, meta_features)))
