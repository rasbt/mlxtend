# Stacking regressor

# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-regressor for stacking regression
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..externals import six
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
import numpy as np


class StackingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):

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
    train_meta_features : numpy array, shape = [n_samples, len(self.regressors)]
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
    http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/

    """
    def __init__(self, regressors, meta_regressor, verbose=0,
                 store_train_meta_features=False, refit=True):

        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.named_regressors = {key: value for
                                 key, value in
                                 _name_estimators(regressors)}
        self.named_meta_regressor = {'meta-%s' % key: value for
                                     key, value in
                                     _name_estimators([meta_regressor])}
        self.verbose = verbose
        self.store_train_meta_features = store_train_meta_features
        self.refit = refit

    def fit(self, X, y):
        """Learn weight coefficients from training data for each regressor.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values.

        Returns
        -------
        self : object

        """
        if self.refit:
            self.regr_ = [clone(clf) for clf in self.regressors]
            self.meta_regr_ = clone(self.meta_regressor)
        else:
            self.clfs_ = self.regressors
            self.meta_clf_ = self.meta_regressor

        if self.verbose > 0:
            print("Fitting %d regressors..." % (len(self.regressors)))

        for regr in self.regr_:

            if self.verbose > 0:
                i = self.regr_.index(regr) + 1
                print("Fitting regressor%d: %s (%d/%d)" %
                      (i, _name_estimators((regr,))[0][0], i, len(self.regr_)))

            if self.verbose > 2:
                if hasattr(regr, 'verbose'):
                    regr.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(_name_estimators((regr,))[0][1])

            regr.fit(X, y)

        meta_features = self.predict_meta_features(X)
        self.meta_regr_.fit(meta_features, y)

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
        if not deep:
            return super(StackingRegressor, self).get_params(deep=False)
        else:
            out = self.named_regressors.copy()
            for name, step in six.iteritems(self.named_regressors):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            out.update(self.named_meta_regressor.copy())
            for name, step in six.iteritems(self.named_meta_regressor):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            for key, value in six.iteritems(super(StackingRegressor,
                                            self).get_params(deep=False)):
                out['%s' % key] = value

            return out

    def predict_meta_features(self, X):
        """ Get meta-features of test-data.

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
            of regressors.

        """
        check_is_fitted(self, 'regr_')
        return np.column_stack([r.predict(X) for r in self.regr_])

    def predict(self, X):
        """ Predict target values for X.

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
        check_is_fitted(self, 'regr_')
        meta_features = self.predict_meta_features(X)
        return self.meta_regr_.predict(meta_features)
