# Out-of-fold stacking regressor
#
# For explanation of approach, see:
# dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/#Stacking
#
# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-regressor for out-of-fold stacking regression
# Authors:
#  Eike Dehling <e.e.dehling@gmail.com>
#  Sebastian Raschka <https://sebastianraschka.com>
#
# License: BSD 3 clause

from ..externals.estimator_checks import check_is_fitted
from ..externals import six
from ..externals.name_estimators import _name_estimators
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.model_selection._split import check_cv

import numpy as np


class StackingCVRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    """A 'Stacking Cross-Validation' regressor for scikit-learn estimators.

    New in mlxtend v0.7.0

    Notes
    -------
    The StackingCVRegressor uses scikit-learn's check_cv
    internally, which doesn't support a random seed. Thus
    NumPy's random seed need to be specified explicitely for
    deterministic behavior, for instance, by setting
    np.random.seed(RANDOM_SEED)
    prior to fitting the StackingCVRegressor

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
    use_features_in_secondary : bool (default: False)
        If True, the meta-regressor will be trained both on
        the predictions of the original regressors and the
        original dataset.
        If False, the meta-regressor will be trained only on
        the predictions of the original regressors.
    shuffle : bool (default: True)
        If True,  and the `cv` argument is integer, the training data will
        be shuffled at fitting stage prior to cross-validation. If the `cv`
        argument is a specific cross validation technique, this argument is
        omitted.
    store_train_meta_features : bool (default: False)
        If True, the meta-features computed from the training data
        used for fitting the
        meta-regressor stored in the `self.train_meta_features_` array,
        which can be
        accessed after calling `fit`.
    refit : bool (default: True)
        Clones the regressors for stacking regression if True (default)
        or else uses the original ones, which will be refitted on the dataset
        upon calling the `fit` method. Setting refit=False is
        recommended if you are working with estimators that are supporting
        the scikit-learn fit/predict API interface but are not compatible
        to scikit-learn's `clone` function.

    Attributes
    ----------
    train_meta_features : numpy array, shape = [n_samples, n_regressors]
        meta-features for training data, where n_samples is the
        number of samples
        in training data and len(self.regressors) is the number of regressors.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/

    """
    def __init__(self, regressors, meta_regressor, cv=5,
                 shuffle=True,
                 use_features_in_secondary=False,
                 store_train_meta_features=False,
                 refit=True):

        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.named_regressors = {key: value for
                                 key, value in
                                 _name_estimators(regressors)}
        self.named_meta_regressor = {'meta-%s' % key: value for
                                     key, value in
                                     _name_estimators([meta_regressor])}
        self.cv = cv
        self.shuffle = shuffle
        self.use_features_in_secondary = use_features_in_secondary
        self.store_train_meta_features = store_train_meta_features
        self.refit = refit

    def fit(self, X, y, groups=None):
        """ Fit ensemble regressors and the meta-regressor.

        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : numpy array, shape = [n_samples]
            Target values.

        groups : numpy array/None, shape = [n_samples]
            The group that each sample belongs to. This is used by specific
            folding strategies such as GroupKFold()

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

        kfold = check_cv(self.cv, y)
        if isinstance(self.cv, int):
            # Override shuffle parameter in case of self generated
            # cross-validation strategy
            kfold.shuffle = self.shuffle

        meta_features = np.zeros((X.shape[0], len(self.regressors)))

        #
        # The outer loop iterates over the base-regressors. Each regressor
        # is trained cv times and makes predictions, after which we train
        # the meta-regressor on their combined results.
        #
        for i, regr in enumerate(self.regressors):
            #
            # In the inner loop, each model is trained cv times on the
            # training-part of this fold of data; and the holdout-part of data
            # is used for predictions. This is repeated cv times, so in
            # the end we have predictions for each data point.
            #
            # Advantage of this complex approach is that data points we're
            # predicting have not been trained on by the algorithm, so it's
            # less susceptible to overfitting.
            #
            for train_idx, holdout_idx in kfold.split(X, y, groups):
                instance = clone(regr)
                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                meta_features[holdout_idx, i] = y_pred

        # save meta-features for training data
        if self.store_train_meta_features:
            self.train_meta_features_ = meta_features

        # Train meta-model on the out-of-fold predictions
        if self.use_features_in_secondary:
            self.meta_regr_.fit(np.hstack((X, meta_features)), y)
        else:
            self.meta_regr_.fit(meta_features, y)

        # Retrain base models on all data
        for regr in self.regr_:
            regr.fit(X, y)

        return self

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

        #
        # First we make predictions with the base-models then we predict with
        # the meta-model from that info.
        #

        check_is_fitted(self, 'regr_')

        meta_features = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])

        if self.use_features_in_secondary:
            return self.meta_regr_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_regr_.predict(meta_features)

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
        return np.column_stack([regr.predict(X) for regr in self.regr_])

    def get_params(self, deep=True):
        #
        # Return estimator parameter names for GridSearch support.
        #
        if not deep:
            return super(StackingCVRegressor, self).get_params(deep=False)
        else:
            out = self.named_regressors.copy()
            for name, step in six.iteritems(self.named_regressors):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            out.update(self.named_meta_regressor.copy())
            for name, step in six.iteritems(self.named_meta_regressor):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            for key, value in six.iteritems(super(StackingCVRegressor,
                                            self).get_params(deep=False)):
                out['%s' % key] = value

            return out
