# Out-of-fold stacking regressor
#
# For explanation of approach, see:
# dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/#Stacking
#
# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-regressor for out-of-fold stacking regression
# Author: Eike Dehling <e.e.dehling@gmail.com>
#
# License: BSD 3 clause

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.model_selection import KFold
from ..externals import six
from ..externals.name_estimators import _name_estimators
import numpy as np


class StackingCVRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5, use_features_in_secondary=False):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.named_regressors = {key: value for
                                 key, value in
                                 _name_estimators(regressors)}
        self.named_meta_regressor = {'meta-%s' % key: value for
                                     key, value in
                                     _name_estimators([meta_regressor])}
        self.n_folds = n_folds
        self.use_features_in_secondary = use_features_in_secondary

    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        meta_features = np.zeros((X.shape[0], len(self.regressors)))

        #
        # The outer loop iterates over the base-regressors. Each regressor
        # is trained n_folds times and makes predictions, after which we train
        # the meta-regressor on their combined results.
        #
        for i, regr in enumerate(self.regressors):
            #
            # In the inner loop, each model is trained n_folds times on the
            # training-part of this fold of data; and the holdout-part of data
            # is used for predictions. This is repeated n_folds times, so in
            # the end we have predictions for each data point.
            #
            # Advantage of this complex approach is that data points we're
            # predicting have not been trained on by the algorithm, so it's
            # less susceptible to overfitting.
            #
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(regr)
                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                meta_features[holdout_idx, i] = y_pred

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
        #
        # First we make predictions with the base-models then we predict with
        # the meta-model from that info.
        #
        meta_features = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])

        if self.use_features_in_secondary:
            return self.meta_regr_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_regr_.predict(meta_features)

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
            return out
