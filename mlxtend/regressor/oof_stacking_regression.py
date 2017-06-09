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

from sklearn.base import BaseEstimator as BaseEst
from sklearn.base import RegressorMixin as RegMix
from sklearn.base import TransformerMixin as TFMix
from sklearn.base import clone
from sklearn.model_selection import KFold
from ..externals import six
from ..externals.name_estimators import _name_estimators
import numpy as np


class OutOfFoldStackingRegressor(BaseEst, RegMix, TFMix):
    def __init__(self, regressors, meta_regressor, n_folds=5):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.named_regressors = {key: value for
                                 key, value in
                                 _name_estimators(regressors)}
        self.named_meta_regressor = {'meta-%s' % key: value for
                                     key, value in
                                     _name_estimators([meta_regressor])}
        self.n_folds = n_folds

    def fit(self, X, y):
        self.regr_ = [list() for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        #
        # The outer loop iterates over the base-regressors. Each regressor
        # is trained and makes predictions, after which we train the
        # meta-regressor on their combined results.
        #
        for i, clf in enumerate(self.regressors):
            #
            # In the inner loop, each model is trained n_folds times on the
            # training-part of this fold of data; and the holdout-part of data
            # is used for predictions. This is repeated n_folds times, so in
            # the end we have predictions for each data point.
            #
            # Advantage of this complex approach is that data points we're
            # predicting for have not been seen by the algorithm, so it's less
            # susceptible to overfitting.
            #
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(clf)
                self.regr_[i].append(instance)

                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                out_of_fold_predictions[holdout_idx, i] = y_pred

        self.meta_regr_.fit(out_of_fold_predictions, y)

        return self

    def predict(self, X):
        #
        # First we make predictions with the base-models (n_folds times per
        # model, averaged) then we predict with the meta-model from that info.
        #
        meta_features = np.column_stack([
            np.column_stack([r.predict(X) for r in regrs]).mean(axis=1)
            for regrs in self.regr_
        ])

        return self.meta_regr_.predict(meta_features)

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support."""
        if not deep:
            return super(OutOfFoldStackingRegressor, self)\
                .get_params(deep=False)
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
