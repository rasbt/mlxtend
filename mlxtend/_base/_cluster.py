# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Base Clusteer (Clutering Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from time import time


class _Cluster(object):

    def __init__(self):
        pass

    def fit(self, X, init_params=True):
        """Learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        init_params : bool (default: True)
            Re-initializes model parameters prior to fitting.
            Set False to continue training with weights from
            a previous model fitting.

        Returns
        -------
        self : object

        """
        self._is_fitted = False
        self._check_arrays(X=X)
        if hasattr(self, 'self.random_seed') and self.random_seed:
            self._rgen = np.random.RandomState(self.random_seed)
        self._init_time = time()
        self._fit(X=X, init_params=init_params)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict targets from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        target_values : array-like, shape = [n_samples]
          Predicted target values.

        """
        self._check_arrays(X=X)
        if not self._is_fitted:
            raise AttributeError('Model is not fitted, yet.')
        return self._predict(X)
