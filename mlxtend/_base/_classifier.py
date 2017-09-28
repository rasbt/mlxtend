# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Base Clusteer (Clutering Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from time import time


class _Classifier(object):

    def __init__(self):
        pass

    def _check_target_array(self, y, allowed=None):
        if not np.issubdtype(y[0], int):
            raise AttributeError('y must be an integer array.\nFound %s'
                                 % y.dtype)
        found_labels = np.unique(y)
        if (found_labels < 0).any():
            raise AttributeError('y array must not contain negative labels.'
                                 '\nFound %s' % found_labels)
        if allowed is not None:
            found_labels = tuple(found_labels)
            if found_labels not in allowed:
                raise AttributeError('Labels not in %s.\nFound %s'
                                     % (allowed, found_labels))

    def score(self, X, y):
        """ Compute the prediction accuracy

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values (true class labels).

        Returns
        ---------
        acc : float
            The prediction accuracy as a float
            between 0.0 and 1.0 (perfect score).

        """
        y_pred = self.predict(X)
        acc = np.sum(y == y_pred, axis=0) / float(X.shape[0])
        return acc

    def fit(self, X, y, init_params=True):
        """Learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        init_params : bool (default: True)
            Re-initializes model parameters prior to fitting.
            Set False to continue training with weights from
            a previous model fitting.

        Returns
        -------
        self : object

        """
        self._is_fitted = False
        self._check_arrays(X=X, y=y)
        self._check_target_array(y)
        if hasattr(self, 'self.random_seed') and self.random_seed:
            self._rgen = np.random.RandomState(self.random_seed)
        self._init_time = time()
        self._fit(X=X, y=y, init_params=init_params)
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
